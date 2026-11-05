# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""/ws/browser-recording, writing real datasets with lerobot's writer. Only the
Hugging Face calls are replaced."""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from lelab import browser_recording
from lelab.browser_recording import MOTORS, BrowserRecordingError, decode_frame, parse_start
from lerobot.datasets import LeRobotDataset

ACCOUNT = "shared-account"
WIDTH, HEIGHT = 128, 96
START = {
    "type": "start",
    "dataset_name": "my_arm_k3f9x2",
    "task": "Pick up the cube",
    "cameras": {"top": {"width": WIDTH, "height": HEIGHT}},
}


def joints(value: float) -> dict[str, float]:
    return dict.fromkeys(MOTORS, value)


def jpeg(width: int = WIDTH, height: int = HEIGHT) -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.full((height, width, 3), 128, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def frame_message(value: float, images: dict[str, bytes] | None = None) -> bytes:
    images = {"top": jpeg()} if images is None else images
    header = json.dumps(
        {
            "state": joints(value),
            "action": joints(value + 1),
            "images": [{"name": name, "size": len(data)} for name, data in images.items()],
        }
    ).encode()
    return struct.pack("<I", len(header)) + header + b"".join(images.values())


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(browser_recording, "HF_LEROBOT_HOME", tmp_path)
    monkeypatch.setattr(browser_recording, "cached_whoami", lambda: {"name": ACCOUNT})
    return tmp_path


@pytest.fixture
def push(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    spy = MagicMock(name="push_to_hub")
    monkeypatch.setattr(LeRobotDataset, "push_to_hub", spy)
    return spy


def open_dataset(home: Path) -> LeRobotDataset:
    # pyav, as torchcodec needs a shared FFmpeg that not every machine has.
    return LeRobotDataset(
        f"{ACCOUNT}/my_arm_k3f9x2", root=home / ACCOUNT / "my_arm_k3f9x2", video_backend="pyav"
    )


def start(ws) -> dict:
    ws.send_text(json.dumps(START))
    return ws.receive_json()


def record_episode(ws, value: float, frames: int = 3) -> dict:
    for _ in range(frames):
        ws.send_bytes(frame_message(value))
    ws.send_text(json.dumps({"type": "save_episode"}))
    return ws.receive_json()


def finish(ws) -> list[dict]:
    ws.send_text(json.dumps({"type": "finish"}))
    replies = []
    while not replies or replies[-1]["type"] != "finished":
        replies.append(ws.receive_json())
    return replies


@pytest.mark.parametrize(
    "change",
    [
        {"dataset_name": "../escape"},
        {"dataset_name": "a/b"},
        {"dataset_name": "two--dashes"},
        {"dataset_name": "two..dots"},
        {"dataset_name": "x" * 97},
        {"dataset_name": None},
        {"task": "  "},
        {"fps": 0},
        {"fps": 30.5},
        {"cameras": {"../top": {"width": WIDTH, "height": HEIGHT}}},
        {"cameras": {"top": {"width": 0, "height": HEIGHT}}},
    ],
)
def test_parse_start_rejects(change: dict) -> None:
    with pytest.raises(BrowserRecordingError):
        parse_start({**START, **change})


def test_parse_start_defaults() -> None:
    assert parse_start(START) == ("my_arm_k3f9x2", "Pick up the cube", 30, {"top": (HEIGHT, WIDTH)})


def test_decode_frame() -> None:
    state, action, images = decode_frame(frame_message(5.0), {"top": (HEIGHT, WIDTH)})
    assert state.tolist() == [5.0] * 6
    assert action.tolist() == [6.0] * 6
    assert images["top"].shape == (HEIGHT, WIDTH, 3)


@pytest.mark.parametrize(
    "payload, cameras",
    [
        (b"", {}),
        (struct.pack("<I", 99) + b"{}", {}),
        (struct.pack("<I", 2) + b"[]", {}),
        (frame_message(float("nan"), {}), {}),
        (frame_message(1.0, {}), {"top": (HEIGHT, WIDTH)}),
        (frame_message(1.0, {"other": jpeg()}), {"top": (HEIGHT, WIDTH)}),
        (frame_message(1.0, {"top": b"not a jpeg"}), {"top": (HEIGHT, WIDTH)}),
        (frame_message(1.0, {"top": jpeg(64, 48)}), {"top": (HEIGHT, WIDTH)}),
    ],
)
def test_decode_frame_rejects(payload: bytes, cameras: dict) -> None:
    with pytest.raises(BrowserRecordingError):
        decode_frame(payload, cameras)


def test_records_and_uploads_privately(client: TestClient, home: Path, push: MagicMock) -> None:
    with client.websocket_connect("/ws/browser-recording") as ws:
        assert start(ws) == {"type": "ready", "dataset_repo_id": f"{ACCOUNT}/my_arm_k3f9x2"}
        assert record_episode(ws, 10.0)["frames"] == 3
        replies = finish(ws)

    assert [r.get("phase") for r in replies[:-1]] == ["saving", "uploading"]
    assert replies[-1] == {
        "type": "finished",
        "dataset_repo_id": f"{ACCOUNT}/my_arm_k3f9x2",
        "saved_episodes": 1,
        "uploaded": True,
        "upload_error": None,
    }
    assert push.call_args.kwargs["private"] is True

    dataset = open_dataset(home)
    assert dataset.num_episodes == 1
    assert dataset.num_frames == 3
    assert dataset.meta.robot_type == "so_follower"
    assert dataset.meta.fps == 30
    item = dataset[0]
    assert item["observation.state"].tolist() == [10.0] * 6
    assert item["action"].tolist() == [11.0] * 6
    assert item["task"] == "Pick up the cube"
    assert item["observation.images.top"].shape == (3, HEIGHT, WIDTH)


def test_discard_and_delete_episodes(client: TestClient, home: Path, push: MagicMock) -> None:
    with client.websocket_connect("/ws/browser-recording") as ws:
        start(ws)
        record_episode(ws, 1.0)
        ws.send_bytes(frame_message(2.0))
        ws.send_text(json.dumps({"type": "discard_episode"}))
        assert ws.receive_json() == {"type": "episode_discarded", "saved_episodes": 1}
        record_episode(ws, 3.0)
        record_episode(ws, 4.0)
        ws.send_text(json.dumps({"type": "delete_episode", "episode": 2}))
        assert ws.receive_json() == {"type": "episode_deleted", "episode": 2, "saved_episodes": 2}
        assert finish(ws)[-1]["saved_episodes"] == 2

    dataset = open_dataset(home)
    assert dataset.num_episodes == 2
    assert [dataset[i]["observation.state"][0].item() for i in (0, 3)] == [1.0, 4.0]


def test_finish_keeps_unsaved_frames(client: TestClient, home: Path, push: MagicMock) -> None:
    with client.websocket_connect("/ws/browser-recording") as ws:
        start(ws)
        ws.send_bytes(frame_message(1.0))
        assert finish(ws)[-1]["saved_episodes"] == 1


def test_finish_without_episodes_leaves_nothing(client: TestClient, home: Path, push: MagicMock) -> None:
    with client.websocket_connect("/ws/browser-recording") as ws:
        start(ws)
        ws.send_text(json.dumps({"type": "save_episode"}))
        assert ws.receive_json() == {"type": "episode_empty", "saved_episodes": 0}
        replies = finish(ws)
    assert replies[-1]["uploaded"] is False
    assert not (home / ACCOUNT / "my_arm_k3f9x2").exists()
    push.assert_not_called()


def test_failed_upload_keeps_the_recording(client: TestClient, home: Path, push: MagicMock) -> None:
    push.side_effect = RuntimeError("Hub is down")
    with client.websocket_connect("/ws/browser-recording") as ws:
        start(ws)
        record_episode(ws, 1.0)
        reply = finish(ws)[-1]
    assert reply["uploaded"] is False
    assert reply["upload_error"] == "Hub is down"
    assert (home / ACCOUNT / "my_arm_k3f9x2" / "meta" / "info.json").exists()


def test_disconnect_keeps_saved_episodes_without_uploading(
    client: TestClient, home: Path, push: MagicMock
) -> None:
    with client.websocket_connect("/ws/browser-recording") as ws:
        start(ws)
        record_episode(ws, 1.0)
        ws.send_bytes(frame_message(2.0))
        # So the frame above is handled before the socket closes.
        ws.send_text(json.dumps({"type": "delete_episode", "episode": 9}))
        assert ws.receive_json()["type"] == "error"

    # The server closes the dataset on a worker thread once the session ends.
    episodes = home / ACCOUNT / "my_arm_k3f9x2" / "meta" / "episodes"
    deadline = time.monotonic() + 10
    while not any(episodes.rglob("*.parquet")) and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.2)

    push.assert_not_called()
    dataset = open_dataset(home)
    assert dataset.num_episodes == 1
    assert dataset.num_frames == 3


def test_two_sessions_at_once(client: TestClient, home: Path, push: MagicMock) -> None:
    with client.websocket_connect("/ws/browser-recording") as first:
        start(first)
        with client.websocket_connect("/ws/browser-recording") as second:
            second.send_text(json.dumps({**START, "dataset_name": "other_arm_a1b2c3"}))
            assert second.receive_json()["type"] == "ready"
            record_episode(second, 1.0)
            assert finish(second)[-1]["uploaded"] is True
        record_episode(first, 2.0)
        assert finish(first)[-1]["uploaded"] is True
    assert (home / ACCOUNT / "other_arm_a1b2c3" / "meta" / "info.json").exists()
    assert (home / ACCOUNT / "my_arm_k3f9x2" / "meta" / "info.json").exists()


def test_existing_name_is_refused(client: TestClient, home: Path, push: MagicMock) -> None:
    (home / ACCOUNT / "my_arm_k3f9x2").mkdir(parents=True)
    with client.websocket_connect("/ws/browser-recording") as ws:
        reply = start(ws)
    assert reply["type"] == "error"
    assert "already exists" in reply["message"]


def test_server_not_logged_in(client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(browser_recording, "cached_whoami", lambda: None)
    with client.websocket_connect("/ws/browser-recording") as ws:
        reply = start(ws)
    assert reply["type"] == "error"
    assert "logged in" in reply["message"]


def test_first_message_must_be_start(client: TestClient, home: Path) -> None:
    with client.websocket_connect("/ws/browser-recording") as ws:
        ws.send_bytes(frame_message(1.0))
        assert ws.receive_json()["type"] == "error"
