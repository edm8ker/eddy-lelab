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
"""Record a dataset for an arm that the browser drives (browser mode).

The page reads the arm and its cameras and sends each frame over
/ws/browser-recording. This server writes them with lerobot's own dataset
writer and uploads the result, privately, to the Hugging Face account it is
logged in to, so the arm can be plugged into a different computer than the one
running this server.

Each connection has its own dataset and shares no state with lelab/record.py,
so several browsers can record at once. See handle_browser_recording for the
protocol.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import re
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.feature_utils import hw_to_dataset_features

from .record import _apply_deleted_episodes
from .utils.config import with_lelab_tag
from .utils.hf_auth import cached_whoami

logger = logging.getLogger(__name__)

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# SOFollower.name: what LeLab's own recordings store as robot_type.
ROBOT_TYPE = "so_follower"
DEFAULT_FPS = 30
MAX_FPS = 60
MAX_CAMERAS = 4
MAX_CAMERA_SIDE = 1920
# Hugging Face's repo name rules.
MAX_NAME_LENGTH = 96
DATASET_NAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
CAMERA_NAME = re.compile(r"^[A-Za-z0-9_]+$")
IMAGE_WRITER_THREADS_PER_CAMERA = 4

_HEADER_LENGTH = struct.Struct("<I")


class BrowserRecordingError(Exception):
    """A problem to report to the page, ending the session."""


def _is_side(value: Any) -> bool:
    return type(value) is int and 0 < value <= MAX_CAMERA_SIDE


def parse_start(message: dict[str, Any]) -> tuple[str, str, int, dict[str, tuple[int, int]]]:
    """Check a start message. Returns (dataset name, task, fps, {camera: (height, width)})."""
    name = message.get("dataset_name")
    if (
        not isinstance(name, str)
        or len(name) > MAX_NAME_LENGTH
        or not DATASET_NAME.match(name)
        or ".." in name
        or "--" in name
    ):
        raise BrowserRecordingError(
            "dataset_name must be letters, digits, '-', '_' or '.', "
            f"at most {MAX_NAME_LENGTH} characters, without '--' or '..'."
        )
    task = message.get("task")
    if not isinstance(task, str) or not task.strip():
        raise BrowserRecordingError("task must describe what the arm is doing.")
    fps = message.get("fps", DEFAULT_FPS)
    if type(fps) is not int or not 0 < fps <= MAX_FPS:
        raise BrowserRecordingError(f"fps must be a whole number from 1 to {MAX_FPS}.")

    raw_cameras = message.get("cameras", {})
    if not isinstance(raw_cameras, dict) or len(raw_cameras) > MAX_CAMERAS:
        raise BrowserRecordingError(f"cameras must map up to {MAX_CAMERAS} names to their sizes.")
    cameras: dict[str, tuple[int, int]] = {}
    for camera, size in raw_cameras.items():
        if (
            not CAMERA_NAME.match(camera)
            or not isinstance(size, dict)
            or not _is_side(size.get("height"))
            or not _is_side(size.get("width"))
        ):
            raise BrowserRecordingError(f"Camera {camera!r} needs a plain name, a width and a height.")
        cameras[camera] = (size["height"], size["width"])
    return name, task.strip(), fps, cameras


def _joint_values(header: dict[str, Any], key: str) -> np.ndarray:
    values = header.get(key)
    if not isinstance(values, dict) or any(
        not isinstance(values.get(motor), int | float) or not math.isfinite(values[motor]) for motor in MOTORS
    ):
        raise BrowserRecordingError(f"Frame {key} must give a number for each of {list(MOTORS)}.")
    return np.array([values[motor] for motor in MOTORS], dtype=np.float32)


def decode_frame(
    payload: bytes, cameras: dict[str, tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Unpack one frame from the page.

    Layout: a little-endian uint32 header length, the UTF-8 JSON header
    {"state": {motor: value}, "action": {motor: value}, "images": [{"name",
    "size"}]}, then each camera's JPEG bytes in header order. Returns (state,
    action, RGB frames). The joints are in lerobot's units: degrees, and 0-100
    for the gripper.
    """
    import cv2

    if len(payload) < _HEADER_LENGTH.size:
        raise BrowserRecordingError("Frame message is too short.")
    (header_length,) = _HEADER_LENGTH.unpack_from(payload)
    body_start = _HEADER_LENGTH.size + header_length
    if body_start > len(payload):
        raise BrowserRecordingError("Frame header is truncated.")
    try:
        header = json.loads(payload[_HEADER_LENGTH.size : body_start].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise BrowserRecordingError("Frame header is not valid JSON.") from e
    if not isinstance(header, dict):
        raise BrowserRecordingError("Frame header is not valid JSON.")

    state = _joint_values(header, "state")
    action = _joint_values(header, "action")

    images: dict[str, np.ndarray] = {}
    offset = body_start
    for entry in header.get("images") or []:
        name, size = entry.get("name"), entry.get("size")
        if name not in cameras or type(size) is not int or size < 0 or offset + size > len(payload):
            raise BrowserRecordingError(f"Unexpected or truncated camera image {name!r}.")
        bgr = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8, count=size, offset=offset), cv2.IMREAD_COLOR
        )
        offset += size
        if bgr is None:
            raise BrowserRecordingError(f"Camera {name!r} sent an image that isn't a valid JPEG.")
        height, width = cameras[name]
        if bgr.shape[:2] != (height, width):
            raise BrowserRecordingError(
                f"Camera {name!r} sent {bgr.shape[1]}x{bgr.shape[0]}, the recording expects {width}x{height}."
            )
        # lerobot's OpenCVCamera records RGB.
        images[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    missing = set(cameras) - set(images)
    if missing:
        raise BrowserRecordingError(f"Frame is missing camera(s): {', '.join(sorted(missing))}.")
    return state, action, images


class BrowserRecording:
    """One browser's dataset, from its start message to its upload."""

    def __init__(self, repo_id: str, task: str, fps: int, cameras: dict[str, tuple[int, int]]) -> None:
        self.repo_id = repo_id
        self.task = task
        self.cameras = cameras
        self.root = Path(HF_LEROBOT_HOME) / repo_id
        self.closed = False
        # Dataset episode indices in recording order, minus the deleted ones.
        self.kept: list[int] = []
        self.deleted: set[int] = set()

        if self.root.exists():
            raise BrowserRecordingError(f"A dataset named {repo_id} already exists on this server.")
        motors: dict[str, Any] = {f"{motor}.pos": float for motor in MOTORS}
        shapes = {camera: (height, width, 3) for camera, (height, width) in cameras.items()}
        # Not streaming_encoding: its encoder drops a frame when its queue
        # stays full, which suits a live camera but not this socket, where
        # frames can arrive in a burst after the network falls behind. A
        # dropped frame would put the video out of step with the joints.
        self.dataset: LeRobotDataset | None = LeRobotDataset.create(
            repo_id,
            fps,
            root=self.root,
            robot_type=ROBOT_TYPE,
            features={
                **hw_to_dataset_features(motors, "action", True),
                **hw_to_dataset_features({**motors, **shapes}, "observation", True),
            },
            use_videos=True,
            image_writer_threads=IMAGE_WRITER_THREADS_PER_CAMERA * len(cameras),
            streaming_encoding=False,
        )

    @property
    def saved_episodes(self) -> int:
        return len(self.kept)

    def add_frame(self, payload: bytes) -> None:
        state, action, images = decode_frame(payload, self.cameras)
        self.dataset.add_frame(
            {
                "action": action,
                "observation.state": state,
                **{f"observation.images.{name}": image for name, image in images.items()},
                "task": self.task,
            }
        )

    def _save_pending(self) -> int:
        """Save the frames sent so far as an episode. Returns how many there were."""
        if not self.dataset.has_pending_frames():
            return 0
        frames = self.dataset.writer.episode_buffer["size"]
        self.dataset.save_episode()
        self.kept.append(self.dataset.meta.total_episodes - 1)
        return frames

    def save_episode(self) -> dict[str, Any]:
        frames = self._save_pending()
        if not frames:
            return {"type": "episode_empty", "saved_episodes": self.saved_episodes}
        return {"type": "episode_saved", "frames": frames, "saved_episodes": self.saved_episodes}

    def discard_episode(self) -> dict[str, Any]:
        if self.dataset.has_pending_frames():
            self.dataset.clear_episode_buffer()
        return {"type": "episode_discarded", "saved_episodes": self.saved_episodes}

    def delete_episode(self, episode: Any) -> dict[str, Any]:
        """Delete a saved episode (1-based, as shown: episode 3 = the third
        kept one). The files are edited when the session ends, as lerobot can
        only edit a finalized dataset."""
        if type(episode) is not int or not 1 <= episode <= self.saved_episodes:
            raise BrowserRecordingError(f"Episode {episode} hasn't been saved.")
        self.deleted.add(self.kept.pop(episode - 1))
        return {"type": "episode_deleted", "episode": episode, "saved_episodes": self.saved_episodes}

    def _close(self) -> None:
        self.closed = True
        self.dataset.finalize()
        if self.dataset.meta.total_episodes == 0:
            shutil.rmtree(self.root, ignore_errors=True)
            self.dataset = None
        elif self.deleted:
            self.dataset = _apply_deleted_episodes(self.dataset, sorted(self.deleted))

    def finish(self) -> None:
        """Keep the frames sent so far as a last episode and finish the files."""
        self._save_pending()
        self._close()

    def abandon(self) -> None:
        """The page went away: keep the saved episodes on this server, without
        the unfinished one, and don't upload. /upload-dataset can do that later."""
        if self.dataset.has_pending_frames():
            self.dataset.clear_episode_buffer()
        self._close()

    def upload(self) -> None:
        self.dataset.push_to_hub(tags=with_lelab_tag(None), private=True)


def _parse_text(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise BrowserRecordingError("Text messages must be JSON.") from e
    if not isinstance(data, dict):
        raise BrowserRecordingError("Text messages must be JSON objects.")
    return data


def _account_name() -> str:
    info = cached_whoami()
    if not info or not info.get("name"):
        raise BrowserRecordingError(
            "This server isn't logged in to Hugging Face, so it can't save recordings."
        )
    return info["name"]


async def _start_session(message: dict[str, Any]) -> BrowserRecording:
    if message.get("type") != "start":
        raise BrowserRecordingError('The first message must be {"type": "start", ...}.')
    name, task, fps, cameras = parse_start(message)
    # The server picks the account, so a page can only write to this server's.
    account = await asyncio.to_thread(_account_name)
    return await asyncio.to_thread(BrowserRecording, f"{account}/{name}", task, fps, cameras)


async def _finish_session(websocket: WebSocket, session: BrowserRecording) -> None:
    await websocket.send_json({"type": "phase", "phase": "saving"})
    await asyncio.to_thread(session.finish)
    uploaded = False
    upload_error = None
    if session.dataset is not None:
        await websocket.send_json({"type": "phase", "phase": "uploading"})
        try:
            await asyncio.to_thread(session.upload)
            uploaded = True
        except Exception as e:
            # The recording is safe on this server: report it, don't lose it.
            logger.exception("Could not upload %s", session.repo_id)
            upload_error = str(e)
    await websocket.send_json(
        {
            "type": "finished",
            "dataset_repo_id": session.repo_id,
            "saved_episodes": session.saved_episodes,
            "uploaded": uploaded,
            "upload_error": upload_error,
        }
    )
    await websocket.close()


async def handle_browser_recording(websocket: WebSocket) -> None:
    """Serve one browser recording session over an accepted-on-entry WebSocket.

    Protocol: the page sends {"type": "start", "dataset_name", "task", "fps",
    "cameras": {name: {"width", "height"}}}; the server answers {"type":
    "ready", "dataset_repo_id"}. Each binary message is a frame of the current
    episode (see decode_frame). Then, each answered once the frames sent before
    it are done:

    - {"type": "save_episode"}: {"type": "episode_saved", "frames",
      "saved_episodes"}, or {"type": "episode_empty", "saved_episodes"}
    - {"type": "discard_episode"}: {"type": "episode_discarded", "saved_episodes"}
    - {"type": "delete_episode", "episode"}: {"type": "episode_deleted",
      "episode", "saved_episodes"}
    - {"type": "finish"}: {"type": "phase", "phase": "saving" | "uploading"},
      then {"type": "finished", "dataset_repo_id", "saved_episodes", "uploaded",
      "upload_error"}. Frames still unsaved become the last episode.

    Closing the socket without finish keeps the saved episodes on this server
    and uploads nothing. Problems arrive as {"type": "error"} and end the session.
    """
    await websocket.accept()
    session: BrowserRecording | None = None
    try:
        first = await websocket.receive()
        if first["type"] == "websocket.disconnect":
            return
        if first.get("text") is None:
            raise BrowserRecordingError('The first message must be {"type": "start", ...}.')
        session = await _start_session(_parse_text(first["text"]))
        logger.info("Browser recording started: %s", session.repo_id)
        await websocket.send_json({"type": "ready", "dataset_repo_id": session.repo_id})

        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("text") is None:
                await asyncio.to_thread(session.add_frame, message.get("bytes") or b"")
                continue
            command = _parse_text(message["text"])
            kind = command.get("type")
            if kind == "save_episode":
                await websocket.send_json(await asyncio.to_thread(session.save_episode))
            elif kind == "discard_episode":
                await websocket.send_json(await asyncio.to_thread(session.discard_episode))
            elif kind == "delete_episode":
                await websocket.send_json(session.delete_episode(command.get("episode")))
            elif kind == "finish":
                await _finish_session(websocket, session)
                break
            else:
                raise BrowserRecordingError(f"Unknown message type {kind!r}.")
    except WebSocketDisconnect:
        pass
    except BrowserRecordingError as e:
        await _send_error(websocket, str(e))
    except Exception as e:
        logger.exception("Browser recording failed")
        await _send_error(websocket, f"Recording failed on the server: {e}")
    finally:
        if session is not None and not session.closed:
            try:
                await asyncio.to_thread(session.abandon)
            except Exception:
                logger.exception("Could not close the browser recording %s", session.repo_id)


async def _send_error(websocket: WebSocket, message: str) -> None:
    with contextlib.suppress(Exception):
        await websocket.send_json({"type": "error", "message": message})
        await websocket.close()
