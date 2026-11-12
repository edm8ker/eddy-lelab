"""Start, watch and stop the cloud GPU that runs a policy for the arm.

The cloud side is a Hugging Face Job running remote_policy_server.py in the
lerobot GPU image, with its port exposed through the HF jobs proxy (reachable
only with the account's token). LeLab keeps at most one such server; the arm
loop (remote_rollout.py) talks to it once it reports ready.

Picking another checkpoint while the server is on swaps the model on the same
machine (POST /load) instead of starting a new one, which skips the queue and
machine start.

States reported to the app: off, starting (waiting for / starting the
machine), loading (model downloading or loading), ready, error; plus "steps"
for the checklist (wait for a GPU, start the machine, download the model,
load it), with "last time" durations.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# L4: enough GPU memory (24 GB) and RAM (30 GB; loading pi0.5 briefly needs
# ~19 GB) for every policy LeLab trains, at the lowest price that fits.
DEFAULT_FLAVOR = "l4x1"
PRICE_PER_HOUR = {"l4x1": 0.80, "a10g-large": 1.50, "l40sx1": 1.80, "a100-large": 2.50}
IMAGE = "huggingface/lerobot-gpu:latest"
PORT = 8000
# The server turns itself off after this long without arm requests, and the
# job has a hard limit on top, so a forgotten session can't keep billing.
IDLE_S = 600
JOB_TIMEOUT = "3h"

_lock = threading.Lock()
_server: dict[str, Any] = {}


def _hf():
    from huggingface_hub import HfApi, get_token

    return HfApi(), get_token()


def _model_location(policy_ref: str) -> tuple[str, str]:
    """(repo, subfolder) of a cloud training's checkpoint ref."""
    from .rollout import _HUB_REF_RE, _HUB_ROOT_REF_RE

    m = _HUB_REF_RE.match(policy_ref)
    if m:
        return m.group("repo"), f"checkpoints/{m.group('step_dir')}/pretrained_model"
    m = _HUB_ROOT_REF_RE.match(policy_ref)
    if m:
        return m.group("repo"), ""
    raise ValueError("Only trainings saved on Hugging Face can run on the cloud GPU.")


def _call(url: str, token: str, path: str, body: dict | None = None, timeout: float = 8) -> dict[str, Any] | None:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url + path, data=data, method="POST" if body is not None else "GET")
    req.add_header("Authorization", f"Bearer {token}")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


# ---- "last time" durations for the checklist ------------------------------


def _timings_file() -> Path:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return Path(HF_LEROBOT_HOME) / "lelab_cloud_gpu_timings.json"


def _load_timings() -> dict[str, Any]:
    try:
        return json.loads(_timings_file().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_timing(key: str, step: str, seconds: float) -> None:
    try:
        timings = _load_timings()
        timings.setdefault(key, {})[step] = round(seconds, 1)
        _timings_file().write_text(json.dumps(timings, indent=1), encoding="utf-8")
    except Exception:
        pass


# ---- start / swap / stop ---------------------------------------------------


def start(policy_ref: str, flavor: str = DEFAULT_FLAVOR) -> dict[str, Any]:
    repo, subfolder = _model_location(policy_ref)
    with _lock:
        running = dict(_server)
    if running.get("job_id") and running.get("flavor") == flavor:
        if running.get("policy_ref") == policy_ref:
            return status()
        # Same machine, another model: swap it there (no queue, no machine
        # start). If the server isn't reachable yet, status() sends the swap
        # once it is.
        with _lock:
            _server.update(policy_ref=policy_ref, swap_requested=time.time(), swap_sent=False)
        logger.info("Cloud GPU: switching model to %s", policy_ref)
        return status()
    if running.get("job_id"):
        stop()

    api, token = _hf()
    code = (Path(__file__).parent / "remote_policy_server.py").read_text(encoding="utf-8")
    job = api.run_job(
        image=IMAGE,
        command=["python", "-u", "-c", code],
        env={"MODEL_REPO": repo, "MODEL_SUBFOLDER": subfolder, "IDLE_S": str(IDLE_S), "PORT": str(PORT)},
        secrets={"HF_TOKEN": token},
        flavor=flavor,
        expose=[PORT],
        timeout=JOB_TIMEOUT,
        labels={"lelab": "remote-policy"},
    )
    with _lock:
        _server.clear()
        _server.update(
            job_id=job.id,
            job_url=job.url,
            policy_ref=policy_ref,
            flavor=flavor,
            started_at=time.time(),
            url=None,
            running_at=None,
            reachable_at=None,
            swap_sent=True,  # the first model is loaded from the job's settings
        )
    logger.info("Cloud GPU requested: job %s (%s) for %s", job.id, flavor, policy_ref)
    return status()


def stop() -> dict[str, Any]:
    with _lock:
        job_id = _server.get("job_id")
        _server.clear()
    if job_id:
        api, _ = _hf()
        try:
            api.cancel_job(job_id=job_id)
            logger.info("Cloud GPU stopped: job %s", job_id)
        except Exception as exc:
            logger.warning("Could not cancel cloud GPU job %s: %s", job_id, exc)
    return {"state": "off"}


# ---- status ----------------------------------------------------------------


def _steps(server: dict, stage: str, health: dict | None, now: float) -> list[dict[str, Any]]:
    """Checklist: wait for a GPU, start the machine, download, load, ready."""
    timings = _load_timings()
    machine_last = timings.get(f"machine:{server['flavor']}", {})
    model_last = timings.get(server["policy_ref"], {})
    swapped = server.get("swap_requested") is not None

    def step(step_id, label, state, seconds=None, last=None, detail=None):
        item = {"id": step_id, "label": label, "state": state}
        if seconds is not None:
            item["seconds"] = round(seconds, 1)
        if last is not None:
            item["last_seconds"] = last
        if detail:
            item["detail"] = detail
        return item

    out = []
    if not swapped:
        running_at, reachable_at = server.get("running_at"), server.get("reachable_at")
        if stage != "RUNNING":
            out.append(step("queue", "Wait for a free GPU on Hugging Face", "active",
                            now - server["started_at"], machine_last.get("queue")))
        else:
            out.append(step("queue", "Wait for a free GPU on Hugging Face", "done",
                            (running_at or now) - server["started_at"]))
        if stage != "RUNNING":
            out.append(step("machine", "Start the cloud machine", "pending"))
        elif reachable_at is None:
            out.append(step("machine", "Start the cloud machine", "active",
                            now - (running_at or now), machine_last.get("machine")))
        else:
            out.append(step("machine", "Start the cloud machine", "done", reachable_at - (running_at or reachable_at)))

    phase = (health or {}).get("phase")
    times = (health or {}).get("times") or {}
    model_matches = (health or {}).get("model") == _model_key(server["policy_ref"])
    elapsed = (health or {}).get("phase_elapsed")
    if health is None or not model_matches or phase == "starting":
        download_state, load_state = "pending", "pending"
    elif phase == "downloading":
        download_state, load_state = "active", "pending"
    elif phase == "loading":
        download_state, load_state = "done", "active"
    elif phase == "ready":
        download_state, load_state = "done", "done"
    else:
        download_state, load_state = "done", "pending"

    detail = None
    progress = (health or {}).get("download") or {}
    if download_state == "active" and progress.get("total"):
        detail = f"{progress['done'] / 1e9:.1f} of {progress['total'] / 1e9:.1f} GB"
    out.append(step("download", "Download the model", download_state,
                    elapsed if download_state == "active" else (times.get("download") if download_state == "done" else None),
                    model_last.get("download"), detail))
    out.append(step("load", "Load the AI on the GPU", load_state,
                    elapsed if load_state == "active" else (times.get("load") if load_state == "done" else None),
                    model_last.get("load")))
    return out


def _model_key(policy_ref: str) -> str:
    repo, sub = _model_location(policy_ref)
    return f"{repo}@{sub}"


def status() -> dict[str, Any]:
    with _lock:
        server = dict(_server)
    if not server.get("job_id"):
        return {"state": "off"}

    now = time.time()
    base = {
        "policy_ref": server["policy_ref"],
        "flavor": server["flavor"],
        "price_per_hour": PRICE_PER_HOUR.get(server["flavor"]),
        "elapsed_s": round(now - server["started_at"]),
        "job_url": server.get("job_url"),
    }
    api, token = _hf()
    try:
        info = api.inspect_job(job_id=server["job_id"])
    except Exception as exc:
        return {**base, "state": "starting", "message": f"Could not reach Hugging Face: {exc}", "steps": []}

    stage = info.status.stage
    if stage in ("ERROR", "COMPLETED", "CANCELED", "DELETED"):
        with _lock:
            if _server.get("job_id") == server["job_id"]:
                _server.clear()
        if stage == "ERROR":
            return {**base, "state": "error", "message": info.status.message or "The cloud GPU stopped with an error."}
        # COMPLETED: the server turned itself off after being idle.
        return {"state": "off", "message": "The cloud GPU turned itself off."}

    url = (getattr(info.status, "expose_urls", None) or [None])[0]
    health = None
    if stage == "RUNNING":
        with _lock:
            if _server.get("job_id") == server["job_id"] and _server.get("running_at") is None:
                _server["running_at"] = now
                _save_timing(f"machine:{server['flavor']}", "queue", now - server["started_at"])
            server = dict(_server)
        if url:
            health = _call(url, token, "/health")
            with _lock:
                if _server.get("job_id") == server["job_id"]:
                    _server["url"] = url
                    if health is not None and _server.get("reachable_at") is None:
                        _server["reachable_at"] = now
                        _save_timing(f"machine:{server['flavor']}", "machine", now - (_server["running_at"] or now))
                server = dict(_server)
            # A model switch asked for before the server could be reached.
            if health is not None and not server.get("swap_sent"):
                repo, sub = _model_location(server["policy_ref"])
                if _call(url, token, "/load", {"repo": repo, "subfolder": sub}) is not None:
                    with _lock:
                        _server["swap_sent"] = True
                    health = _call(url, token, "/health")

    steps = _steps(server, stage, health, now)
    if health is not None and health.get("phase") == "ready" and health.get("model") == _model_key(server["policy_ref"]):
        # Remember how long this model took, for "last time" next time.
        times = health.get("times") or {}
        with _lock:
            saved = _server.get("timings_saved_for")
            if saved != server["policy_ref"]:
                _server["timings_saved_for"] = server["policy_ref"]
                for step_id in ("download", "load"):
                    if step_id in times:
                        _save_timing(server["policy_ref"], step_id, times[step_id])
        return {**base, "state": "ready", "steps": steps}
    if health is not None and health.get("error") and health.get("model") == _model_key(server["policy_ref"]):
        return {**base, "state": "error", "message": health["error"], "steps": steps}
    if health is None:
        return {**base, "state": "starting", "steps": steps}
    return {**base, "state": "loading", "steps": steps}


def ready_url(policy_ref: str) -> str:
    """The server address to drive the arm with, or ValueError with a reason."""
    current = status()
    if current.get("state") != "ready":
        raise ValueError("The cloud GPU isn't ready yet. Start it first and wait until it says ready.")
    if current.get("policy_ref") != policy_ref:
        raise ValueError("The cloud GPU has a different training loaded. Load this one first.")
    with _lock:
        return _server["url"]


# ---- arm loop in the browser ------------------------------------------------

# Same limit as remote_rollout.py's requests.
ACT_TIMEOUT_S = 5.0
_act_client = None


class CloudActError(Exception):
    """The cloud GPU didn't answer a plan request, or answered with an error."""


def _http():
    """One connection to the cloud GPU, kept open between plan requests: a new
    TLS handshake to the jobs proxy per plan would add to every round trip."""
    global _act_client
    import httpx

    with _lock:
        if _act_client is None:
            _act_client = httpx.Client(timeout=ACT_TIMEOUT_S)
        return _act_client


def act(policy_ref: str, body: dict[str, Any]) -> dict[str, Any]:
    """Pass one plan request from an arm the browser drives to the cloud GPU.

    The browser can't call the cloud GPU itself: the jobs proxy needs this
    server's Hugging Face token. The body is remote_policy_server.py's /act
    request ({"images", "state", "task"}), and its reply comes back as is.
    Unlike ready_url, this doesn't ask Hugging Face for the job's state on
    every request (they come about once a second): the page keeps polling
    /remote-server/status, which keeps the address current, and the cloud
    GPU itself answers 503 while a model is still loading.
    """
    with _lock:
        server = dict(_server)
    if not server.get("job_id") or not server.get("url"):
        raise ValueError("The cloud GPU isn't ready yet. Start it first and wait until it says ready.")
    if server.get("policy_ref") != policy_ref:
        raise ValueError("The cloud GPU has a different training loaded. Load this one first.")
    _, token = _hf()
    import httpx

    try:
        reply = _http().post(server["url"] + "/act", json=body, headers={"Authorization": f"Bearer {token}"})
    except httpx.HTTPError as exc:
        raise CloudActError(f"The cloud GPU did not answer: {type(exc).__name__}") from exc
    if reply.status_code != 200:
        try:
            detail = reply.json().get("error")
        except ValueError:
            detail = reply.text[:200]
        raise CloudActError(f"The cloud GPU answered {reply.status_code}: {detail}")
    return reply.json()
