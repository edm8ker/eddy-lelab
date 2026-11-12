# Hosted server setup

How LeLab runs on a hosted server (staging today, production next) so that
eddy-fe can use it for recording and training. Repeat every section on each
new server.

## How requests reach LeLab

```
browser (eddy-fe) ──https/wss──> Cloudflare ──> nginx :443 ──> LeLab 127.0.0.1:8000
```

- LeLab listens only on `127.0.0.1:8000`, a fixed value in
  `leLab-main/lelab/scripts/lelab.py`. nginx is the only way in from outside.
- PM2 keeps LeLab running, using `leLab-main/ecosystem.config.js`.
- eddy-fe finds LeLab through `VUE_APP_LELAB_BASE_URL`, for example
  `https://staging-lelab.edm8ker.com`. It builds the `wss://` address for
  WebSockets from it, so set only the `https://` one.

## 1. Python environment

LeLab must run from its own virtual environment in `leLab-main/.venv`, never
from a shared Python such as Anaconda base. Anaconda base has packages built
for NumPy 1.x (`numexpr`, `bottleneck`), and LeRobot installs NumPy 2. LeLab
then fails at startup with `AttributeError: _ARRAY_API not found`.

Create the environment once, with Python 3.12 or newer (`pyproject.toml`
requires it). Anaconda's Python is fine for creating it:

```bash
cd ~/staging/eddy-lelab/leLab-main
/home/ubuntu/anaconda3/bin/python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

- Don't add `--system-site-packages`. It brings the conflicting packages back.
- The install downloads LeRobot from GitHub (`lerobot @ git+...@v0.6.0`), so the
  server needs to reach github.com.
- `.venv/` is in `leLab-main/.gitignore`. To start over, delete it and run the
  commands again. Don't edit anything inside it.
- After pulling changes to `pyproject.toml`, run the last `pip install -e .`
  again.

## 2. Hugging Face login

Recording saves datasets under the server's Hugging Face account, and Finish
uploads them there as private datasets. Cloud training uses the same account.
Without a login, Start Recording fails with "This server isn't logged in to
Hugging Face".

Log in as the same Linux user that runs PM2:

```bash
cd ~/staging/eddy-lelab/leLab-main
.venv/bin/hf auth whoami     # shows the account, or that you're not logged in
.venv/bin/hf auth login      # paste a token with write access
```

Everyone using the server shares this account. See
[hosted-server-access-control.md](hosted-server-access-control.md) for what
that allows.

## 3. LeLab settings (`.env`)

`leLab-main/.env` holds LeLab's settings, such as the Hugging Face cache paths
(`HF_HOME`, `HF_LEROBOT_HOME`, `HF_LEROBOT_CALIBRATION`). On a server, every
path in it must exist on that server, not on a developer's computer. It is not
committed; create it on each server.

## 4. PM2

`ecosystem.config.js` starts LeLab with `leLab-main/.venv/bin/python`. Start it
the first time, or after changing `ecosystem.config.js`:

```bash
cd ~/staging/eddy-lelab/leLab-main
pm2 delete lelab              # skip on the very first start
pm2 start ecosystem.config.js
pm2 save                      # keeps it across reboots (with `pm2 startup`)
```

`pm2 restart lelab` is enough after a code change. It does not pick up changes
to `ecosystem.config.js` itself, which is why the steps above delete and start.

Check that PM2 uses the virtual environment:

```bash
pm2 describe lelab | grep "script path"   # must end in leLab-main/.venv/bin/python
pm2 logs lelab                            # live log
```

`start.sh` in the repo root runs the same environment without PM2, for a quick
manual start.

## 5. nginx

nginx must pass WebSocket upgrades through. Recording
(`/ws/browser-recording`) and the live arm data (`/ws/joint-data`) are
WebSockets. Without these lines they fail with a 404, and eddy-fe shows "Could
not reach the recording server".

In the `http { }` block of `/etc/nginx/nginx.conf`, unless it is already there:

```nginx
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}
```

In the site's `server { }` block:

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection $connection_upgrade;
    proxy_set_header Host $host;
    proxy_read_timeout 3600s;   # long recording sessions
    proxy_send_timeout 3600s;
}
```

Apply it:

```bash
sudo nginx -t && sudo systemctl reload nginx
```

In Cloudflare, **Network → WebSockets** must be on. It is on by default.

## Running a trained AI (Challenge 5 Step 4)

The arm is plugged into the student's computer, so the browser runs the arm
loop itself (`policyRunner.js` in eddy-fe, a port of `remote_rollout.py`). The
AI runs on a Hugging Face cloud GPU that LeLab starts with the server's
account. About once a second the browser sends the camera pictures and joint
positions to `POST /remote-server/act`, and LeLab passes them on to the cloud
GPU, because only LeLab has the token the GPU needs.

- Only trainings done on the Hugging Face cloud can run. "This computer" isn't
  offered in browser mode: it would be the LeLab server, which can't reach the
  arm.
- The server's Hugging Face account needs access to Jobs, with credits.
  The cloud GPU costs about $0.80 an hour and turns itself off after 10 minutes
  without requests.
- Each request is about 100 KB, under nginx's default 1 MB body limit.

## 6. Check the server

Run these on the server. Each must pass before the next one means anything.

```bash
# LeLab is up
curl -sS http://127.0.0.1:8000/health

# LeLab accepts WebSockets: expect "101 Switching Protocols", then a
# timeout after 3 s (the connection stays open, which is correct)
curl -i --http1.1 --max-time 3 \
  -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
  http://127.0.0.1:8000/ws/joint-data
```

Then run the same WebSocket check against the public address
(`https://staging-lelab.edm8ker.com/ws/joint-data`). It must also return `101`.
Finally, record and finish one short dataset from eddy-fe.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `_ARRAY_API not found` or "compiled using NumPy 1.x" at startup | LeLab is running on Anaconda base | Sections 1 and 4; check the PM2 script path |
| Public WebSocket returns 404, local one returns 101 | nginx drops the upgrade | Section 5 |
| Both WebSocket checks return 404 | uvicorn has no WebSocket library | Rerun `pip install -e .` in `.venv` (installs `websockets`), then restart |
| Every public request returns `error code: 502` | Cloudflare can't reach the server: LeLab or nginx is down | `pm2 status`, `pm2 logs lelab`, `sudo systemctl status nginx` |
| "This server isn't logged in to Hugging Face" | No Hugging Face login for the PM2 user | Section 2 |
| Step 4: "Follower calibration file not found: …/so_follower/<name>.json" | eddy-fe is too old: it asks LeLab to drive the arm, which is plugged into the student's computer | Deploy an eddy-fe with the browser AI run (see below) |
| Step 4: "This LeLab server can't run the AI for the browser yet" | LeLab has no `/remote-server/act` | Pull, then `pm2 restart lelab` |

eddy-fe shows "Could not connect to the follower arm" for **any** request that
never got an answer from LeLab, including training requests, which don't use
the arm. When you see it, check the browser's Network tab and the rows above
before checking the arm.
