# Improvement: access control on the hosted server

**Status:** deferred. Not part of the browser-recording work.

## Why this matters

LeLab was built to run on one person's computer, so its API has no login. On a
hosted server, every student reaches the same API, and all of them share one
Hugging Face account. `server.py` also allows requests from any website
(`allow_origins=["*"]`).

## What anyone who can reach the server can do

| Endpoint | Effect on a shared server |
|---|---|
| `POST /hf-auth/login` | Replaces the shared Hugging Face token, so later uploads and cloud training use the caller's account. |
| `POST /delete-dataset` with `delete_hub: true` | Deletes any student's dataset from Hugging Face. `_delete_hub_repo` only checks that the repo belongs to the logged-in account, and on a shared account every repo does. |
| `DELETE /jobs/{job_id}` with `delete_hub=true` | Deletes a training and its cloud model repo, with the same ownership check. |
| `POST /jobs/{job_id}/stop` | Stops any student's training. |
| `GET /datasets`, `GET /jobs` | List every student's datasets and trainings. |

## Suggested minimum

1. Add a server setting that marks the server as hosted.
2. When it is on, turn off `POST /hf-auth/login` and Hugging Face deletion
   (`delete_hub` on `/delete-dataset` and `/jobs/{job_id}`).
3. An admin logs in to the shared account once on the server.
4. eddy-fe hides its Hugging Face login box and any delete-from-Hub option.

This protects the shared token and stops Hugging Face deletions. It does not
identify students, so they could still stop each other's trainings or delete
each other's copies on the server.

## Longer term

Give each request a verified identity, for example eddy's login checked by the
server, and limit datasets and trainings to their owner. Restrict
`allow_origins` to eddy's own sites at the same time. Review the `/system/*`
endpoints too, which also have no login.
