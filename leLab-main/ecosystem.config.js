const path = require("path");

module.exports = {
  apps: [
    {
      name: "lelab",
      // Use this checkout's own virtualenv, never a shared Python (e.g. Anaconda
      // base, whose NumPy 1.x-built packages break under NumPy 2). Create it
      // once with: python3.12 -m venv .venv && .venv/bin/pip install -e .
      // `python -m` from this directory runs the code in this checkout.
      script: path.join(__dirname, ".venv", "bin", "python"),
      args: "-m lelab.scripts.lelab --no-open",
      interpreter: "none",
      cwd: __dirname, // the launcher reads .env from here
      env: {
        // No display on a server; lets pygame start for the gamepad endpoints.
        SDL_VIDEODRIVER: "dummy",
      },
      // pm2's default (1.6 s) is shorter than the backend's graceful shutdown.
      kill_timeout: 5000,
    },
  ],
};