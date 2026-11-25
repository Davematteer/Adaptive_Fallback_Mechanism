"""Host entry point: starts the dashboard and the experiment manager.

Run from inside the host/ folder with the CARLA environment active:
    source ~/carla-env/bin/activate
    pip install -r requirements.txt
    python main.py --jetson-host 192.168.55.1
Then open http://localhost:8000 in a browser on Windows.
"""

import argparse
import faulthandler
import os

import uvicorn

from dashboard.main import create_app
from experiments.scenario_manager import ScenarioManager


def parse_args():
    p = argparse.ArgumentParser(description="CARLA experiment host")
    # From WSL, CARLA on Windows is reached via the Windows host IP ($WIN_HOST), not localhost.
    p.add_argument("--carla-host", default=os.environ.get("WIN_HOST", "localhost"))
    p.add_argument("--carla-port", type=int, default=2000)
    p.add_argument("--jetson-host", default="192.168.55.1", help="Jetson USB-C network address")
    p.add_argument("--jetson-port", type=int, default=5555)
    p.add_argument("--port", type=int, default=8000, help="dashboard port")
    return p.parse_args()


def main():
    # If the CARLA client library ever crashes the process (segfault), print the Python stack first.
    faulthandler.enable()
    args = parse_args()
    manager = ScenarioManager(args.carla_host, args.carla_port, args.jetson_host, args.jetson_port)
    uvicorn.run(create_app(manager), host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
