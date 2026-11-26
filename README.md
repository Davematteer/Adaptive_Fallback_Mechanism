

# Adaptive Fallback Mechanism

A fault-tolerant autonomous driving platform.

A capstone project: a self-driving car in the CARLA simulator, driven by an NVIDIA Jetson Orin Nano, with a Basys 3 FPGA planned as an independent safety monitor. The goal is to test what happens when the driving computer fails, and to show that a separate, simpler piece of hardware can detect the failure and bring the car to a safe stop.




https://github.com/user-attachments/assets/f8ea4369-0652-4741-8c39-33d4869731aa



## Architecture

```
┌────────────────────────┐   camera, LiDAR, car state   ┌──────────────────────────┐
│  PC (host)             │ ───────────────────────────► │  Jetson Orin Nano        │
│  CARLA simulator       │                              │  perception, planning,   │
│  experiment dashboard  │ ◄─────────────────────────── │  control                 │
└────────────────────────┘   steer, throttle, brake     └────────────┬─────────────┘
                                                                     │ heartbeat (planned)
                                                        ┌────────────▼─────────────┐
                                                        │  Basys 3 FPGA (planned)  │
                                                        │  watchdog + fallback     │
                                                        └──────────────────────────┘
```

The PC and the Jetson talk over a USB-C network link (TCP). Every simulation step, the host sends a camera frame, a LiDAR sweep and the car's state; the Jetson replies with a driving command and a heartbeat. If the heartbeat stops, the fallback takes over and stops the car.

## What it does

**Jetson (`jetson/`)**
- **Perception:** YOLO26n object detection, UFLDv2 lane detection, road segmentation and depth estimation, all running as TensorRT engines. LiDAR obstacles come from geometry alone (ground removal and clustering, no ML).
- **Planning:** follows the detected lane, and uses the route through junctions and unmarked bends. Stops for pedestrians, follows traffic, and tells a traffic queue apart from a real blockage before swerving or reversing.
- **Control:** pure pursuit steering and speed control.
- **Health:** a heartbeat that only advances when the whole pipeline completes, plus injectable faults (hang, crash, communication loss) for experiments.

**Host (`host/`)**
- Runs CARLA in synchronous mode and streams sensor data to the Jetson.
- Shortest-route planning (Dijkstra) over CARLA's road network.
- A watchdog and command arbitrator that switch to the fallback when the Jetson stops responding.
- A web dashboard: experiment setup, fault injection, a route map with selectable destinations, a live camera view with detection and lane overlays, a LiDAR bird's-eye view, and CSV logging of every run.

## Repository layout

```
host/                 runs on the PC
  carla_manager/      world, vehicle, sensors, traffic, pedestrians, route planning
  command_manager/    watchdog and command arbitration (Jetson or fallback)
  communication/      TCP link to the Jetson
  dashboard/          FastAPI dashboard and live streams
  experiments/        experiment configuration, run loop, fault injection
jetson/               runs on the Jetson
  perception/         object, lane, road, depth and LiDAR perception
  planning/           path, obstacle tracking, manoeuvres
  control/            pure pursuit and speed control
  monitoring/         pipeline heartbeat and fault injection
  communication/      TCP server
  tools/              offline accuracy check on saved frames
```

## Running it

**Host** (Python 3.10, CARLA 0.9.15 running):
```bash
cd host
pip install -r requirements.txt
python main.py --jetson-host 192.168.55.1
```
Then open http://localhost:8000.

**Jetson** (JetPack with TensorRT; model engines in `~/models/`):
```bash
cd jetson
pip install -r requirements.txt
python3 main.py
```

Model files (`.onnx`, `.engine`) are not included in this repository; they are exported from the pretrained models and built on the Jetson with `trtexec`.

## Status

The simulator, Jetson driving pipeline and dashboard are working. Next: the Basys 3 FPGA as the independent hardware watchdog and fallback controller.
