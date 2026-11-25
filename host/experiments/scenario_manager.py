"""Experiment configuration, the experiment state machine and the main run loop.

All CARLA calls happen in one background thread (the run loop). Dashboard
requests only change state or set flags, so they never race with the simulation.
"""

import csv
import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from carla_manager.pedestrian_manager import PedestrianManager
from carla_manager.route_planner import RouteFollower, RoutePlanner
from carla_manager.sensor_manager import SensorManager
from carla_manager.traffic_manager import TrafficManager
from carla_manager.vehicle_manager import VehicleManager
from carla_manager.world_manager import WorldManager, carla_reachable
from command_manager.command_arbitrator import CommandArbitrator
from communication.jetson_connection import JetsonConnection
from experiments.fault_injector import FaultInjector

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

# CARLA objects from runs whose cleanup failed (CARLA crashed or hung). They are kept referenced on
# purpose: when Python frees them, their C++ destructors contact the dead server, throw an exception
# Python cannot catch, and abort the whole host app ("terminate called ... TimeoutException").
_ABANDONED = []


@dataclass
class ExperimentConfig:
    map: str = "Town04"
    spawn_index: int = 0
    destination_index: int = -1  # spawn point to drive to (shortest route, Dijkstra); -1 = just follow the road
    spawn_lateral_offset_m: float = 0.0  # start this far right (+) / left (-) of the lane centre (recovery tests)
    target_speed_mps: float = 8.0
    fault: str = "pipeline_hang"
    duration_s: float = 120.0  # simulated seconds before the run stops by itself
    step_s: float = 0.1  # simulated time per step: 0.1 = 10 Hz (this lab PC's GPU can't sustain 20 Hz with a camera)
    fallback_enabled: bool = True  # False = baseline: on failure the car keeps its last command
    camera_width: int = 640
    camera_height: int = 360
    pedestrians: int = 0  # walkers that cross the ego's lane ahead (0 = none)
    pedestrian_trigger_m: float = 35.0  # a walker starts crossing when the car is this close
    pedestrian_behaviour: str = "cross"  # "cross", or "stop_in_lane" (stands in the lane for 15 s)
    blocked_lane_car: bool = False  # a parked car in the ego lane 60 m ahead (to test swerving / reversing)
    red_light_queue: bool = False  # cars queued at a red light ahead (tests waiting in traffic, no reversing)
    block_all_lanes: bool = False  # with blocked_lane_car: also park cars beside it, so there's no gap (reversing)
    vehicles: int = 0  # other cars on autopilot near the ego (0 = none)
    path_source: str = "perception"  # "perception": Jetson drives from lanes/road; "map": CARLA's waypoints
    lidar: bool = True  # roof LiDAR, streamed to the Jetson with each camera frame
    time_scale: float = 1.0  # 1 = real time; 0.3 = slow motion (sim runs at 30 %), e.g. to test a slower Jetson stand-in
    at_split: str = "right"  # where the ego lane forks (e.g. exit ramp): follow the "right" or "left" branch

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        """Load a saved config so an experiment can be repeated with identical settings."""
        return cls(**json.loads(Path(path).read_text()))


class State(str, Enum):
    IDLE = "idle"
    CONFIGURED = "configured"
    READY = "ready"
    RUNNING = "running"
    FAULT_INJECTED = "fault_injected"
    FPGA_TAKEOVER = "fpga_takeover"
    STOPPED = "stopped"


# Which states each state may move to. Anything else is rejected.
TRANSITIONS = {
    State.IDLE: {State.CONFIGURED},
    State.CONFIGURED: {State.CONFIGURED, State.READY},
    State.READY: {State.RUNNING, State.STOPPED},
    State.RUNNING: {State.FAULT_INJECTED, State.FPGA_TAKEOVER, State.STOPPED},
    State.FAULT_INJECTED: {State.FPGA_TAKEOVER, State.STOPPED},
    State.FPGA_TAKEOVER: {State.STOPPED},
    State.STOPPED: {State.IDLE},
}


class InvalidTransition(Exception):
    pass


class ScenarioManager:
    def __init__(self, carla_host: str, carla_port: int, jetson_host: str, jetson_port: int):
        self.carla_host, self.carla_port = carla_host, carla_port
        self.jetson_host, self.jetson_port = jetson_host, jetson_port
        self.state = State.IDLE
        self.config = None
        self.arbitrator = CommandArbitrator()
        self.connection = None
        self.fault_injector = None
        self._stop_requested = threading.Event()
        self._lock = threading.Lock()  # guards state changes
        self._map_cache = {}  # (town, spawn) -> map info for the dashboard
        self._world_mgr = None  # ONE CARLA client for the whole app (see _world_manager)
        self._carla_lock = threading.Lock()  # map loads and run set-up never overlap on that client
        self._map_job_lock = threading.Lock()
        self._map_job = None  # the background map load in progress: {"key": (town, spawn)} or {"error": ...}
        self._reset_run_data()

    # ---------- dashboard actions ----------

    def configure(self, config: ExperimentConfig) -> None:
        self._transition(State.CONFIGURED)
        self.config = config
        self.arbitrator.fallback_enabled = config.fallback_enabled
        # The watchdog counts wall-clock time: in slow motion the Jetson gets proportionally longer per frame.
        self.arbitrator.stale_after_s = 0.5 / max(config.time_scale, 0.05)

    def start(self) -> None:
        if self.state is not State.CONFIGURED:
            hint = " - click Reset, then Configure" if self.state is State.STOPPED else " - click Configure first"
            raise InvalidTransition(f"can only start from 'configured', not '{self.state.value}'{hint}")
        self._reset_run_data()
        self._stop_requested.clear()
        threading.Thread(target=self._run, daemon=True).start()

    def inject_fault(self) -> None:
        self._transition(State.FAULT_INJECTED)
        self.metrics["fault"] = self.config.fault
        self.metrics["fault_sent_at"] = self.fault_injector.inject(self.config.fault)
        self._event(f"fault injected: {self.config.fault}")

    def stop(self) -> None:
        if self.state not in (State.READY, State.RUNNING, State.FAULT_INJECTED, State.FPGA_TAKEOVER):
            raise InvalidTransition(f"nothing to stop in state '{self.state.value}'")
        self._stop_requested.set()  # the run loop tears everything down and moves to STOPPED

    def reset(self) -> None:
        """Back to IDLE for the next experiment. Clears the fallback latch.

        Later this must also reset the FPGA's fallback latch explicitly:
        restarting CARLA does not reset the FPGA.
        """
        self._transition(State.IDLE)
        self.arbitrator.reset()
        self.config = None

    def view(self):
        """Latest camera frame + the Jetson's detections for it (see JetsonConnection.view)."""
        return self.connection.view() if self.connection else None

    def view_maps(self) -> dict:
        """Newest downscaled segmentation / depth maps from the Jetson, for the dashboard overlays."""
        return dict(self.connection.view_maps) if self.connection else {}

    def _world_manager(self) -> WorldManager:
        """The app's single CARLA connection, created on first use and reused for every run.

        Creating a carla.Client per run and dropping it afterwards occasionally aborted the whole
        host process (a native client thread calling std::terminate during the client's teardown);
        CARLA's own examples use one client for the program's lifetime.
        """
        if self._world_mgr is None:
            self._world_mgr = WorldManager(self.carla_host, self.carla_port)
        else:
            self._world_mgr.world = self._world_mgr.client.get_world()  # the map may have changed
        return self._world_mgr

    def map_info(self, town: str, spawn_index: int) -> dict:
        """Road lines, the start and suggested destinations for the dashboard map.

        Loading a town takes CARLA 30-120 s, so it runs in the background, one load at a time: the first
        call starts it and returns {"status": "loading"}; the page asks again until the result is ready.
        """
        key = (town, spawn_index)
        if key in self._map_cache:
            return {"status": "ready", **self._map_cache[key]}
        with self._map_job_lock:
            job = self._map_job
            if job and job["key"] == key:
                if job.get("error"):
                    self._map_job = None
                    raise ConnectionError(job["error"])
                return {"status": "loading", "town": town}
            if job and not job.get("error"):
                raise InvalidTransition(f"still loading {job['key'][0]}: wait for it to finish, then try again")
            if self.state not in (State.IDLE, State.CONFIGURED, State.STOPPED):
                raise InvalidTransition("can't load a map while an experiment is running")
            if not carla_reachable(self.carla_host, self.carla_port):
                raise ConnectionError(f"CARLA is not running at {self.carla_host}:{self.carla_port}")
            self._map_job = {"key": key}
        threading.Thread(target=self._load_map_job, args=(town, spawn_index), daemon=True).start()
        return {"status": "loading", "town": town}

    def _load_map_job(self, town: str, spawn_index: int) -> None:
        key = (town, spawn_index)
        try:
            with self._carla_lock:  # never at the same time as a run being set up (or another load)
                carla_map = self._world_manager().load_map(town).get_map()
                planner = RoutePlanner(carla_map)
                spawns = carla_map.get_spawn_points()
                start = spawns[spawn_index % len(spawns)]
                self._map_cache[key] = {
                    "town": town, "spawn_index": spawn_index,
                    "start": [round(start.location.x, 1), round(start.location.y, 1), round(start.rotation.yaw)],
                    "destinations": planner.destinations(start.location, spawns),
                    "roads": planner.road_lines(),
                }
            with self._map_job_lock:
                self._map_job = None
        except Exception as exc:  # noqa: BLE001 - reported to the page on its next poll
            with self._map_job_lock:
                self._map_job = {"key": key, "error": f"{type(exc).__name__}: {exc}"}

    def route_info(self) -> dict:
        """The current run's planned route (thinned) and where the car is, for the dashboard map."""
        pts = self.route.points[::3] if self.route else []
        return {"experiment_id": self.experiment_id, "route": [[round(p[0], 1), round(p[1], 1)] for p in pts],
                "ego": self.ego_pose}

    def status(self) -> dict:
        cmd = self.connection.latest_command if self.connection else None
        age = self.connection.command_age_s() if self.connection else None
        return {
            "state": self.state.value,
            "experiment_id": self.experiment_id,
            "config": asdict(self.config) if self.config else None,
            "owner": self.arbitrator.owner.value,
            "latch_reason": self.arbitrator.latch_reason,
            "speed_mps": round(self.speed_mps, 2),
            "ego": self.ego_pose,
            "route_remaining_m": self.route_remaining,
            "jetson_heartbeat": cmd.get("heartbeat") if cmd else None,
            "command_age_ms": None if age is None else round(age * 1000),
            "metrics": self.metrics,
            "events": self.events[-10:],
            "error": self.error,
        }

    # ---------- run loop (background thread) ----------

    def _run(self) -> None:
        world_mgr = vehicles = sensors = pedestrians = traffic = world = vehicle = None
        try:
            # 1. Configure CARLA and connect to the Jetson.
            if not carla_reachable(self.carla_host, self.carla_port):
                raise ConnectionError(f"CARLA is not running at {self.carla_host}:{self.carla_port}: "
                                      "start CarlaUE4.exe, wait for the map to appear, then Start again")
            world_mgr = self._world_manager()
            with self._carla_lock:  # waits for a dashboard map load still in progress
                world = world_mgr.load_map(self.config.map)
            world_mgr.enable_sync_mode(self.config.step_s)
            vehicles = VehicleManager(world)
            vehicle = vehicles.spawn(spawn_index=self.config.spawn_index,
                                     lateral_offset_m=self.config.spawn_lateral_offset_m)
            # In synchronous mode a new actor reports location (0, 0, 0) until the world ticks once. Tick now,
            # or the pedestrians / traffic / parked car below get placed relative to the map origin.
            world_mgr.tick()
            sensors = SensorManager(world)
            sensors.attach_camera(vehicle, self.config.camera_width, self.config.camera_height)
            sensors.attach_collision_sensor(vehicle)
            pedestrians = PedestrianManager(world)
            if self.config.pedestrians > 0:
                for message in pedestrians.spawn_crossers(vehicle, self.config.pedestrians,
                                                          behaviour=self.config.pedestrian_behaviour):
                    self._event(message)
            traffic = TrafficManager(world_mgr.client, world)
            if self.config.vehicles > 0:
                for message in traffic.spawn(vehicle, self.config.vehicles):
                    self._event(message)
            if self.config.red_light_queue:
                for message in traffic.spawn_red_light_queue(vehicle):
                    self._event(message)
            if self.config.blocked_lane_car:
                for message in traffic.spawn_parked(vehicle, all_lanes=self.config.block_all_lanes):
                    self._event(message)
            sensors.attach_rear_sensor(vehicle)
            if self.config.lidar:
                sensors.attach_lidar(vehicle, self.config.step_s)
            if self.config.destination_index >= 0:
                spawns = vehicles.map.get_spawn_points()
                goal = spawns[self.config.destination_index % len(spawns)].location
                points = RoutePlanner(vehicles.map).route(vehicle.get_location(), goal)
                if points is None:
                    raise ValueError(f"spawn point {self.config.destination_index} can't be reached from here")
                self.route = RouteFollower(points)
                self.metrics["route_m"] = round(self.route.length)
                self._event(f"route planned (Dijkstra): {self.route.length:.0f} m to spawn point "
                            f"{self.config.destination_index}")
            self.connection = JetsonConnection(self.jetson_host, self.jetson_port)
            self.connection.connect()
            self.fault_injector = FaultInjector(self.connection)
            self._transition(State.READY)

            # 2. Drive until Stop is clicked or the duration runs out. The run keeps going after a
            #    failure so the car stays visible: stopped by the fallback, or crashed without it.
            self._transition(State.RUNNING)
            self._event("experiment started")
            self._loop(world_mgr, vehicles, sensors, pedestrians, traffic)
        except Exception as exc:  # noqa: BLE001 - any failure is shown on the dashboard instead of killing the thread
            self.error = f"{type(exc).__name__}: {exc}"
            self._event(f"error: {self.error}")
        finally:
            # 3. Always clean up, otherwise CARLA stays frozen in sync mode with leftover actors.
            self._cleanup_carla(world_mgr, pedestrians, sensors, vehicles, traffic, world, vehicle)
            if self.connection:
                self.connection.close()
            with self._lock:
                self.state = State.STOPPED
            self._event("experiment stopped")

    def _cleanup_carla(self, world_mgr, *carla_objects) -> None:
        """Destroy our actors and restore CARLA's settings, unless CARLA has died.

        If CARLA is gone (or any step fails), the objects are parked in _ABANDONED instead of being
        freed, so the host app survives and the next experiment can start once CARLA is back.
        """
        if world_mgr is None:
            return  # never connected: no carla objects exist
        pedestrians, sensors, vehicles, traffic = carla_objects[:4]
        ok = carla_reachable(self.carla_host, self.carla_port)
        if ok:
            world_mgr.set_timeout(5.0)  # a struggling server must not stall each cleanup call for 20 s
            for cleanup in (traffic and traffic.destroy, pedestrians and pedestrians.destroy,
                            sensors and sensors.destroy, vehicles and vehicles.destroy, world_mgr.restore_settings):
                if cleanup:
                    try:
                        cleanup()
                    except Exception as exc:  # noqa: BLE001 - keep cleaning up the rest
                        ok = False
                        self._event(f"cleanup failed: {exc}")
        else:
            self._event("CARLA stopped responding (crashed?): skipped CARLA cleanup. "
                        "Restart CARLA, then Reset -> Configure -> Start")
        world_mgr.set_timeout(20.0)  # back to normal for the next run (the client is reused)
        if not ok:
            _ABANDONED.append((world_mgr, *carla_objects))
            self._world_mgr = None  # CARLA died: the next run connects afresh

    def _loop(self, world_mgr: WorldManager, vehicles: VehicleManager, sensors: SensorManager,
              pedestrians: PedestrianManager, traffic: TrafficManager) -> None:
        dt = world_mgr.world.get_settings().fixed_delta_seconds
        RUNS_DIR.mkdir(exist_ok=True)
        log_path = RUNS_DIR / f"{self.experiment_id}.csv"
        # The run's settings next to its log, so every CSV says which map, spawn point, speed, ... it used.
        log_path.with_suffix(".json").write_text(json.dumps(asdict(self.config), indent=2))
        with open(log_path, "w", newline="") as f:
            log = csv.writer(f)
            log.writerow(["sim_time_s", "wall_time_s", "state", "owner", "speed_mps", "steer", "throttle", "brake",
                          "command_age_ms", "jetson_heartbeat", "decision", "reason", "path_source",
                          "path_note", "lane_width_m", "target_right_m", "gt_lane_offset_m", "obstacle_class", "obstacle_dist_m",
                          "obstacle_dist_geom_m", "n_detections", "inference_ms", "pipeline_ms",
                          "gt_ped_dist_m", "gt_car_ahead_m", "maneuver", "reverse", "rear_clearance_m",
                          "junction_ahead_m", "route_remaining_m", "obstacle_state", "obstacle_id",
                          "obstacle_ever_moved", "queue_detected", "traffic_evidence"])
            sim_time = 0.0
            started = next_tick = time.monotonic()
            while not self._stop_requested.is_set() and sim_time < self.config.duration_s:
                # Pace the loop to real time: one 0.05 s step every 0.05 s of wall-clock time.
                # Without this the sim runs as fast as the PC allows, so motion looks jerky
                # and the Jetson gets less real time per frame than it would in a real car.
                next_tick += dt / max(self.config.time_scale, 0.05)  # time_scale < 1 = slow motion
                time.sleep(max(0.0, next_tick - time.monotonic()))
                frame = world_mgr.tick()
                sim_time += dt

                # Send the newest camera frame + vehicle state to the Jetson.
                state = vehicles.state()
                state["target_speed_mps"] = self.config.target_speed_mps
                state["path_source"] = self.config.path_source
                state["at_split"] = self.config.at_split
                state["sim_time_s"] = round(sim_time, 3)  # exact time base for the Jetson's odometry
                self.ego_pose = [round(state["x"], 1), round(state["y"], 1), round(state["yaw_deg"])]
                if self.route:  # the planned route replaces "follow the road" waypoints
                    state.update(self.route.update(state["x"], state["y"]))
                    self.route_remaining = state["route_remaining_m"]
                    if state["route_remaining_m"] < 4.0 and abs(state["speed_mps"]) < 0.3:
                        self.metrics["destination_reached_s"] = round(sim_time, 1)
                        self._event(f"DESTINATION REACHED after {sim_time:.0f} s ({self.route.length:.0f} m route)")
                        self._stop_requested.set()
                state["rear_clearance_m"] = sensors.rear_clearance_m()  # rear parking sensor, None = clear
                self.speed_mps = state["speed_mps"]
                self.connection.send_frame(frame, state, sensors.latest_image, sensors.camera_info,
                                           sensors.latest_lidar, sensors.lidar_mount)

                # Let pedestrians step out when the car gets close (ground truth distance for the log).
                ped_dist, messages = pedestrians.update(vehicles.vehicle.get_location(),
                                                        self.config.pedestrian_trigger_m, sim_time)
                for message in messages:
                    self._event(message)

                # Pick whose command to apply (Jetson, or fallback once latched).
                age = self.connection.command_age_s()  # logged
                command = self.arbitrator.select(self.connection.latest_command, self.connection.unanswered_s())
                vehicles.apply(command)
                vehicles.follow_with_spectator()  # chase view in the CARLA window
                self._track_outcome(state, sensors, sim_time)
                jetson = self.connection.latest_command or {}
                obstacle = self._track_perception(jetson, ped_dist)
                lane_offset = vehicles.lane_offset()
                car_ahead = traffic.nearest_ahead(vehicles.vehicle)
                self._track_driving(jetson, lane_offset, car_ahead)
                path = jetson.get("path") or {}
                blockage = jetson.get("blockage") or {}

                log.writerow([round(sim_time, 3), round(time.monotonic() - started, 3),
                              self.state.value, self.arbitrator.owner.value,
                              round(self.speed_mps, 3), command.get("steer"), command.get("throttle"),
                              command.get("brake"), None if age is None else round(age * 1000),
                              jetson.get("heartbeat"), jetson.get("decision"), jetson.get("reason"),
                              path.get("source"), path.get("note"), path.get("lane_width_m"),
                              (path.get("target") or [None, None])[1], round(lane_offset, 3),
                              obstacle.get("class"), obstacle.get("distance_m"), obstacle.get("distance_geom_m"),
                              jetson.get("n_detections"), jetson.get("inference_ms"), jetson.get("pipeline_ms"),
                              None if ped_dist is None else round(ped_dist, 2),
                              None if car_ahead is None else round(car_ahead, 2),
                              jetson.get("maneuver"), command.get("reverse", False), state.get("rear_clearance_m"),
                              state.get("junction_ahead_m"), state.get("route_remaining_m"),
                              blockage.get("state"), blockage.get("id"), blockage.get("ever_moved"),
                              blockage.get("queue"), "; ".join(blockage.get("evidence") or [])])
        self._event(f"log saved to runs/{log_path.name}")

    def _track_outcome(self, state: dict, sensors: SensorManager, sim_time: float) -> None:
        """Record each key moment once: failure detected, takeover, vehicle stopped, collision."""
        arb, m = self.arbitrator, self.metrics

        if arb.failure_detected_at and "failure_detected_at" not in m:
            m["failure_detected_at"] = arb.failure_detected_at
            m["failure_xy"] = (state["x"], state["y"])
            if m.get("fault_sent_at"):
                m["detection_s"] = round(arb.failure_detected_at - m["fault_sent_at"], 3)
            if arb.fallback_enabled:
                self._transition(State.FPGA_TAKEOVER)
                m["takeover_at"] = arb.latched_at
                self._event(f"fallback took over: {arb.latch_reason}")
            else:
                self._event(f"Jetson failure detected ({arb.latch_reason}); fallback disabled, "
                            "vehicle keeps its last command")

        if "failure_xy" in m and "stopping_distance_m" not in m and self.speed_mps < 0.1:
            x0, y0 = m["failure_xy"]
            m["stopping_distance_m"] = round(math.hypot(state["x"] - x0, state["y"] - y0), 2)
            self._event(f"vehicle at rest {m['stopping_distance_m']} m after the failure")

        if sensors.first_collision and "collision" not in m:
            m["collision"] = {"with": sensors.first_collision, "sim_time_s": round(sim_time, 2)}
            self._event(f"COLLISION with {sensors.first_collision}")

    def _track_perception(self, jetson: dict, ped_dist: float | None) -> dict:
        """Log stop/go changes from the Jetson's planner and the closest the car came to a pedestrian."""
        obstacle = jetson.get("obstacle") or {}
        decision = jetson.get("decision")
        if decision and decision != self._last_decision:
            if decision == "stop":
                where = f" ({obstacle['class']} {obstacle['distance_m']} m ahead)" if obstacle else ""
                self._event(f"Jetson: STOP - {jetson.get('reason') or 'stopping'}{where}")
            elif decision == "slow":
                self._event(f"Jetson: SLOW - {jetson.get('reason') or 'following the road fallback'}")
            elif self._last_decision in ("stop", "slow"):
                self._event("Jetson: GO - path clear")
            self._last_decision = decision
        if ped_dist is not None:
            m = self.metrics
            m["min_pedestrian_dist_m"] = round(min(ped_dist, m.get("min_pedestrian_dist_m", ped_dist)), 2)
        return obstacle

    def _track_driving(self, jetson: dict, lane_offset: float, car_ahead: float | None) -> None:
        """How well the car keeps its lane (CARLA ground truth), and which path source the Jetson uses."""
        m = self.metrics
        self._offset_samples += 1
        n = self._offset_samples
        m["mean_abs_lane_offset_m"] = round(m.get("mean_abs_lane_offset_m", 0.0) + (abs(lane_offset) - m.get("mean_abs_lane_offset_m", 0.0)) / n, 3)
        m["max_abs_lane_offset_m"] = round(max(abs(lane_offset), m.get("max_abs_lane_offset_m", 0.0)), 3)
        if abs(lane_offset) > 1.0 and not self._departed:
            self._departed = True
            self._event(f"LANE DEPARTURE: {lane_offset:+.2f} m from the lane centre")
        elif abs(lane_offset) < 0.5:
            self._departed = False
        maneuver = jetson.get("maneuver")
        if maneuver and maneuver != self._last_maneuver:
            if maneuver.startswith("swerve") or maneuver in ("reverse", "traffic_wait", "recovery_failed"):
                counts = m.setdefault("maneuvers", {})
                counts[maneuver] = counts.get(maneuver, 0) + 1
                self._event(f"Jetson: {maneuver.upper().replace('_', ' ')} - {jetson.get('reason')}")
            self._last_maneuver = maneuver
        source = (jetson.get("path") or {}).get("source")
        if source and source != self._last_source:
            self._event(f"Jetson path source: {source}")
            self._last_source = source
        if source:
            counts = m.setdefault("path_source_steps", {})
            counts[source] = counts.get(source, 0) + 1
        if car_ahead is not None:
            m["min_car_ahead_m"] = round(min(car_ahead, m.get("min_car_ahead_m", car_ahead)), 2)

    # ---------- helpers ----------

    def _transition(self, new: State) -> None:
        with self._lock:
            if new not in TRANSITIONS[self.state]:
                raise InvalidTransition(f"cannot go from '{self.state.value}' to '{new.value}'")
            self.state = new

    def _event(self, text: str) -> None:
        self.events.append(f"{time.strftime('%H:%M:%S')}  {text}")

    def _reset_run_data(self) -> None:
        self.experiment_id = time.strftime("exp_%Y%m%d_%H%M%S")
        self.metrics, self.events, self.error, self.speed_mps = {}, [], None, 0.0
        self.route = self.route_remaining = self.ego_pose = None
        self._last_decision = self._last_source = self._last_maneuver = None
        self._departed = False
        self._offset_samples = 0
