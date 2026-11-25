"""World-level CARLA control: connecting, loading maps, synchronous mode and ticking."""

import socket
import time

import carla


def carla_reachable(host: str, port: int, timeout_s: float = 2.0) -> bool:
    """Is a CARLA server accepting connections? A plain TCP check that creates no carla objects,
    so it is safe to call when CARLA has crashed (carla objects can abort the process then)."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


class WorldManager:
    def __init__(self, host: str = "localhost", port: int = 2000, timeout_s: float = 20.0):
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout_s)
        self._timeout_s = timeout_s
        self.world = self.client.get_world()

    def load_map(self, map_name: str) -> carla.World:
        """Load `map_name` (e.g. "Town04") unless it is already the current map."""
        # Map names come back as "Carla/Maps/Town04"; loading takes a while, so skip it if possible.
        if not self.world.get_map().name.endswith(map_name):
            self.client.set_timeout(120.0)  # map loading is slow on integrated graphics
            self.world = self.client.load_world(map_name)
            # load_world() can return before the switch has finished: the map read straight afterwards was
            # still the OLD town (wrong roads/destinations on the dashboard, wrong spawn points for a run).
            # Wait until CARLA really reports the new town.
            deadline = time.monotonic() + 120.0
            while not self.world.get_map().name.endswith(map_name):
                if time.monotonic() > deadline:
                    raise RuntimeError(f"CARLA did not switch to {map_name} within 120 s")
                time.sleep(1.0)
                self.world = self.client.get_world()
            self.client.set_timeout(self._timeout_s)
        return self.world

    def enable_sync_mode(self, fixed_dt_s: float = 0.05) -> None:
        """In synchronous mode the simulation only advances when we call tick().

        Every step is then exactly `fixed_dt_s` of simulated time (0.05 s = 20 Hz),
        which makes experiments repeatable regardless of how fast the PC is.
        """
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_dt_s
        self.world.apply_settings(settings)

    def set_timeout(self, timeout_s: float) -> None:
        self.client.set_timeout(timeout_s)

    def tick(self) -> int:
        """Advance the simulation by one step and return the new frame number."""
        return self.world.tick()

    def restore_settings(self) -> None:
        """Put CARLA back in normal (asynchronous) mode so it does not freeze waiting for ticks.

        Always switches to async explicitly rather than restoring "whatever it was before":
        if an earlier run crashed and left CARLA in sync mode, that is not worth restoring.
        """
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
