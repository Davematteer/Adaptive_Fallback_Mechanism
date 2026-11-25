"""TCP link from the host to the Jetson.

The host connects OUT to the Jetson (the Jetson runs the server). Because Windows
starts the connection, the Windows firewall does not block it.

Wire format (must match jetson/communication/communication_server.py):
    [4-byte big-endian header length][JSON header][payload of header["size"] bytes]
"""

import json
import socket
import struct
import threading
import time
from collections import OrderedDict


def send_msg(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    header = dict(header, size=len(payload))
    raw = json.dumps(header).encode()
    sock.sendall(struct.pack("!I", len(raw)) + raw + payload)


def recv_msg(sock: socket.socket) -> tuple[dict, bytes]:
    (length,) = struct.unpack("!I", _recv_exact(sock, 4))
    header = json.loads(_recv_exact(sock, length))
    return header, _recv_exact(sock, header.get("size", 0))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed by peer")
        buf.extend(chunk)
    return bytes(buf)


class JetsonConnection:
    def __init__(self, host: str, port: int):
        self.host, self.port = host, port
        self.sock = None
        self.latest_command = None      # last {"steer", "throttle", "brake", ...} from the Jetson
        self.command_received_at = None  # time.monotonic() when it arrived
        self._send_lock = threading.Lock()  # the run loop and the dashboard both send
        self._in_flight_since = None     # when the unanswered frame was sent, if any
        self.dead = False                # True once the Jetson stops accepting data
        self._sent_images = OrderedDict()  # frame_id -> carla.Image, last few sent (dashboard camera view)
        self._images_lock = threading.Lock()
        self.view_maps = {}  # newest downscaled segmentation / depth maps from the Jetson (not sent every frame)

    SEND_TIMEOUT_S = 2.0  # a send blocked this long means the Jetson has stopped reading

    def connect(self, timeout_s: float = 5.0) -> None:
        self.sock = socket.create_connection((self.host, self.port), timeout=timeout_s)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # send small messages immediately

        # Handshake. The Jetson's OS accepts connections even when the app itself is frozen
        # (e.g. still stuck from a previous pipeline_hang fault), so require a reply.
        send_msg(self.sock, {"type": "hello"})
        try:
            reply, _ = recv_msg(self.sock)
        except TimeoutError:
            self.close()
            raise ConnectionError(f"Jetson app did not answer within {timeout_s:.0f} s. "
                                  "It may be frozen from a previous fault: restart it on the Jetson.") from None
        if reply.get("type") != "ready":
            self.close()
            raise ConnectionError(f"unexpected handshake reply from Jetson: {reply}")

        self.dead = False
        self.sock.settimeout(self.SEND_TIMEOUT_S)
        threading.Thread(target=self._read_loop, daemon=True).start()

    def send_frame(self, frame_id: int, state: dict, image, camera: dict | None = None,
                   lidar=None, lidar_mount: dict | None = None) -> bool:
        """Send one camera frame + vehicle state. Returns False if skipped.

        Only one frame is allowed "in flight": if the Jetson hasn't answered the
        previous one yet, this frame is dropped. That stops a slow Jetson from
        building up a queue of old frames (and ever-growing latency).
        """
        if self._in_flight_since and time.monotonic() - self._in_flight_since < 1.0:
            return False
        header = {"type": "frame", "frame_id": frame_id, "state": state, "sent_at": time.time()}
        payload = b""
        if image is not None:
            header.update(width=image.width, height=image.height, encoding="bgra8", camera=camera or {})
            payload = bytes(image.raw_data)
        header["image_size"] = len(payload)  # payload = [image][lidar xyz float32]
        if lidar is not None and len(lidar):
            payload += lidar.astype("float32").tobytes()
            header["lidar_mount"] = lidar_mount or {}
        self._send(header, payload)
        self._in_flight_since = time.monotonic()
        if image is not None:
            with self._images_lock:
                self._sent_images[frame_id] = image
                while len(self._sent_images) > 10:
                    self._sent_images.popitem(last=False)
        return True

    def view(self):
        """(frame_id, carla.Image, command) for the dashboard: the image the latest command answers,
        so its bounding boxes line up. Falls back to the newest sent image. None if nothing sent yet."""
        command = self.latest_command or {}
        with self._images_lock:
            if not self._sent_images:
                return None
            frame_id = command.get("frame_id")
            if frame_id not in self._sent_images:
                frame_id = next(reversed(self._sent_images))
                command = {}  # boxes belong to a different frame: don't draw them on this one
            return frame_id, self._sent_images[frame_id], command

    def send_control(self, message: dict) -> None:
        """Send a non-frame message, e.g. a fault-injection request."""
        self._send(message)

    def command_age_s(self) -> float | None:
        """Seconds since the last command arrived, or None if none has arrived yet (for logging)."""
        return None if self.command_received_at is None else time.monotonic() - self.command_received_at

    def unanswered_s(self) -> float | None:
        """How long the Jetson has been sitting on a frame it was sent without answering.

        This, not command_age_s, is what the watchdog should use: if the HOST pauses
        (e.g. CARLA stalls for a second), no frame is outstanding, so the Jetson is
        not blamed for the host's own delay.
        None until the first command arrives (start-up is not a failure); 0 when nothing is outstanding.
        """
        if self.command_received_at is None:
            return None
        if self._in_flight_since is None:
            return 0.0
        return time.monotonic() - self._in_flight_since

    def _send(self, header: dict, payload: bytes = b"") -> None:
        """Send, but never block the run loop for more than SEND_TIMEOUT_S.

        A Jetson that stops reading is a Jetson failure, which is exactly what the
        experiment tests. So instead of crashing the run, mark the link dead and stop
        sending; the arbitrator then sees the commands go stale and the fallback takes over.
        """
        if self.dead:
            return
        with self._send_lock:
            try:
                send_msg(self.sock, header, payload)
            except (TimeoutError, OSError):
                self.dead = True

    def _read_loop(self) -> None:
        """Background thread: store every command the Jetson sends back."""
        while self.sock is not None:
            try:
                header, _ = recv_msg(self.sock)
            except TimeoutError:
                continue  # nothing received for a while; keep waiting
            except (ConnectionError, OSError):
                return  # connection closed; the arbitrator will notice the commands going stale
            if header.get("type") == "command":
                view = header.get("view") or {}
                for key in ("seg", "depth", "lidar"):
                    if key in view:
                        self.view_maps[key] = {"data": view[key], "size": view.get("size"),
                                               "frame_id": header.get("frame_id")}
                self.latest_command = header
                self.command_received_at = time.monotonic()
                self._in_flight_since = None

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()
            self.sock = None
