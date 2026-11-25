"""Shortest-route planning on CARLA's road network (Dijkstra).

The map's topology is a list of lane segments (start waypoint -> end waypoint). Each
segment becomes a graph edge weighted by its length; segments join where one ends and
the next starts (including the connecting lanes inside junctions). Dijkstra from the
car's segment gives the shortest drivable route to any destination; where several
routes lead there, the shortest one wins.

Lane changes are not part of the graph: the route stays in the car's lane, which is what
the Jetson's lane following does too. Destinations are offered only if they are
reachable that way.
"""

import heapq
import math
from itertools import pairwise

import carla

STEP_M = 2.0


def _key(loc: carla.Location) -> tuple[int, int, int]:
    """Graph node id: a location rounded to 10 cm (segment ends meet at the same point)."""
    return round(loc.x * 10), round(loc.y * 10), round(loc.z)


def _lane_key(wp: carla.Waypoint) -> tuple[int, int, int]:
    return wp.road_id, wp.section_id, wp.lane_id


class RoutePlanner:
    def __init__(self, carla_map: carla.Map):
        self.map = carla_map
        self.segments = []   # dicts: start, end (node keys), points [(x, y, z, is_junction)], length
        self.out_edges = {}  # node key -> [segment index]
        self.by_lane = {}    # (road, section, lane) -> [segment index]
        for start_wp, end_wp in carla_map.get_topology():
            wps = [start_wp] + start_wp.next_until_lane_end(STEP_M)
            pts = [(w.transform.location.x, w.transform.location.y, w.transform.location.z, w.is_junction) for w in wps]
            end = end_wp.transform.location
            if math.dist(pts[-1][:2], (end.x, end.y)) > 0.5:
                pts.append((end.x, end.y, end.z, end_wp.is_junction))
            length = sum(math.dist(a[:2], b[:2]) for a, b in pairwise(pts))
            seg = {"start": _key(start_wp.transform.location), "end": _key(end), "points": pts,
                   "length": max(length, 0.1), "lane": _lane_key(start_wp)}
            idx = len(self.segments)
            self.segments.append(seg)
            self.out_edges.setdefault(seg["start"], []).append(idx)
            self.by_lane.setdefault(seg["lane"], []).append(idx)

    # ------------------------------------------------------------------
    def locate(self, location: carla.Location) -> tuple[int, int] | None:
        """(segment index, point index) of the nearest point on the segment of the lane at `location`."""
        wp = self.map.get_waypoint(location)
        candidates = self.by_lane.get(_lane_key(wp)) or range(len(self.segments))
        best = None
        for si in candidates:
            for pi, p in enumerate(self.segments[si]["points"]):
                d = math.dist(p[:2], (location.x, location.y))
                if best is None or d < best[0]:
                    best = (d, si, pi)
        return None if best is None else (best[1], best[2])

    def distances_from(self, start: tuple[int, int]) -> dict:
        """Dijkstra: shortest driving distance from `start` to the start node of every reachable segment."""
        si, pi = start
        seg = self.segments[si]
        to_end = sum(math.dist(a[:2], b[:2]) for a, b in zip(seg["points"][pi:], seg["points"][pi + 1:]))
        dist, prev = {seg["end"]: to_end}, {seg["end"]: (None, si)}
        queue = [(to_end, seg["end"])]
        while queue:
            d, node = heapq.heappop(queue)
            if d > dist.get(node, math.inf):
                continue
            for nxt in self.out_edges.get(node, []):
                nd, end = d + self.segments[nxt]["length"], self.segments[nxt]["end"]
                if nd < dist.get(end, math.inf):
                    dist[end], prev[end] = nd, (node, nxt)
                    heapq.heappush(queue, (nd, end))
        return {"dist": dist, "prev": prev, "start": start}

    def route_length(self, tree: dict, goal: tuple[int, int]) -> float | None:
        """Length of the shortest route to `goal` (segment, point), or None if unreachable."""
        gsi, gpi = goal
        seg = self.segments[gsi]
        into = sum(math.dist(a[:2], b[:2]) for a, b in zip(seg["points"][:gpi], seg["points"][1:gpi + 1]))
        ssi, spi = tree["start"]
        if gsi == ssi and gpi >= spi:  # further along the segment we're already on
            return into - sum(math.dist(a[:2], b[:2]) for a, b in zip(seg["points"][:spi], seg["points"][1:spi + 1]))
        d = tree["dist"].get(seg["start"])
        return None if d is None else d + into

    def route(self, start_loc: carla.Location, goal_loc: carla.Location) -> list | None:
        """Shortest route as points [(x, y, is_junction)] about STEP_M apart, or None if unreachable."""
        start, goal = self.locate(start_loc), self.locate(goal_loc)
        if start is None or goal is None:
            return None
        tree = self.distances_from(start)
        if self.route_length(tree, goal) is None:
            return None
        ssi, spi = start
        gsi, gpi = goal
        if gsi == ssi and gpi >= spi:
            pts = self.segments[ssi]["points"][spi:gpi + 1]
        else:
            chain, node = [gsi], self.segments[gsi]["start"]  # walk back through the predecessors
            while True:
                prev_node, seg_idx = tree["prev"][node]
                if prev_node is None:
                    break
                chain.append(seg_idx)
                node = prev_node
            chain.append(ssi)
            chain.reverse()
            pts = list(self.segments[ssi]["points"][spi:])
            for seg_idx in chain[1:-1]:
                pts += self.segments[seg_idx]["points"][1:]
            pts += self.segments[gsi]["points"][1:gpi + 1]
        return [(p[0], p[1], p[3]) for p in pts]

    # ------------------------------------------------------------------
    def destinations(self, start_loc: carla.Location, spawn_points, min_route_m: float = 300.0,
                     min_apart_m: float = 150.0, count: int = 8) -> list[dict]:
        """Spawn points worth driving to: reachable, at least `min_route_m` away by road and at least
        `min_apart_m` from each other in a straight line (picked by farthest-point sampling)."""
        start = self.locate(start_loc)
        if start is None:
            return []
        tree = self.distances_from(start)
        cands = []
        for i, sp in enumerate(spawn_points):
            goal = self.locate(sp.location)
            length = self.route_length(tree, goal) if goal else None
            if length is not None and length >= min_route_m and sp.location.distance(start_loc) >= min_apart_m:
                cands.append({"spawn_index": i, "x": round(sp.location.x, 1), "y": round(sp.location.y, 1),
                              "route_m": round(length)})
        chosen = []
        while cands and len(chosen) < count:
            def gap(c):
                return min((math.dist((c["x"], c["y"]), (o["x"], o["y"])) for o in chosen), default=c["route_m"])
            best = max(cands, key=gap)
            if chosen and gap(best) < min_apart_m:
                break
            chosen.append(best)
            cands.remove(best)
        for n, c in enumerate(sorted(chosen, key=lambda c: c["route_m"])):
            c["label"] = chr(ord("A") + n)
        return sorted(chosen, key=lambda c: c["route_m"])

    def road_lines(self, step: int = 3) -> list[list[list[float]]]:
        """Every lane centreline, thinned, for drawing the map on the dashboard."""
        return [[[round(p[0], 1), round(p[1], 1)] for p in seg["points"][::step] + [seg["points"][-1]]]
                for seg in self.segments]


class RouteFollower:
    """Tracks the car's progress along a planned route and gives the next stretch of it."""

    def __init__(self, points: list):
        self.points = points
        self.cum = [0.0]
        for a, b in pairwise(points):
            self.cum.append(self.cum[-1] + math.dist(a[:2], b[:2]))
        self.idx = 0

    @property
    def length(self) -> float:
        return self.cum[-1]

    def update(self, x: float, y: float, ahead_points: int = 25) -> dict:
        """Advance to the nearest route point (searching forward only) and return what the Jetson needs."""
        window = range(self.idx, min(len(self.points), self.idx + 40))
        self.idx = min(window, key=lambda i: math.dist(self.points[i][:2], (x, y)))
        ahead = self.points[self.idx + 1: self.idx + 1 + ahead_points]
        junction = next((self.cum[i] - self.cum[self.idx] for i in range(self.idx, min(len(self.points), self.idx + 30))
                         if self.points[i][2]), None)
        return {"waypoints": [[round(p[0], 2), round(p[1], 2)] for p in ahead],
                "junction_ahead_m": None if junction is None else round(junction, 1),
                "route_remaining_m": round(self.length - self.cum[self.idx] + math.dist(self.points[self.idx][:2], (x, y)), 1)}
