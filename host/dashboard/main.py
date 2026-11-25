"""Builds the FastAPI app: API routes, the status WebSocket and a minimal control page."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from dashboard import camera, routes, websocket

# Deliberately bare-bones: enough to drive experiments. A proper UI can replace it later.
PAGE = """<!doctype html>
<title>AV Test Platform</title>
<style>
  body { font: 14px system-ui, sans-serif; max-width: 760px; margin: 24px auto; padding: 0 16px; }
  label { display: inline-block; width: 150px; } input, select { margin: 3px 0; }
  button { margin: 8px 6px 8px 0; padding: 6px 12px; }
  pre { background: #f3f4f6; padding: 12px; overflow-x: auto; }
  #cam { width: 100%; max-width: 640px; aspect-ratio: 16 / 9; background: #111; display: block; }
  #camInfo { font-family: monospace; margin: 4px 0 8px; }
  #layers { margin: 6px 0; } #layers label { width: auto; margin-right: 12px; }
  #mapbox { margin: 8px 0; } #map { width: 100%; max-width: 640px; aspect-ratio: 4 / 3; background: #1d2430;
    display: block; cursor: crosshair; border-radius: 4px; } #mapInfo { font-family: monospace; margin: 4px 0; }
  .stop { color: #c00; font-weight: bold; } .go { color: #080; font-weight: bold; } .slow { color: #c80; font-weight: bold; } .reverse { color: #70c; font-weight: bold; }
</style>
<h2>Autonomous Vehicle Test Platform</h2>
<form id="cfg">
  <label>Map</label><input name="map" value="Town04"><br>
  <label>Spawn point</label><input name="spawn_index" type="number" value="0">
    offset <input name="spawn_lateral_offset_m" type="number" step="0.1" value="0" style="width:60px"> m (+ right, to test recovery)<br>
  <label>Target speed (m/s)</label><input name="target_speed_mps" type="number" step="0.5" value="8"><br>
  <label>Path source</label><select name="path_source">
    <option value="perception">perception - Jetson follows lanes / road it sees</option>
    <option value="map">map - CARLA's lane-centre waypoints (comparison)</option></select><br>
  <label>At lane splits</label><select name="at_split">
    <option value="right">keep right (e.g. follow an exit-only lane onto its ramp)</option>
    <option value="left">keep left (stay on the through road)</option></select><br>
  <label>Fault</label><select name="fault" id="fault"></select><br>
  <label>Fallback</label><input type="checkbox" id="fallback" checked> enabled
    (untick for a baseline run: the car keeps its last command and crashes)<br>
  <label>Duration (sim s)</label><input name="duration_s" type="number" value="120"><br>
  <label>Pedestrians</label><input name="pedestrians" type="number" min="0" value="0"> crossing the road ahead<br>
  <label>Ped behaviour</label><select name="pedestrian_behaviour">
    <option value="cross">cross the road</option>
    <option value="stop_in_lane">stop in the lane for 15 s (test swerve / reverse)</option></select><br>
  <label>Blocked lane</label><input type="checkbox" id="blockedcar"> parked car in the ego lane 60 m ahead
    <input type="checkbox" id="blockall"> and in the lanes beside it (no gap: tests reversing)<br>
  <label>Red-light queue</label><input type="checkbox" id="redqueue"> cars waiting at the next traffic light (30 s red)<br>
  <label>Ped trigger (m)</label><input name="pedestrian_trigger_m" type="number" step="1" value="35">
    (a pedestrian starts crossing when the car is this close)<br>
  <label>Vehicles</label><input name="vehicles" type="number" min="0" value="0"> other cars on autopilot nearby<br>
  <label>Time scale</label><input name="time_scale" type="number" step="0.1" min="0.05" max="1" value="1">
    (1 = real time, 0.3 = slow motion)<br>
  <label>Step (s)</label><input name="step_s" type="number" step="0.01" value="0.1"> (0.1 = 10 Hz, 0.05 = 20 Hz)<br>
  <input type="hidden" name="destination_index" id="destination" value="-1">
</form>
<div id="mapbox">
  <button type="button" onclick="loadMap()">Load map</button> pick a destination (letters), then Configure.
  <button type="button" onclick="pickDestination(null)">No destination</button>
  <canvas id="map" width="640" height="480"></canvas>
  <div id="mapInfo">map: click "Load map" (uses the Map and Spawn point above; loading a town takes a while)</div>
</div>
<button onclick="configure()">Configure</button>
<button onclick="post('start')">Start</button>
<button onclick="post('fault')">Inject fault</button>
<button onclick="post('stop')">Stop</button>
<button onclick="post('reset')">Reset</button>
<p id="msg"></p>
<div id="layers">Show:
  <label><input type="checkbox" data-layer="boxes" checked> objects</label>
  <label><input type="checkbox" data-layer="lanes" checked> lanes</label>
  <label><input type="checkbox" data-layer="path" checked> path</label>
  <label>segmentation <select data-layer="seg"><option value="off">off</option>
    <option value="road" selected>road</option><option value="all">all classes</option></select></label>
  <label><input type="checkbox" data-layer="depth"> depth</label>
  <label><input type="checkbox" data-layer="lidar" checked> LiDAR view</label>
</div>
<canvas id="cam" width="640" height="360"></canvas>
<canvas id="bev" width="320" height="320" style="width:320px;max-width:100%;background:#0e1117;margin-top:6px;border-radius:4px"></canvas>
<div id="camInfo">camera: waiting for a run...</div>
<pre id="status">connecting...</pre>
<script>
const msg = document.getElementById('msg');
const TEXT_FIELDS = ['map', 'fault', 'path_source', 'at_split', 'pedestrian_behaviour'];
async function post(action, body) {
  const r = await fetch('/api/' + action, {method: 'POST', headers: {'Content-Type': 'application/json'},
                                          body: body ? JSON.stringify(body) : undefined});
  msg.textContent = r.ok ? '' : 'Error: ' + (await r.json()).detail;
}
function configure() {
  const f = new FormData(document.getElementById('cfg')), cfg = {};
  for (const [k, v] of f) cfg[k] = TEXT_FIELDS.includes(k) ? v : Number(v);
  cfg.fallback_enabled = document.getElementById('fallback').checked;
  cfg.blocked_lane_car = document.getElementById('blockedcar').checked;
  cfg.block_all_lanes = document.getElementById('blockall').checked;
  cfg.red_light_queue = document.getElementById('redqueue').checked;
  post('config', cfg);
}
fetch('/api/faults').then(r => r.json()).then(faults => {
  for (const name in faults) document.getElementById('fault').add(new Option(name + ' - ' + faults[name], name));
});
const ws = new WebSocket(`ws://${location.host}/ws`);
ws.onmessage = e => document.getElementById('status').textContent = JSON.stringify(JSON.parse(e.data), null, 2);
ws.onclose = () => document.getElementById('status').textContent = 'disconnected from host';

// ---- overlay layers (remembered in this browser) ----
const layers = {};
for (const el of document.querySelectorAll('[data-layer]')) {
  const key = el.dataset.layer;
  try { const saved = localStorage.getItem('layer_' + key); if (saved !== null) { if (el.type === 'checkbox') el.checked = saved === '1'; else el.value = saved; } } catch (e) {}
  const read = () => { layers[key] = el.type === 'checkbox' ? el.checked : el.value; };
  read();
  el.addEventListener('change', () => { read(); try { localStorage.setItem('layer_' + key, el.type === 'checkbox' ? (el.checked ? '1' : '0') : el.value); } catch (e) {} redraw(); });
}

// Cityscapes colours (class id -> RGB); 0 = road
const PALETTE = [[128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],[153,153,153],[250,170,30],[220,220,0],
  [107,142,35],[152,251,152],[70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],[0,60,100],[0,80,100],[0,0,230],[119,11,32]];
const LANE_COLOURS = {ego_left: '#00ff00', ego_right: '#ff3030', outer_left: '#3080ff', outer_right: '#ffff00'};
function mapCanvas(map, colour) {  // base64 uint8 map -> small canvas coloured by `colour(value) -> [r,g,b,a]`
  const [w, h] = map.size, bytes = Uint8Array.from(atob(map.data), c => c.charCodeAt(0));
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const g = c.getContext('2d'), img = g.createImageData(w, h);
  for (let i = 0; i < bytes.length; i++) img.data.set(colour(bytes[i]), i * 4);
  g.putImageData(img, 0, 0); return c;
}
const maps = {seg: null, depth: null};
let segRaw = null;
function buildSeg() {
  if (!segRaw || layers.seg === 'off') { maps.seg = null; return; }
  maps.seg = mapCanvas(segRaw, v => layers.seg === 'road' ? (v === 0 ? [0, 255, 90, 90] : [0, 0, 0, 0]) : [...(PALETTE[v] || [0, 0, 0]), 130]);
}
function depthColour(v) {  // v = metres * 4; near = yellow, far = dark purple
  const t = Math.min(1, (v / 4) / 40);
  return [Math.round(255 * (1 - t * 0.75)), Math.round(220 * (1 - t) ** 1.5), Math.round(80 + 100 * t), 150];
}

// ---- live camera ----
const cam = document.getElementById('cam'), ctx = cam.getContext('2d'), camInfo = document.getElementById('camInfo');
let meta = null, frame = null;
function redraw() {
  if (!frame) return;
  buildSeg();
  const m = meta, W = cam.width, H = cam.height;
  ctx.drawImage(frame, 0, 0);
  ctx.imageSmoothingEnabled = false;
  if (maps.seg) ctx.drawImage(maps.seg, 0, 0, W, H);
  ctx.imageSmoothingEnabled = true;
  if (layers.depth && maps.depth) ctx.drawImage(maps.depth, 0, 0, W, H);
  if (!m) return;
  if (layers.lanes) for (const [name, pts] of Object.entries(m.lanes || {})) {
    ctx.fillStyle = LANE_COLOURS[name] || '#fff';
    for (const [u, v] of pts) { ctx.beginPath(); ctx.arc(u, v, 3, 0, 7); ctx.fill(); }
  }
  const path = m.path || {};
  if (layers.path && path.centreline_px && path.centreline_px.length > 1) {
    ctx.strokeStyle = '#00e5ff'; ctx.lineWidth = 3; ctx.beginPath();
    path.centreline_px.forEach(([u, v], i) => i ? ctx.lineTo(u, v) : ctx.moveTo(u, v)); ctx.stroke();
  }
  if (layers.path && path.target_px) {
    ctx.strokeStyle = '#ff00ff'; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(path.target_px[0], path.target_px[1], 8, 0, 7); ctx.stroke();
  }
  const ob = m.obstacle && m.obstacle.bbox ? m.obstacle.bbox.join() : null;
  ctx.font = '13px monospace'; ctx.textBaseline = 'bottom';
  if (layers.boxes) for (const d of m.detections) {
    const [x1, y1, x2, y2] = d.bbox, hit = ob && d.bbox.join() === ob;
    ctx.strokeStyle = hit ? '#ff2020' : '#20e040'; ctx.lineWidth = hit ? 3 : 1.5;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `${d.class} ${d.confidence.toFixed(2)}` + (hit ? ` · ${m.obstacle.distance_m} m` : '');
    const w = ctx.measureText(label).width + 6;
    ctx.fillStyle = hit ? '#ff2020' : '#20e040'; ctx.fillRect(x1, Math.max(0, y1 - 16), w, 16);
    ctx.fillStyle = '#000'; ctx.fillText(label, x1 + 3, Math.max(16, y1));
  }
}
function connectCamera() {
  const cws = new WebSocket(`ws://${location.host}/ws/camera`);
  cws.binaryType = 'blob';
  cws.onmessage = async e => {
    if (typeof e.data === 'string') {
      meta = JSON.parse(e.data);
      if (meta.maps.seg) segRaw = meta.maps.seg;
      if (meta.maps.depth) maps.depth = mapCanvas(meta.maps.depth, depthColour);
      if (meta.maps.lidar) lidarPts = new Int16Array(Uint8Array.from(atob(meta.maps.lidar.data), c => c.charCodeAt(0)).buffer);
      drawBev();
      return;
    }
    frame = await createImageBitmap(e.data);
    if (cam.width !== frame.width) { cam.width = frame.width; cam.height = frame.height; }
    redraw();
    drawBev();
    const m = meta || {}, t = m.timings_ms || {}, path = m.path || {};
    const dec = m.decision ? `<span class="${m.decision}">${m.decision.toUpperCase()}</span>` + (m.reason ? ` (${m.reason})` : '') : '-';
    const models = Object.entries(t).map(([k, v]) => `${k} ${v}`).join(' · ') || '-';
    const man = m.maneuver && m.maneuver !== 'drive' ? ` · <b>${m.maneuver.toUpperCase().replace('_', ' ')}</b>` : '';
    camInfo.innerHTML = `${dec}${man} · path: ${path.source ?? '-'} · ${(m.detections || []).length} objects<br>` +
                        `ms: ${models} · pipeline ${m.pipeline_ms ?? '-'} · frame ${m.frame_id ?? '-'}`;
  };
  cws.onclose = () => setTimeout(connectCamera, 2000);  // reconnect if the host restarts
}
connectCamera();

// ---- LiDAR bird's-eye view: grey = above-ground points, orange = clusters, red = obstacle in the path,
//      cyan = planned path, white = our car. Up = ahead; 1 px = 0.15 m (about 40 m ahead, 8 m behind).
const bev = document.getElementById('bev'), bctx = bev.getContext('2d');
let lidarPts = null;
function drawBev() {
  bev.style.display = layers.lidar ? 'block' : 'none';
  if (!layers.lidar) return;
  const S = 1 / 0.15, cx = bev.width / 2, cy = bev.height - 50, m = meta || {};
  const B = (ahead, right) => [cx + right * S, cy - ahead * S];
  bctx.fillStyle = '#0e1117'; bctx.fillRect(0, 0, bev.width, bev.height);
  bctx.fillStyle = '#8b95a5';
  if (lidarPts) for (let i = 0; i + 1 < lidarPts.length; i += 2) { const [x, y] = B(lidarPts[i] / 10, lidarPts[i + 1] / 10); bctx.fillRect(x, y, 1.5, 1.5); }
  const cl = (m.path && m.path.centreline) || [];
  if (cl.length > 1) { bctx.strokeStyle = '#00e5ff'; bctx.lineWidth = 2; bctx.beginPath(); cl.forEach(([a, r], i) => i ? bctx.lineTo(...B(a, r)) : bctx.moveTo(...B(a, r))); bctx.stroke(); }
  const ob = m.obstacle && m.obstacle.extent ? m.obstacle.extent.join() : null;
  for (const c of m.lidar_clusters || []) {
    const [a0, a1, r0, r1] = c.extent, hit = ob && c.extent.join() === ob, [x0, y0] = B(a1, r0), [x1, y1] = B(a0, r1);
    bctx.strokeStyle = hit ? '#ff3030' : '#f59e0b'; bctx.lineWidth = hit ? 2.5 : 1.2; bctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  }
  const [ex, ey] = B(2.4, -1.0); bctx.fillStyle = '#fff'; bctx.fillRect(ex, ey, 2 * S, 4.8 * S);
}

// ---- route map: roads, start, destinations (Dijkstra shortest route), planned route, live car ----
const mapC = document.getElementById('map'), mctx = mapC.getContext('2d'), mapInfo = document.getElementById('mapInfo');
let mapData = null, view = null, routeLine = [], routeRun = null, ego = null, chosen = null;
async function loadMap() {
  const f = document.getElementById('cfg');
  const town = f.elements.map.value, spawn = Number(f.elements.spawn_index.value);
  const url = `/api/map?town=${encodeURIComponent(town)}&spawn=${spawn}`, started = Date.now();
  let d;
  while (true) {  // the host loads the town in the background; ask again until it's ready
    const r = await fetch(url);
    if (!r.ok) { mapInfo.textContent = 'Error: ' + (await r.json()).detail; return; }
    d = await r.json();
    if (d.status === 'ready') break;
    mapInfo.textContent = `loading ${town} in CARLA... ${Math.round((Date.now() - started) / 1000)} s (can take a minute or two)`;
    await new Promise(res => setTimeout(res, 2000));
  }
  mapData = d;
  let xs = [], ys = [];
  for (const line of mapData.roads) for (const [x, y] of line) { xs.push(x); ys.push(y); }
  const minx = Math.min(...xs), maxx = Math.max(...xs), miny = Math.min(...ys), maxy = Math.max(...ys);
  const sc = Math.min((mapC.width - 30) / (maxx - minx), (mapC.height - 30) / (maxy - miny));
  view = {sc, ox: 15 - minx * sc + (mapC.width - 30 - (maxx - minx) * sc) / 2, oy: 15 - miny * sc + (mapC.height - 30 - (maxy - miny) * sc) / 2};
  pickDestination(null);
  mapInfo.textContent = `${mapData.town}: ${mapData.destinations.length} destinations reachable from spawn ${mapData.spawn_index} (≥ 300 m by road)`;
}
const P = (x, y) => [view.ox + x * view.sc, view.oy + y * view.sc];
function pickDestination(d) {
  chosen = d;
  document.getElementById('destination').value = d ? d.spawn_index : -1;
  if (mapData) mapInfo.textContent = d ? `destination ${d.label}: spawn point ${d.spawn_index}, ${d.route_m} m by the shortest route`
                                       : 'no destination: the car just follows the road';
  drawMap();
}
mapC.addEventListener('click', e => {
  if (!mapData) return;
  const b = mapC.getBoundingClientRect(), mx = (e.clientX - b.left) * mapC.width / b.width, my = (e.clientY - b.top) * mapC.height / b.height;
  let best = null, bd = 1e9;
  for (const d of mapData.destinations) { const [px, py] = P(d.x, d.y), dd = Math.hypot(px - mx, py - my); if (dd < bd) { bd = dd; best = d; } }
  if (best && bd < 25) pickDestination(best);
});
function drawMap() {
  mctx.fillStyle = '#1d2430'; mctx.fillRect(0, 0, mapC.width, mapC.height);
  if (!mapData) return;
  mctx.strokeStyle = '#6b7686'; mctx.lineWidth = 1;
  for (const line of mapData.roads) { mctx.beginPath(); line.forEach(([x, y], i) => i ? mctx.lineTo(...P(x, y)) : mctx.moveTo(...P(x, y))); mctx.stroke(); }
  if (routeLine.length > 1) {
    mctx.strokeStyle = '#00e5ff'; mctx.lineWidth = 3; mctx.beginPath();
    routeLine.forEach(([x, y], i) => i ? mctx.lineTo(...P(x, y)) : mctx.moveTo(...P(x, y))); mctx.stroke();
  }
  const [sx, sy] = P(mapData.start[0], mapData.start[1]);
  mctx.fillStyle = '#3b82f6'; mctx.beginPath(); mctx.arc(sx, sy, 6, 0, 7); mctx.fill();
  mctx.font = 'bold 12px system-ui'; mctx.textAlign = 'center'; mctx.textBaseline = 'middle';
  for (const d of mapData.destinations) {
    const [px, py] = P(d.x, d.y), on = chosen && chosen.spawn_index === d.spawn_index;
    mctx.fillStyle = on ? '#ef4444' : '#f59e0b'; mctx.beginPath(); mctx.arc(px, py, on ? 10 : 8, 0, 7); mctx.fill();
    mctx.fillStyle = '#000'; mctx.fillText(d.label, px, py + 0.5);
  }
  if (ego) { const [ex, ey] = P(ego[0], ego[1]); mctx.fillStyle = '#fff'; mctx.beginPath(); mctx.arc(ex, ey, 5, 0, 7); mctx.fill(); }
}
setInterval(async () => {  // live car position and the run's planned route
  if (!mapData) return;
  try {
    const r = await (await fetch('/api/route')).json();
    ego = r.ego;
    if (r.experiment_id !== routeRun || (r.route.length && !routeLine.length)) { routeRun = r.experiment_id; routeLine = r.route; }
    drawMap();
  } catch (e) {}
}, 1000);
drawMap();
</script>
"""


def create_app(manager) -> FastAPI:
    app = FastAPI(title="AV Test Platform")
    app.state.manager = manager  # shared by every route and the WebSocket
    app.include_router(routes.router)
    app.include_router(websocket.router)
    app.include_router(camera.router)
    app.add_api_route("/", lambda: HTMLResponse(PAGE), include_in_schema=False)
    return app
