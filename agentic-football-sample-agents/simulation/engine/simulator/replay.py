"""Write a self-contained HTML replay of a simulated match (open it in any browser)."""

import json

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg:#0f1512; --panel:#18211c; --text:#e8efe9; --muted:#9aa89f; --pitch:#1f5135;
          --line:#cfe3d4; --home:#5aa9ff; --away:#ff6b6b; --ball:#ffffff; --hold:#ffd54a; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width:960px; margin:0 auto; padding:16px; }
  header { display:flex; flex-wrap:wrap; align-items:baseline; gap:8px 16px; margin-bottom:12px; }
  h1 { font-size:20px; margin:0; }
  .score { font-size:28px; font-weight:700; font-variant-numeric:tabular-nums; }
  .home { color:var(--home); } .away { color:var(--away); }
  .clock { color:var(--muted); font-variant-numeric:tabular-nums; }
  canvas { width:100%; height:auto; display:block; border-radius:8px; background:var(--pitch); }
  .controls { display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin:12px 0; }
  button, select { background:var(--panel); color:var(--text); border:1px solid #2c3a31;
                   border-radius:6px; padding:6px 12px; font:inherit; cursor:pointer; }
  input[type=range] { flex:1; min-width:160px; }
  ul { list-style:none; padding:0; margin:0; display:grid; gap:4px; }
  li { background:var(--panel); padding:6px 10px; border-radius:6px; cursor:pointer; }
  li span { color:var(--muted); font-variant-numeric:tabular-nums; margin-right:8px; }
  .note { color:var(--muted); font-size:12px; margin-top:12px; }
</style></head>
<body><main>
  <header>
    <h1><span class="home">__HOME__</span> vs <span class="away">__AWAY__</span></h1>
    <div class="score" id="score">0 - 0</div>
    <div class="clock" id="clock">0:00</div>
  </header>
  <canvas id="pitch" width="1100" height="700" aria-label="Match replay"></canvas>
  <div class="controls">
    <button id="play">Pause</button>
    <select id="speed" aria-label="Playback speed">
      <option value="1">1x</option><option value="2">2x</option>
      <option value="4" selected>4x</option><option value="8">8x</option>
    </select>
    <input type="range" id="seek" min="0" value="0" aria-label="Match time">
  </div>
  <ul id="goals"></ul>
  <p class="note">Simulated with the local stand-in engine (simulator/engine.py), not the real game server.
  Gold ring: ball carrier. Numbers: player index (0 = keeper).</p>
</main>
<script>
const F = __FRAMES__, GOALS = __GOALS__;
const cv = document.getElementById("pitch"), g = cv.getContext("2d");
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const X = x => (x + 55) * 10, Y = y => (35 - y) * 10;
const seek = document.getElementById("seek"); seek.max = F.length - 1;
let i = 0, playing = true;

function pitch() {
  g.fillStyle = css("--pitch"); g.fillRect(0, 0, 1100, 700);
  g.strokeStyle = css("--line"); g.lineWidth = 2;
  g.strokeRect(2, 2, 1096, 696);
  g.beginPath(); g.moveTo(550, 0); g.lineTo(550, 700); g.stroke();
  g.beginPath(); g.arc(550, 350, 60, 0, 2 * Math.PI); g.stroke();
  for (const s of [-1, 1]) {
    const gx = X(s * 55), bx = X(s * (55 - 16.5));
    g.strokeRect(Math.min(gx, bx), Y(20), 165, 400);
    g.lineWidth = 6; g.beginPath(); g.moveTo(gx, Y(3.66)); g.lineTo(gx, Y(-3.66)); g.stroke(); g.lineWidth = 2;
  }
}

function draw() {
  const [t, hs, as, bx, by, ps, holder] = F[i];
  pitch();
  ps.forEach(([x, y], k) => {
    g.beginPath(); g.arc(X(x), Y(y), 13, 0, 2 * Math.PI);
    g.fillStyle = css(k < 5 ? "--home" : "--away"); g.fill();
    if (k === holder) { g.lineWidth = 4; g.strokeStyle = css("--hold"); g.stroke(); g.lineWidth = 2; }
    g.fillStyle = "#0b0f0d"; g.font = "bold 13px system-ui"; g.textAlign = "center"; g.textBaseline = "middle";
    g.fillText(k % 5, X(x), Y(y) + 1);
  });
  g.beginPath(); g.arc(X(bx), Y(by), 6, 0, 2 * Math.PI); g.fillStyle = css("--ball"); g.fill();
  document.getElementById("score").textContent = hs + " - " + as;
  document.getElementById("clock").textContent = Math.floor(t / 60) + ":" + String(Math.floor(t % 60)).padStart(2, "0");
  seek.value = i;
}

let last = 0;
function loop(now) {
  const speed = +document.getElementById("speed").value;
  if (playing && now - last > 200 / speed) { i = Math.min(F.length - 1, i + 1); last = now; draw(); }
  if (i === F.length - 1) { playing = false; document.getElementById("play").textContent = "Replay"; }
  requestAnimationFrame(loop);
}
document.getElementById("play").onclick = e => {
  if (i === F.length - 1) i = 0;
  playing = !playing; e.target.textContent = playing ? "Pause" : "Play";
};
seek.oninput = () => { i = +seek.value; draw(); };
const list = document.getElementById("goals");
GOALS.forEach(([t, team, scorer, score]) => {
  const li = document.createElement("li");
  li.innerHTML = `<span>${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}</span>`
    + `<b class="${team === 0 ? "home" : "away"}">GOAL</b> ${score} `
    + (scorer === null ? "(own goal / deflection)" : `(player ${scorer})`);
  li.onclick = () => { i = Math.max(0, F.findIndex(f => f[0] >= t) - 25); draw(); };
  list.appendChild(li);
});
draw(); requestAnimationFrame(loop);
</script></body></html>
"""


def write_replay(path: str, result) -> None:
    goals = [[e[0], e[2], e[3], e[4]] for e in result.events if e[1] == "goal"]
    html = (_PAGE.replace("__TITLE__", f"{result.home} vs {result.away} replay")
            .replace("__HOME__", result.home).replace("__AWAY__", result.away)
            .replace("__FRAMES__", json.dumps(result.frames, separators=(",", ":")))
            .replace("__GOALS__", json.dumps(goals)))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
