window.drawScene = function(canvas, scene, cast, qstate) {
  var ctx = canvas.getContext('2d');
  var W = canvas.width, H = canvas.height;
  var art = (window.HILBERT_ART && window.HILBERT_ART[scene.id]) || ['#1b2340', '#3a2c22', '#f4c95d', 50, 62, 70, 'The Bell Tower', []];
  var sky = art[0], ground = art[1], light = art[2], px = art[3], py = art[4], sc = (art[5] || 60) / 100, landmark = art[6] || 'The Bell Tower', actors = art[7] || [];
  var cx = W * px / 100, cy = H * py / 100;
  var g = ctx.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, sky); g.addColorStop(0.62, shade(sky, -18)); g.addColorStop(1, ground);
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  var rg = ctx.createRadialGradient(cx, cy, 4, cx, cy, Math.max(W, H) * 0.75);
  rg.addColorStop(0, hexA(light, 0.55)); rg.addColorStop(0.35, hexA(light, 0.16)); rg.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = rg; ctx.fillRect(0, 0, W, H);
  drawStars(ctx, W, H, scene.id);
  drawHaze(ctx, W, H, sky, ground);
  drawSilhouette(ctx, W, H, ground, scene.id);
  var seed = hash(scene.id);
  var ringCols = ['#ffd166', '#ef476f', '#06d6a0', '#118ab2', '#f78c6b', '#c77dff'];
  var rc = ringCols[seed % ringCols.length];
  var rings = qstate && (qstate.born || qstate.clues) ? 4 : 2;
  for (var i = 0; i < rings; i++) {
    ctx.beginPath();
    ctx.strokeStyle = hexA(rc, 0.5 - i * 0.11);
    ctx.lineWidth = 3 + (i % 2) * 2;
    ctx.ellipse(cx, cy, (30 + i * 26) * sc + 18, (12 + i * 10) * sc + 8, 0, 0, Math.PI * 2);
    ctx.stroke();
  }
  drawLandmark(ctx, cx, cy, sc, landmark, light, ground, seed);
  drawActors(ctx, cx, cy, sc, actors, W, H, seed);
};
function hash(s) { var h = 2166136261; for (var i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return Math.abs(h >>> 0); }
function hexA(hex, a) { var c = parseHex(hex); return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')'; }
function shade(hex, amt) { var c = parseHex(hex); return 'rgb(' + clamp(c[0] + amt) + ',' + clamp(c[1] + amt) + ',' + clamp(c[2] + amt) + ')'; }
function parseHex(hex) { var h = String(hex || '#000000').replace('#', ''); if (h.length === 3) h = h.split('').map(function (x) { return x + x; }).join(''); var n = parseInt(h.slice(0, 6), 16) || 0; return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function clamp(v) { return Math.max(0, Math.min(255, v | 0)); }
function drawStars(ctx, W, H, id) { var r = hash(id + 'stars'); for (var i = 0; i < 46; i++) { var x = ((r * (i + 7) * 2654435761) >>> 0) % W; var y = (((r * (i + 13) * 40503) >>> 0) % Math.floor(H * 0.42)); var s = 0.6 + ((r + i * 31) % 3) * 0.5; ctx.fillStyle = 'rgba(255,255,255,' + (0.35 + ((i * 17) % 50) / 100) + ')'; ctx.beginPath(); ctx.arc(x, y, s, 0, Math.PI * 2); ctx.fill(); } }
function drawHaze(ctx, W, H, sky, ground) { var g = ctx.createLinearGradient(0, H * 0.42, 0, H * 0.78); g.addColorStop(0, 'rgba(255,255,255,0)'); g.addColorStop(0.5, hexA(ground, 0.28)); g.addColorStop(1, 'rgba(255,255,255,0)'); ctx.fillStyle = g; ctx.fillRect(0, H * 0.42, W, H * 0.36); }
function drawSilhouette(ctx, W, H, ground, id) { var r = hash(id + 'sil'); ctx.fillStyle = 'rgba(8,10,18,0.82)'; ctx.beginPath(); ctx.moveTo(0, H * 0.78); for (var x = 0; x <= W; x += 26) { var h = H * (0.66 + (((r + x * 7) % 23) / 100)); ctx.lineTo(x, h); } ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath(); ctx.fill(); }
function drawLandmark(ctx, cx, cy, sc, name, light, ground, seed) { var s = Math.max(0.35, sc); ctx.save(); ctx.translate(cx, cy); ctx.scale(s, s); ctx.fillStyle = '#141a2e'; ctx.strokeStyle = hexA(light, 0.9); ctx.lineWidth = 2; var n = name.toLowerCase(); if (n.indexOf('tower') >= 0) { tower(ctx, light); } else if (n.indexOf('chamber') >= 0 || n.indexOf('ledger') >= 0 || n.indexOf('coin') >= 0) { chamber(ctx, light, seed); } else if (n.indexOf('stair') >= 0) { stairs(ctx, light, seed); } else if (n.indexOf('merchant') >= 0 || n.indexOf('exchange') >= 0 || n.indexOf('bargain') >= 0) { merchant(ctx, light, seed); } else if (n.indexOf('square') >= 0 || n.indexOf('hour') >= 0 || n.indexOf('watcher') >= 0) { hour(ctx, light, seed); } else { tower(ctx, light); } ctx.restore(); }
function tower(ctx, light) { ctx.fillRect(-16, -78, 32, 78); ctx.beginPath(); ctx.moveTo(-22, -78); ctx.lineTo(0, -108); ctx.lineTo(22, -78); ctx.closePath(); ctx.fill(); ctx.fillRect(-4, -96, 8, 18); ctx.beginPath(); ctx.arc(0, -34, 9, 0, Math.PI * 2); ctx.stroke(); ctx.fillStyle = hexA(light, 0.95); ctx.beginPath(); ctx.arc(0, -34, 4, 0, Math.PI * 2); ctx.fill(); }
function chamber(ctx, light, seed) { ctx.fillRect(-34, -52, 68, 52); ctx.beginPath(); ctx.moveTo(-40, -52); ctx.lineTo(0, -84); ctx.lineTo(40, -52); ctx.closePath(); ctx.fill(); ctx.fillRect(-8, -30, 16, 30); ctx.strokeRect(-24, -44, 12, 10); ctx.strokeRect(12, -44, 12, 10); }
function stairs(ctx, light, seed) { for (var i = 0; i < 6; i++) { ctx.fillRect(-34 + i * 12, -8 - i * 11, 12, 11); } ctx.beginPath(); ctx.moveTo(-40, 0); ctx.lineTo(40, 0); ctx.lineTo(40, 6); ctx.lineTo(-40, 6); ctx.closePath(); ctx.fill(); ctx.strokeStyle = hexA(light, 0.8); ctx.beginPath(); ctx.moveTo(-34, -70); ctx.lineTo(34, -70); ctx.stroke(); }
function merchant(ctx, light, seed) { ctx.fillRect(-30, -46, 60, 46); ctx.beginPath(); ctx.moveTo(-36, -46); ctx.lineTo(0, -72); ctx.lineTo(36, -46); ctx.closePath(); ctx.fill(); ctx.fillRect(-6, -22, 12, 22); ctx.strokeRect(-20, -38, 10, 8); ctx.strokeRect(10, -38, 10, 8); }
function hour(ctx, light, seed) { ctx.beginPath(); ctx.arc(0, -40, 26, 0, Math.PI * 2); ctx.fill(); ctx.fillStyle = '#141a2e'; ctx.beginPath(); ctx.arc(0, -40, 20, 0, Math.PI * 2); ctx.fill(); ctx.strokeStyle = hexA(light, 0.95); ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(0, -40); ctx.lineTo(0, -54); ctx.moveTo(0, -40); ctx.lineTo(9, -34); ctx.stroke(); }
function drawActors(ctx, cx, cy, sc, actors, W, H, seed) { var n = actors.length || 1; for (var i = 0; i < n; i++) { var side = i % 2 === 0 ? -1 : 1; var x = cx + side * (34 + i * 22) * Math.max(0.6, sc); var y = cy + 4; drawFigure(ctx, x, y, 0.8 + (i % 3) * 0.18, i === 0 ? '#f4c95d' : i === 1 ? '#ef476f' : '#06d6a0', seed + i); } }
function drawFigure(ctx, x, y, s, col, seed) { ctx.save(); ctx.translate(x, y); ctx.scale(s, s); ctx.fillStyle = '#0b0f1d'; ctx.beginPath(); ctx.arc(0, -34, 7, 0, Math.PI * 2); ctx.fill(); ctx.beginPath(); ctx.moveTo(-9, -27); ctx.quadraticCurveTo(0, -33, 9, -27); ctx.lineTo(12, 0); ctx.lineTo(-12, 0); ctx.closePath(); ctx.fill(); ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(8, -18); ctx.lineTo(16, -10); ctx.stroke(); ctx.beginPath(); ctx.arc(17, -9, 3.4, 0, Math.PI * 2); ctx.fillStyle = col; ctx.fill(); ctx.restore(); }
