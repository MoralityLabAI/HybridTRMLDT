window.drawScene = function (canvas, scene, cast, qstate) {
  var W = canvas.width, H = canvas.height, ctx = canvas.getContext('2d');
  var seed = 0; for (var i = 0; i < scene.id.length; i++) seed += scene.id.charCodeAt(i) * (i + 1);
  var art = window.HILBERT_ART[scene.id];
  if (!art || !art.length) art = ['#1a2a4a', '#2b3a5a', '#ffd97a', 50, 62, 40, 'Bell Tower', ['c1', 'c2', 'c3']];
  var sky = art[0], ground = art[1], light = art[2], xP = art[3], yP = art[4], sc = art[5] / 100, name = art[6], ids = art[7];
  var cx = xP * W / 100, cy = yP * H / 100;
  var g = ctx.createLinearGradient(0, 0, 0, H); g.addColorStop(0, sky); g.addColorStop(1, ground); ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  var rg = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(W, H) * 0.7); rg.addColorStop(0, light + 'aa'); rg.addColorStop(1, 'rgba(0,0,0,0)'); ctx.fillStyle = rg; ctx.fillRect(0, 0, W, H);
  for (var s = 0; s < 24; s++) { var sx = (seed * 7 + s * 31) % W, sy = (seed * 13 + s * 17) % (H * 0.5); ctx.fillStyle = 'rgba(255,255,255,' + (0.2 + (s % 3) * 0.1) + ')'; ctx.fillRect(sx, sy, 2, 2); }
  ctx.fillStyle = ground; ctx.fillRect(0, H * 0.7, W, H * 0.3);
  var style = name.indexOf('Archive') >= 0 ? 0 : name.indexOf('Bell') >= 0 || name.indexOf('Tower') >= 0 ? 1 : name.indexOf('Plaza') >= 0 || name.indexOf('Market') >= 0 ? 2 : name.indexOf('Under') >= 0 || name.indexOf('Door') >= 0 ? 3 : name.indexOf('Page') >= 0 || name.indexOf('Ledger') >= 0 ? 4 : 5;
  var k = (seed % 7) + 1;
  ctx.save(); ctx.translate(cx, cy); ctx.scale(sc, sc);
  if (style === 0) { ctx.fillStyle = light; ctx.fillRect(-80, -120, 160, 240); ctx.fillStyle = 'rgba(0,0,0,0.3)'; for (var r = 0; r < 5; r++) ctx.fillRect(-70 + r * 28, -100 + r * 40, 140, 18); }
  else if (style === 1) { ctx.fillStyle = light; ctx.fillRect(-30, -160, 60, 280); ctx.fillRect(-50, -180, 100, 30); ctx.fillStyle = 'rgba(0,0,0,0.4)'; ctx.fillRect(-20, -150, 40, 40); }
  else if (style === 2) { ctx.fillStyle = light; for (var b = 0; b < 4; b++) ctx.fillRect(-90 + b * 50, -80, 40, 160); ctx.fillStyle = 'rgba(0,0,0,0.3)'; ctx.fillRect(-100, 80, 200, 20); }
  else if (style === 3) { ctx.fillStyle = light; ctx.fillRect(-60, -40, 120, 120); ctx.fillStyle = 'rgba(0,0,0,0.5)'; ctx.fillRect(-40, -20, 80, 80); }
  else if (style === 4) { ctx.fillStyle = light; ctx.fillRect(-70, -60, 140, 180); ctx.fillStyle = 'rgba(0,0,0,0.3)'; for (var l = 0; l < 6; l++) ctx.fillRect(-60 + l * 20, -50 + l * 25, 120, 10); }
  else { ctx.fillStyle = light; ctx.fillRect(-40, -100, 80, 200); ctx.fillStyle = 'rgba(0,0,0,0.3)'; ctx.fillRect(-50, -120, 100, 20); }
  ctx.restore();
  for (var a = 0; a < cast.length; a++) { var c = cast[a]; if (ids.indexOf(c.id) >= 0) { var px = cx + (a - 1) * 60, py = H * 0.85; ctx.fillStyle = light; ctx.beginPath(); ctx.arc(px, py - 40, 12, 0, 7); ctx.fill(); ctx.fillStyle = 'rgba(0,0,0,0.6)'; ctx.fillRect(px - 18, py - 30, 36, 50); ctx.fillStyle = light; ctx.fillRect(px + 14, py - 20, 8, 30); } }
  var born = qstate && qstate.born ? qstate.born : {};
  var clues = (qstate && Array.isArray(qstate.clues) ? qstate.clues.length : 0);
  var ring = Math.max(1, Math.min(6, clues + (born ? 1 : 0)));
  for (var w = 0; w < ring; w++) { ctx.strokeStyle = 'hsla(' + ((seed + w * 40) % 360) + ',80%,60%,0.5)'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(cx, cy, 40 + w * 25, 0, 7); ctx.stroke(); }
};
