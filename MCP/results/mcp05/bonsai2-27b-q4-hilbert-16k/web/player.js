(function () {
  'use strict';
  var W = window.HILBERT_WORLD;
  var sceneIndex = 0;
  var clues = new Set();
  var ending = null;
  var amp = [[1, 0], [0, 0], [0, 0], [0, 0]];

  function applyOperator(name) {
    var seq = W.math.operators[name] || [];
    for (var k = 0; k < seq.length; k++) {
      var g = seq[k];
      if (g === 'I') continue;
      if (g === 'CX01') { var t = amp[2]; amp[2] = amp[3]; amp[3] = t; continue; }
      if (g === 'CX10') { var u = amp[1]; amp[1] = amp[3]; amp[3] = u; continue; }
      var sym = g[0];
      var bit = (g[1] === '0' ? 2 : 1);
      for (var i = 0; i < 4; i++) {
        if ((i & bit) !== 0) continue;
        var j = i | bit;
        var ar = amp[i][0], ai = amp[i][1];
        var br = amp[j][0], bi = amp[j][1];
        if (sym === 'H') {
          amp[i] = [(ar + br) / Math.SQRT2, (ai + bi) / Math.SQRT2];
          amp[j] = [(ar - br) / Math.SQRT2, (ai - bi) / Math.SQRT2];
        } else if (sym === 'X') {
          amp[i] = [br, bi];
          amp[j] = [ar, ai];
        } else if (sym === 'Z') {
          amp[j] = [-br, -bi];
        } else if (sym === 'S') {
          amp[j] = [-bi, br];
        }
      }
    }
  }

  function prob(a, P) {
    if (P === '00' || P === '01' || P === '10' || P === '11') {
      var idx = ['00', '01', '10', '11'].indexOf(P);
      return a[idx][0] * a[idx][0] + a[idx][1] * a[idx][1];
    }
    var axis = P[0], bit = (P[1] === '0' ? 2 : 1), sign = P[2], sum = 0;
    if (axis === 'Z') {
      for (var i = 0; i < 4; i++) {
        if (Boolean(i & bit) === (sign === '-')) sum += a[i][0] * a[i][0] + a[i][1] * a[i][1];
      }
      return sum;
    }
    if (axis === 'X') {
      var sx = (sign === '+' ? 1 : -1);
      for (var ix = 0; ix < 4; ix++) {
        if ((ix & bit) !== 0) continue;
        var jx = ix | bit;
        sum += ((a[ix][0] + sx * a[jx][0]) * (a[ix][0] + sx * a[jx][0]) + (a[ix][1] + sx * a[jx][1]) * (a[ix][1] + sx * a[jx][1])) / 2;
      }
      return sum;
    }
    if (axis === 'Y') {
      var sy = (sign === '+' ? 1 : -1);
      for (var iy = 0; iy < 4; iy++) {
        if ((iy & bit) !== 0) continue;
        var jy = iy | bit;
        sum += ((a[iy][0] + sy * a[jy][1]) * (a[iy][0] + sy * a[jy][1]) + (a[iy][1] - sy * a[jy][0]) * (a[iy][1] - sy * a[jy][0])) / 2;
      }
      return sum;
    }
    return 0;
  }

  function getEligibleSecrets() {
    return Object.keys(W.math.secrets).filter(function (id) {
      var r = W.math.secrets[id];
      return r.requires.every(function (c) { return clues.has(c); }) &&
        r.projectors.every(function (p) {
          var v = prob(amp, p[0]);
          return v >= p[1] - 1e-9 && v <= p[2] + 1e-9;
        });
    }).sort();
  }

  window.gameState = function () {
    return {
      scene: ending ? ending : (sceneIndex < W.scenes.length ? W.scenes[sceneIndex].id : 'enc_finale'),
      index: sceneIndex,
      born: { '00': prob(amp, '00'), '01': prob(amp, '01'), '10': prob(amp, '10'), '11': prob(amp, '11') },
      clues: Array.from(clues).sort(),
      eligibleSecrets: getEligibleSecrets(),
      ending: ending
    };
  };

  function render() {
    var titleEl = document.getElementById('scene-title');
    var textEl = document.getElementById('scene-text');
    var choicesEl = document.getElementById('choices');
    choicesEl.innerHTML = '';
    if (!ending && sceneIndex < W.scenes.length) {
      var sc = W.scenes[sceneIndex];
      titleEl.textContent = sc.title;
      textEl.textContent = sc.text;
      for (var j = 0; j < 3; j++) {
        let opt = sc.options[j];
        let btn = document.createElement('button');
        btn.setAttribute('data-option', String(j));
        btn.textContent = opt[0];
        btn.addEventListener('click', function () {
          applyOperator(opt[2]);
          if (opt[3]) clues.add(opt[3]);
          document.getElementById('reaction').textContent = opt[1];
          sceneIndex += opt[4];
          render();
        });
        choicesEl.appendChild(btn);
      }
      window.drawScene(document.getElementById('art'), sc, W.cast, window.gameState());
    } else if (!ending && sceneIndex >= W.scenes.length) {
      titleEl.textContent = 'The City That Lost Its Bell';
      textEl.textContent = 'The bell has fallen silent. Choose your ending.';
      let elig = getEligibleSecrets();
      for (var e = 0; e < W.endings.length; e++) {
        let eo = W.endings[e];
        if (!eo.secret || elig.includes(eo.id)) {
          let btn = document.createElement('button');
          btn.setAttribute('data-ending', eo.id);
          btn.textContent = eo.title;
          btn.addEventListener('click', function () {
            ending = eo.id;
            render();
          });
          choicesEl.appendChild(btn);
        }
      }
      window.drawScene(document.getElementById('art'), { id: 'enc_finale', title: 'Finale' }, W.cast, window.gameState());
    } else {
      let endObj = null;
      for (var f = 0; f < W.endings.length; f++) {
        if (W.endings[f].id === ending) { endObj = W.endings[f]; break; }
      }
      titleEl.textContent = endObj.title;
      textEl.textContent = endObj.text;
      window.drawScene(document.getElementById('art'), endObj, W.cast, window.gameState());
    }
  }

  document.getElementById('restart').addEventListener('click', function () {
    sceneIndex = 0;
    amp = [[1, 0], [0, 0], [0, 0], [0, 0]];
    clues = new Set();
    ending = null;
    render();
  });

  render();
})();