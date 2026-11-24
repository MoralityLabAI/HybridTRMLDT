// Interactive Hilbert storyworld: The City That Lost Its Night Bell
// Pure JavaScript runtime for the player.js entry point.
// Runtime data (scenes, endings, math) lives in window.HILBERT_WORLD from world-data.js.

(function () {
  'use strict';

  var sceneIndex = 0;
  var clues = new Set();
  var ending = null;
  var amp = [[1, 0], [0, 0], [0, 0], [0, 0]]; // 0='00', 1='01', 2='10', 3='11'

  function prob(state, P) {
    if (P === '00' || P === '01' || P === '10' || P === '11') {
      var idx = ['00', '01', '10', '11'].indexOf(P);
      return state[idx][0] * state[idx][0] + state[idx][1] * state[idx][1];
    }
    var axis = P[0];
    var bit = (P[1] === '0' ? 2 : 1);
    var sign = P[2];
    var sum = 0;
    var i, j;
    if (axis === 'Z') {
      for (i = 0; i < 4; i++) {
        if ((Boolean(i & bit) === (sign === '-'))) {
          sum += state[i][0] * state[i][0] + state[i][1] * state[i][1];
        }
      }
      return sum;
    }
    if (axis === 'X') {
      var sx = (sign === '+' ? 1 : -1);
      for (i = 0; i < 4; i++) {
        if ((i & bit) === 0) {
          j = i | bit;
          sum += ((state[i][0] + sx * state[j][0]) * (state[i][0] + sx * state[j][0]) +
                  (state[i][1] + sx * state[j][1]) * (state[i][1] + sx * state[j][1])) / 2;
        }
      }
      return sum;
    }
    if (axis === 'Y') {
      var sy = (sign === '+' ? 1 : -1);
      for (i = 0; i < 4; i++) {
        if ((i & bit) === 0) {
          j = i | bit;
          sum += ((state[i][0] + sy * state[j][1]) * (state[i][0] + sy * state[j][1]) +
                  (state[i][1] - sy * state[j][0]) * (state[i][1] - sy * state[j][0])) / 2;
        }
      }
      return sum;
    }
    return 0;
  }

  function applyOperator(operatorName) {
    var seq = (window.HILBERT_WORLD && window.HILBERT_WORLD.math && window.HILBERT_WORLD.math.operators[operatorName]) || [];
    for (var k = 0; k < seq.length; k++) {
      var g = seq[k];
      if (g === 'I') continue;
      if (g === 'CX01') {
        var tmp = amp[2]; amp[2] = amp[3]; amp[3] = tmp;
        continue;
      }
      if (g === 'CX10') {
        var tmp2 = amp[1]; amp[1] = amp[3]; amp[3] = tmp2;
        continue;
      }
      var sym = g.charAt(0);
      var bit = (g.charAt(1) === '0' ? 2 : 1);
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

  function getEligibleSecrets() {
    var secrets = window.HILBERT_WORLD.math.secrets || {};
    return Object.keys(secrets).filter(function (id) {
      var r = secrets[id];
      var requiresOk = r.requires.every(function (c) { return clues.has(c); });
      if (!requiresOk) return false;
      return r.projectors.every(function (proj) {
        var P = proj[0], lo = proj[1], hi = proj[2];
        var v = prob(amp, P);
        return v >= lo - 1e-9 && v <= hi + 1e-9;
      });
    }).sort();
  }

  window.gameState = function () {
    var scenes = window.HILBERT_WORLD.scenes || [];
    var sceneId;
    if (ending) {
      sceneId = ending;
    } else if (sceneIndex < scenes.length) {
      sceneId = scenes[sceneIndex].id;
    } else {
      sceneId = 'enc_finale';
    }
    return {
      scene: sceneId,
      index: sceneIndex,
      born: {
        '00': prob(amp, '00'),
        '01': prob(amp, '01'),
        '10': prob(amp, '10'),
        '11': prob(amp, '11')
      },
      clues: Array.from(clues).sort(),
      eligibleSecrets: getEligibleSecrets(),
      ending: ending
    };
  };

  function clearChoices() {
    var choices = document.getElementById('choices');
    while (choices.firstChild) choices.removeChild(choices.firstChild);
  }

  function render() {
    var titleEl = document.getElementById('scene-title');
    var textEl = document.getElementById('scene-text');
    var reactionEl = document.getElementById('reaction');
    var readoutEl = document.getElementById('quantum-readout');
    var choicesEl = document.getElementById('choices');
    var scenes = window.HILBERT_WORLD.scenes || [];

    if (!ending && sceneIndex < scenes.length) {
      var sc = scenes[sceneIndex];
      titleEl.textContent = sc.title;
      textEl.textContent = sc.text;
      clearChoices();
      var opts = sc.options || [];
      for (var j = 0; j < 3 && j < opts.length; j++) {
        (function (idx) {
          var btn = document.createElement('button');
          btn.setAttribute('data-option', String(idx));
          btn.textContent = opts[idx][0];
          btn.addEventListener('click', function () {
            var opt = opts[idx];
            applyOperator(opt[2]);
            if (opt[3]) clues.add(opt[3]);
            reactionEl.textContent = opt[1] || '';
            sceneIndex += (typeof opt[4] === 'number' ? opt[4] : 1);
            render();
          });
          choicesEl.appendChild(btn);
        })(j);
      }
      if (typeof window.drawScene === 'function') {
        window.drawScene(document.getElementById('art'), sc, window.HILBERT_WORLD.cast, window.gameState());
      }
      return;
    }

    if (!ending && sceneIndex >= scenes.length) {
      titleEl.textContent = 'The Night Bell';
      textEl.textContent = 'The city waits for its bell to ring again. Which secret does the quantum state reveal?';
      clearChoices();
      var elig = getEligibleSecrets();
      var endings = window.HILBERT_WORLD.endings || [];
      for (var e = 0; e < endings.length; e++) {
        (function (endObj) {
          if (endObj.secret && elig.indexOf(endObj.id) === -1) return;
          var btn = document.createElement('button');
          btn.setAttribute('data-ending', endObj.id);
          btn.textContent = endObj.title;
          btn.addEventListener('click', function () {
            ending = endObj.id;
            render();
          });
          choicesEl.appendChild(btn);
        })(endings[e]);
      }
      if (typeof window.drawScene === 'function') {
        window.drawScene(document.getElementById('art'), { id: 'enc_finale', title: 'Finale' }, window.HILBERT_WORLD.cast, window.gameState());
      }
      return;
    }

    // Ending state
    var endObj = null;
    var allEndings = window.HILBERT_WORLD.endings || [];
    for (var k = 0; k < allEndings.length; k++) {
      if (allEndings[k].id === ending) { endObj = allEndings[k]; break; }
    }
    if (endObj) {
      titleEl.textContent = endObj.title;
      textEl.textContent = endObj.text;
    } else {
      titleEl.textContent = 'The City That Lost Its Night Bell';
      textEl.textContent = 'The bell rings once more, and the city remembers its silence.';
    }
    clearChoices();
    if (typeof window.drawScene === 'function') {
      window.drawScene(document.getElementById('art'), endObj || { id: ending, title: 'Ending' }, window.HILBERT_WORLD.cast, window.gameState());
    }
  }

  var restartBtn = document.getElementById('restart');
  if (restartBtn) {
    restartBtn.addEventListener('click', function () {
      sceneIndex = 0;
      clues = new Set();
      ending = null;
      amp = [[1, 0], [0, 0], [0, 0], [0, 0]];
      var reactionEl = document.getElementById('reaction');
      if (reactionEl) reactionEl.textContent = '';
      render();
    });
  }

  render();
})();
