const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

// Exercise the actual rendering functions without starting camera polling.
const source = fs.readFileSync(require.resolve('../web/app.js'), 'utf8');
function loadFunction(name, context) {
  const start = source.indexOf(`function ${name}(`);
  const end = source.indexOf('\n}', start) + 2;
  vm.runInNewContext(source.slice(start, end), context);
  return context[name];
}

test('screen badge distinguishes looking, unknown and a different target', () => {
  const badge = loadFunction('screenBadge', {});
  assert.equal(badge({calibrated:true}, 'TV', 'tv'), 'Olhando');
  assert.equal(badge({calibrated:true}, 'NOTEBOOK', 'notebook'), 'Olhando');
  assert.equal(badge({calibrated:true}, 'TV', 'notebook'), 'Não olhando');
  assert.equal(badge({calibrated:true}, 'INDETERMINADO', 'tv'), 'Indeterminado');
  assert.equal(badge({calibrated:false}, 'INDETERMINADO', 'tv'), 'Sem calibração');
});

test('recognized screen remains explicit when its coordinate estimate is bad or unavailable', () => {
  const line = loadFunction('estimateLine', {
    esc:String, levelNames:{baixa:'precisão baixa'}, cameraColors:{validation:'1,2,3'},
    cameraNames:{validation:'Câmera da apresentação'}, partShort:{}
  });
  const estimate = {role:'validation', validation:{level:'baixa'},
    gaze:{surface:'tv', visible:true, inside:true, position_inside:false, x:1.3, y:.5}};
  assert.match(line(estimate), /olhando para esta tela/);
  assert.match(line(estimate), /posição imprecisa/);
  assert.doesNotMatch(line(estimate), /fora desta tela/);
  estimate.gaze = {visible:false, inside:true, reason:'Posição do olhar indisponível'};
  assert.match(line(estimate), /superfície reconhecida/);
  assert.match(line(estimate), /Posição do olhar indisponível/);
});

test('canvas shows recognized screen without inventing a dot when location is missing', () => {
  const text = [], arcs = [];
  const ctx = {clearRect(){},fillRect(){},strokeRect(){},beginPath(){},moveTo(){},lineTo(){},stroke(){},fill(){},
    arc(...args){arcs.push(args);},fillText(value){text.push(value);}};
  const draw = loadFunction('drawScreen', {
    $:() => ({width:600,height:340,getContext:()=>ctx}), window:{devicePixelRatio:1},
    performance:{now:()=>1000}, liveTrails:{}, cameraColors:{validation:'1,2,3'}, smoothed:()=>null
  });
  draw('tv', {calibrated:true, estimates:[{role:'validation', gaze:{visible:false,inside:true}}]}, true);
  assert.ok(text.includes('Olhando para a TV'));
  assert.ok(text.includes('Posição do olhar indisponível'));
  assert.equal(arcs.length, 0);
  text.length = 0;
  draw('tv', {calibrated:true, estimates:[]}, false);
  assert.equal(text.length, 0);
});
