const test=require('node:test');
const assert=require('node:assert/strict');
const {classify,decision,project}=require('../web/ray-overlay.js');

test('task color does not require choosing an exact target within the task',()=>{
  const result=decision({target:'INDETERMINADO',regions:{task:{target:'DENTRO',reason:'região calibrada'}}},'task');
  assert.equal(result.state,'green');assert.match(result.label,/alvo exato incerto/);
});
test('an unknown calibrated region is not replaced by an optimistic exact label',()=>{
  const result=decision({target:'TV',regions:{task:{target:'INDETERMINADO',reason:'calibração pendente'}}},'task');
  assert.equal(result.state,'unknown');assert.equal(result.label,'calibração pendente');
});

test('screen reference is green only for notebook/TV, unknown is not red',()=>{
  assert.equal(classify('NOTEBOOK'),'green');assert.equal(classify('TV'),'green');
  assert.equal(classify('NAO'),'red');assert.equal(classify('OUTRO_LOCAL'),'red');
  assert.equal(classify('INDETERMINADO'),'unknown');assert.equal(classify(null),'unknown');
});
test('robot and task references explicitly change the green targets',()=>{
  assert.equal(classify('NAO','robot'),'green');assert.equal(classify('TV','robot'),'red');
  assert.equal(classify('NAO','task'),'green');assert.equal(classify('TV','task'),'green');
  assert.equal(classify('OUTRO_LOCAL','task'),'red');
});
test('rays follow image coordinates with letterboxing, not the full stretched box',()=>{
  const [ray]=project([[.25,.25],[.75,.25]],[.5,0],320,200,640,480);
  assert.ok(Math.abs(ray.x-93.333333)<.001);
  assert.ok(Math.abs(ray.y-50)<.001);
  assert.ok(ray.endX>ray.x);assert.equal(ray.endY,ray.y);
});
test('near frontal direction draws a frontal marker and invalid geometry is rejected',()=>{
  assert.equal(project([[.4,.4],[.6,.4]],[0,0],320,200,640,480)[0].frontal,true);
  assert.deepEqual(project([[NaN,.5],[2,.5]],[1,1],320,200,640,480),[]);
  assert.deepEqual(project(null,[1,1],320,200,640,480),[]);
});
test('endpoints are clipped to the source image, including a long diagonal',()=>{
  const rays=project([[.99,.99],[.01,.01]],[1,1],320,200,640,480);
  for(const r of rays){assert.ok(r.endX<=293.334);assert.ok(r.endY<=200);assert.ok(r.endX>=26.666);}
});
