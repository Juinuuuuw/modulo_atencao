"use strict";
const byId=id=>document.getElementById(id);
const display=new URLSearchParams(location.search).get("display")==="tv"?"tv":"notebook";
const clientKey=`observa-display-client-${display}`;
const client=sessionStorage.getItem(clientKey)||(typeof window.crypto?.randomUUID==="function"?window.crypto.randomUUID():`display-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`);
sessionStorage.setItem(clientKey,client);
let enabled=false,ready=false,current=null,busy=false,lastCueId=null,speechTimer=null;
let screenState=null,participantLoaded=false,reticleTarget=null,reticlePoint=null,lastPaint=null;
let layoutSignature=null,layoutStableSince=performance.now();
const testPoints=[[.5,.5],[.25,.25],[.75,.25],[.75,.75],[.25,.75]];
let testIndex=0,referencePoint=testPoints[0],previewBusy=false;
byId("display-mode").value=new URLSearchParams(location.search).get("mode")==="calibration"?"calibration":"gaze";
function placePoint(marker){byId("fixation-point").style.left=`${marker[0]*100}%`;byId("fixation-point").style.top=`${marker[1]*100}%`;}
function mode(){return byId("display-mode").value;}
function setReticle(gaze,checking=false){
  reticleTarget=gaze?.visible?{x:gaze.x,y:gaze.y,received:performance.now(),age:gaze.age||0}:null;
  byId("gaze-reticle").dataset.checking=String(checking);
  byId("gaze-reticle").dataset.verified=String(gaze?.verified===true);
  if(!reticleTarget){byId("gaze-reticle").hidden=true;reticlePoint=null;}
}
function errorText(gaze,point){
  if(!gaze?.visible||!point)return "";
  return `Distância estimada ao amarelo: ${Math.round(Math.hypot((gaze.x-point[0])*innerWidth,(gaze.y-point[1])*innerHeight))} px`;
}
const partTitles={head:"Cabeça do robô","right-arm":"Braço do robô à sua esquerda",torso:"Peito do robô","left-arm":"Braço do robô à sua direita","right-leg":"Perna do robô à sua esquerda","left-leg":"Perna do robô à sua direita"};
const partSpeech={head:"Olhe para a cabeça do robô.","right-arm":"Olhe para o braço do robô à sua esquerda.",torso:"Olhe para o peito do robô.","left-arm":"Olhe para o braço do robô à sua direita.","right-leg":"Olhe para a perna do robô à sua esquerda.","left-leg":"Olhe para a perna do robô à sua direita."};
const roleNames={notebook:"câmera do notebook",nao:"câmera do NAO",validation:"câmera da apresentação"};
const surfaceNames={notebook:"Notebook",tv:"Apresentação / TV",robot:"Robô NAO"};
function surfaceParam(){return byId("position-surface").value==="robot"?"robot":display;}
// A tela escolhida define onde o mapa é salvo. Antes ela vinha só do endereço, e uma janela
// arrastada para a TV com ?display=notebook gravava o mapa como notebook.
byId("position-surface").value=new URLSearchParams(location.search).get("surface")==="robot"?"robot":display;
function updateHint(){
  const surface=byId("position-surface").value,role=byId("position-role").value;
  byId("position-hint").textContent=surface==="robot"?"Calibre também uma tela com esta mesma câmera: é a comparação com a tela que separa “olhando o robô” de “olhando a tela”."
    :surface==="notebook"&&role==="validation"?"A câmera da apresentação fica na TV. Se esta janela está na TV, escolha “Tela da apresentação (TV)”.":
    surface==="tv"&&role==="notebook"?"Esta janela será tratada como a TV. Confira se ela está na tela da apresentação.":"";
}
function participantParam(){return byId("position-participant").value.trim();}
function updateCalibrateLabel(){const robot=surfaceParam()==="robot";byId("calibrate-position").textContent=robot?"Calibrar e validar o robô · 6 partes + 6":"Calibrar e validar esta tela · 9 pontos + 4";}
function spatialCue(cal){const robot=cal.surface==="robot";return {active:true,cue_id:cal.cue_id,robot,step:{target:robot?"NAO":display==="tv"?"TV":"NOTEBOOK",marker:cal.marker,part:cal.part,point:robot?(partTitles[cal.part]||"o robô").toLowerCase():"o ponto amarelo",speech:robot?(partSpeech[cal.part]||"Olhe para o robô."):"Olhe para o ponto amarelo."},index:cal.mode==="check"?cal.index-cal.fit_total:cal.index,total:cal.mode==="check"?cal.check_total:cal.fit_total,mode:cal.mode,paused:false,collecting:cal.phase==="collecting",remaining:cal.remaining,spatial:true,valid_samples:cal.valid_samples,minimum_samples:cal.minimum_samples,reason:cal.reason};}
function precisionLabel(model){
  return model?.validation_text&&model.validation_text!=="não medida"?`Validação: ${model.validation_text}`:"Ainda sem validação · use Validar de novo";
}
function robotView(data){
  const gaze=data.gaze;
  byId("fixation-point").hidden=true;setReticle(null);
  byId("phase-label").textContent="ROBÔ NAO · PARTE ESTIMADA DO OLHAR";
  byId("standby").hidden=!!gaze.visible||gaze.outside;
  byId("standby-title").textContent=data.model.ready?"Aguardando leitura dos olhos":"Calibre o robô para esta câmera";
  byId("standby-description").textContent=data.calibration?.phase==="failed"?data.calibration.reason:gaze.reason;
  byId("physical-instruction").hidden=!(gaze.visible||gaze.outside);
  const onRobot=gaze.visible&&gaze.inside;
  byId("robot-guide").removeAttribute("hidden");byId("robot-guide").dataset.part=onRobot&&gaze.part||"";
  byId("instruction-title").textContent=!onRobot?"Fora do robô":gaze.part?`Agora: ${partTitles[gaze.part]}`:"Olhando o robô";
  const partNote=onRobot&&gaze.part_reliable===false?` · esta câmera não distinguiu as partes na validação (${Math.round(6*gaze.part_accuracy)}/6)`:"";
  const closer=gaze.closer_to?` · leitura mais compatível com ${surfaceNames[gaze.closer_to]}`:"";
  byId("instruction-description").textContent=`${roleNames[data.model.role]} · ${precisionLabel(data.model)}${partNote}${closer}`;
  const dot=byId("robot-gaze-dot");dot.toggleAttribute("hidden",!onRobot);
  if(gaze.visible){dot.setAttribute("cx",String(Math.max(0,Math.min(1,gaze.x))*320));dot.setAttribute("cy",String(Math.max(0,Math.min(1,gaze.y))*380));}
  byId("target-status").textContent="Olhe para o robô físico; o desenho mostra a estimativa. Não confirma atenção.";
  byId("verify-position").disabled=true;byId("free-position").disabled=true;
  byId("measure-position").disabled=!data.model.ready;
}
function trackingView(data){
  if(data.model.surface==="robot")return robotView(data);
  byId("robot-gaze-dot").setAttribute("hidden","");
  byId("fixation-point").hidden=!data.model.ready||!referencePoint;byId("physical-instruction").hidden=true;
  if(data.model.ready&&referencePoint)placePoint(referencePoint);
  byId("phase-label").textContent=referencePoint?"CONFERÊNCIA · OLHE PARA O AMARELO":"POSIÇÃO ESTIMADA DO OLHAR";
  const gaze=data.gaze;
  byId("standby").hidden=!!gaze.visible;
  byId("standby-title").textContent=data.model.ready?"Aguardando leitura dos olhos":"Calibre a posição desta tela";
  byId("standby-description").textContent=data.calibration?.phase==="failed"?data.calibration.reason:gaze.reason;
  const color=gaze.verified?"Verde":"Laranja";
  byId("target-status").textContent=gaze.visible?(referencePoint?`${errorText(gaze,referencePoint)} · Amarelo: referência / ${color}: estimativa · ${precisionLabel(data.model)}`:precisionLabel(data.model)):gaze.reason;
  setReticle(gaze);
  byId("verify-position").disabled=!data.model.ready;
  byId("free-position").disabled=!data.model.ready;
  byId("measure-position").disabled=!data.model.ready;
}
byId("setup-title").textContent=display==="tv"?"Esta janela deve ficar na TV":"Esta janela deve ficar no notebook";
byId("setup-description").textContent="Calibre cada câmera nesta tela ou no robô. A validação vem logo depois da calibração, e o resultado fica salvo para o participante.";
byId("voice-option").hidden=display!=="notebook";
byId("display-label").textContent=display==="tv"?"APRESENTAÇÃO / TV":"NOTEBOOK DO PARTICIPANTE";
// Na TV, a câmera presa à tela de apresentação é a referência natural para a posição.
if(display==="tv")byId("position-role").value="validation";
byId("position-surface").addEventListener("change",()=>{
  const surface=byId("position-surface").value;
  // Trocar de tela recarrega a página como aquela tela (identidade da janela e tamanho).
  if(surface!=="robot"&&surface!==display){location.replace(`/alvo?display=${surface}`);return;}
  updateCalibrateLabel();updateHint();referencePoint=surface==="robot"?null:referencePoint;void poll();
});
byId("position-role").addEventListener("change",()=>{updateHint();void poll();});
updateCalibrateLabel();updateHint();
function renderLibrary(data){
  const names=data.participants||[];
  const list=byId("known-participants");const signature=names.join("|");
  if(list.dataset.signature!==signature){list.innerHTML=names.map(n=>`<option value="${n.replace(/"/g,"&quot;")}"></option>`).join("");list.dataset.signature=signature;}
  const entries=Object.values(data.library||{});
  byId("position-library").innerHTML=entries.length?`<p><strong>Já calibrado para ${escapeHtml(data.participant||"")}</strong></p>`+entries.map(e=>`<p>${surfaceNames[e.surface]} · ${roleNames[e.role]} · <span class="${e.validation?.level||""}">${escapeHtml(e.validation_text)}</span></p>`).join(""):"<p>Nada calibrado ainda para este participante.</p>";
}
function escapeHtml(value){return String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
async function request(path,data){
  const response=await fetch(`/api${path}`,data?{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)}:{});
  if(response.status===401){byId("target-login").hidden=false;byId("target-setup").hidden=true;byId("target-stage").hidden=true;throw new Error("Informe o código do servidor");}
  const value=await response.json();if(!response.ok)throw new Error(value.error||"Falha no servidor");return value;
}
byId("target-login-form").addEventListener("submit",async event=>{
  event.preventDefault();try{await request("/login",{key:byId("target-key").value});byId("target-key").value="";byId("target-login").hidden=true;byId("target-setup").hidden=false;}
  catch(error){byId("target-login-error").textContent=error.message;}
});
function instruct(cue){
  ready=false;clearTimeout(speechTimer);
  if("speechSynthesis" in window)speechSynthesis.cancel();
  if(!cue?.active||cue.paused||!enabled||document.hidden)return;
  if(display!=="notebook"||!byId("voice-enabled").checked||!("speechSynthesis" in window)){ready=true;return;}
  const cueId=cue.cue_id, utterance=new SpeechSynthesisUtterance(cue.step.speech);
  utterance.lang="pt-BR";utterance.rate=1.05;
  const voice=speechSynthesis.getVoices().find(v=>v.lang.toLowerCase()==="pt-br");if(voice)utterance.voice=voice;
  const done=()=>{if(current?.cue_id===cueId){ready=true;clearTimeout(speechTimer);}};
  utterance.onend=done;utterance.onerror=done;speechSynthesis.speak(utterance);
  speechTimer=setTimeout(()=>{if(current?.cue_id===cueId){speechSynthesis.cancel();ready=true;byId("target-status").textContent="Leia a indicação: voz indisponível ou não concluída.";}},6000);
}
function render(cue){
  byId("fixation-point").hidden=true;byId("physical-instruction").hidden=true;byId("standby").hidden=true;
  if(!cue.active){byId("standby").hidden=false;byId("standby-title").textContent=cue.complete?(cue.error?"Ajuste necessário":"Etapas concluídas"):"Pronto";byId("standby-description").textContent=cue.error||(cue.complete?"Use Esc e volte à apresentação.":"Aguarde o operador.");byId("phase-label").textContent="OBSERVA";return;}
  byId("phase-label").textContent=`${cue.mode==="check"?"VALIDAÇÃO":"CALIBRAÇÃO"} ${cue.index+1}/${cue.total} · ${cue.paused?"PAUSADO":cue.collecting?`${cue.remaining.toFixed(1)} s`:"PREPARE O OLHAR"}`;
  if(cue.paused){byId("standby").hidden=false;byId("standby-title").textContent="Pausado";byId("standby-description").textContent="Aguarde o operador retomar.";return;}
  const isOwnScreen=(display==="notebook"&&cue.step.target==="NOTEBOOK")||(display==="tv"&&cue.step.target==="TV");
  if(isOwnScreen){placePoint(cue.step.marker);byId("fixation-point").hidden=false;byId("target-status").textContent=cue.spatial?`${cue.valid_samples}/${cue.minimum_samples} leituras mínimas · ${cue.reason}. O amarelo indica onde olhar; não é confirmação de acerto.`:"Fixe o olhar no centro do ponto amarelo.";}
  else if((display==="notebook"||cue.spatial)&&["NAO","OUTRO_LOCAL"].includes(cue.step.target)){
    byId("physical-instruction").hidden=false;byId("robot-gaze-dot").setAttribute("hidden","");
    const part=cue.step.part;
    byId("instruction-title").textContent=partTitles[part]||"Olhe para fora dos alvos";
    byId("instruction-description").textContent=cue.spatial?`Olhe para a parte física do robô, não para este desenho. ${cue.collecting?`${cue.valid_samples}/${cue.minimum_samples} leituras mínimas · ${cue.reason}`:cue.reason}`:`Olhe para ${cue.step.point}.`;
    byId("robot-guide").toggleAttribute("hidden",cue.step.target!=="NAO");byId("robot-guide").dataset.part=part||"";
    byId("target-status").textContent="Olhe para o robô ou o local físico indicado, não para este desenho.";
  }else{byId("target-status").textContent=`Olhe para ${cue.step.point}.`;}
}
async function activateDisplay(){
  try{await document.documentElement.requestFullscreen();}catch{byId("setup-error").textContent="Tela cheia indisponível. Mantenha esta janela maximizada.";}
  enabled=true;byId("target-setup").hidden=true;byId("target-stage").hidden=false;lastCueId=null;void poll();
  document.body.classList.add("tracking");
}
byId("enable-display").addEventListener("click",activateDisplay);
byId("calibrate-position").addEventListener("click",async()=>{
  try{
    if(screenState?.calibration?.active&&screenState.calibration.display===display)await request("/screen-gaze",{action:"cancel",display});
    byId("display-mode").value="gaze";
    testIndex=0;referencePoint=testPoints[0];
    await activateDisplay();
    await request("/screen-gaze",{action:"start",display,surface:surfaceParam(),role:byId("position-role").value,participant:participantParam(),check:true});
    lastCueId=null;
  }catch(error){enabled=false;document.body.classList.remove("tracking");byId("target-stage").hidden=true;byId("target-setup").hidden=false;byId("setup-error").textContent=error.message;}
});
byId("display-mode").addEventListener("change",()=>{byId("position-options").hidden=mode()!=="gaze";});
byId("position-options").hidden=mode()!=="gaze";
byId("verify-position").addEventListener("click",()=>{testIndex=(testIndex+1)%testPoints.length;referencePoint=testPoints[testIndex];void poll();});
byId("free-position").addEventListener("click",()=>{referencePoint=null;void poll();});
byId("measure-position").addEventListener("click",async()=>{
  try{await request("/screen-gaze",{action:"check",display,surface:surfaceParam(),role:byId("position-role").value,participant:participantParam()});lastCueId=null;void poll();}
  catch(error){byId("target-status").textContent=error.message;}
});
byId("retry-position").addEventListener("click",()=>{byId("calibrate-position").click();});
byId("target-settings").addEventListener("click",async()=>{
  if(screenState?.calibration?.active&&screenState.calibration.display===display){try{await request("/screen-gaze",{action:"cancel",display});}catch{}}
  enabled=false;ready=false;reticleTarget=null;byId("gaze-reticle").hidden=true;byId("target-stage").hidden=true;byId("target-setup").hidden=false;if("speechSynthesis" in window)speechSynthesis.cancel();
  document.body.classList.remove("tracking");
});
document.addEventListener("visibilitychange",()=>{ready=false;if(!document.hidden)lastCueId=null;});
byId("voice-enabled").addEventListener("change",()=>{lastCueId=null;});
async function poll(){
  if(busy)return;busy=true;
  try{
    const [normal,data]=await Promise.all([request("/calibration/cue"),request(`/screen-gaze?display=${display}&surface=${surfaceParam()}&role=${byId("position-role").value}&participant=${encodeURIComponent(participantParam())}`)]);
    screenState=data;
    byId("position-reading").textContent=data.reading?`${data.reading.valid?"✓":"⚠"} ${data.reading.reason}`:"Verificando leitura dos olhos…";
    if(!participantLoaded){byId("position-participant").value=data.participant||"";participantLoaded=true;}
    byId("position-info").textContent=data.calibration?.phase==="failed"&&data.calibration.display===display?data.calibration.reason:data.model.ready?`${surfaceNames[data.model.surface]} com a ${roleNames[data.model.role]}: calibrado. ${precisionLabel(data.model)}`:`${surfaceNames[surfaceParam()]} com a ${roleNames[byId("position-role").value]}: ainda não calibrado para este participante.`;
    renderLibrary(data);
    const spatial=data.calibration?.active&&data.calibration.display===display;
    current=mode()==="gaze"?(spatial?spatialCue(data.calibration):{active:false}):normal;
    if(enabled){
      if(mode()==="gaze"&&!spatial)trackingView(data);
      else{
        setReticle(null);render(current);
        byId("verify-position").disabled=true;byId("free-position").disabled=true;byId("measure-position").disabled=true;
        if(spatial&&data.calibration.mode==="check"&&data.calibration.surface!=="robot"){
          // A estimativa não aparece durante a validação: o participante tendia a seguir o ponto azul.
          byId("target-status").textContent=`Validação · ${data.calibration.valid_samples}/${data.calibration.minimum_samples} leituras mínimas · olhe para o amarelo até ele mudar. O resultado aparece no fim.`;
        }
      }
      if(current.cue_id!==lastCueId){lastCueId=current.cue_id;instruct(current);}
    }
    const layout=[innerWidth,innerHeight,window.screenX,window.screenY],signature=JSON.stringify(layout);
    if(signature!==layoutSignature){layoutSignature=signature;layoutStableSince=performance.now();}
    await request("/calibration/display",{display,client,cue_id:current.cue_id||null,visible:enabled&&!document.hidden,ready:enabled&&ready&&!document.hidden&&performance.now()-layoutStableSince>=600,fullscreen:!!document.fullscreenElement,layout});
  }catch(error){ready=false;reticleTarget=null;byId("gaze-reticle").hidden=true;byId("target-status").textContent=error.message;lastCueId=null;}
  finally{busy=false;}
}
setInterval(()=>{void poll();},200);void poll();
function animateReticle(now){
  const fresh=reticleTarget&&reticleTarget.age+(now-reticleTarget.received)/1000<=.7;
  const visible=enabled&&mode()==="gaze"&&!document.hidden&&fresh;
  byId("gaze-reticle").hidden=!visible;
  if(visible){
    const dt=lastPaint===null?16:Math.min(100,now-lastPaint),alpha=1-Math.exp(-dt/75);
    if(!reticlePoint)reticlePoint={x:reticleTarget.x,y:reticleTarget.y};
    reticlePoint.x+=(reticleTarget.x-reticlePoint.x)*alpha;reticlePoint.y+=(reticleTarget.y-reticlePoint.y)*alpha;
    byId("gaze-reticle").style.left=`${Math.max(.01,Math.min(.99,reticlePoint.x))*100}%`;byId("gaze-reticle").style.top=`${Math.max(.01,Math.min(.99,reticlePoint.y))*100}%`;
  }else reticlePoint=null;
  lastPaint=now;requestAnimationFrame(animateReticle);
}
requestAnimationFrame(animateReticle);
async function previewWebcam(){
  if(previewBusy||document.hidden||!byId("target-login").hidden)return;
  previewBusy=true;
  try{
    const role=screenState?.reading?.role||byId("position-role").value;
    const response=await fetch(`/api/preview/${role}`);
    if(!response.ok||!response.headers.get("Content-Type")?.startsWith("image/jpeg"))throw new Error("Aguardando imagem da webcam");
    const overlay=JSON.parse(response.headers.get("X-Eye-Overlay")||"{}");
    const bitmap=await createImageBitmap(await response.blob());
    const canvas=byId("webcam-canvas");canvas.height=Math.round(320*bitmap.height/bitmap.width);
    const ctx=canvas.getContext("2d");ctx.drawImage(bitmap,0,0,canvas.width,canvas.height);bitmap.close();
    for(const eye of overlay.eyes||[]){ctx.beginPath();ctx.arc(eye[0]*canvas.width,eye[1]*canvas.height,5,0,Math.PI*2);ctx.strokeStyle="#63ddff";ctx.lineWidth=2;ctx.stroke();}
    byId("webcam-status").textContent=screenState?.reading?.reason||"Webcam conectada · mantenha só o participante no enquadramento";
  }catch(error){byId("webcam-status").textContent=error.message;byId("webcam-canvas").getContext("2d").clearRect(0,0,320,byId("webcam-canvas").height);}
  finally{previewBusy=false;}
}
setInterval(()=>{void previewWebcam();},160);void previewWebcam();
