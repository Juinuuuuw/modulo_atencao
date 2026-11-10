"use strict";
const $ = id => document.getElementById(id);
const roles = ["notebook", "nao", "validation"];
const labels = ["NOTEBOOK", "TV", "NAO", "OUTRO_LOCAL", "INDETERMINADO"];
const names = {NOTEBOOK:"Notebook", TV:"Apresentação / TV", NAO:"Robô NAO", OUTRO_LOCAL:"Outro local", INDETERMINADO:"Indeterminado"};
let state, activeTab = "capture", sourceSignature = "", sourceInitialized = false, authenticated = false;
const attentionNames = {EM_FOCO_NOTEBOOK:"Foco no notebook",EM_FOCO_TV:"Foco na apresenta\u00e7\u00e3o",EM_FOCO_ROBO:"Foco no rob\u00f4",DISPERSAO_EXTERNA:"Dispers\u00e3o externa",SLEEPY:"Olhos fechados prolongadamente",BAIXA_ATIVIDADE_DO_OLHAR:"Baixa atividade do olhar",INDETERMINADO:"Indeterminado"};
let participantInitialized=false;
let selectedSession = null, annotations = [], report = null, playing = false, playOrigin = 0, dirty = false;
let agreementReport = null, consensusSources = null;
let frameIndex = {}, lastFrameKey = "", loadedAnnotator = "";
const previewInFlight = new Set();
const previewSequences = {};
const rayFrames = {};
const roleNames = {notebook:"Notebook", nao:"NAO", validation:"Apresentação"};
function paintRay(role){
  const canvas=$(`ray-${role}`),badge=$(`ray-status-${role}`),image=$(`preview-${role}`);
  const enabled=$("show-eye-rays").checked;
  canvas.hidden=!enabled;badge.hidden=!enabled;
  const cached=rayFrames[role];
  const selected=activeTab==="live"?$("lg-participant-select").value:"";
  const recent=cached&&(!selected||cached.metadata?.participant===selected)&&cached.source===state?.assignments[role]&&performance.now()-cached.receivedAt<=400&&state?.cameras[role]?.online;
  const metadata=recent?cached.metadata:{target:"INDETERMINADO",reason:"Aguardando leitura dos olhos",eyes:null,direction:null};
  const result=GazeOverlay.paint(canvas,image,metadata,$("ray-reference").value,enabled);
  badge.className=`ray-status ${result.state}`;
  if(badge.textContent!==result.label)badge.textContent=result.label;
}
const paintRays = () => roles.forEach(paintRay);
$("show-eye-rays").addEventListener("change",paintRays);
$("ray-reference").addEventListener("change",paintRays);
window.addEventListener("resize",paintRays);
// O mesmo bloco de câmeras aparece em "Câmeras e sessão" e em "Olhar ao vivo".
function placeCameras(){const slot=$(`camera-slot-${activeTab}`);if(slot&&$("camera-block").parentElement!==slot){slot.appendChild($("camera-block"));requestAnimationFrame(paintRays);}}
placeCameras();
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pct = value => value == null ? "—" : `${(100 * value).toFixed(1)}%`;
const seconds = value => Number(value).toFixed(2);
const clock = value => `${Math.floor(value / 60).toString().padStart(2,"0")}:${Math.floor(value % 60).toString().padStart(2,"0")}`;

function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").className = error ? "notice error" : "notice";
  $("notice").hidden = false;
}
async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, {...options, headers:{"Content-Type":"application/json", ...options.headers}});
  const data = await response.json();
  if (!response.ok) {
    if (response.status === 401 && path !== "/login") showLogin();
    throw new Error(data.error || `Erro ${response.status}`);
  }
  return data;
}
const post = (path, data = {}) => api(path, {method:"POST", body:JSON.stringify(data)});
function action(id, handler) {
  $(id).addEventListener("click", async () => {
    try { await handler(); } catch (error) { notice(error.message, true); }
  });
}
function showLogin() { authenticated = false; $("login").hidden = false; $("workspace").hidden = true; }
async function enter(key) {
  if (key) await post("/login", {key});
  await poll();
  authenticated = true; $("login").hidden = true; $("workspace").hidden = false;
  $("nao-command").textContent = `python nao_client.py --server http://IP_DO_SERVIDOR:${location.port || "8765"}`;
}
$("login-form").addEventListener("submit", async event => {
  event.preventDefault();
  try { await enter($("access-key").value); $("access-key").value = ""; }
  catch (error) { $("login-error").textContent = error.message; }
});
action("logout", async () => { await post("/logout"); showLogin(); });
document.querySelectorAll("[data-tab]").forEach(button => button.addEventListener("click", () => {
  activeTab = button.dataset.tab;
  document.querySelectorAll(".tab").forEach(tab => { tab.hidden = tab.id !== `tab-${activeTab}`; });
  document.querySelectorAll(".nav").forEach(nav => nav.classList.toggle("active", nav === button));
  $("page-title").textContent = {capture:"Câmeras e sessão",live:"Olhar ao vivo",dashboard:"Dados e análises",review:"Validação humana",connect:"Conectar dispositivos"}[activeTab];
  placeCameras();
  if (activeTab === "live") { resetLiveTrails(); resizeLiveCanvases(); }
  if (activeTab !== "review") stopPlayback();
  if (activeTab === "dashboard") refreshDashboard().catch(error => notice(error.message, true));
  if (activeTab === "review") refreshSessions().catch(error => notice(error.message, true));
}));

// ---- Dados e análises: consultas agregadas no índice SQLite local ----
const cameraDisplayNames = {notebook:"Notebook",nao:"Robô NAO",validation:"Apresentação"};
let dashboardData = null;
function dashboardDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString("pt-BR", {dateStyle:"short",timeStyle:"short"});
}
function renderDashboard() {
  const data = dashboardData, overview = data.overview;
  $("dashboard-overview").innerHTML = [["Sessões",overview.sessions],["Participantes",overview.participants],
    ["Tempo coletado",clock(overview.duration)],["Previsões",Number(overview.predictions).toLocaleString("pt-BR")]]
    .map(([label,value])=>`<div class="metric"><strong>${esc(value)}</strong><span>${label}</span></div>`).join("");
  const select = $("dashboard-session"), previous = data.selected?.id || select.value;
  select.innerHTML = data.sessions.map(row=>`<option value="${esc(row.id)}">${esc(row.participant)} · ${dashboardDate(row.started_utc)}</option>`).join("") || '<option value="">Nenhuma sessão disponível</option>';
  if (previous && data.sessions.some(row=>row.id===previous)) select.value=previous;
  $("dashboard-sessions").innerHTML = data.sessions.map(row=>`<tr><td>${esc(row.participant)}</td><td>${dashboardDate(row.started_utc)}</td><td>${clock(row.duration||0)}</td><td>${Number(row.predictions).toLocaleString("pt-BR")}</td><td>${Number(row.frames).toLocaleString("pt-BR")}</td><td>${esc(row.status)}</td><td><button class="secondary dash-open" data-session="${esc(row.id)}">Analisar</button></td></tr>`).join("") || '<tr><td colspan="7">Nenhuma sessão foi indexada ainda.</td></tr>';
  const selected = data.selected;
  $("dashboard-content").hidden = !selected;
  $("dashboard-session-info").textContent = selected ? `${selected.participant} · ${clock(selected.duration||0)} · ${Number(selected.predictions).toLocaleString("pt-BR")} previsões · ${Number(selected.frames).toLocaleString("pt-BR")} quadros` : "Finalize uma gravação para visualizar as análises.";
  if (!selected) return;
  const duration = Math.max(.001, Number(selected.duration||0));
  $("target-distribution").innerHTML = labels.map(target=>{
    const value=Number(selected.target_seconds[target]||0);
    return `<div class="distribution-row"><div><span class="chart-dot target-${target}"></span><strong>${names[target]}</strong><span>${seconds(value)} s · ${pct(value/duration)}</span></div><progress class="analysis-progress target-${target}" max="${duration}" value="${value}">${pct(value/duration)}</progress></div>`;
  }).join("");
  const attentionStates=selected.attention_states||[], attentionTotal=attentionStates.reduce((sum,row)=>sum+Number(row.duration||0),0);
  $("attention-state-distribution").innerHTML = attentionStates.map(row=>{
    const value=Number(row.duration||0), share=attentionTotal?value/attentionTotal:null;
    return `<div class="distribution-row"><div><strong>${esc(attentionNames[row.state]||row.state)}</strong><span>${seconds(value)} s · ${pct(share)} · ${Number(row.events||0)} intervalo(s)</span></div><progress class="analysis-progress" max="${Math.max(.001,attentionTotal)}" value="${value}">${pct(share)}</progress></div>`;
  }).join("") || '<p class="muted">Esta sess\u00e3o ainda n\u00e3o possui estados temporais registrados.</p>';
  $("camera-quality").innerHTML = Object.entries(selected.cameras).map(([role,value])=>{
    const total=value.samples||0, valid=value.valid_samples||0, validRatio=total?valid/total:null;
    return `<div class="camera-quality-row"><div><strong>${esc(cameraDisplayNames[role]||role)}</strong><span>${total.toLocaleString("pt-BR")} amostras · ${pct(value.data_loss)} de perda</span></div><progress class="analysis-progress" max="${Math.max(1,total)}" value="${valid}">${pct(validRatio)} v\u00e1lidas</progress></div>`;
  }).join("");
  $("dashboard-events").innerHTML = selected.events.map(row=>`<p><strong>${seconds(row.t)} s · ${esc(row.kind)}</strong><br>${esc(row.note||"Sem nota")}</p>`).join("") || '<p class="muted">Nenhum marco registrado.</p>';
  $("dashboard-reports").innerHTML = selected.reports.map(row=>`<div class="report-summary"><strong>${esc(row.annotator)}</strong><span>Acurácia ${pct(row.accuracy)} · cobertura ${pct(row.coverage)} · macro F1 ${pct(row.macro_f1)} · kappa ${row.kappa==null?"—":Number(row.kappa).toFixed(3)}</span><span>IoU temporal média ${pct(row.temporal_mean_iou)} · F1 temporal @ 0,5 ${pct(row.temporal_f1_05)}</span></div>`).join("") || '<p class="muted">Calcule uma validação humana para exibir as métricas aqui.</p>';
}
async function refreshDashboard(sessionId = "") {
  dashboardData = await api(`/dashboard${sessionId?`?session_id=${encodeURIComponent(sessionId)}`:""}`);
  renderDashboard();
}
action("refresh-dashboard", async()=>refreshDashboard($("dashboard-session").value));
$("dashboard-session").addEventListener("change",()=>refreshDashboard($("dashboard-session").value).catch(error=>notice(error.message,true)));
$("dashboard-sessions").addEventListener("click",event=>{const id=event.target.dataset.session;if(id)refreshDashboard(id).catch(error=>notice(error.message,true));});

function renderSources() {
  const signature = JSON.stringify(state.nodes.map(n => [n.id,n.online,n.cameras]));
  if (signature === sourceSignature && sourceInitialized) return;
  sourceSignature = signature;
  for (const role of roles) {
    const select = $(`source-${role}`);
    const selected = sourceInitialized ? select.value : state.assignments[role] || "";
    select.innerHTML = '<option value="">Selecione uma câmera…</option>' + state.nodes.map(node =>
      `<optgroup label="${esc(node.name)}${node.online ? "" : " · desconectado"}">${node.cameras.map(camera =>
        `<option value="${esc(`${node.id}:${camera.id}`)}">${esc(node.name)} / ${esc(camera.name)}${node.online ? "" : " (offline)"}</option>`).join("")}</optgroup>`).join("");
    select.value = selected;
  }
  sourceInitialized = true;
}
function renderState() {
  if (state.error) notice(state.error, true);
  renderSources();
  const recording = !!state.recording;
  if(!participantInitialized){if(!$("participant").value&&state.calibration_participant)$("participant").value=state.calibration_participant;participantInitialized=true;}
  const task=state.live.regions?.task;
  const cal=state.calibration;
  $("attention-panel").className=`attention-panel ${!state.fit_report?'unknown':task?.target==='DENTRO'?'green':task?.target==='FORA'?'red':'unknown'}`;
  if(!state.fit_report){
    const started=cal?.counts&&Object.values(cal.counts).some(counts=>Object.values(counts).some(n=>n>0));
    $("attention-title").textContent=started?'Calibração em andamento':'Ainda não há modelo do olhar';
    $("attention-message").textContent=started?'Conclua as fixações e a verificação pelo painel.':cal?.automatic?'A sequência não coletou amostras: aguarda as janelas de ponto. Use “Calibrar nesta disposição” para seguir só pelo painel.':'Calibre a disposição atual. Antes disso, a cor não identifica telas ou robô.';
  }else{
    $("attention-title").textContent=task?.target==='DENTRO'?'Olhar voltado à tarefa':task?.target==='FORA'?'Olhar fora da tarefa':'Olhar ainda inconclusivo';
    $("attention-message").textContent=task?.target==='DENTRO'?'Notebook, TV e NAO contam como tarefa. A identificação exata do alvo aparece abaixo.':task?.reason||state.live.reason;
  }
  $("start-scene").disabled=recording||!!cal?.started;
  $("start-scene").textContent=cal?.profile==='assisted'&&!cal.complete?'Continuar calibração':'Calibrar nesta disposição';
  $("server-status").textContent = recording ? "● Gravando" : "● Servidor conectado";
  $("server-status").className = `badge ${recording ? "recording" : "online"}`;
  for (const role of roles) {
    const camera = state.cameras[role];
    $(`status-${role}`).textContent = camera.online ? `Ao vivo · ${Number(camera.fps || 0).toFixed(1)} FPS` : "Sem sinal";
    $(`status-${role}`).className = `badge ${camera.online ? "online" : ""}`;
    $(`source-${role}`).disabled = recording;
    const prediction = camera.prediction;
    const targetDetail = activeTab === "live" && liveData?.source === "posição"
      ? `Alvo combinado: ${names[liveTarget(liveData)]} · `
      : prediction ? `${names[prediction.target]} · ${prediction.reason} · ` : "Sem modelo de alvos · ";
    $(`detail-${role}`).textContent = camera.error || (camera.online
      ? `${targetDetail}envio ${camera.transport_ms ?? "—"} ms · sincronia ±${camera.sync_ms} ms (estimada)`
      : "Conecte o cliente e aplique a câmera selecionada");
    if (!camera.online) { $(`preview-${role}`).hidden = true; paintRay(role); }
  }
  $("save-cameras").disabled = recording;
  $("live-target").textContent = names[state.live.target] || state.live.target;
  $("live-task").textContent=task?.target==='DENTRO'?'Tarefa: dentro (notebook / TV / NAO)':task?.target==='FORA'?'Tarefa: fora':'Tarefa: indeterminado';
  $("live-reason").textContent = `${state.live.reason} · escore ${Number(state.live.score).toFixed(2)} (não é probabilidade)`;
  $("start-recording").disabled = recording || !state.check_report?.sufficient;
  $("stop-recording").disabled = !recording;
  $("recording-time").textContent = recording ? clock(state.recording.elapsed) : "00:00";
  $("participant").disabled = recording;
  $("calibration-owner").textContent = state.calibration_participant ? `Calibração de ${state.calibration_participant}` : "Participante ainda não calibrado.";
  $("begin-fit").disabled = recording;
  $("begin-check").disabled = recording || !state.fit_report;
  const autoActive=!!state.calibration?.automatic&&!state.calibration.complete;
  $("begin-quick").disabled=recording||!!cal?.started;
  $("begin-auto").disabled=recording||autoActive;
  $("begin-fit").disabled=recording||autoActive;
  $("begin-check").disabled=recording||!state.fit_report||autoActive;
  $("pause-calibration").disabled=recording||!autoActive;
  $("pause-calibration").textContent=state.calibration?.paused?"Retomar":"Pausar";
  $("next-participant").disabled=recording;
  $("display-readiness").textContent=[['notebook','Notebook'],['tv','TV']].map(([key,label])=>`${label}: ${state.displays?.[key]?.online?'janela visível':'abra e ative a tela do ponto'}`).join(' · ');
  document.querySelectorAll("[data-event]").forEach(b => { b.disabled = !recording; });
  renderCalibration();
  $("node-list").innerHTML = state.nodes.length ? state.nodes.map(node =>
    `<div class="node-row"><div><strong>${esc(node.name)}</strong><p>${node.cameras.map(c => esc(c.name)).join(" · ")}${node.error ? `<br>${esc(node.error)}` : ""}</p></div><span class="badge ${node.online ? "online" : ""}">${node.online ? "Conectado" : "Offline"}</span></div>`).join("") : "Nenhum cliente conectado.";
}
function renderCalibration() {
  const cal = state.calibration;
  const autoRunning=cal?.automatic&&!cal.complete&&!cal.paused;
  $("capture-step").disabled = !cal || cal.started !== null || cal.complete || autoRunning;
  $("retry-capture").disabled = !cal || cal.started !== null || !!state.recording || autoRunning;
  $("finish-calibration").disabled = !cal || cal.started !== null || !!state.recording || autoRunning;
  if (cal) {
    const step = cal.plan[cal.index];
    $("calibration-box").className = `calibration-box${cal.error ? " error" : ""}`;
    $("calibration-box").innerHTML = `<span class="eyebrow">${cal.mode === "check" ? "VERIFICAÇÃO · DADOS INDEPENDENTES" : "AJUSTE DOS ALVOS"} · ${Math.min(cal.index + 1,cal.plan.length)}/${cal.plan.length}</span><h3>${cal.error ? "Revise as amostras" : cal.paused?"Pausado":step ? names[step.target] : "Etapas concluídas"}</h3><p>${esc(cal.error || (step ? `Olhe para ${step.point}. ${cal.started !== null ? `Coletando · ${cal.remaining.toFixed(1)} s restantes` : cal.paused?'Ajuste o enquadramento e retome.':cal.automatic?(cal.wait_reason||'Preparando a próxima indicação…'):"Prepare o participante e clique em coletar."}` : "Confira os resultados abaixo."))}</p>`;
    $("cal-progress").max=cal.duration;
    $("cal-progress").value = cal.started !== null ? cal.duration - cal.remaining : 0;
    $("capture-step").textContent=`Participante pronto · coletar ${cal.duration} segundos`;
    const signature = JSON.stringify(cal.plan);
    if ($("retry-step").dataset.plan !== signature) {
      $("retry-step").innerHTML = cal.plan.map((s,i) => `<option value="${i}">${i+1}. ${names[s.target]} · ${esc(s.point)}</option>`).join("");
      $("retry-step").dataset.plan = signature;
    }
    $("calibration-counts").innerHTML = roles.filter(role => state.assignments[role]).map(role => `<p>${roleNames[role].toUpperCase()}: ${Object.entries(cal.counts[role]).map(([k,v]) => `${names[k]} ${v}`).join(" · ")}</p>`).join("");
  }else{
    $("calibration-box").className="calibration-box";
    $("calibration-box").innerHTML='<span class="eyebrow">PREPARAR PARTICIPANTE</span><h3>Notebook · TV · NAO</h3><p>Abra as duas telas de ponto e inicie a calibração rápida.</p>';
    $("calibration-counts").textContent="";$("cal-progress").value=0;
  }
  const fit = state.fit_report, check = state.check_report;
  $("check-summary").innerHTML = (fit ? `<p><strong>Câmeras calibradas:</strong> ${fit.usable_cameras.map(esc).join(", ")}. ${fit.warnings.map(esc).join(". ")}</p>` : "") + (check ? `<p><strong>Verificação independente:</strong> alvo exato ${pct(check.accuracy)} · cobertura ${pct(check.coverage)} · macro F1 ${pct(check.macro_f1)}.</p><p>${check.task_metrics?`Tarefa × fora: acurácia ${pct(check.task_metrics.accuracy)} · cobertura ${pct(check.task_metrics.coverage)}. `:""}${check.sufficient ? "Todos os alvos têm pelo menos 5 avaliações." : "Faltam avaliações de algum alvo; repita as etapas."} Limite de 65% é operacional para o piloto, não validação científica.</p>` : "");
}
async function poll() { state = await api("/state"); renderState(); }
async function refreshPreviews() {
  if (!authenticated || document.hidden || !["capture","live"].includes(activeTab) || !state) return;
  await Promise.all(roles.filter(role => state.cameras[role].online && !previewInFlight.has(role)).map(async role => {
    previewInFlight.add(role);
    const source = state.assignments[role];
    const previous = previewSequences[role];
    const query = new URLSearchParams();
    if (previous?.source === source) query.set("after", previous.sequence);
    if (activeTab === "live" && $("lg-participant-select").value) query.set("participant", $("lg-participant-select").value);
    try {
      const response = await fetch(`/api/preview/${role}?${query}`, {signal:AbortSignal.timeout(1500)});
      if (!response.ok || response.status === 204) return;
      const blob = await response.blob(), image = $(`preview-${role}`), oldUrl = image.dataset.blob;
      const nextUrl = URL.createObjectURL(blob);
      image.src = nextUrl; image.dataset.blob = nextUrl;
      try { await image.decode(); image.hidden = false; }
      finally { if (oldUrl) URL.revokeObjectURL(oldUrl); }
      {
        const encoded=response.headers.get("X-Eye-Overlay");
        let metadata=null;
        try{metadata=encoded?JSON.parse(encoded):null;}catch{metadata=null;}
        rayFrames[role]={metadata,source,receivedAt:performance.now()};paintRay(role);
      }
      const sequence = response.headers.get("X-Frame-Sequence");
      if (sequence !== null) previewSequences[role] = {source,sequence:Number(sequence)};
    } catch {
      // Uma câmera lenta não impede a atualização das demais ou dos controles.
    } finally { previewInFlight.delete(role); }
  }));
}
action("save-cameras", async () => { await post("/configure", {assignments:Object.fromEntries(roles.map(r => [r,$(`source-${r}`).value || null]))}); notice("Câmeras aplicadas. Aguarde os clientes enviarem as imagens; a calibração anterior foi descartada."); await poll(); });
async function beginAssisted(){
  if(state.calibration?.profile!=='assisted'||state.calibration.complete){
    const participant=$("participant").value.trim()||state.calibration_participant;
    if(!participant){$("participant").focus();throw new Error('Informe o identificador do participante antes de calibrar');}
    $("participant").value=participant;
    await post("/calibration",{action:"fit",profile:"assisted",automatic:false,chain_check:true,participant});await poll();
  }
  document.querySelector('.calibration-panel').scrollIntoView({behavior:'smooth',block:'start'});
  notice('Olhe para o alvo físico indicado e clique em “Participante pronto · coletar”. Não é necessário abrir outras janelas.');
}
action("begin-quick",beginAssisted);
action("start-scene",beginAssisted);
action("begin-auto", async()=>{await post("/calibration",{action:"fit",profile:"quick",automatic:true,participant:$("participant").value.trim()||state.calibration_participant});notice("Sequência automática iniciada. Mantenha as duas telas do ponto visíveis; as instruções saem pelo notebook.");await poll();});
action("pause-calibration",async()=>{await post("/calibration",{action:state.calibration.paused?"resume":"pause"});await poll();});
action("next-participant",async()=>{await post("/calibration",{action:"cancel"});$("participant").value="";$("participant").focus();notice("Calibração anterior descartada. Informe a próxima pessoa e inicie a sequência rápida.");await poll();});
action("begin-fit", async () => { await post("/calibration",{action:"fit",profile:$("calibration-profile").value,participant:$("participant").value.trim()}); notice("Ajuste manual iniciado. Os pontos aparecem nas telas preparadas."); await poll(); });
action("begin-check", async () => { await post("/calibration",{action:"check",profile:state.calibration?.profile||$("calibration-profile").value,participant:$("participant").value.trim()}); notice("Verificação independente iniciada. Estas amostras não ajustam o modelo."); await poll(); });
action("capture-step", async () => { await post("/calibration",{action:"capture"}); await poll(); });
action("retry-capture", async () => { await post("/calibration",{action:"capture",index:Number($("retry-step").value)}); await poll(); });
action("finish-calibration", async () => { await post("/calibration",{action:"finish"}); await poll(); });
action("start-recording", async () => { await post("/recording/start",{participant:$("participant").value,override:$("override").value}); notice("Gravação iniciada. As três câmeras e previsões estão sendo salvas no servidor."); await poll(); });
action("stop-recording", async () => { await post("/recording/stop"); notice("Sessão encerrada. Abra Validação humana para revisar."); await poll(); });
document.querySelectorAll("[data-event]").forEach(button => button.addEventListener("click", async () => {
  try { await post("/events",{kind:button.dataset.event,note:$("event-note").value}); notice(`Marco ${button.textContent} registrado.`); $("event-note").value=""; }
  catch (error) { notice(error.message,true); }
}));
action("show-connection", async () => { const value = await api("/connection"); $("connection-details").hidden = false; $("connection-details").textContent = `Código: ${value.key}`; });

async function refreshSessions() {
  const sessions = await api("/sessions"), select = $("session-list"), previous = select.value;
  select.innerHTML = '<option value="">Selecione uma sessão</option>' + sessions.map(s => `<option value="${s.id}" ${s.status === "recording" ? "disabled" : ""}>${esc(s.participant)} · ${esc(new Date(s.started_utc).toLocaleString("pt-BR"))} · ${s.status === "complete" ? "concluída" : s.status === "interrupted" ? "interrompida" : "gravando"}</option>`).join("");
  select.value = previous;
}
action("refresh-sessions",refreshSessions);
$("session-list").addEventListener("change", async () => {
  if (dirty && !confirm("Há anotações não salvas. Trocar de sessão e descartá-las?")) { $("session-list").value = selectedSession?.metadata.id || ""; return; }
  stopPlayback(); $("validation-report").hidden = true; $("review-content").hidden = true;
  if (!$("session-list").value) return;
  try {
    selectedSession = await api(`/sessions/${$("session-list").value}`);
    const meta = selectedSession.metadata;
    frameIndex = Object.fromEntries(roles.map(role => [role,selectedSession.frames.filter(f => f.role === role).sort((a,b) => a.t-b.t)]));
    $("session-info").textContent = `${meta.participant} · ${seconds(meta.duration)} s · ${selectedSession.frames.length} quadros. ${meta.status === "interrupted" ? "Sessão interrompida: confira as lacunas." : ""}`;
    $("timeline").max = meta.duration; $("timeline").value = 0; lastFrameKey = "";
    $("csv-download").href = `/api/sessions/${meta.id}/predictions.csv`;
    $("session-events").innerHTML = selectedSession.events.map(e => `<p>${seconds(e.t)} s · ${esc(e.kind)} · ${esc(e.note)}</p>`).join("") || "Nenhum marco registrado.";
    report = null; agreementReport = null; consensusSources = null;
    $("report-download").disabled = true; $("agreement-report").hidden = true;
    $("prepare-consensus").disabled = true; refreshAnnotatorChoices(); loadAnnotations();
    $("review-content").hidden = false; renderFrame();
  } catch(error) { notice(error.message,true); }
});
function loadAnnotations() {
  loadedAnnotator = $("annotator").value.trim();
  annotations = structuredClone(selectedSession.annotations[loadedAnnotator] || []);
  consensusSources = selectedSession.annotation_metadata?.[loadedAnnotator]?.sources || null;
  dirty = false; renderAnnotations();
}
function refreshAnnotatorChoices() {
  const values=Object.keys(selectedSession?.annotations||{}).filter(name=>selectedSession.annotations[name]?.length&&selectedSession.annotation_metadata?.[name]?.kind!=="consensus");
  for(const id of ["agreement-first","agreement-second"]){
    const select=$(id), previous=select.value;
    select.innerHTML='<option value="">Selecione</option>'+values.map(name=>`<option value="${esc(name)}">${esc(name)}</option>`).join("");
    if(values.includes(previous))select.value=previous;
  }
  if(values.length>=2){
    if(!$("agreement-first").value)$("agreement-first").value=values[0];
    if(!$("agreement-second").value)$("agreement-second").value=values.find(value=>value!==$("agreement-first").value)||"";
  }
}
action("load-annotations", async () => {
  if (!selectedSession) throw new Error("Selecione uma sessão");
  if (dirty && !confirm("Descartar as alterações não salvas deste avaliador?")) return;
  loadAnnotations();
});
function renderAnnotations() {
  $("annotation-list").innerHTML = annotations.map((row,index) => `<div class="annotation-row ${row.needs_review?'needs-review':''}"><span>${seconds(row.start)}–${seconds(row.end)} s</span><strong>${names[row.target]}</strong>${row.needs_review?'<em>Revisar divergência</em>':''}<button data-edit="${index}" class="secondary" aria-label="Editar intervalo ${index+1}">Editar</button><button data-remove="${index}" aria-label="Remover intervalo ${index+1}">Remover</button></div>`).join("") || '<p class="small muted">Nenhum intervalo anotado.</p>';
  $("annotation-status").textContent = `${loadedAnnotator || "Informe o avaliador"} · ${annotations.length} intervalos${dirty ? " · alterações não salvas" : ""}`;
  $("validation-report").hidden = true; report = null; $("report-download").disabled = true;
}
$("annotation-list").addEventListener("click", event => {
  const index = event.target.dataset.remove;
  if (index !== undefined) { annotations.splice(Number(index),1); dirty = true; renderAnnotations(); }
  const editIndex = event.target.dataset.edit;
  if (editIndex !== undefined) {
    const row=annotations.splice(Number(editIndex),1)[0];
    $("interval-start").value=seconds(row.start); $("interval-end").value=seconds(row.end);
    $("interval-target").value=row.target; $("timeline").value=row.start; dirty=true; renderAnnotations(); renderFrame();
  }
});
action("add-interval", async () => {
  if ($("annotator").value.trim() !== loadedAnnotator) throw new Error("Carregue as anotações desse avaliador antes de editar");
  const row = {start:Number($("interval-start").value),end:Number($("interval-end").value),target:$("interval-target").value};
  if (!Number.isFinite(row.start) || !Number.isFinite(row.end) || row.start < 0 || row.start >= row.end || row.end > selectedSession.metadata.duration) throw new Error("Informe um intervalo válido dentro da sessão");
  if (annotations.some(a => row.start < a.end && a.start < row.end)) throw new Error("Este intervalo se sobrepõe a outro do avaliador");
  annotations.push(row); annotations.sort((a,b) => a.start-b.start); dirty = true; renderAnnotations();
});
async function saveAnnotations() {
  const annotator = $("annotator").value.trim();
  if (!annotator || annotator !== loadedAnnotator) throw new Error("Informe o avaliador e carregue suas anotações antes de salvar");
  if(consensusSources&&annotations.some(row=>row.needs_review))throw new Error("Revise todos os intervalos divergentes antes de salvar o consenso");
  const clean=annotations.map(({start,end,target})=>({start,end,target}));
  if(consensusSources){
    await post(`/sessions/${selectedSession.metadata.id}/consensus`,{annotator,first:consensusSources[0],second:consensusSources[1],intervals:clean});
    selectedSession.annotation_metadata ||= {};
    selectedSession.annotation_metadata[annotator]={kind:"consensus",sources:[...consensusSources]};
  }else await post(`/sessions/${selectedSession.metadata.id}/annotations`,{annotator,intervals:clean});
  selectedSession.annotations[annotator] = structuredClone(clean); annotations=structuredClone(clean);
  dirty = false; refreshAnnotatorChoices(); renderAnnotations();
}
action("save-annotations",async () => { await saveAnnotations(); notice("Anotações salvas no servidor."); });
action("evaluate",async () => {
  await saveAnnotations();
  report = await api(`/sessions/${selectedSession.metadata.id}/report?annotator=${encodeURIComponent(loadedAnnotator)}`);
  renderReport(report); $("report-download").disabled = false;
});
function renderReport(value) {
  $("validation-report").hidden = false;
  $("validation-report").innerHTML = `<h2>Comparação com ${esc(value.annotator)}</h2><p class="small muted">Matriz ponderada pela duração em segundos. Abstenções contam no denominador da acurácia. Trechos humanos indeterminados são excluídos das métricas.</p><div class="metrics">${[["Acurácia",pct(value.accuracy)],["Cobertura automática",pct(value.coverage)],["Macro F1",pct(value.macro_f1)],["Kappa",value.kappa == null ? "—" : value.kappa.toFixed(3)]].map(([label,v])=>`<div class="metric"><strong>${v}</strong><span>${label}</span></div>`).join("")}</div><p>${seconds(value.annotated_seconds)} s anotados de ${seconds(value.session_seconds)} s (${pct(value.annotation_coverage)}). ${seconds(value.ambiguous_seconds)} s ambíguos. Acurácia apenas nos trechos com previsão: ${pct(value.selective_accuracy)}.</p><h3>Matriz de confusão · linhas = humano / colunas = sistema</h3><div class="table-scroll"><table><thead><tr><th>Segundos</th>${labels.map(l=>`<th>${names[l]}</th>`).join("")}</tr></thead><tbody>${labels.slice(0,4).map(t=>`<tr><th>${names[t]}</th>${labels.map(p=>`<td>${value.matrix[t][p].toFixed(2)}</td>`).join("")}</tr>`).join("")}</tbody></table></div><table><thead><tr><th>Alvo</th><th>Precisão</th><th>Revocação</th><th>F1</th><th>Suporte (s)</th></tr></thead><tbody>${Object.entries(value.per_target).map(([label,m])=>`<tr><td>${names[label]}</td><td>${pct(m.precision)}</td><td>${pct(m.recall)}</td><td>${pct(m.f1)}</td><td>${m.support.toFixed(2)}</td></tr>`).join("")}</tbody></table>`;
}
action("compare-annotators",async()=>{
  if(!selectedSession)throw new Error("Selecione uma sessão");
  const first=$("agreement-first").value,second=$("agreement-second").value;
  if(!first||!second||first===second)throw new Error("Selecione dois avaliadores diferentes");
  agreementReport=await api(`/sessions/${selectedSession.metadata.id}/agreement?first=${encodeURIComponent(first)}&second=${encodeURIComponent(second)}`);
  renderAgreement(agreementReport); $("prepare-consensus").disabled=!agreementReport.consensus_draft.length;
});
function renderAgreement(value){
  const disagreements=value.disagreements||[];
  $("agreement-report").hidden=false;
  $("agreement-report").innerHTML=`<div class="metrics">${[["Concordância",pct(value.agreement)],["Kappa",value.kappa==null?"—":Number(value.kappa).toFixed(3)],["Cobertura conjunta",pct(value.joint_coverage)],["Tempo comparável",`${seconds(value.comparable_seconds)} s`]].map(([label,v])=>`<div class="metric"><strong>${v}</strong><span>${label}</span></div>`).join("")}</div><p>${seconds(value.joint_seconds)} s cobertos pelos dois avaliadores; ${seconds(value.indeterminate_seconds)} s possuem ao menos uma marcação indeterminada.</p><h3>IoU temporal por alvo</h3><div class="agreement-iou">${Object.entries(value.per_target_iou).map(([target,iou])=>`<span><strong>${esc(names[target])}</strong> ${pct(iou)}</span>`).join("")}</div><h3>Divergências para revisão (${disagreements.length})</h3><div class="disagreement-list">${disagreements.map((row,index)=>`<button class="disagreement-row" data-disagreement="${index}"><span>${seconds(row.start)}–${seconds(row.end)} s</span><strong>${esc(names[row.first])} × ${esc(names[row.second])}</strong><small>abrir no vídeo</small></button>`).join("")||'<p class="small muted">Nenhuma divergência na cobertura comparável.</p>'}</div>`;
}
$("agreement-report").addEventListener("click",event=>{
  const index=event.target.closest("[data-disagreement]")?.dataset.disagreement;
  if(index===undefined)return;
  const row=agreementReport.disagreements[Number(index)];
  $("timeline").value=row.start; $("interval-start").value=seconds(row.start); $("interval-end").value=seconds(row.end);
  stopPlayback(); renderFrame();
});
action("prepare-consensus",async()=>{
  if(!agreementReport?.consensus_draft?.length)throw new Error("Calcule a concordância primeiro");
  if(dirty&&!confirm("Descartar as alterações não salvas e preparar o consenso?"))return;
  const safe=value=>value.replace(/[^a-zA-Z0-9_-]/g,"_");
  const consensusName=`CONSENSO_${safe(agreementReport.first)}_${safe(agreementReport.second)}`.slice(0,80);
  $("annotator").value=consensusName; loadedAnnotator=consensusName;
  consensusSources=[agreementReport.first,agreementReport.second];
  annotations=structuredClone(agreementReport.consensus_draft); dirty=true; renderAnnotations();
  notice("Rascunho de consenso preparado. Use Editar nos intervalos destacados e salve somente após resolver cada divergência.");
});
action("report-download",async () => {
  if (!report) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:"application/json"}));
  const a=document.createElement("a"); a.href=url; a.download=`validacao-${selectedSession.metadata.id}-${loadedAnnotator.replace(/[^a-zA-Z0-9_-]/g,"_")}.json`; a.click(); URL.revokeObjectURL(url);
});
function renderFrame() {
  if (!selectedSession) return;
  const t=Number($("timeline").value), role=$("review-role").value, frames=frameIndex[role];
  $("review-clock").textContent = `${seconds(t)} s`;
  let lo=0,hi=frames.length;
  while(lo<hi) {const mid=(lo+hi)>>1; if(frames[mid].t<t) lo=mid+1; else hi=mid;}
  const candidates=[frames[lo],frames[lo-1]].filter(Boolean).sort((a,b)=>Math.abs(a.t-t)-Math.abs(b.t-t));
  const frame=candidates[0], available=frame && Math.abs(frame.t-t)<=0.5;
  $("review-frame").hidden=!available; $("frame-missing").hidden=!!available;
  if (!available) { $("frame-missing").textContent="Sem quadro próximo deste instante · lacuna na gravação"; $("frame-time").textContent="Nenhuma imagem propagada sobre a lacuna."; lastFrameKey=""; return; }
  const key=`${role}/${frame.sequence}`;
  if(key!==lastFrameKey) {$("review-frame").src=`/api/sessions/${selectedSession.metadata.id}/frame/${key}`; lastFrameKey=key;}
  $("frame-time").textContent=`Quadro em ${seconds(frame.t)} s · diferença ${seconds(frame.t-t)} s · incerteza de sincronização ±${(frame.sync_uncertainty_s*1000).toFixed(1)} ms (estimada)`;
}
function stopPlayback() { playing=false; $("play").textContent="Reproduzir"; }
action("play", async () => { if(playing) stopPlayback(); else {playing=true;playOrigin=performance.now()/1000-Number($("timeline").value);$("play").textContent="Pausar";} });
$("timeline").addEventListener("input",()=>{stopPlayback();renderFrame();});
$("review-role").addEventListener("change",()=>{lastFrameKey="";renderFrame();});
function moveTime(delta) { stopPlayback(); $("timeline").value=Math.max(0,Math.min(Number($("timeline").max),Number($("timeline").value)+delta));renderFrame(); }
action("back-frame",async()=>moveTime(-0.2)); action("next-frame",async()=>moveTime(0.2));
action("mark-start",async()=>{$("interval-start").value=seconds($("timeline").value);});
action("mark-end",async()=>{$("interval-end").value=seconds($("timeline").value);});
setInterval(()=>{if(playing){const t=performance.now()/1000-playOrigin;$("timeline").value=t;renderFrame();if(t>=Number($("timeline").max))stopPlayback();}},125);
// ---- Olhar ao vivo: calibrações de posição do participante, uma estimativa por câmera ----
const liveTrails = {};
let liveData = null, liveBusy = false;
const cameraColors = {notebook: "99,221,255", nao: "199,155,255", validation: "255,207,104"};
const cameraNames = {notebook: "Câmera do notebook", nao: "Câmera do NAO", validation: "Câmera da apresentação"};
const partNames = {head: "cabeça", "right-arm": "braço direito do robô (à esquerda do participante)", torso: "peito / tronco",
  "left-arm": "braço esquerdo do robô (à direita do participante)", "left-leg": "perna esquerda do robô (à direita do participante)",
  "right-leg": "perna direita do robô (à esquerda do participante)"};
const partShort = {head: "cabeça", "right-arm": "braço direito", torso: "tronco", "left-arm": "braço esquerdo", "left-leg": "perna esquerda", "right-leg": "perna direita"};
const levelNames = {alta: "precisão alta", moderada: "precisão moderada", baixa: "precisão baixa"};
function resetLiveTrails() { for (const key of Object.keys(liveTrails)) delete liveTrails[key]; }
$("lg-participant-select").addEventListener("change", () => { resetLiveTrails(); liveData = null; void pollLive(); });
function resizeLiveCanvases() {
  for (const display of ["notebook", "tv"]) {
    const canvas = $(`lg-canvas-${display}`), box = canvas.parentElement.getBoundingClientRect();
    if (!box.width) continue;
    const layout = liveData?.screens?.[display]?.estimates?.[0]?.layout || [1920, 1080];
    const width = Math.round(box.width), height = Math.round(width * layout[1] / layout[0]);
    const ratio = window.devicePixelRatio || 1;
    canvas.style.height = `${height}px`;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    }
  }
}
window.addEventListener("resize", () => { if (activeTab === "live") resizeLiveCanvases(); });
function median(values) { const sorted = [...values].sort((a, b) => a - b); return sorted[Math.floor(sorted.length / 2)]; }
function smoothed(key) {
  // O servidor já filtra as leituras no tempo; a mediana das 3 últimas só tira o resto
  // do tremor sem somar muito atraso.
  const recent = (liveTrails[key] || []).slice(-3);
  return recent.length ? {x: median(recent.map(p => p.x)), y: median(recent.map(p => p.y))} : null;
}
function drawScreen(display, screen, highlighted) {
  const canvas = $(`lg-canvas-${display}`), ctx = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
  if (!w || !h) return;
  const ratio = window.devicePixelRatio || 1;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0f2428"; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(160,200,195,.16)"; ctx.lineWidth = 1;
  for (const f of [1 / 3, 2 / 3]) { ctx.beginPath(); ctx.moveTo(f * w, 0); ctx.lineTo(f * w, h); ctx.moveTo(0, f * h); ctx.lineTo(w, f * h); ctx.stroke(); }
  ctx.strokeStyle = highlighted ? "#42f59b" : "rgba(160,200,195,.35)"; ctx.lineWidth = (highlighted ? 4 : 1.5) * ratio;
  ctx.strokeRect(ctx.lineWidth / 2, ctx.lineWidth / 2, w - ctx.lineWidth, h - ctx.lineWidth);
  if (!screen?.calibrated) {
    ctx.fillStyle = "#8fb0aa"; ctx.font = `${13 * ratio}px Segoe UI, sans-serif`; ctx.textAlign = "center";
    ctx.fillText("Nenhuma câmera calibrada nesta tela", w / 2, h / 2); return;
  }
  const now = performance.now(), clamp = v => Math.max(0, Math.min(1, v));
  let offscreen = false, located = false;
  for (const estimate of screen.estimates) {
    const key = `${display}:${estimate.role}`, color = cameraColors[estimate.role];
    const trail = (liveTrails[key] || []).filter(p => now - p.t <= 2000);
    liveTrails[key] = trail;
    trail.forEach((p, i) => {
      const alpha = .1 + .4 * (i / Math.max(1, trail.length - 1)) * (1 - (now - p.t) / 2000);
      ctx.fillStyle = `rgba(${color},${alpha.toFixed(3)})`;
      ctx.beginPath(); ctx.arc(clamp(p.x) * w, clamp(p.y) * h, 4 * ratio, 0, Math.PI * 2); ctx.fill();
    });
    const point = estimate.gaze?.visible ? smoothed(key) : null;
    if (!point) continue;
    const x = clamp(point.x) * w, y = clamp(point.y) * h;
    if (estimate.gaze.inside === false || !highlighted) {
      // A zona diz que o olhar está em outra superfície: só uma marca apagada, sem anel.
      ctx.fillStyle = `rgba(${color},.35)`; ctx.beginPath(); ctx.arc(x, y, 4 * ratio, 0, Math.PI * 2); ctx.fill();
      continue;
    }
    located = true;
    ctx.strokeStyle = `rgb(${color})`; ctx.lineWidth = 2.5 * ratio;
    ctx.beginPath(); ctx.arc(x, y, 14 * ratio, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = `rgb(${color})`; ctx.beginPath(); ctx.arc(x, y, 4 * ratio, 0, Math.PI * 2); ctx.fill();
    if (point.x < 0 || point.x > 1 || point.y < 0 || point.y > 1) offscreen = true;
  }
  if (offscreen) {
    ctx.fillStyle = "#ffd28f"; ctx.font = `${12 * ratio}px Segoe UI, sans-serif`; ctx.textAlign = "left";
    ctx.fillText("posição imprecisa (mostrada na borda)", 10 * ratio, 18 * ratio);
  }
  if (highlighted && !located) {
    ctx.fillStyle = "#b9f7d8"; ctx.font = `${14 * ratio}px Segoe UI, sans-serif`; ctx.textAlign = "center";
    ctx.fillText(display === "tv" ? "Olhando para a TV" : "Olhando para o notebook", w / 2, h / 2);
    ctx.font = `${12 * ratio}px Segoe UI, sans-serif`;
    ctx.fillText("Posição do olhar indisponível", w / 2, h / 2 + 22 * ratio);
  }
}
function estimateLine(estimate) {
  const gaze = estimate.gaze, level = estimate.validation?.level;
  const quality = level ? levelNames[level] : "sem validação";
  const surfaceLabel = {notebook: "o notebook", tv: "a apresentação", robot: "o robô"};
  const closer = gaze.closer_to ? ` (mais compatível com ${surfaceLabel[gaze.closer_to]})` : "";
  // Duas superfícies na mesma direção para esta câmera (ex.: robô na frente da TV): ela não
  // separa as duas sozinha; a decisão de cima usa as outras câmeras.
  const also = gaze.ambiguous_with ? ` · também compatível com ${surfaceLabel[gaze.ambiguous_with]}` : "";
  const where = !gaze.visible ? `${gaze.inside ? "superfície reconhecida · " : ""}${esc(gaze.reason)}`
    : gaze.surface === "robot" ? (!gaze.inside ? `fora do robô${closer}` : (gaze.part ? `olhando: ${esc(partShort[gaze.part])}`
      : gaze.part_reliable === false ? `olhando o robô · parte não confiável (${Math.round(6 * gaze.part_accuracy)}/6 na validação)` : "olhando o robô") + also)
    : gaze.inside ? `olhando para esta tela · ${gaze.position_inside === false ? "posição imprecisa (fora do mapa)" : `x ${(gaze.x * 100).toFixed(0)}% · y ${(gaze.y * 100).toFixed(0)}%`}${also}` : `esta câmera não confirma esta tela${closer}`;
  return `<span class="cam-line"><span class="cam-dot" style="background:rgb(${cameraColors[estimate.role]})"></span>${esc(cameraNames[estimate.role])}: ${where} <span class="muted">· ${esc(quality)}</span></span>`;
}
function renderRobot(robot) {
  const looking = robot.looking, part = robot.part;
  $("lg-card-robot").classList.toggle("looking", looking);
  for (const [id, key] of [["lr-head", "head"], ["lr-leg-left", "left-leg"], ["lr-leg-right", "right-leg"],
                           ["lr-arm-left", "left-arm"], ["lr-arm-right", "right-arm"], ["lr-body", "torso"]]) {
    $(id).classList.toggle("active", looking && part === key);
  }
  $("lr-dots").innerHTML = robot.estimates.filter(e => e.gaze.visible && e.gaze.inside).map(e => {
    const p = smoothed(`robot:${e.role}`) || e.gaze, x = Math.max(0, Math.min(1, p.x)) * 320, y = Math.max(0, Math.min(1, p.y)) * 380;
    return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="11" fill="none" stroke="rgb(${cameraColors[e.role]})" stroke-width="4"/>`;
  }).join("");
  $("lg-badge-robot").textContent = !robot.calibrated && liveData.source !== "classificador" ? "Sem calibração" : looking ? "Olhando" : "Não olhando";
  $("lg-badge-robot").className = `badge ${looking ? "online" : robot.calibrated ? "" : "warn"}`;
  $("lg-status-robot").innerHTML = robot.calibrated ? robot.estimates.map(estimateLine).join("")
    : "Calibre o robô na página de alvo: <em>O que calibrar → Robô NAO</em>.";
}
function renderLive() {
  const data = liveData; if (!data) return;
  // "stable" tem histerese: um alvo novo precisa durar 0,35 s para aparecer.
  const fused = liveTarget(data), part = data.stable ? data.stable.part : data.robot.part;
  const pending = data.stable?.pending ? ` · confirmando: ${names[data.stable.pending] || data.stable.pending}` : "";
  $("lg-target").textContent = fused === "NAO" && part ? `Robô NAO · ${partShort[part]}` : (names[fused] || fused);
  $("lg-target").className = fused === "INDETERMINADO" ? "unknown" : "";
  $("lg-reason").textContent = data.source === "posição" ? `${data.reason}${pending} · pelas calibrações de posição de ${data.participant}`
    : data.source === "classificador" ? `${data.reason} · pelo classificador de alvos (sem calibração de posição)`
    : "Nenhuma calibração para este participante. Calibre cada câmera na página de alvo (/alvo).";
  $("lg-recording").textContent = data.recording ? "● Gravando" : "Sem gravação";
  $("lg-recording").className = `badge ${data.recording ? "recording" : ""}`;
  const select = $("lg-participant-select"), signature = data.participants.join("|");
  if (select.dataset.signature !== signature) {
    const chosen = select.value;
    select.innerHTML = `<option value="">Último calibrado${data.participants.length ? "" : " (nenhum)"}</option>` +
      data.participants.map(p => `<option value="${esc(p)}">${esc(p)}</option>`).join("");
    select.value = data.participants.includes(chosen) ? chosen : "";
    select.dataset.signature = signature;
  }
  $("lg-participant").textContent = data.participant ? `Mostrando ${data.participant}` : "";
  for (const display of ["notebook", "tv"]) {
    const screen = data.screens[display], highlighted = fused === (display === "tv" ? "TV" : "NOTEBOOK");
    $(`lg-card-${display}`).classList.toggle("looking", highlighted);
    $(`lg-badge-${display}`).textContent = screenBadge(screen, fused, display);
    $(`lg-badge-${display}`).className = `badge ${highlighted ? "online" : fused === "INDETERMINADO" ? "warn" : ""}`;
    $(`lg-status-${display}`).innerHTML = !screen.calibrated
      ? `Calibre esta tela em <a href="/alvo?display=${display}" target="_blank" rel="noopener">/alvo?display=${display}</a>, uma câmera de cada vez.`
      : screen.estimates.map(estimateLine).join("");
    drawScreen(display, screen, highlighted);
  }
  renderRobot({...data.robot, looking: fused === "NAO", part});
}
function liveTarget(data) { return data.stable?.target ?? data.target; }
function screenBadge(screen, target, display) {
  if (!screen.calibrated) return "Sem calibração";
  if (target === (display === "tv" ? "TV" : "NOTEBOOK")) return "Olhando";
  if (target === "INDETERMINADO") return "Indeterminado";
  return "Não olhando";
}
async function pollLive() {
  if (!authenticated || activeTab !== "live" || document.hidden || liveBusy) return;
  liveBusy = true;
  try {
    const chosen = $("lg-participant-select").value;
    const response = await api(`/gaze/live${chosen ? `?participant=${encodeURIComponent(chosen)}` : ""}`);
    if (chosen !== $("lg-participant-select").value) return;
    liveData = response;
    const now = performance.now();
    const estimates = [...liveData.screens.notebook.estimates.map(e => ["notebook", e]), ...liveData.screens.tv.estimates.map(e => ["tv", e]),
                       ...liveData.robot.estimates.map(e => ["robot", e])];
    for (const [surface, estimate] of estimates) {
      const key = `${surface}:${estimate.role}`, gaze = estimate.gaze, trail = (liveTrails[key] ||= []);
      if (gaze?.visible && trail.at(-1)?.sequence !== gaze.sequence) trail.push({x: gaze.x, y: gaze.y, t: now, sequence: gaze.sequence});
      while (trail.length && now - trail[0].t > 2000) trail.shift();
    }
    resizeLiveCanvases(); renderLive();
  } catch (error) {
    $("lg-reason").textContent = error.message;
  } finally { liveBusy = false; }
}
setInterval(() => { void pollLive(); }, 100);
// Redesenha para o rastro esmaecer mesmo quando a leitura some.
setInterval(() => { if (activeTab === "live" && liveData) for (const d of ["notebook", "tv"]) drawScreen(d, liveData.screens[d], liveTarget(liveData) === (d === "tv" ? "TV" : "NOTEBOOK")); }, 250);
let polling=false;
setInterval(async()=>{if(!authenticated||polling)return;polling=true;try{await poll();}catch(error){$("server-status").textContent="Servidor indisponível";$("server-status").className="badge";}finally{polling=false;}},800);
setInterval(()=>{void refreshPreviews();},50);
setInterval(()=>{if(authenticated&&["capture","live"].includes(activeTab)){for(const role of roles){if(rayFrames[role]&&performance.now()-rayFrames[role].receivedAt>400)paintRay(role);}}},150);
window.addEventListener("beforeunload",event=>{if(dirty){event.preventDefault();event.returnValue="";}});
(async()=>{const fragment=new URLSearchParams(location.hash.slice(1)), key=fragment.get("access"); if(key)history.replaceState(null,"",location.pathname);try{await enter(key);}catch{showLogin();}})();
