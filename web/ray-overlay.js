"use strict";
(() => {
  const COLORS={green:"#3bdf85",red:"#ff6565",unknown:"#aab9c6"};
  const TARGET_NAMES={NOTEBOOK:"Notebook",TV:"TV",NAO:"NAO",OUTRO_LOCAL:"Outro local",INDETERMINADO:"Indeterminado"};
  function classify(target,reference="screens"){
    if(!["NOTEBOOK","TV","NAO","OUTRO_LOCAL"].includes(target))return "unknown";
    const accepted=reference==="robot"?["NAO"]:reference==="task"?["NOTEBOOK","TV","NAO"]:["NOTEBOOK","TV"];
    return accepted.includes(target)?"green":"red";
  }
  function decision(metadata,reference="task"){
    const region=metadata?.regions?.[reference];
    const state=region?(region.target==="DENTRO"?"green":region.target==="FORA"?"red":"unknown"):classify(metadata?.target,reference);
    const base=reference==="robot"?["Voltado ao NAO","Fora do NAO"]:reference==="screens"?["Nas telas","Fora das telas"]:["Na tarefa","Fora da tarefa"];
    const exact=TARGET_NAMES[metadata?.target];
    return {state,color:COLORS[state],label:state==="unknown"?(region?.reason||metadata?.reason||"Aguardando leitura dos olhos"):
      `${base[state==="green"?0:1]} · ${exact&&metadata.target!=="INDETERMINADO"?exact:"alvo exato incerto"}`,approximate:true};
  }
  function project(eyes,direction,width,height,sourceWidth,sourceHeight){
    if(!eyes||eyes.length!==2||!sourceWidth||!sourceHeight)return [];
    const scale=Math.min(width/sourceWidth,height/sourceHeight);
    const renderedWidth=sourceWidth*scale,renderedHeight=sourceHeight*scale;
    const ox=(width-renderedWidth)/2,oy=(height-renderedHeight)/2;
    const validDirection=Array.isArray(direction)&&direction.length===2&&direction.every(Number.isFinite);
    const dx=validDirection?Math.max(-1,Math.min(1,direction[0])):0;
    const dy=validDirection?Math.max(-1,Math.min(1,direction[1])):0;
    const magnitude=Math.hypot(dx,dy),factor=magnitude>1?1/magnitude:1;
    const length=Math.min(renderedWidth,renderedHeight)*.28;
    return eyes.filter(p=>Array.isArray(p)&&p.length===2&&p.every(Number.isFinite)&&p.every(v=>v>=0&&v<=1)).map(p=>{
      const x=ox+p[0]*renderedWidth,y=oy+p[1]*renderedHeight;
      return {x,y,endX:Math.max(ox,Math.min(ox+renderedWidth,x+dx*factor*length)),
              endY:Math.max(oy,Math.min(oy+renderedHeight,y+dy*factor*length)),frontal:validDirection&&magnitude<.12,hasDirection:validDirection};
    });
  }
  function paint(canvas,image,metadata,reference="screens",enabled=true){
    const box=canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);
    const pixelWidth=Math.round(box.width*dpr),pixelHeight=Math.round(box.height*dpr);
    if(canvas.width!==pixelWidth)canvas.width=pixelWidth;
    if(canvas.height!==pixelHeight)canvas.height=pixelHeight;
    const ctx=canvas.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,box.width,box.height);
    const result=decision(metadata,reference),{color}=result;
    if(enabled&&metadata){
      const rays=project(metadata.eyes,metadata.direction,box.width,box.height,image.naturalWidth,image.naturalHeight);
      for(const ray of rays){
        ctx.lineCap="round";ctx.lineJoin="round";
        const path=()=>{
          ctx.beginPath();ctx.arc(ray.x,ray.y,4,0,Math.PI*2);
          if(ray.hasDirection&&!ray.frontal){
            const angle=Math.atan2(ray.endY-ray.y,ray.endX-ray.x),head=6;
            ctx.moveTo(ray.x,ray.y);ctx.lineTo(ray.endX,ray.endY);
            ctx.moveTo(ray.endX-head*Math.cos(angle-.5),ray.endY-head*Math.sin(angle-.5));
            ctx.lineTo(ray.endX,ray.endY);ctx.lineTo(ray.endX-head*Math.cos(angle+.5),ray.endY-head*Math.sin(angle+.5));
          }else if(ray.frontal){ctx.moveTo(ray.x+8,ray.y);ctx.arc(ray.x,ray.y,8,0,Math.PI*2);}
        };
        path();ctx.strokeStyle="rgba(0,0,0,.75)";ctx.lineWidth=4.5;ctx.stroke();
        path();ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();
      }
    }
    return result;
  }
  const api={COLORS,classify,decision,project,paint};
  if(typeof module==="object"&&module.exports)module.exports=api;else window.GazeOverlay=api;
})();
