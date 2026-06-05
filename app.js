const sz = 320;
const infoUrl = "./models/effb4_320.json";

// UI
const fileImg = document.getElementById("fileImg");
const outImg = document.getElementById("outImg");
const orig = document.getElementById("orig"); const octx = orig.getContext("2d");
const heat = document.getElementById("heat"); const hctx = heat.getContext("2d");
const overlay = document.getElementById("overlay"); const ovctx = overlay.getContext("2d");
const saveHeatImg = document.getElementById("saveHeatImg");
const saveOverlayImg = document.getElementById("saveOverlayImg");

// state
let sessionCeleb = null;
let sessionFFPP = null;
let info = null;
let imageBitmap = null;

async function loadInfo() {
  const r = await fetch(infoUrl);
  info = await r.json();
}
loadInfo();

function providers(ep) { return ["webgpu", "wasm"]; }

async function ensureSession() {
  if (sessionCeleb && sessionFFPP) return;
  sessionCeleb = await ort.InferenceSession.create(`./models/effb4_320.onnx`, {
    executionProviders: providers(), graphOptimizationLevel: "all",
  });
  sessionFFPP = await ort.InferenceSession.create(`./models/effb4_ffpp_320.onnx`, {
    executionProviders: providers(), graphOptimizationLevel: "all",
  });
}

function centerCropToCanvas(img, size, ctx) {
  const s = Math.min(img.width, img.height);
  const sx = (img.width - s)/2, sy = (img.height - s)/2;
  ctx.clearRect(0,0,size,size);
  ctx.drawImage(img, sx, sy, s, s, 0, 0, size, size);
}

function toCHWFloat32(imgData, mean, std) {
  const { data, width, height } = imgData;
  const N = width*height;
  const out = new Float32Array(3*N);
  const m = mean ?? [0.485,0.456,0.406];
  const s = std  ?? [0.229,0.224,0.225];
  let jR=0, jG=N, jB=2*N;
  for (let i=0;i<N;i++) {
    const r=data[4*i]/255, g=data[4*i+1]/255, b=data[4*i+2]/255;
    out[jR++]=(r-m[0])/s[0]; out[jG++]=(g-m[1])/s[1]; out[jB++]=(b-m[2])/s[2];
  }
  return out;
}

function sigmoid(x){ return 1/(1+Math.exp(-x)); }

function upsampleBilinear(src, Wf, Hf, W, H){
  const dst = new Float32Array(W*H);
  const xratio = (Wf-1)/(W-1), yratio = (Hf-1)/(H-1);
  for (let y=0;y<H;y++){
    const gy=y*yratio, y0=Math.floor(gy), y1=Math.min(Hf-1,y0+1), ly=gy-y0;
    for (let x=0;x<W;x++){
      const gx=x*xratio, x0=Math.floor(gx), x1=Math.min(Wf-1,x0+1), lx=gx-x0;
      const i00=y0*Wf+x0, i01=y0*Wf+x1, i10=y1*Wf+x0, i11=y1*Wf+x1;
      const v0=src[i00]*(1-lx)+src[i01]*lx, v1=src[i10]*(1-lx)+src[i11]*lx;
      dst[y*W+x]=v0*(1-ly)+v1*ly;
    }
  }
  return dst;
}

function normalize01(arr){
  let min=Infinity,max=-Infinity;
  for (const v of arr){ if(v<min)min=v; if(v>max)max=v; }
  const d=(max-min)||1; const out=new Float32Array(arr.length);
  for (let i=0;i<arr.length;i++) out[i]=(arr[i]-min)/d;
  return out;
}

function colormapJet(v){
  const four=4*v;
  const r=Math.min(1,Math.max(0,Math.min(four-1.5,-four+4.5)));
  const g=Math.min(1,Math.max(0,Math.min(four-0.5,-four+3.5)));
  const b=Math.min(1,Math.max(0,Math.min(four+0.5,-four+2.5)));
  return [Math.round(r*255),Math.round(g*255),Math.round(b*255)];
}

function paintHeat(mask01, ctx, W, H){
  const img=ctx.createImageData(W,H);
  for (let i=0;i<W*H;i++){
    const [r,g,b]=colormapJet(mask01[i]);
    img.data[4*i]=r; img.data[4*i+1]=g; img.data[4*i+2]=b; img.data[4*i+3]=255;
  }
  ctx.putImageData(img,0,0);
}

function blendOverlay(baseCtx, heatCtx, outCtx, alpha=0.45){
  const base=baseCtx.getImageData(0,0,sz,sz);
  const hm=heatCtx.getImageData(0,0,sz,sz);
  const out=outCtx.createImageData(sz,sz);
  for (let i=0;i<sz*sz;i++){
    out.data[4*i  ]=(1-alpha)*base.data[4*i  ]+alpha*hm.data[4*i  ];
    out.data[4*i+1]=(1-alpha)*base.data[4*i+1]+alpha*hm.data[4*i+1];
    out.data[4*i+2]=(1-alpha)*base.data[4*i+2]+alpha*hm.data[4*i+2];
    out.data[4*i+3]=255;
  }
  outCtx.putImageData(out,0,0);
}

async function forwardCHW(chw, targetSession){
  const t = new ort.Tensor("float32", chw, [1,3,sz,sz]);
  return await targetSession.run({ input: t }); 
}

fileImg.addEventListener("change", async (e) => {
  const f = e.target.files?.[0];
  if (!f) return;
  imageBitmap = await createImageBitmap(f);
  await runOnce();
});

async function runOnce(){
  try{
    await ensureSession();
    centerCropToCanvas(imageBitmap, sz, octx);
    const imgData = octx.getImageData(0,0,sz,sz);
    const baseCHW = toCHWFloat32(imgData, info?.mean, info?.std);

    const t0 = performance.now();
    const out1 = await forwardCHW(baseCHW, sessionCeleb);
    const prob1 = sigmoid(out1.logits.data[0]);

    const out2 = await forwardCHW(baseCHW, sessionFFPP);
    const prob2 = 1.0 - sigmoid(out2.logits.data[0]);

    const weightCeleb = 0.4;
    const weightFFPP  = 0.6;
    const ensembleProb = (prob1 * weightCeleb) + (prob2 * weightFFPP);

    const threshold = 0.5;
    const label = ensembleProb >= threshold ? "FAKE" : "REAL";
    const pill = `<span class="pill ${label==='FAKE'?'bad':'ok'}" style="font-size: 1.2rem; padding: 10px 24px;">${label}</span>`;
    
    outImg.innerHTML = `
      <div style="display: flex; flex-direction: column; align-items: center; text-align: center; gap: 12px; width: 100%;">
        <div style="margin-bottom: 4px;">${pill}</div>
        <div style="font-size: 0.9rem; color: var(--text-muted); display: flex; gap: 15px; justify-content: center;">
          <span>Celeb-DF: <b>${prob1.toFixed(3)}</b></span>
          <span>FF++: <b>${prob2.toFixed(3)}</b></span>
        </div>
        <div style="font-size: 1.1rem; color: var(--text-main); border-top: 1px solid var(--border); padding-top: 8px; width: 80%;">
          Итог: <span style="font-weight: 700; color: ${label === 'FAKE' ? '#f87171' : '#34d399'}">${ensembleProb.toFixed(3)}</span>
        </div>
        <div style="font-size: 0.75rem; color: var(--text-muted); opacity: 0.7;">
          time: ${(performance.now() - t0).toFixed(1)} ms
        </div>
      </div>
    `;

    const feat = out1.feat; const [_, C, Hf, Wf] = feat.dims;
    const arr = feat.data;
    const K = 32; 

    const means = new Float32Array(C);
    for (let c=0;c<C;c++){
      let s=0; for (let i=0;i<Hf*Wf;i++) s += Math.abs(arr[c*Hf*Wf+i]);
      means[c] = s/(Hf*Wf);
    }
    const idx = Array.from(means.keys()).sort((a,b)=>means[b]-means[a]).slice(0,K);

    const heatAcc = new Float32Array(sz*sz);
    for (const c of idx){
      const ch = arr.subarray(c*Hf*Wf, (c+1)*Hf*Wf);
      let up = upsampleBilinear(ch, Wf, Hf, sz, sz);
      up = normalize01(up);
      const masked = new ImageData(sz, sz);
      for (let i=0;i<sz*sz;i++){
        const m = up[i];
        const r = imgData.data[4*i], g = imgData.data[4*i+1], b = imgData.data[4*i+2];
        masked.data[4*i]=r*m; masked.data[4*i+1]=g*m; masked.data[4*i+2]=b*m; masked.data[4*i+3]=255;
      }
      const outM = await forwardCHW(toCHWFloat32(masked, info?.mean, info?.std), sessionCeleb);
      const p_c = sigmoid(outM.logits.data[0]); 
      for (let i=0;i<sz*sz;i++) heatAcc[i] += p_c * up[i];
    }

    for (let i=0;i<heatAcc.length;i++) heatAcc[i] = Math.max(0, heatAcc[i]);
    const heat01 = normalize01(heatAcc);
    paintHeat(heat01, hctx, sz, sz);
    blendOverlay(octx, hctx, ovctx, 0.45);

    saveHeatImg.disabled = false;
    saveOverlayImg.disabled = false;
    saveHeatImg.onclick = () => {
      const a=document.createElement("a"); a.href=heat.toDataURL("image/png"); a.download="heatmap.png"; a.click();
    };
    saveOverlayImg.onclick = () => {
      const a=document.createElement("a"); a.href=overlay.toDataURL("image/png"); a.download="overlay.png"; a.click();
    };
  }catch(e){
    console.error(e);
    outImg.textContent = "Ошибка: " + e.message;
  }
}