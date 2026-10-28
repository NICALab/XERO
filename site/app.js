/* Buildless, browser-local interactions. Scientific values are precomputed. */
const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const mediaRoot = 'https://xero-viewer.pages.dev/project-media/';
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let motionPaused = reducedMotion.matches || Boolean(navigator.connection?.saveData);
const visible = new Set();

function loadVideo(video) {
  if (video.dataset.src && video.getAttribute('src') !== video.dataset.src) {
    video.src = video.dataset.src;
    video.load();
  }
}
function updateMotionButton() {
  const button = $('#motion-toggle');
  button.textContent = motionPaused ? 'Play motion ▶' : 'Pause motion Ⅱ';
  button.setAttribute('aria-pressed', String(motionPaused));
}
function maybePlay(video) {
  if (!motionPaused && !document.hidden && visible.has(video)) {
    loadVideo(video);
    video.play().catch(() => { /* Native controls remain available. */ });
  }
}
const observer = new IntersectionObserver(entries => {
  for (const entry of entries) {
    const video = entry.target;
    if (entry.isIntersecting) { visible.add(video); maybePlay(video); }
    else { visible.delete(video); video.pause(); }
  }
}, {threshold: 0.12});
$$('video[data-src]').forEach(video => {
  observer.observe(video);
  video.addEventListener('pointerdown', () => loadVideo(video), {passive:true});
  video.addEventListener('keydown', () => loadVideo(video));
  video.addEventListener('error', () => {
    video.title = 'Video could not load. The poster shows the selected recording.';
  });
});
$('#motion-toggle').addEventListener('click', () => {
  motionPaused = !motionPaused; updateMotionButton();
  $$('video').forEach(video => motionPaused ? video.pause() : maybePlay(video));
});
document.addEventListener('visibilitychange', () => {
  $$('video').forEach(video => document.hidden ? video.pause() : maybePlay(video));
});
reducedMotion.addEventListener('change', event => {
  if (event.matches) {motionPaused=true;updateMotionButton();$$('video').forEach(video=>video.pause());}
});
updateMotionButton();

const cases = {
  calcium: {type:'Functional imaging',title:'Calcium activity, cell by cell.',description:'Mouse visual cortex. Matched images, magnified ROIs and fluorescence traces.',bound:'1%'},
  voltage: {type:'Fast functional imaging',title:'Follow the fast signals.',description:'Zebrafish spinal cord. Activity overlays, matched ROIs and optical voltage transients.',bound:'0.5%'},
  structural: {type:'Four-color structural imaging',title:'Structure in every channel.',description:'ELAST brain tissue. A four-channel overlay, full field of view and matched magnified regions.',bound:'2%'},
  molecular: {type:'Volumetric RNA imaging',title:'Small spots. Rich information.',description:'EASI-FISH. A z-stack of punctate RNA signals with matched full-field and magnified views.',bound:'2%'},
  'two-color': {type:'Whole-brain functional imaging',title:'Neurons and glia, together.',description:'Zebrafish calcium imaging. Two-channel reconstruction and corresponding activity traces.',bound:'1%'}
};
function selectCase(key) {
  const selected = cases[key]; if (!selected) return;
  $$('.tabs button').forEach(button => {const active=button.dataset.case===key;button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1;});
  $('#gallery-panel').setAttribute('aria-labelledby','tab-'+key);
  $('#gallery-type').textContent=selected.type;$('#gallery-title').textContent=selected.title;
  $('#gallery-description').textContent=selected.description;$('#gallery-bound').textContent=selected.bound;
  const video=$('#gallery-video');video.pause();video.removeAttribute('src');video.load();
  video.poster='assets/'+key+'.jpg';video.dataset.src=mediaRoot+key+'.mp4';
  video.setAttribute('aria-label',selected.description);maybePlay(video);
}
$$('.tabs button').forEach((button,i,buttons) => {
  button.addEventListener('click',()=>selectCase(button.dataset.case));
  button.addEventListener('keydown',event=>{
    let next=i;
    if(event.key==='ArrowRight')next=(i+1)%buttons.length;
    else if(event.key==='ArrowLeft')next=(i-1+buttons.length)%buttons.length;
    else if(event.key==='Home')next=0;
    else if(event.key==='End')next=buttons.length-1;
    else return;
    event.preventDefault();buttons[next].focus();selectCase(buttons[next].dataset.case);
  });
});

const comparison=$('#comparison');
let comparisonPointer=null;
function setComparison(value) {
  const split=Math.max(0,Math.min(100,value));
  comparison.style.setProperty('--split',split+'%');
  comparison.setAttribute('aria-valuenow',String(Math.round(split)));
  comparison.setAttribute('aria-valuetext',`${Math.round(split)}% denoised, ${100-Math.round(split)}% XERO`);
}
function moveComparison(event) {
  const rect=comparison.getBoundingClientRect();
  if(rect.width>0)setComparison(100*(event.clientX-rect.left)/rect.width);
}
comparison.addEventListener('pointerdown',event=>{
  if(!event.isPrimary || event.button!==0 || comparisonPointer!==null)return;
  comparisonPointer=event.pointerId;
  comparison.focus({preventScroll:true});
  comparison.setPointerCapture(event.pointerId);
  comparison.classList.add('dragging');
  moveComparison(event);
});
comparison.addEventListener('pointermove',event=>{
  if(event.pointerId===comparisonPointer)moveComparison(event);
});
function finishComparison(event) {
  if(event.pointerId!==comparisonPointer)return;
  comparisonPointer=null;
  comparison.classList.remove('dragging');
  if(comparison.hasPointerCapture(event.pointerId))comparison.releasePointerCapture(event.pointerId);
}
comparison.addEventListener('pointerup',event=>{
  if(event.pointerId===comparisonPointer)moveComparison(event);
  finishComparison(event);
});
comparison.addEventListener('pointercancel',finishComparison);
comparison.addEventListener('lostpointercapture',finishComparison);
comparison.addEventListener('dragstart',event=>event.preventDefault());
comparison.addEventListener('keydown',event=>{
  const value=parseFloat(comparison.style.getPropertyValue('--split'));
  const step=event.shiftKey?10:1;
  const next={ArrowLeft:value-step,ArrowDown:value-step,ArrowRight:value+step,ArrowUp:value+step,Home:0,End:100}[event.key];
  if(next===undefined)return;
  event.preventDefault();setComparison(next);
});
$$('[data-zoom]').forEach(button=>button.addEventListener('click',()=>{
  $('#comparison').style.setProperty('--zoom',button.dataset.zoom);
  $$('[data-zoom]').forEach(b=>{b.classList.toggle('selected',b===button);b.setAttribute('aria-pressed',String(b===button));});
}));

const storage={
  embryo:{name:'Whole-embryo anatomy',before:178016606040,after:1529490391},
  calcium:{name:'Mouse two-photon calcium',before:56811490816,after:1620698209},
  kidney:{name:'Expanded kidney tissue',before:174749878256,after:3319972873},
  voltage:{name:'Zebrafish spinal-cord voltage',before:7256309760,after:294076874}
};
$$('[data-storage]').forEach(button=>button.addEventListener('click',()=>{
  const data=storage[button.dataset.storage];
  $$('[data-storage]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});
  $('#storage-name').textContent=data.name;
  $('#size-before').innerHTML=(data.before/1e9).toFixed(2)+' <small>GB</small>';
  $('#size-after').innerHTML=(data.after/1e9).toFixed(2)+' <small>GB</small>';
  $('#size-fill').style.width=(100*data.after/data.before)+'%';
  $('#size-saving').textContent=(100*(1-data.after/data.before)).toFixed(1)+'% less data';
}));

async function drawTrace() {
  const response=await fetch('assets/traces.json');if(!response.ok)throw Error('Trace unavailable');
  const d=await response.json(), svg=$('#trace-plot'), ns='http://www.w3.org/2000/svg';
  const values=[...d.denoised,...d.xero],low=Math.floor(Math.min(...values)/5)*5,high=Math.ceil(Math.max(...values)/5)*5;
  const x=t=>50+930*t/d.time_s.at(-1),y=v=>174-146*(v-low)/(high-low||1);
  function element(type,attrs,text){const e=document.createElementNS(ns,type);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;svg.append(e);return e;}
  for(let i=0;i<=3;i++){const v=low+(high-low)*i/3,yy=y(v);element('line',{x1:50,x2:980,y1:yy,y2:yy,stroke:'#cad6c2','stroke-width':.6});element('text',{x:38,y:yy+4,'text-anchor':'end',fill:'#5e7356','font-size':11},v.toFixed(0));}
  for(const t of [0,10,20,30])if(t<=d.time_s.at(-1)+.1)element('text',{x:x(t),y:198,'text-anchor':'middle',fill:'#5e7356','font-size':11},t+' s');
  for(const[key,color,dash]of [['denoised','#2a6652',''],['xero','#bd581b','5 4']]){
    const path=d[key].map((v,i)=>(i?'L':'M')+x(d.time_s[i]).toFixed(2)+','+y(v).toFixed(2)).join(' ');
    element('path',{d:path,fill:'none',stroke:color,'stroke-width':key==='denoised'?1.8:1.4,'stroke-dasharray':dash,'stroke-linecap':'round'});
  }
}
drawTrace().catch(()=>{$('#trace-plot').setAttribute('aria-label','Trace could not load');});
