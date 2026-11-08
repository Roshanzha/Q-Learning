'use strict';
const $=id=>document.getElementById(id),{Environment,Agent}=CatchLearning;
const env=new Environment(),trainingEnv=new Environment();let agent=new Agent();
const canvas=$('arena'),ctx=canvas.getContext('2d');
let mode='human',phase='ready',duration=60,remaining=60,score=0,caught=0,missed=0,accumulator=0,training=false,trainStart=0,trainTarget=0,last=0,flash=0,saveAvailable=true;
const held=new Set();
try{const saved=localStorage.getItem('catch-lab-model-v1');if(saved)agent.restore(JSON.parse(saved));}catch{saveAvailable=false;}
function persist(){try{localStorage.setItem('catch-lab-model-v1',JSON.stringify(agent.data()));saveAvailable=true;}catch{saveAvailable=false;}}
function overlay(title,text,label,icon='↔'){ $('overlay').hidden=false;$('overlayTitle').textContent=title;$('overlayText').textContent=text;$('start').textContent=label;$('overlayIcon').textContent=icon;$('overlayHint').textContent=mode==='human'?'Use ← → or A / D to move':'The agent chooses the highest Q-value.';}
function ready(){phase='ready';remaining=duration;score=caught=missed=0;env.reset();held.clear();accumulator=0;overlay(mode==='human'?'Your move.':'Watch it think.',mode==='human'?'Catch the apple before it hits the ground. Every catch +10. Every miss −10.':agent.episodes?'Your trained agent is ready. Let’s put its policy to the test.':'This agent is untrained. Watch random decisions, or train it first.','Start a round',mode==='human'?'↔':'✦');sync();}
function start(){if(training)return;if(phase==='paused'){phase='playing';}else{score=caught=missed=0;remaining=duration;accumulator=0;env.reset();phase='playing';}held.clear();last=performance.now();$('overlay').hidden=true;$('feedback').textContent=mode==='human'?'Follow the apple. Hold a direction to move.':'Greedy policy active. The agent is using what it learned.';sync();}
function pause(){if(phase==='playing'){phase='paused';held.clear();overlay('Take a breather.','Your game and timer are paused. Pick up where you left off.','Resume round','Ⅱ');}else if(phase==='paused')start();sync();}
function sync(){
  $('score').textContent=score;$('caught').innerHTML=caught+'<span class="score-unit"> apples</span>';
  const secs=Math.ceil(remaining);$('timer').textContent=String(Math.floor(secs/60)).padStart(2,'0')+':'+String(secs%60).padStart(2,'0');document.querySelector('.timer').classList.toggle('urgent',secs<=10);
  $('gameStatus').textContent=training?'TRAINING IN PROGRESS':{ready:'READY TO CATCH',playing:mode==='human'?'YOUR TURN':'AGENT IN ACTION',paused:'TAKING A BREAK',ended:'ROUND COMPLETE'}[phase];
  $('pause').disabled=!['playing','paused'].includes(phase)||training;$('pause').textContent=phase==='paused'?'▷ Resume':'Ⅱ Pause';$('start').disabled=training;$('restart').disabled=training;$('duration').disabled=phase==='playing'||phase==='paused'||training;
  $('episodes').textContent=agent.episodes.toLocaleString();$('epsilon').innerHTML=(agent.epsilon*100).toFixed(agent.epsilon<.1?1:0)+'<span>%</span>';
  const values=agent.values(env.state),best=Math.max(...values);values.forEach((v,i)=>{$('q'+i).querySelector('strong').textContent=v.toFixed(2);$('q'+i).classList.toggle('best',v===best&&best!==0);});
  $('stateLabel').textContent=`State: basket ${env.basket} · apple ${env.apple} · row ${env.row}`;
}
function chart(){const history=agent.history;$('chartEmpty').style.display=history.length?'none':'';if(!history.length){$('chartLine').setAttribute('d','');$('chartArea').setAttribute('d','');$('trainRate').textContent='Awaiting training';return;}
  const points=history.map((p,i)=>[history.length===1?160:i/(history.length-1)*320,90-p.rate*80]);const path=points.map((p,i)=>(i?'L':'M')+p.join(' ')).join(' ');$('chartLine').setAttribute('d',path);$('chartArea').setAttribute('d',path+` L${points.at(-1)[0]} 100 L${points[0][0]} 100 Z`);$('trainRate').textContent=Math.round(history.at(-1).rate*100)+'% recent catches';}
function stopTraining(){training=false;persist();$('train').innerHTML='✦ Train agent <span>+30,000 episodes</span>';$('trainProgress').hidden=true;$('trainNote').textContent=saveAvailable?'Progress saved in this browser. Switch to AI plays.':'Trained for this session. Export to keep your model.';sync();chart();}
function train(){if(training){stopTraining();return;}if(phase==='playing')pause();training=true;trainStart=agent.episodes;trainTarget=trainStart+30000;$('train').textContent='Stop training · keep progress';$('trainProgress').hidden=false;$('trainNote').textContent='Learning from rewards, one apple at a time…';sync();}
function draw(){
  const w=630,h=450,cw=w/9,ch=h/9;ctx.clearRect(0,0,w,h);ctx.fillStyle='#edf1e5';ctx.fillRect(0,0,w,h);
  ctx.strokeStyle='#dce5d1';ctx.lineWidth=1;for(let i=1;i<9;i++){ctx.beginPath();ctx.moveTo(i*cw,0);ctx.lineTo(i*cw,h);ctx.stroke();ctx.beginPath();ctx.moveTo(0,i*ch);ctx.lineTo(w,i*ch);ctx.stroke();}
  ctx.fillStyle='#e0e8ce';ctx.fillRect(0,400,w,50);ctx.setLineDash([3,5]);ctx.strokeStyle='#c0cfae';ctx.beginPath();ctx.moveTo(0,400);ctx.lineTo(w,400);ctx.stroke();ctx.setLineDash([]);
  const x=env.apple*cw+cw/2,y=env.row*ch+ch/2;
  ctx.fillStyle='#233b2010';ctx.beginPath();ctx.ellipse(x,431,18,5,0,0,Math.PI*2);ctx.fill();
  ctx.strokeStyle='#697443';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x,y-13);ctx.quadraticCurveTo(x-2,y-22,x+4,y-24);ctx.stroke();
  ctx.fillStyle='#6e9250';ctx.beginPath();ctx.ellipse(x+8,y-20,8,4,-.5,0,Math.PI*2);ctx.fill();
  ctx.fillStyle='#e87854';ctx.beginPath();ctx.moveTo(x,y-12);ctx.bezierCurveTo(x-28,y-26,x-26,y+19,x-6,y+20);ctx.quadraticCurveTo(x,y+17,x+6,y+20);ctx.bezierCurveTo(x+26,y+19,x+28,y-26,x,y-12);ctx.fill();
  ctx.fillStyle='#f6ae86';ctx.beginPath();ctx.ellipse(x-10,y-5,3,6,.4,0,Math.PI*2);ctx.fill();
  const bx=env.basket*cw+cw/2;ctx.fillStyle=flash>0?'#829e50':'#355b43';ctx.beginPath();ctx.roundRect(bx-26,416,52,26,6);ctx.fill();ctx.strokeStyle='#57754b';ctx.lineWidth=2;for(let i=-14;i<=14;i+=14){ctx.beginPath();ctx.moveTo(bx+i,420);ctx.lineTo(bx+i,439);ctx.stroke();}ctx.fillStyle='#214d38';ctx.beginPath();ctx.roundRect(bx-30,412,60,8,4);ctx.fill();
}
function tick(now){const dt=last?Math.min((now-last)/1000,.25):0;last=now;
  if(training){const until=performance.now()+9;while(agent.episodes<trainTarget&&performance.now()<until)agent.trainEpisode(trainingEnv);$('trainProgress').value=agent.episodes-trainStart;chart();if(agent.episodes>=trainTarget)stopTraining();}
  if(phase==='playing'&&!training){const active=Math.min(dt,remaining);remaining=Math.max(0,remaining-active);accumulator+=active;flash=Math.max(0,flash-dt);while(accumulator>=.2){accumulator-=.2;const action=mode==='ai'?agent.action(env.state):1+Number(held.has('right'))-Number(held.has('left'));const result=env.step(action);if(result.done){score+=result.reward;if(result.reward>0){caught++;flash=.3;}else missed++;$('feedback').textContent=result.reward>0?'Nice catch! +10 points. Keep it going.':'Missed this one. −10 points. Next apple, fresh chance.';env.reset();}}
    if(remaining<=0){phase='ended';held.clear();overlay('Round complete!',`${score} points · ${caught} caught · ${missed} missed · ${caught+missed?Math.round(caught/(caught+missed)*100):0}% catch rate`,'Play again','✓');}}
  sync();draw();requestAnimationFrame(tick);
}
$('start').onclick=start;$('pause').onclick=pause;$('restart').onclick=()=>{if(!training){ready();start();}};$('train').onclick=train;
for(const m of ['human','ai'])$(m+'Mode').onclick=()=>{if(training)return;mode=m;for(const n of ['human','ai']){$(n+'Mode').classList.toggle('selected',m===n);$(n+'Mode').setAttribute('aria-pressed',String(m===n));}ready();};
$('duration').onchange=()=>{duration=Number($('duration').value);ready();};
for(const dir of ['left','right']){const button=$(dir);button.onpointerdown=e=>{e.preventDefault();button.setPointerCapture(e.pointerId);held.add(dir);};button.onpointerup=button.onpointercancel=button.onlostpointercapture=()=>held.delete(dir);}
window.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;const k=e.key.toLowerCase();if(['arrowleft','arrowright','a','d','p','r',' '].includes(k)){e.preventDefault();if(['arrowleft','a'].includes(k))held.add('left');if(['arrowright','d'].includes(k))held.add('right');if(!e.repeat&&(k==='p'||k===' '))pause();if(!e.repeat&&k==='r'&&!training){ready();start();}}});
window.addEventListener('keyup',e=>{const k=e.key.toLowerCase();if(['arrowleft','a'].includes(k))held.delete('left');if(['arrowright','d'].includes(k))held.delete('right');});
window.addEventListener('blur',()=>{held.clear();if(phase==='playing')pause();});document.addEventListener('visibilitychange',()=>{if(document.hidden){held.clear();if(phase==='playing')pause();if(training)persist();}});
$('reset').onclick=()=>{if(training||!confirm('Reset the learned Q-table and training history?'))return;agent=new Agent();persist();chart();ready();$('trainNote').textContent='A fresh agent. Ready to learn from scratch.';};
$('export').onclick=()=>{const blob=new Blob([JSON.stringify(agent.data(),null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='catch-lab-q-model.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
if(agent.episodes)$('trainNote').textContent='Your saved agent is back. Ready when you are.';
chart();ready();requestAnimationFrame(tick);

