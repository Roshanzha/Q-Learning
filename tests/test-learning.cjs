const assert=require('node:assert/strict');
const {Agent,Environment}=require('../docs/learning.js');
function rng(seed){return()=>{seed=(1664525*seed+1013904223)>>>0;return seed/4294967296;};}
const env=new Environment(rng(42));env.basket=env.apple=0;
for(let i=0;i<7;i++)assert.equal(env.step(0).reward,0);
assert.equal(env.step(0).reward,10);assert.equal(env.basket,0);
assert.throws(()=>env.step(1));env.reset();env.basket=8;env.apple=0;
for(let i=0;i<7;i++)env.step(2);assert.equal(env.step(2).reward,-10);assert.equal(env.basket,8);
const agent=new Agent(rng(7));agent.learn([0,0,7],1,10,[0,0,8],true);assert.equal(agent.values([0,0,7])[1],2);
for(let i=0;i<30000;i++)agent.trainEpisode(env);
function evaluate(random){const e=new Environment(rng(100)),r=rng(101);let wins=0;for(let i=0;i<2000;i++){e.reset();for(;;){const result=e.step(random?Math.floor(r()*3):agent.action(e.state));if(result.done){wins+=result.reward>0;break;}}}return wins/2000;}
const baseline=evaluate(true),trained=evaluate(false);assert.ok(trained>.9);assert.ok(trained>baseline+.6);
const copy=new Agent();copy.restore(JSON.parse(JSON.stringify(agent.data())));assert.deepEqual(copy.data(),agent.data());assert.throws(()=>copy.restore({version:4}));
console.log(`PASS: boundaries, rewards, terminal update, model round-trip. Random ${(baseline*100).toFixed(1)}%; trained ${(trained*100).toFixed(1)}%.`);

