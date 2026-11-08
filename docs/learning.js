(function(root){
  'use strict';
  class Environment {
    constructor(rng=Math.random){this.rng=rng;this.reset();}
    reset(){this.basket=Math.floor(this.rng()*9);this.apple=Math.floor(this.rng()*9);this.row=0;this.done=false;return this.state;}
    get state(){return [this.basket,this.apple,this.row];}
    step(action){
      if(this.done)throw Error('Reset before stepping a terminal episode');
      if(![0,1,2].includes(action))throw Error('Invalid action');
      this.basket=Math.max(0,Math.min(8,this.basket+action-1));this.row++;
      this.done=this.row===8;
      return {state:this.state,reward:this.done?(this.basket===this.apple?10:-10):0,done:this.done};
    }
  }
  class Agent {
    constructor(rng=Math.random){this.rng=rng;this.q={};this.episodes=0;this.epsilon=1;this.alpha=.2;this.gamma=.95;this.history=[];this.batchWins=0;}
    values(s){return this.q[s.join(',')]||[0,0,0];}
    action(s,explore=false){if(explore&&this.rng()<this.epsilon)return Math.floor(this.rng()*3);const q=this.values(s),best=Math.max(...q),ties=[0,1,2].filter(a=>q[a]===best);return ties[Math.floor(this.rng()*ties.length)];}
    learn(s,a,r,next,done){const key=s.join(','),q=this.q[key]||(this.q[key]=[0,0,0]);const target=done?r:r+this.gamma*Math.max(...this.values(next));q[a]+=this.alpha*(target-q[a]);}
    trainEpisode(env){let s=env.reset();for(;;){const a=this.action(s,true),result=env.step(a);this.learn(s,a,result.reward,result.state,result.done);s=result.state;if(result.done){this.batchWins+=result.reward>0?1:0;break;}}
      this.episodes++;this.epsilon=Math.max(.03,this.epsilon*.9998);
      if(this.episodes%500===0){this.history.push({episode:this.episodes,rate:this.batchWins/500});this.batchWins=0;}
    }
    data(){return {version:1,q:this.q,episodes:this.episodes,epsilon:this.epsilon,alpha:this.alpha,gamma:this.gamma,history:this.history,batchWins:this.batchWins};}
    restore(data){
      if(data.version!==1||!Number.isInteger(data.episodes)||data.episodes<0||!data.q||!Array.isArray(data.history))throw Error('Invalid model');
      for(const [k,v]of Object.entries(data.q)){if(!/^[0-8],[0-8],[0-7]$/.test(k)||!Array.isArray(v)||v.length!==3||!v.every(Number.isFinite))throw Error('Invalid Q-table');}
      if(!Number.isFinite(data.epsilon)||data.epsilon<.03||data.epsilon>1||!Number.isInteger(data.batchWins)||data.batchWins<0||data.batchWins>500||!data.history.every(p=>Number.isFinite(p.episode)&&Number.isFinite(p.rate)&&p.rate>=0&&p.rate<=1))throw Error('Invalid training data');
      this.q=data.q;this.episodes=data.episodes;this.epsilon=data.epsilon;this.history=data.history;this.batchWins=data.batchWins;
    }
  }
  root.CatchLearning={Environment,Agent};
  if(typeof module!=='undefined')module.exports={Environment,Agent};
})(typeof globalThis!=='undefined'?globalThis:this);
