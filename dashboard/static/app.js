// BTCUSDT 5m ML bot dashboard. Vanilla JS + ECharts.
const $ = (s) => document.querySelector(s);
const charts = {};
const IND_LABEL = { rsi:"RSI(14)", macd:"MACD", atr:"ATR%", bb:"Bollinger pos", vwap:"VWAP off", htf:"HTF trend" };

function mk(id) { if (!charts[id]) charts[id] = echarts.init($('#'+id)); return charts[id]; }

function ts(v){ return new Date(v).toISOString().replace('T',' ').slice(0,16); }

// ---- navigation ----
function openView(v){
  document.querySelectorAll('nav button').forEach(x=>x.classList.remove('active'));
  const btn=document.querySelector('nav button[data-view="'+v+'"]'); if(btn) btn.classList.add('active');
  document.querySelectorAll('main section').forEach(s=>s.classList.add('hidden'));
  $('#view-'+v).classList.remove('hidden');
  if(v==='chart') renderChart();
  if(v==='backtest') renderBacktest();
  if(v==='grid') renderGrid();
  if(v==='wf') renderWF();
  if(v==='live') renderLive();
}
document.querySelectorAll('nav button').forEach(b=>{
  b.onclick=()=>{ history.replaceState(null,'','#'+b.dataset.view); openView(b.dataset.view); };
});
const _views=['chart','backtest','grid','wf','live'];
const _init=(''+location.hash).slice(1);
openView(_views.includes(_init)?_init:'chart');

// ---- shared fetch ----
async function get(u){ const r=await fetch(u); return r.json(); }

// ---- CHART ----
let candleData={}, indData={}, predData={};
async function renderChart(){
  const limit=$('#chart-limit').value;
  const [ohlcv, ind, preds] = await Promise.all([
    get('/api/ohlcv?limit='+limit), get('/api/indicators?limit='+limit), get('/api/predictions?limit='+limit)]);
  candleData=ohlcv; indData=ind; predData=preds;
  const ch=mk('candle');
  const cats=ohlcv.map(o=>ts(o.ts));
  const k=[];
  ohlcv.forEach(o=>k.push([o.open,o.close,o.low,o.high]));
  const series=[{name:'Candle', type:'candlestick', data:k}];
  // signals
  const longs=[], shorts=[]; const maxPre=(1+ohlcv.length)*0;
  ohlcv.forEach((o,i)=>{
    const p=preds.find(x=>x.ts===o.ts);
    if(p){ if(p.position>0) longs.push({xAxis:i, yAxis:o.low*0.9995, symbol:'arrowUp', symbolSize:10, color:'#3fb950'});
           if(p.position<0) shorts.push({xAxis:i, yAxis:o.high*1.0005, symbol:'arrowDown', symbolSize:10, color:'#f85149'}); }
  });
  series.push({name:'Long', type:'scatter', data:longs, z:5, symbol:'triangle', symbolRotate:180},
              {name:'Short', type:'scatter', data:shorts, z:5, symbol:'triangle'});
  // price-line indicators (SMA/EMA approximation via rolling handled server-side? use vwap/bb here)
  if($('[data-ind="vwap"]').checked){
    const vw=indData.map((r,i)=>[i, r.vwap_off? (ohlcv[i].close*(1+r.vwap_off)/10):null]);
    series.push({name:'VWAP(S/10)', type:'line', data:vw, showSymbol:false, lineStyle:{width:1}});
  }
  ch.setOption({
    animation:false, tooltip:{trigger:'axis', axisPointer:{type:'cross'}},
    legend:{textStyle:{color:'#c9d1d9'}},
    xAxis:{data:cats, type:'category', axisLabel:{color:'#c9d1d9'}},
    yAxis:{scale:true, axisLabel:{color:'#c9d1d9'}},
    dataZoom:[{type:'inside'},{type:'slider', height:18, bottom:0}],
    series
  },true);
}

// ---- BACKTEST ----
async function renderBacktest(){
  const bt=await get('/api/backtest?limit=6000');
  const m=bt.pooled;
  $('#bt-metrics').innerHTML =
    ['sharpe','sortino','total_return','max_drawdown','win_rate','num_trades','folds_positive']
      .map(k=>`<span class="metric">${k.replace(/_/g,' ')}<b class="${(typeof m[k+'_mean']==='number'&&m[k+'_mean']<0)?'neg':'pos'}">${typeof m[k+'_mean']==='number'?m[k+'_mean'].toFixed(3):m[k+'_mean']}</b></span>`).join('');
  const cats=bt.equity.map(e=>ts(e.ts));
  const ch=mk('equity');
  ch.setOption({
    animation:false, tooltip:{trigger:'axis'},
    legend:{data:['Strategy','Buy&Hold'],textStyle:{color:'#c9d1d9'}},
    xAxis:{type:'category',data:cats,axisLabel:{color:'#c9d1d9'}},
    yAxis:{scale:true,axisLabel:{color:'#c9d1d9'}},
    dataZoom:[{type:'inside'}],
    series:[
      {name:'Strategy',type:'line',showSymbol:false,data:bt.equity.map(e=>e.equity),lineStyle:{color:'#58a6ff'}},
      {name:'Buy&Hold',type:'line',showSymbol:false,data:bt.buyhold.map(e=>e.equity),lineStyle:{color:'#8957e5',width:1}}
    ]
  },true);
}

// ---- GRID ----
async function renderGrid(){
  const g=await get('/api/grid'); const bc=g.best_config;
  $('#grid-best').innerHTML=`<pre style="font-size:12px;color:#c9d1d9">${JSON.stringify(bc,null,2)}</pre>`+
    `<p class="status">ranked ${g.num_configs_ranked} configs · top score ${g.top_score?.toFixed? g.top_score.toFixed(3):g.top_score}`+
    ` · bottom ${g.bottom_score?.toFixed? g.bottom_score.toFixed(3):g.bottom_score}</p>`;
  const wf=await get('/api/walkforward');
  const ch=mk('grid-scores');
  ch.setOption({
    animation:false, tooltip:{trigger:'axis'},
    legend:{data:['Sharpe','MaxDD'],textStyle:{color:'#c9d1d9'}},
    xAxis:{type:'category',data:wf.map((f,i)=>'fold'+(i+1)),axisLabel:{color:'#c9d1d9'}},
    yAxis:[{type:'value',axisLabel:{color:'#c9d1d9'}},{type:'value',axisLabel:{color:'#c9d1d9'}}],
    series:[
      {name:'Sharpe',type:'bar',data:wf.map(f=>f.metrics.sharpe)},
      {name:'MaxDD',type:'line',yAxisIndex:1,showSymbol:false,data:wf.map(f=>f.metrics.max_drawdown)}
    ]
  },true);
}

// ---- WALK-FORWARD ----
async function renderWF(){
  const folds=await get('/api/walkforward');
  const rows=folds.map(f=>Object.assign({fold:'fold '+(folds.indexOf(f)+1),
    window:f.ts&&f.ts.length?ts(f.ts[0]).slice(0,10)+' → '+ts(f.ts[f.ts.length-1]).slice(0,10):''},f.metrics));
  if(!rows.length) return;
  const keys=['fold','window','sharpe','sortino','total_return','max_drawdown','win_rate','num_trades'];
  let html='<table><tr>'+keys.map(k=>'<th>'+k.replace(/_/g,' ')+'</th>').join('')+'</tr>';
  rows.forEach(r=>{
    html+='<tr>'+keys.map(k=>'<td class="'+((k==='sharpe'||k==='total_return')&&r[k]<0?'neg':(r[k]>0?'pos':''))+'">'+
      (typeof r[k]==='number'?r[k].toFixed(3):r[k])+'</td>').join('')+'</tr>';
  });
  html+='</table>'; $('#wf-table').innerHTML=html;

  drawWfPerFold(folds);
}

function drawWfPerFold(folds){
  const grid=$('#wf-equity-grid'); if(!grid) return;
  const withEq=folds.filter(f=>f&&f.equity&&f.buyhold);
  if(grid.childElementCount!==withEq.length){
    grid.innerHTML='';
    withEq.forEach((f,i)=>{
      const tile=document.createElement('div'); tile.className='wf-fold';
      tile.innerHTML=`<h4>fold ${folds.indexOf(f)+1} · ${ts(f.ts[0]).slice(0,10)}→${ts(f.ts[f.ts.length-1]).slice(0,10)}</h4><div class="wf-fold-chart"></div>`;
      grid.appendChild(tile);
    });
  }
  withEq.forEach((f,i)=>{
    const el=grid.children[i].querySelector('.wf-fold-chart');
    let ch=echarts.getInstanceByDom(el);
    if(!ch) ch=echarts.init(el);
    ch.setOption({
      animation:false, tooltip:{trigger:'axis'},
      legend:{data:['Strategy','Buy&Hold'],textStyle:{color:'#8b949e',fontSize:10}},
      xAxis:{type:'category',data:f.ts.map(t=>ts(t)),axisLabel:{color:'#6e7681',fontSize:9},axisLine:{lineStyle:{color:'#30363d'}}},
      yAxis:{scale:true,axisLabel:{color:'#6e7681',fontSize:9},splitLine:{lineStyle:{color:'#21262d'}}},
      dataZoom:[{type:'inside'}],
      series:[
        {name:'Strategy',type:'line',showSymbol:false,data:f.equity,lineStyle:{color:'#58a6ff'},areaStyle:{opacity:0.10}},
        {name:'Buy&Hold',type:'line',showSymbol:false,data:f.buyhold,lineStyle:{color:'#8957e5',width:1}}
      ]
    },true);
  });
}

// ---- LIVE ----
async function renderLive(){
  const d=await get('/api/live'); if(!d.live){ $('#live-box').innerHTML='<p class="status">'+d.msg+'</p>'; return; }
  const ohlcv=await get('/api/ohlcv?limit=120');
  const cats=ohlcv.map(o=>ts(o.ts));
  mk('live-candle').setOption({
    animation:false, tooltip:{trigger:'axis'}, xAxis:{type:'category',data:cats},
    yAxis:{scale:true}, dataZoom:[{type:'inside'}],
    series:[{type:'candlestick',data:ohlcv.map(o=>[o.open,o.close,o.low,o.high])}]
  },true);
  const c=ohlcv[ohlcv.length-1];
  $('#live-box').innerHTML=`
    last close <b>$${c.close.toLocaleString()}</b> · model <b>${d.model}</b><br>
    score <b class="${d.score>0?'pos':'neg'}">${d.score}</b> · position
    <b class="${d.position>0?'pos':(d.position<0?'neg':'')}">${d.position>0?'LONG':(d.position<0?'SHORT':'FLAT')}</b>
    · margin ${d.margin} · uptime ${d.uptime_s}s<br>
    model trained ${d.model_ts? new Date(d.model_ts*1000).toLocaleString():'n/a'}
    · rows ${d.train_n||0}
    ${d.retraining? ' · <b class="pos">RETRAINING…</b>':''}
    ${d.last_retrain_err? ' · <span class="neg">retrain err: '+d.last_retrain_err+'</span>':''}`;
  const ind=d.indicators||{};
  mk('live-ind').setOption({
    animation:false, tooltip:{trigger:'axis'},
    xAxis:{type:'category',data:Object.keys(ind)},
    yAxis:{type:'value'}, series:[{type:'bar',data:Object.values(ind)}]
  },true);
}

// ---- indicator checkbox re-render + live polling ----
document.querySelectorAll('[data-ind]').forEach(c=>c.onchange=()=>{ if(!$('#view-chart').classList.contains('hidden')) renderChart(); });
$('#chart-limit').onchange=()=>renderChart();
setInterval(async ()=>{ try{ const h=await get('/api/health'); $('#health').textContent='artifact='+h.artifact; }catch(e){} }, 20000);
setInterval(async ()=>{
  try{ const d=await get('/api/live'); const el=$('#live');
    if(d.live){ el.textContent='LIVE: on · up '+d.uptime_s+'s'; el.className='live on'; }
    else { el.textContent='LIVE: waiting'; el.className='live off'; }
    if(!$('#view-live').classList.contains('hidden')) renderLive();
  }catch(e){}
},5000);
