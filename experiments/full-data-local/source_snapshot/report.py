"""Portable experiment plots and a self-contained interactive HTML report."""
from pathlib import Path
import html
import json
import os
from itertools import cycle

from .io import write_json


def plots(evaluation, training, output):
    os.environ.setdefault('MPLCONFIGDIR', str(Path('.cache/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'sans-serif', 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': '#f7f9fc', 'axes.facecolor': '#f7f9fc', 'axes.titleweight': 'bold'})
    colors = ['#086bcb', '#e5783d', '#8190a2', '#159a87', '#8e69c8', '#d54a67', '#c59b18']
    fig, ax = plt.subplots(figsize=(8, 4.3), layout='constrained')
    for (method, metric), color in zip(evaluation['methods'].items(), cycle(colors)):
        means = metric.get('exact_mean', metric['mean'])
        ax.plot(evaluation['ns'], means, 'o-', label=method, color=color, linewidth=2, markersize=4)
    ax.set(xlabel='Candidates per problem (N)', ylabel='Fraction of problems solved', ylim=(0, 1), title='Best-of-N selection')
    ax.grid(alpha=0.15)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(output/'best_of_n.svg')
    fig.savefig(output/'best_of_n.png', dpi=160)
    plt.close(fig)
    if training:
        fig, ax = plt.subplots(figsize=(8, 3.5), layout='constrained')
        for name, stats in training.items():
            history = stats['history']
            ax.plot([r['epoch'] for r in history], [r['train_loss'] for r in history], 'o-', label=f'{name.upper()} train')
            ax.plot([r['epoch'] for r in history], [r['validation_loss'] for r in history], 'o--', label=f'{name.upper()} validation')
        ax.set(xlabel='Epoch', ylabel='Cross-entropy', title='Training diagnostics')
        ax.legend()
        fig.savefig(output/'training.svg')
        plt.close(fig)


def render_report(output, evaluation, samples=(), training=None, environment=None, title='PRM800K · Local experiment'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    training = training or {}
    environment = environment or {}
    write_json(output/'evaluation.json', evaluation)
    plots(evaluation, training, output)
    data = json.dumps({'samples': list(samples), 'evaluation': evaluation, 'training': training}, ensure_ascii=False).replace('<', '\\u003c')
    table = '\n'.join(f'| {m} | ' + ' | '.join(f'{x:.2%}' for x in v.get('exact_mean', v['mean'])) + ' |'
                      for m, v in evaluation['methods'].items())
    analysis = f'''# {title}

{evaluation['problems']} problems; {evaluation['trials']} subsampling trials; seed {evaluation['seed']}.

| Method | {' | '.join(f'N={n}' for n in evaluation['ns'])} |
|---|{'|'.join('---:' for _ in evaluation['ns'])}|
{table}

Exact expectation is reported for score-based selection when present. Majority,
random and oracle use seeded sampling. Missing/answerless candidates count as
failed generation slots. Standard deviation describes subsampling variability,
not uncertainty over the entire population of MATH problems.

The bundled local experiment uses small models initialized from scratch and
an active-learning-biased subset of the released data. Its PRM/ORM comparison
is a functional demonstration, not evidence for the paper's effect size.
Short context truncation, class imbalance and the small training set limit
quality. The report shows every retained test candidate; no example is chosen
for model success. Validation problems and test problems are excluded from training.

See `prm_training.json`, `orm_training.json`, `environment.json`, `inputs.json`,
`scored.jsonl.gz` and checkpoint files for the corresponding run when available.
'''
    if not training:
        start = analysis.index('The bundled local experiment')
        analysis = analysis[:start] + 'This run evaluates supplied candidate scores. It does not train reward models.\nCheck the adjacent input and source manifests for dataset coverage, generator\nrevision, candidate counts, missing answers and applicable limitations.\n'
    else:
        analysis += '\n## Classification diagnostics\n\n'
        for name, stats in training.items():
            last = stats['history'][-1]
            analysis += f"{name.upper()}: accuracy {last['validation_accuracy']:.2%}; majority-class baseline {last.get('majority_class_baseline', 0):.2%}; balanced accuracy {last.get('balanced_accuracy', 0):.2%}; macro-F1 {last.get('macro_f1', 0):.3f}. Training context truncated for {stats['truncated_training_examples']}/{stats['training_examples']} examples.\n\n"
    (output/'ANALYSIS.md').write_text(analysis, encoding='utf-8')
    page = TEMPLATE.replace('__TITLE__', html.escape(title)).replace('__PROBLEMS__', str(evaluation['problems'])).replace('__DEVICE__', html.escape(str(environment.get('device', 'CPU evaluation')))).replace('__DATA__', data)
    if not training:
        page = page.replace('<img class="chart" style="max-width:760px" src="training.svg" alt="Training and validation loss" onerror="this.hidden=true">', '')
        page = page.replace("Small local models demonstrate the process-supervision workflow. These results do not reproduce GPT-4 training or the paper's 78.2% result. See the separate official-score replay.", "This report evaluates supplied scores. It does not retrain the original models. Read the experiment analysis for source coverage and interpretation.")
    (output/'report.html').write_text(page, encoding='utf-8')
    return output/'report.html'


TEMPLATE = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title><style>
:root{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#1c2d43;background:#f2f5fa;font-size:15px}*{box-sizing:border-box}body{margin:0}aside{position:fixed;inset:0 auto 0 0;width:220px;background:#10283f;color:#becddd;padding:36px 24px}.brand{font-size:24px;color:white;font-weight:750;letter-spacing:-1px}.badge{font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#74d9cc;margin:8px 0 48px}.nav{padding:13px 14px;border-radius:8px;margin:8px 0;display:block;text-decoration:none;color:inherit}.nav:hover,.nav:first-of-type{background:#25445d;color:white}.foot{position:absolute;bottom:24px;font-size:12px;line-height:1.8;max-width:170px}main{margin-left:220px;max-width:1420px;padding:34px 42px}header{display:flex;justify-content:space-between;align-items:center}.eyebrow{color:#118a81;font-size:11px;letter-spacing:2px;text-transform:uppercase;font-weight:750}h1{font-size:31px;letter-spacing:-.8px;margin:12px 0}h2{font-size:19px;margin:0 0 16px}.sub{color:#697b8f;line-height:1.65}.chip{background:#e2f3ed;color:#217d63;border-radius:30px;padding:9px 15px;font-size:12px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:26px 0}.stat,.card{background:white;border:1px solid #dce4ec;border-radius:12px;padding:23px;box-shadow:0 3px 12px #16385104}.stat b{display:block;font-size:29px;letter-spacing:-1px;margin:8px 0}.label{font-size:12px;color:#657a90}.grid{display:grid;grid-template-columns:1.35fr 1fr;gap:22px}.chart{width:100%;border-radius:6px}.method{display:flex;justify-content:space-between;border-bottom:1px solid #e8edf2;padding:12px 0;font-size:13px}.notice{background:#fff7e9;border-left:3px solid #e7b658;padding:14px;margin-top:16px;line-height:1.6;font-size:12px;color:#796034}section{scroll-margin-top:24px;margin:24px 0}select,button{border:1px solid #cbd6e2;padding:11px;border-radius:7px;background:white;color:#243b53;font:inherit}select{width:100%;margin:8px 0 20px}button{cursor:pointer;margin:4px 6px 8px 0;font-size:12px}button.active{background:#086bcb;color:white;border-color:#086bcb}.problem{line-height:1.6;background:#f3f7fc;border-radius:8px;padding:18px;white-space:pre-wrap;font-size:14px}.step{display:grid;grid-template-columns:32px 1fr 92px;gap:12px;align-items:start;padding:15px 0;border-bottom:1px solid #edf1f5;font-size:13px;line-height:1.6}.num{color:#8a9bac;font-size:12px}.prob{font-size:12px;text-align:right;color:#197963}.bar{height:5px;background:#edf0f5;border-radius:3px;margin-top:8px;overflow:hidden}.fill{height:100%;background:#38a992}.answer{font-size:13px;padding:12px 0;color:#64798e}.small{font-size:12px;color:#7e8fa1;line-height:1.7}code{background:#eef2f7;padding:2px 5px;border-radius:4px}.status{font-size:12px;font-weight:600}.ok{color:#148164}.bad{color:#c66243}@media(max-width:900px){aside{display:none}main{margin:0;padding:20px}.grid{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}}
</style><aside><div class="brand">PRM800K<span style="color:#63d7c4">.</span></div><div class="badge">Process supervision lab</div><a class="nav" href="#overview">01 / Experiment overview</a><a class="nav" href="#inspect">02 / Step inspector</a><a class="nav" href="#training">03 / Training diagnostics</a><div class="foot">Let's Verify Step by Step<br>Local, reproducible, inspectable.<br><span style="color:#6c91aa">Official data · Small local models</span></div></aside>
<main><header><div><div class="eyebrow">Reasoning, one step at a time</div><h1>__TITLE__</h1><div class="sub">Train verifiers. Inspect each step. Compare best-of-N selection.</div></div><div class="chip">__DEVICE__ · Recorded run</div></header>
<section id="overview"><div class="stats"><div class="stat"><span class="label">Evaluated problems</span><b>__PROBLEMS__</b><span class="small">See input split provenance</span></div><div class="stat"><span class="label">PRM best-of-N</span><b id="prmMetric">—</b><span class="small">Local checkpoint / released scores</span></div><div class="stat"><span class="label">ORM best-of-N</span><b id="ormMetric">—</b><span class="small">Outcome supervision baseline</span></div><div class="stat"><span class="label">Evaluation seed</span><b id="seed">42</b><span class="small">Sampling without replacement</span></div></div>
<div class="grid"><div class="card"><h2>Search performance</h2><img class="chart" src="best_of_n.svg" alt="Best-of-N evaluation chart"><div class="small">Score-based methods use exact expectation; other methods use seeded trials.</div></div><div class="card"><h2>Selection methods</h2><div id="methods"></div><div class="notice">Small local models demonstrate the process-supervision workflow. These results do not reproduce GPT-4 training or the paper's 78.2% result. See the separate official-score replay.</div></div></div></section>
<section id="inspect" class="card"><div class="eyebrow">From solution to step-level evidence</div><h2 style="margin-top:10px">Inspect a held-out problem</h2><label class="small" for="problemSelect">Choose a problem from the retained test set</label><select id="problemSelect"></select><div id="problem" class="problem"></div><div id="candidates" style="margin-top:16px"></div><div id="answer" class="answer"></div><div class="small">Each probability is P(positive) + P(neutral). Solution ranking uses the sum of log probabilities.</div><div id="steps"></div></section>
<section id="training" class="card"><h2>Training diagnostics & reproducibility</h2><div id="trainingMeta" class="sub"></div><img class="chart" style="max-width:760px" src="training.svg" alt="Training and validation loss" onerror="this.hidden=true"><div class="notice">Raw metrics, model checkpoints, source hashes, data splits and runtime details are retained alongside this report. A positive step score is a model prediction, not a proof of correctness.</div></section><footer class="small">PRM800K / Let's Verify Step by Step · Offline report, no external requests.</footer></main>
<script id="payload" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('payload').textContent);const E=data.evaluation;const pct=x=>(100*x).toFixed(1)+'%';const el=id=>document.getElementById(id);const val=m=>{let v=E.methods[m];return v?(v.exact_mean||v.mean).at(-1):null};['prm','orm'].forEach(m=>{if(val(m)!==null)el(m+'Metric').textContent=pct(val(m))});el('seed').textContent=E.seed;Object.entries(E.methods).forEach(([m,v])=>{const row=document.createElement('div');row.className='method';const name=document.createElement('span');name.textContent=m;const value=document.createElement('b');value.textContent=pct((v.exact_mean||v.mean).at(-1));row.append(name,value);el('methods').append(row)});
const groups=new Map;data.samples.forEach(s=>{if(!groups.has(s.problem))groups.set(s.problem,[]);groups.get(s.problem).push(s)});const problems=[...groups.keys()];problems.forEach((p,i)=>{const o=document.createElement('option');o.value=i;o.textContent=String(i+1).padStart(2,'0')+' / '+p.slice(0,120);el('problemSelect').append(o)});let current=[];function showCandidate(index){const s=current[index];[...el('candidates').children].forEach((b,i)=>b.classList.toggle('active',i===index));el('answer').textContent='Final answer: '+(s.answer??'[no answer]')+'  ·  '+(s.is_correct?'Correct':'Incorrect')+'  ·  PRM log score: '+Number(s.prm_score).toFixed(3);el('answer').className='answer '+(s.is_correct?'ok':'bad');el('steps').replaceChildren();(s.steps||[]).forEach((t,i)=>{const row=document.createElement('div');row.className='step';const n=document.createElement('div');n.className='num';n.textContent=String(i+1).padStart(2,'0');const text=document.createElement('div');text.textContent=t;const prob=document.createElement('div');prob.className='prob';const p=s.step_probabilities?.[i];const score=p?p[1]+p[2]:null;prob.textContent=score===null?'—':pct(score);const bar=document.createElement('div');bar.className='bar';const fill=document.createElement('div');fill.className='fill';fill.style.width=(score===null?0:score*100)+'%';if(score<.5)fill.style.background='#d57654';bar.append(fill);prob.append(bar);row.append(n,text,prob);el('steps').append(row)})}function showProblem(){const p=problems[Number(el('problemSelect').value)||0];if(!p){el('inspect').hidden=true;return}current=groups.get(p);el('problem').textContent=p;el('candidates').replaceChildren();current.forEach((s,i)=>{const b=document.createElement('button');b.textContent='Candidate '+(i+1);b.onclick=()=>showCandidate(i);el('candidates').append(b)});showCandidate(0)}el('problemSelect').onchange=showProblem;showProblem();el('trainingMeta').textContent=Object.entries(data.training).map(([k,v])=>k.toUpperCase()+': '+v.parameters.toLocaleString()+' parameters · '+v.training_examples+' training examples · '+v.epochs+' epochs · '+v.device+' · '+v.seconds.toFixed(1)+'s').join(' / ')||'Evaluation of supplied scores. No model training in this run.';
</script></html>'''
