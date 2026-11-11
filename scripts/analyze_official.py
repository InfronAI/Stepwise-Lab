"""Additional full-release checks: majority, four PRM reductions and dataset coverage."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time

from prm800k.evaluation import indexed_groups, answer_of
from prm800k.grading.math_normalize import normalize_answer
from prm800k.io import read_jsonl, write_json


def main():
    output=Path('experiments/official-replay')
    output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    majority=[]
    counts=[]
    tied=0
    for problem,samples in indexed_groups('.cache/scored.sqlite'):
        answers=defaultdict(list)
        for sample in samples:
            if answer_of(sample) is not None:
                answers[normalize_answer(answer_of(sample))].append(sample['is_correct'])
        if answers:
            top=max(map(len,answers.values()))
            tied_answers=[v for v in answers.values() if len(v)==top]
            value=sum(sum(v)/len(v) for v in tied_answers)/len(tied_answers)
            tied+=len(tied_answers)>1
        else:
            value=0.0
        majority.append(value)
        counts.append(len(samples))
    write_json(output/'majority_at_1860.json',{'exact_mean':sum(majority)/len(majority),
        'problems':len(majority),'tied_winning_answer_groups':tied,'normalization':'upstream math_normalize.normalize_answer',
        'policy':'full population; ties uniformly averaged, no correctness-based tie breaking'})
    best=defaultdict(dict)
    maximum_difference=0.0
    omitted_zero_classes=0
    max_probability_mass_error=0.0
    rows=0
    for row in read_jsonl('.cache/data/scored.jsonl'):
        rows+=1
        if answer_of(row) is None:
            continue
        omitted_zero_classes += sum(len(p)<3 for p in row['rating_probs'])
        max_probability_mass_error = max(max_probability_mass_error, *(abs(sum(p.values())-1.0) for p in row['rating_probs']))
        positive=[p.get('1',0.0) for p in row['rating_probs']]
        acceptable=[p.get('1',0.0)+p.get('0',0.0) for p in row['rating_probs']]
        values={'product_neutral_positive':math.prod(acceptable),'minimum_neutral_positive':min(acceptable),
                'product_neutral_negative':math.prod(positive),'minimum_neutral_negative':min(positive)}
        maximum_difference=max(maximum_difference,abs(values['product_neutral_positive']-row['prm_score']))
        record=best[row['problem']]
        for key,value in values.items():
            current=record.get(key)
            if current is None or value>current[0]:
                record[key]=[value,int(row['is_correct']),1]
            elif value==current[0]:
                current[1]+=int(row['is_correct'])
                current[2]+=1
    reductions={key:sum(record[key][1]/record[key][2] for record in best.values())/len(best) for key in next(iter(best.values()))}
    write_json(output/'aggregation_ablation.json',{'best_of':1860,'problems':len(best),'exact_means':reductions,
         'max_recomputed_vs_released_prm_score_difference':maximum_difference,
         'steps_with_omitted_zero_classes':omitted_zero_classes,'max_probability_mass_error':max_probability_mass_error})
    write_json(output/'coverage.json',{'rows':rows,'problems':len(counts),'min_candidates':min(counts),
         'max_candidates':max(counts),'stored_candidates':sum(counts),'attempted_generations':1860*len(counts),
         'missing_generations':1860*len(counts)-sum(counts),'candidate_count_histogram':dict(sorted(Counter(counts).items())),
         'seconds_for_additional_analysis':time.perf_counter()-start})
    print('Majority:',sum(majority)/len(majority),'PRM reductions:',reductions,flush=True)
    if (output/'evaluation.json').exists():
        data=json.loads((output/'evaluation.json').read_text())
        lines=['\n## Full-release validation\n',f'All {len(counts)} problems and {rows:,} stored candidates were processed. The evaluation population retains {1860*len(counts)-sum(counts):,} missing attempts.\n',
            '| PRM scoring policy | Best-of-1860 |','|---|---:|']
        lines.extend(f'| {key} | {value:.2%} |' for key,value in reductions.items())
        lines.extend(['',f'Normalized majority voting at the full population: **{sum(majority)/len(majority):.2%}**.',
            'The paper reports majority voting at 69.6%; this implementation gives about 69.2% with the released answer strings and upstream normalization. The original voting implementation was not released, so this baseline is not an exact reproduction.',
            'This is score replay of the original released models, not training reproduction.',
            'The source file passed byte-length and Azure Content-MD5 verification; its SHA-256 is in `source.json`.'])
        for dimension in ['subject','level','generator_difficulty']:
            lines.extend(['',f'### {dimension}','', '| Group | Problems | PRM N=1860 | ORM N=1860 |','|---|---:|---:|---:|'])
            for name,record in data['breakdowns'][dimension].items():
                lines.append(f'| {name} | {record["problems"]} | {record["methods"]["prm"][-1]:.2%} | {record["methods"]["orm"][-1]:.2%} |')
        with (output/'ANALYSIS.md').open('a') as stream:
            stream.write('\n'.join(lines)+'\n')

if __name__=='__main__':
    main()
