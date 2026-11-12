"""Step probabilities and the four Appendix F aggregation policies."""
import math
import torch

from .model import batch_tokens, encode


def aggregate(probabilities, neutral_positive=True, reduction='product'):
    if reduction not in ('product', 'minimum'):
        raise ValueError('reduction must be product or minimum')
    if not probabilities:
        return -math.inf
    scores = []
    for negative, neutral, positive in probabilities:
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in (negative, neutral, positive)) or not math.isclose(negative+neutral+positive, 1.0, abs_tol=1e-5):
            raise ValueError('expected normalized [negative, neutral, positive] probabilities')
        scores.append(positive + (neutral if neutral_positive else 0))
    logs = [math.log(max(s, 1e-30)) for s in scores]
    return sum(logs) if reduction == 'product' else min(logs)


@torch.inference_mode()
def predict(model, requests, batch_size=32, details=None):
    device = next(model.parameters()).device
    model.eval()
    result = []
    for offset in range(0, len(requests), batch_size):
        batch = requests[offset:offset+batch_size]
        encoded = [encode(problem, steps, model.config.context) for problem, steps in batch]
        sequences = [item[0] for item in encoded]
        tokens, lengths = batch_tokens(sequences, device)
        output = model(tokens)[torch.arange(len(batch), device=device), lengths-1]
        probabilities = output.softmax(-1).cpu().tolist()
        result.extend(probabilities)
        if details is not None:
            logits = output.cpu().tolist()
            for index, ((problem, steps), (ids, start, truncated), values, probs) in enumerate(zip(batch, encoded, logits, probabilities)):
                details.append({'request_index': offset+index, 'problem': problem, 'steps': steps,
                    'token_ids': ids, 'solution_start': start, 'truncated_solution': truncated,
                    'question_bytes': len(problem.encode()), 'retained_question_bytes': start-2,
                    'logits': values, 'probabilities': probs})
    return result


def score_candidates(samples, prm, orm=None, batch_size=32, trace_dir=None):
    from .tracing import TraceStore
    trace = TraceStore(trace_dir, 'candidate_inference') if trace_dir is not None else None
    output = []
    for candidate_index, sample in enumerate(samples):
        if not sample.get('steps'):
            raise ValueError('A candidate must contain at least one step')
        row = dict(sample)
        steps = sample['steps']
        details = [] if trace else None
        probs = predict(prm, [(sample['problem'], steps[:i+1]) for i in range(len(steps))], batch_size, details)
        row['step_probabilities'] = probs
        row['prm_score'] = aggregate(probs)
        row['prm_minimum'] = aggregate(probs, reduction='minimum')
        row['prm_positive_only'] = aggregate(probs, neutral_positive=False)
        row['prm_positive_minimum'] = aggregate(probs, neutral_positive=False, reduction='minimum')
        if orm is not None:
            orm_details = [] if trace else None
            row['orm_score'] = predict(orm, [(sample['problem'], steps)], batch_size, orm_details)[0][1]
        if trace:
            for index, detail in enumerate(details):
                detail['step_index'] = index
                detail['log_acceptable_probability'] = math.log(max(sum(probs[index][1:]), 1e-30))
                detail['cumulative_prm_score'] = aggregate(probs[:index+1])
            trace.json(f'candidates/{candidate_index:07d}.json.gz', {'candidate': sample,
                'prm_steps': details, 'orm': orm_details if orm is not None else None,
                'reductions': {key: row[key] for key in ('prm_score', 'prm_minimum', 'prm_positive_only', 'prm_positive_minimum')}})
        output.append(row)
    if trace:
        trace.finish(candidates=len(output), step_predictions=sum(len(s['steps']) for s in output),
                     scoring='sum(log(P(neutral)+P(positive))); no future steps in a prefix input')
    return output
