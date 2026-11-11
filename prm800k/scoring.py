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
def predict(model, requests, batch_size=32):
    device = next(model.parameters()).device
    model.eval()
    result = []
    for offset in range(0, len(requests), batch_size):
        batch = requests[offset:offset+batch_size]
        sequences = [encode(problem, steps, model.config.context)[0] for problem, steps in batch]
        tokens, lengths = batch_tokens(sequences, device)
        output = model(tokens)[torch.arange(len(batch), device=device), lengths-1]
        result.extend(output.softmax(-1).cpu().tolist())
    return result


def score_candidates(samples, prm, orm=None, batch_size=32):
    output = []
    for sample in samples:
        if not sample.get('steps'):
            raise ValueError('A candidate must contain at least one step')
        row = dict(sample)
        steps = sample['steps']
        probs = predict(prm, [(sample['problem'], steps[:i+1]) for i in range(len(steps))], batch_size)
        row['step_probabilities'] = probs
        row['prm_score'] = aggregate(probs)
        row['prm_minimum'] = aggregate(probs, reduction='minimum')
        row['prm_positive_only'] = aggregate(probs, neutral_positive=False)
        row['prm_positive_minimum'] = aggregate(probs, neutral_positive=False, reduction='minimum')
        if orm is not None:
            row['orm_score'] = predict(orm, [(sample['problem'], steps)], batch_size)[0][1]
        output.append(row)
    return output
