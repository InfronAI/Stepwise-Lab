"""Memory-bounded LoRA adaptation of the optional local generator.

SFT keeps only answer-correct candidates and masks the problem/prompt. Math
adaptation accepts a user-provided text corpus. The private MathMix corpus is
not included or imitated.
"""
from pathlib import Path
import time

from .generation import DEFAULT_MODEL
from .io import write_json
from .model import device_for, seed_all


def finetune(rows, output, *, mode='sft', model_name=DEFAULT_MODEL, revision=None,
             device='auto', seed=42, epochs=1, context=256, max_examples=32):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    if mode not in ('sft','math') or epochs < 1 or context < 16 or max_examples < 1:
        raise ValueError('valid mode, positive epochs/examples and context >=16 required')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory for a new adapter')
    output.mkdir(parents=True, exist_ok=True)
    device = device_for(device)
    seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=False)
    examples = []
    for row in rows:
        if mode == 'sft':
            # Recheck correctness rather than trusting a supplied boolean.
            from .grading.grader import grade_answer
            if not row.get('answer') or not row.get('ground_truth_answer') or not grade_answer(row['answer'],row['ground_truth_answer']):
                continue
            messages = [{'role':'system','content':'Solve the math problem step by step. Put each step on a separate line and finish with a boxed answer.'},
                        {'role':'user','content':row['problem']}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            solution = '\n'.join(row['steps']) + tokenizer.eos_token
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            solution_ids = tokenizer.encode(solution, add_special_tokens=False)
            ids = (prompt_ids + solution_ids)[:context]
            targets = ([-100]*len(prompt_ids) + solution_ids)[:context]
        else:
            ids = tokenizer.encode(row['text'], add_special_tokens=False)[:context]
            targets = ids.copy()
        if len(ids) < 2 or all(x == -100 for x in targets[1:]):
            continue
        examples.append((ids, targets))
        if len(examples) == max_examples:
            break
    if not examples:
        raise ValueError('No usable examples; provide correct solutions or text, and enough context for solution tokens')
    base = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, trust_remote_code=False,
                                               torch_dtype=torch.float32, attn_implementation='eager')
    resolved = getattr(base.config,'_commit_hash',None)
    model = get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                                         lora_dropout=0.0, target_modules=['q_proj','v_proj'])).to(device)
    model.config.use_cache = False
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    started = time.perf_counter()
    losses = []
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for ids, targets in examples:
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=torch.tensor([ids],device=device), labels=torch.tensor([targets],device=device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            optimizer.step()
            total += loss.item()
        losses.append(total/len(examples))
        print(f'{mode} adapter epoch {epoch+1}: loss={losses[-1]:.4f}',flush=True)
    model.save_pretrained(output)
    metadata = {'mode':mode,'model':model_name,'revision':resolved,'device':str(device),'seed':seed,
                'epochs':epochs,'context':context,'examples':len(examples),'loss':losses,
                'seconds':time.perf_counter()-started,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                'scope':'LoRA local substitute; not the original GPT-4 full-parameter adaptation'}
    write_json(output/'training.json',metadata)
    return metadata
