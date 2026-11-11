"""Optional fresh generation with an openly available local causal LM."""
import re
import time

from .grading.grader import grade_answer
from .model import device_for, seed_all

DEFAULT_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'


def extract_answer(text):
    # Balanced braces, including nested LaTeX fractions. Prefer the last box.
    matches = list(re.finditer(r'\\boxed\s*\{', text))
    if matches:
        start = matches[-1].end()
        depth = 1
        for i in range(start, len(text)):
            depth += (text[i] == '{') - (text[i] == '}')
            if depth == 0:
                return text[start:i].strip()
    match = re.search(r'(?:#\s*Answer|Final answer\s*:)\s*(.+)', text, re.I | re.S)
    return match.group(1).strip() if match else None


def generate(problems, *, model_name=DEFAULT_MODEL, revision=None, device='auto',
             samples_per_problem=4, max_new_tokens=256, seed=42, local_files_only=False, adapter=None, temperature=1.0, top_p=1.0, greedy=False):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError('temperature must be positive and top_p in (0,1]')
    if samples_per_problem < 1 or max_new_tokens < 1:
        raise ValueError('positive sample count and token limit required')
    if greedy and samples_per_problem != 1:
        raise ValueError('Greedy generation requires --samples 1; repeated greedy candidates would be duplicates')
    seed_all(seed)
    device = device_for(device)
    start = time.perf_counter()
    options = dict(revision=revision, local_files_only=local_files_only, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **options)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32,
                                                attn_implementation='eager', **options).to(device).eval()
    resolved = getattr(model.config, '_commit_hash', None)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False).to(device).eval()
    rows = []
    for problem_index, problem in enumerate(problems):
        messages = [{'role': 'system', 'content': 'Solve the math problem step by step. Put each reasoning step on a separate line. Finish with the final answer in \\boxed{}.'},
                    {'role': 'user', 'content': problem['problem']}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        for sample_index in range(samples_per_problem):
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=not greedy,
                                     temperature=None if greedy else temperature, top_p=None if greedy else top_p, top_k=None if greedy else 0,
                                     pad_token_id=tokenizer.eos_token_id)
            continuation = ids[0, inputs['input_ids'].shape[1]:]
            text = tokenizer.decode(continuation, skip_special_tokens=True)
            answer = extract_answer(text)
            truth = problem.get('ground_truth_answer', problem.get('answer'))
            if truth is None:
                raise ValueError('Generation evaluation needs a ground truth answer')
            rows.append({'id': f'generated:{problem_index}:{sample_index}', 'problem': problem['problem'],
                         'steps': [s.strip() for s in text.splitlines() if s.strip()] or ['[empty completion]'],
                         'answer': answer, 'ground_truth_answer': truth,
                         'is_correct': bool(answer is not None and grade_answer(answer, truth)),
                         'generated_tokens': len(continuation), 'hit_token_limit': len(continuation) == max_new_tokens,
                         'source': model_name, 'model_revision': resolved})
            print(f'Generated problem {problem_index+1}, candidate {sample_index+1}/{samples_per_problem}: answer={answer!r}', flush=True)
    return rows, {'model': model_name, 'revision': resolved, 'device': str(device),
                  'seconds': time.perf_counter()-start, 'seed': seed, 'max_new_tokens': max_new_tokens,
                  'samples_per_problem': samples_per_problem, 'adapter': str(adapter) if adapter else None, 'temperature': temperature, 'top_p': top_p, 'greedy': greedy, 'dtype': 'float32'}
