import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluation.evaluate import evaluate
from evaluation.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from evaluation.parser import *
from evaluation.trajectory import *
from evaluation.data_loader import load_data
from evaluation.python_executor import PythonExecutor
# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()
class Args:
    def __init__(self):
        pass

def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True

def compute_accuracy(python_executor, questions: list[str], solutions: list[str], gt_solutions: list[str], all_idxs: list[int]=None, output_file=None, all_mcts_solutions_steps_and_rewards=None, return_scores=False)->float:
    """
    for math dataset
    """
    executor = python_executor
    args = Args()
    args.adapt_few_shot = False
    args.data_name = 'math'
    args.prompt_type = 'pure'
    args.num_shots = 0

    prompt_type = 'pure'
    data_name = 'math'
    # executor = PythonExecutor(get_answer_from_stdout=True)
    # stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "\n\n\n"]
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    end_prompts = []
    questions = [(i, question) for i, question in enumerate(questions)]
    samples = []
    for i, question in questions:
        example = {
            'question': question,
            'solution': gt_solutions[i],
        }
        if all_idxs:
            idx = all_idxs[i]
        else:
            idx = i
        
        example['idx'] = idx
        example['question'] = parse_question(example, data_name)
        if example['question'] == '':
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example['gt_ans'] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        sample = {
            'idx': idx,
            'question': example['question'],
            'gt_cot': gt_cot,
            'gt': gt_ans,
            'prompt': full_prompt,
        }
        samples.append(sample)


    input_prompts = [
        sample['prompt'] for sample in samples
    ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []
    for (i, query), output in zip(remain_prompts, solutions):
        output = output.strip()
        query += output
        end_prompts.append((i, query))
    
    codes = []
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        # Logically, stop words themselves will not be generated
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    results = [
        run_execute(executor, code, prompt_type, data_name) for code in codes
    ]

    all_samples = []
    for i, sample in enumerate(samples):
        code = [codes[i]]
        result = [results[i]]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports})
        if all_mcts_solutions_steps_and_rewards is not None:
            sample.update({"steps": [all_mcts_solutions_steps_and_rewards[i][0]], "rewards": [all_mcts_solutions_steps_and_rewards[i][1]]})
        all_samples.append(sample)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=prompt_type,
        execute=True
    )

    if output_file is not None:
        save_jsonl(all_samples, output_file)
        with open(
            output_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
        ) as f:
            json.dump(result_json, f, indent=4)
    
    scores = []
    if return_scores:
        scores = [sample["score"] for sample in all_samples]
        result_json["scores"] = scores
    # ['acc']
    return result_json


if __name__ == "__main__":
    questions = [
        "Chandra has four bowls.  Each one is a different color (red, blue, yellow, green).  She also has exactly one glass the same color as each bowl.  If she chooses a bowl and a glass from the cupboard, how many pairings are possible?  One such pairing is a blue bowl and a yellow glass.",
        "Chandra has four bowls.  Each one is a different color (red, blue, yellow, green).  She also has exactly one glass the same color as each bowl.  If she chooses a bowl and a glass from the cupboard, how many pairings are possible?  One such pairing is a blue bowl and a yellow glass.",
        "Chandra has four bowls.  Each one is a different color (red, blue, yellow, green).  She also has exactly one glass the same color as each bowl.  If she chooses a bowl and a glass from the cupboard, how many pairings are possible?  One such pairing is a blue bowl and a yellow glass."
    ]
    solutions = [
        "There are four different bowls and four different glasses that Chandra can pick. Since her choices are mutually exclusive, there are $4 \\times 4 = \\boxed{16}$ possible pairings.",
        "There are four different bowls and four different glasses that Chandra can pick. Since her choices are mutually exclusive, there are $4 \\times 4 = \\boxed{16}$ possible pairings.",
        "There are $8\\times 3\\times 4=\\boxed{96}$ ways to make three decisions if there are 8, 3, and 4 options available for the decisions."
    ]
    gt_solutions = [
        "There are four different bowls and four different glasses that Chandra can pick. Since her choices are mutually exclusive, there are $4 \\times 4 = \\boxed{16}$ possible pairings.",
        "There are four different bowls and four different glasses that Chandra can pick. Since her choices are mutually exclusive, there are $4 \\times 4 = \\boxed{16}$ possible pairings.",
        "There are four different bowls and four different glasses that Chandra can pick. Since her choices are mutually exclusive, there are $4 \\times 4 = \\boxed{16}$ possible pairings."
    ]
    compute_accuracy(questions, solutions, gt_solutions)