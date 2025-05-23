from typing import Dict, List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
from tqdm import tqdm, trange
from vllm import LLM, SamplingParams
import os


class ArmoRMPipeline:
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return {"score": score}

def apply_chat_template(
    prompt: str,
    response: str,
):
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]

def predict_scores_with_orm(generate_result_file: Path):
    # Create Reward Model Pipeline
    rm = ArmoRMPipeline("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)
    with open(generate_result_file, "r") as f:
        generate_results = f.readlines()
    generate_results = [json.loads(result) for result in generate_results]
    for i in trange(len(generate_results)):
        result = generate_results[i]
        prompt = result["question"]
        responses = result["code"]
        scores = []
        for response in responses:
            pred = rm(apply_chat_template(prompt, response))
            print(pred)
            score = pred["score"]
            scores.append([score])
        result["pred_score"] = scores
    with open(generate_result_file.parent / "orm_with_pred_score.jsonl", "w") as f:
        for result in generate_results:
            f.write(json.dumps(result) + "\n")

def calculate_step_scores(input_for_prm, model, tokenizer, device):
    km_token_id1 = 12902
    km_token_id2 = 1107
    good_token_id = 648
    bad_token_id = 387
    candidate_token_ids = [good_token_id, bad_token_id]
    inputs = tokenizer.encode_plus(
        input_for_prm,
        return_tensors="pt",
        truncation=True,
        add_special_tokens=False,
    )
    input_id = inputs["input_ids"].to(device)
    # sampling_params = SamplingParams(
    #     max_tokens=0,
    # )        
    with torch.no_grad():
        # outputs = model.generate(input_ids=input_id, sampling_params=sampling_params)
        logits = model(input_id).logits[:, :, candidate_token_ids].to(device)
        # logits = logits[:, :, candidate_token_ids]
        scores = logits.softmax(dim=-1)[:, :, 0]
        # Bug fix: change to step_tag_id or step_tag_id2
        step_scores = scores[(input_id == km_token_id1) | (input_id == km_token_id2)].to(device)
    return step_scores

def predict_scores_with_prm(generate_result_file: Path):
    good_token = '+'
    bad_token = '-'
    km_token = 'ки'
    sep_token = '\n'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/xxx/math-shepherd-mistral-7b-prm"
    # available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    # pipeline_parallel_size = 1
    # model = LLM(
    #     model=model_path,
    #     tensor_parallel_size=len(available_gpus) // pipeline_parallel_size,
    #     pipeline_parallel_size=pipeline_parallel_size,
    #     trust_remote_code=True,
    # )
    model = AutoModelForCausalLM.from_pretrained(model_path).eval()
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    with open(generate_result_file, "r") as f:
        generate_results = f.readlines()
    generate_results = [json.loads(result) for result in generate_results]
    for i in trange(len(generate_results)):
        result = generate_results[i]
        prompt = result["question"]
        responses = result["code"]
        scores = []
        min_scores = []
        avg_scores = []
        complete_scores = []
        final_scores = []
        for response in responses:
            pred = calculate_step_scores(prompt+'\n'+response.replace(km_token, '').replace(sep_token, km_token)+km_token, model, tokenizer=tokenizer, device=device)
            complete_scores.append(pred.tolist())
            min_score = pred.min().item()
            min_scores.append(min_score)
            avg_score = pred.mean().item()
            avg_scores.append(avg_score)
            final_score = pred[-1].item()
            final_scores.append(final_score)
        print('min_score:', min_scores)
        result["pred_score"] = scores
        result["min_score"] = min_scores
        result["avg_score"] = avg_scores
        result["final_score"] = final_scores
        result["complete_score"] = complete_scores
    with open(generate_result_file.parent / "prm_with_pred_score.jsonl", "w") as f:
        for result in generate_results:
            f.write(json.dumps(result) + "\n")

if __name__ == '__main__':
    generate_result_file = Path("/xxx/code/Qwen2.5-Math/evaluation/outputs/xxx/orlhf_checkpoints/checkpoint/llama3-1b-porm_grpo_n16_8samples/math_eval/math/test_direct_-1_seed0_t0.0_s0_e-1.jsonl")
    predict_scores_with_prm(generate_result_file)