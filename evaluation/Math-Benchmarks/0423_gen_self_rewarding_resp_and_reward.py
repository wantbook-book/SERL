from vllm import LLM, SamplingParams
import argparse
from data_loader import load_data
import random
import os
from datetime import datetime
from model_utils import load_hf_lm_and_tokenizer, generate_completions
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import time
import re
# import debugpy 
# debugpy.listen(("localhost", 5678))
# debugpy.wait_for_client()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--reward_n_sampling", default=1, type=int)
    parser.add_argument("--n_sampling_per_prompt", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--self_reward_prompt_file", type=str, default="self_reward_prompt.txt")

    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args



def setup(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if args.use_vllm:
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            trust_remote_code=True,
        )
        # tokenizer = None
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path
        )
    else:
        assert False, "should set use_vllm=True"

    main(llm, tokenizer, args)
    

def read_txt(file_path):
    assert str(file_path).endswith(".txt")
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data

def extract_score(input_str: str)->float:
    # reward_regex_str = "Score: {reward}".format(reward = "([0-9\.]+)")
    reward_regex_str = r"Score: ([0-9\.]+)"
    result = re.search(rf"{reward_regex_str}", input_str)

    if result is None or len(result.groups()) == 0:
        return None

    if not result.group(1).isnumeric():
        return None

    return float(result.group(1))

def avg(lst):
    if len(lst) == 0:
        return 0
    return sum(lst) / len(lst)

def main(llm, tokenizer, args):
    # Load input file data; it already contains problems and generated code answers
    data_list = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            item['code'] = item['code'][:args.n_sampling_per_prompt]
            data_list.append(item)

    # load self reward prompt
    self_reward_prompt = read_txt(args.self_reward_prompt_file)

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    # if args.prompt_type in ["cot"]:
    #     stop_words.append("\n\nQuestion:")
    # if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
    #     stop_words.extend(["\n\n---", "```output"])
    # elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
    #     stop_words.extend(["Instruction", "Response"])
    # elif "jiuzhang" in args.prompt_type:
    #     stop_words.append("\n\n## Question")
    # elif "numina" in args.prompt_type:
    #     stop_words.append("\n### Problem")
    # elif "pure" in args.prompt_type:
    stop_words.append("\n\n\n")

    # Concatenate all prompts
    input_prompts = []
    for data in data_list:
        for code in data['code']:
            question = data.get('question', data.get('problem', ''))
            assert question != "", "question is empty"
            input_prompts.append(self_reward_prompt.format(prompt=question, response=code))
    
    # generate outputs
    if args.use_vllm:
        outputs = llm.generate(
            input_prompts,
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call,
                # max_tokens=128,
                n=args.reward_n_sampling,
                stop=stop_words, #+['## Step 2:'],
                stop_token_ids=(
                    [151645, 151643]
                    if "qwen2" in args.model_name_or_path.lower()
                    else None
                ),
            ),
        )
        outputs = [o.text for output in outputs for o in output.outputs]
    else:
        assert False, "should set use_vllm=True"
        # outputs = generate_completions(
        #     model=llm,
        #     tokenizer=tokenizer,
        #     prompts=input_prompts,
        #     max_new_tokens=args.max_tokens_per_call,
        #     batch_size=16,
        #     stop_id_sequences=stop_words,
        # )
    assert len(outputs) == len(input_prompts)*args.reward_n_sampling, f"len(outputs) = {len(outputs)}, len(input_prompts) = {len(input_prompts)}"

    # extract rewards
    n_sampling_rewards = []
    for output in outputs:
        reward = extract_score(output)
        if reward is None:
            print("No reward found in output")
            n_sampling_rewards.append(0)
        else:
            if reward > 5.0:
                reward = 5.0
            elif reward < 0.0:
                reward = 0.0
            
            n_sampling_rewards.append(reward)
    
    n_samples_per_prompt = len(data_list[0]['code'])
    rewards = []
    for i in range(len(data_list)):
        for j in range(n_samples_per_prompt):
            code_idx = i*n_samples_per_prompt+j
            rewards.append(avg(n_sampling_rewards[code_idx*args.reward_n_sampling:(code_idx+1)*args.reward_n_sampling]))

    # save outputs
    with open(args.output_file, "w", encoding="utf-8") as f:
        for i, data in enumerate(data_list):
            data['self_reward_rewards'] = rewards[i*n_samples_per_prompt:(i+1)*n_samples_per_prompt]
            data['self_reward_resps'] = outputs[i*n_samples_per_prompt:(i+1)*n_samples_per_prompt]
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)

