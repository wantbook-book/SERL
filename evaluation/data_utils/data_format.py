import re
import json
from collections import defaultdict, Counter
import random
def parse_to_train_format(data_path, output_path):
    with open(data_path, "r") as f:
        data = f.readlines()
    instructions = []
    for i, line in enumerate(data):
        item = json.loads(line)
        question = item.get("problem", "")
        if question == "":
            question = item.get("question", "")
        if question == "":
            question = item.get("instruction", "")

        assert question != "", "question is empty"
        instructions.append({
            "idx": i,
            "problem": question,
            "answer": "",
            "solution": "",
        })
    with open(output_path, "w") as f:
        for instruction in instructions:
            f.write(json.dumps(instruction) + "\n")
    
    exit()


def convert_to_sft_format(input_file, output_file):
    data_list = []
    with open(input_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            data_list.append({
                "instruction": entry["problem"],
                "input": "",
                "output": entry["solution"],
            })
    # Save the preprocessed data to a new JSON file
    with open(output_file, 'w') as f:
        for entry in data_list:
            f.write(json.dumps(entry) + '\n')
    exit()

def convert_bon_eval_results_to_filtered_data(file_path, output_path, origin_data_file):
    data_list = []
    idx2data = {}
    with open(origin_data_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            idx2data[entry['idx']] = entry

    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Extract the final answer from the solution string
            data_list.append(idx2data[entry['idx']])

    with open(output_path, 'w') as f:
        for entry in data_list:
            f.write(json.dumps(entry) + '\n')
    exit()

def append_answer(file_path, output_path, origin_data_file):
    idx2data = {}
    with open(origin_data_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            idx2data[entry['idx']] = entry

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Extract the final answer from the solution string
            entry['answer'] = idx2data[entry['idx']]['answer']
            data.append(entry)

    # Save the preprocessed data to a new JSON file
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    exit()

def append_solution(file_path, output_path, origin_data_file):
    idx2data = {}
    with open(origin_data_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            idx2data[entry['idx']] = entry

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Extract the final answer from the solution string
            entry['solution'] = idx2data[entry['idx']]['solution']
            data.append(entry)

    # Save the preprocessed data to a new JSON file
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    exit()

def append_idx(file_path, output_path):
    data_list = []
    idx = 0
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            entry['idx'] = idx
            data_list.append(entry)
            idx += 1
    with open(output_path, 'w') as f:
        for entry in data_list:
            f.write(json.dumps(entry) + '\n')
    exit()

def append_codes(file_path, output_path, origin_data_file):
    idx2data = {}
    with open(origin_data_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            idx2data[entry['idx']] = entry

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Extract the final answer from the solution string
            entry['code'] = idx2data[entry['idx']]['code']
            data.append(entry)

    # Save the preprocessed data to a new JSON file
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    exit()


if __name__ == "__main__":
    # file_path = "/xxx/SEO/openrlhf/dataset/math/numina_math_cot_train.jsonl"
    # output_path = "/xxx/SEO/openrlhf/dataset/math/numina_math_cot_train_with_idx.jsonl"
    # append_idx(
    #     file_path=file_path,
    #     output_path=output_path
    # )

    # file_path = '/xxx/Math-Verify/outputs/xxx/model/Llama-3.2-3B-Instruct/math_eval_bon_32/math/train_with_idx_filter_0_2_0_8.jsonl'
    # output_path = '/xxx/Math-Verify/outputs/xxx/model/Llama-3.2-3B-Instruct/math_eval_bon_32/math/train_with_idx_filter_0_2_0_8_data.jsonl'
    # origin_data_file = '/xxx/MCGEP_back/dataset/math/train_with_idx.jsonl'
    # convert_bon_eval_results_to_filtered_data(
    #     file_path=file_path,
    #     output_path=output_path,
    #     origin_data_file=origin_data_file
    # )

    # file_path = "/xxx/Math-Verify/outputs/xxx/model/Llama-3.2-3B-Instruct/math_eval_bon_32/math/train_with_idx_pure_-1_seed0_t1.0_s0_e-1_maj_eval.jsonl"
    # origin_data_file = '/xxx/MCGEP_back/dataset/math/train_with_idx.jsonl'
    # output_path = "/xxx/Math-Verify/outputs/xxx/model/Llama-3.2-3B-Instruct/math_eval_bon_32/math/train_with_idx_pure_-1_seed0_t1.0_s0_e-1_maj_eval_with_answer.jsonl"
    # append_answer(
    #     file_path=file_path,
    #     output_path=output_path,
    #     origin_data_file=origin_data_file
    # )

    # file_path = "/xxx/Math-Verify/outputs/xxx/model/Llama-3.2-3B-Instruct/math_eval_bon_32/math/train_with_idx_pure_-1_seed0_t1.0_s0_e-1_maj_eval_with_answer.jsonl"
    # output_path = "/xxx/Math-Verify/outputs/xxx/model/Llama-3.2-3B-Instruct/math_eval_bon_32/math/train_with_idx_pure_-1_seed0_t1.0_s0_e-1_maj_eval_with_answer_codes.jsonl"
    # origin_data_file = "/xxx/Math-Verify/outputs/xxx/model/Llama-3.2-3B-Instruct/math_eval_bon_32/math/train_with_idx_pure_-1_seed0_t1.0_s0_e-1.jsonl"
    # append_codes(
    #     file_path=file_path,
    #     output_path=output_path,
    #     origin_data_file=origin_data_file
    # )

    # file_path = "/xxx/SEO/Math-Verify/outputs/xxx/model/Llama-3.2-3B-Instruct/math_eval_bon_32/math/train_with_idx_pure_-1_seed0_t1.0_s0_e-1.jsonl"
    # output_path = "/xxx/SEO/Math-Verify/outputs/xxx/model/Llama-3.2-3B-Instruct/math_eval_bon_32/math/train_with_idx_pure_-1_seed0_t1.0_s0_e-1.jsonl"
    # origin_data_file = "/xxx/MCGEP_back/dataset/math/train_with_idx.jsonl"
    # append_solution(
    #     file_path=file_path,
    #     output_path=output_path,
    #     origin_data_file=origin_data_file
    # )


    # file_path = "/xxx/MCGEP_back/dataset/math/train_with_idx.jsonl"
    # output_file = "/xxx/MCGEP_back/dataset/math/train_sft.jsonl"
    # input_file = "/xxx/SEO/openrlhf/dataset/math/train_with_idx.jsonl"
    # output_file = "/xxx/SEO/openrlhf/dataset/math/train_with_idx_sft.jsonl"
    # convert_to_sft_format(
    #     input_file=input_file,
    #     output_file=output_file
    # )


    # # input_file = "/xxx/SEO/Math-Verify/outputs/xxx/orlhf_checkpoints/checkpoint/llama3-3b-0208seed_gen7500_maj_filtered_0208_random_bon_maj_bs16/global_step200_hf/math_eval_bon_32/math/seo_0208_500seed_gen7500_iter1_pure_-1_seed0_t1.0_s0_e-1_maj_eval_filtered_0208.jsonl"
    # # input_file = "/xxx/SEO/self-instruct/data/llama3_2_3b_generations/self_math_rewards_0208_500seed_gen7500_iter1/2025-04-26_04-13-59/machine_generated_instructions.jsonl"
    # input_file = "/xxx/SEO/self-instruct/data/qwen_2_5_7B_generations/qwen_seo_0208_500seed_gen7500_iter0/2025-04-27_15-45-21/machine_generated_instructions.jsonl"
    # # output_file = "/xxx/SEO/self-instruct/data/llama3_2_3b_generations/self_math_rewards_0208_500seed_gen7500_iter1/2025-04-26_04-13-59/math/self_math_rewards_0208_500seed_gen7500_iter1.jsonl"
    # output_file = "/xxx/SEO/self-instruct/data/qwen_2_5_7B_generations/qwen_seo_0208_500seed_gen7500_iter0/2025-04-27_15-45-21/math/seo_0208_500seed_gen7500_iter0_online.jsonl"
    # parse_to_train_format(
    #     data_path=input_file,
    #     output_path=output_file
    # )


