import torch
import re
import sys
sys.path.append('.')
from evaluation.parser import extract_answer, strip_string
from collections import defaultdict, Counter

def group_pred(preds, strip=True, use_symbol=False):
    """
    groups: {pred: [idx list]}
    """
    orginal_preds = preds
    if not use_symbol:
        if strip:
            preds = [strip_string(pred) for pred in preds]
        cnt = Counter(preds)
        majority = cnt.most_common(1)[0][0]
        groups = defaultdict(list)
        for idx, pred in enumerate(preds):
            groups[pred].append(idx)
        return groups, orginal_preds[groups[majority][0]]

    # groups = defaultdict(list)
    # for idx, pred in enumerate(preds):
    #     found_group = False
    #     if strip:
    #         pred = strip_string(pred)
    #     for group_pred in groups:
    #         try:
    #             if math_equal_timeout(pred, group_pred):
    #                 groups[group_pred].append(idx)
    #                 found_group = True
    #                 break
    #         except:
    #             continue
    #     if not found_group:
    #         groups[pred].append(idx)
    # # get the key of the longest group
    # majority = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)[0][0]
    # majority = orginal_preds[groups[majority][0]]
    return groups, majority

def reward_func(queries: list[str], prompts: list[str], labels: list[str])->torch.Tensor:
    """
    queries: list of strings, each string is a question and response
    # The input is 'samples', which contains a micro_rollout_batch_size number of items,
    # but here we expect n_samples_per_prompt.
    # n_samples_per_prompt needs to be divisible by micro_rollout_batch_size—ideally, they should be equal.
    """
    
    preds = []
    gt_reward_list = []
    # Iterate over each question-response pair
    for query, label in zip(queries, labels):
        extracted_answer = extract_answer(query, data_name='math')
        if label != "":
            gt_answer = extract_answer(label, data_name='math')
            reward = 1.0 if extracted_answer == gt_answer else 0.0
            gt_reward_list.append(reward)
        preds.append(extracted_answer)
        
    if not gt_reward_list:
        groups, majority_pred = group_pred(preds, strip=True, use_symbol=False)
        reward_list = [0.0]*len(preds)
        for idx in groups[majority_pred]:
            reward_list[idx] = 1.0
    else:
        reward_list = gt_reward_list
    
    return torch.tensor(reward_list)


if __name__ == '__main__':
    response = "\\boxed$123$"

    queries = ["What is the value of $\\boxed{123}$?", "What is the value of $\\boxed{456}$?", "What is the value of $\\boxed{123}$?", "The final answer is \\boxed{77}"]

    print(reward_func(queries, [], []))