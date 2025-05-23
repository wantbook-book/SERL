import torch
import re
import sys
sys.path.append('.')
from evaluation.parser import extract_answer
def extract_answer_from_boxed(response: str)->str:
    ans = response.split("boxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    pred = a
    return pred



def reward_func(queries: list[str], prompts: list[str], labels: list[str])->torch.Tensor:
    """
    queries: list of strings, each string is a question and response
    """
    reward_list = []
    
    # Iterate over each question-response pair
    for query, label in zip(queries, labels):
        extracted_answer = extract_answer(query, data_name='math')
        gt_answer = extract_answer(label, data_name='math')

        reward = 1.0 if extracted_answer == gt_answer else 0.0

        reward_list.append(reward)
    
    return torch.tensor(reward_list)


if __name__ == '__main__':
    response = "\\boxed$123$"
    prompts = ["What is the value of $\\boxed{123}$?", "What is the value of $\\boxed{456}$?"]
    labels = ["123", "456"]
    print(extract_answer_from_boxed(response))
    print(reward_func([], prompts, labels))