import torch
from math_verify import math_metric, LatexExtractionConfig, ExprExtractionConfig

def reward_func(queries: list[str], prompts: list[str], labels: list[str])->torch.Tensor:
    """
    queries: list of strings, each string is a question and response
    # The input is 'samples', which contains a micro_rollout_batch_size number of items,
    # but here we expect n_samples_per_prompt.
    # n_samples_per_prompt needs to be divisible by micro_rollout_batch_size—ideally, they should be equal.
    """
    # Create the verification function
    verify_func = math_metric(
        # gold_extraction_target=(LatexExtractionConfig(), ExprExtractionConfig()),
        gold_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),  
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        aggregation_function=max,
        precision=6
    )

    # Iterate over each question-response pair
    equal_matrix = [[False]*len(queries) for _ in range(len(queries))]
    for i in range(len(queries)):
        for j in range(i, len(queries)):
            if i == j:
                equal_matrix[i][j] = True
                equal_matrix[j][i] = True
            else:
                is_true = False
                try:
                    # is_true = verify(preds[i], preds[j])
                    grade, extracted_answers = verify_func([queries[i]], [queries[j]])
                    is_true = grade == 1.0
                    if not is_true:
                        grade, extracted_answers = verify_func([queries[j]], [queries[i]])
                        is_true = grade == 1.0
                except:
                    # print(f"maj reward: Error verifying answers: {e}")
                    # print("="*20+"maj reward error"+"="*20)
                    # print(queries[i])
                    # print(queries[j])
                    is_true = False
                equal_matrix[i][j] = is_true
                equal_matrix[j][i] = is_true

    maj_count = [0]*len(queries)
    for i in range(len(queries)):
        maj_count[i] = sum(equal_matrix[i])
    # The majority index with the highest count
    majority_pred_idx = maj_count.index(max(maj_count))
    reward_list = [0.0]*len(queries)
    for idx in range(len(queries)):
        if equal_matrix[majority_pred_idx][idx]:
            reward_list[idx] = 1.0
    
    return torch.tensor(reward_list)


if __name__ == '__main__':
    response = "\\boxed$123$"

    queries = [
        'The three-digit integer $63\\underline{\\hphantom{0}}$ is a multiple of 3. What is the greatest possible difference between two of the possibilities for the units digit?$$\n\n## Step 1: Recall the rule for divisibility by 3\nA number is divisible by 3 if the sum of its digits is divisible by 3.\n\n## Step 2: Apply the rule to the given number\nTo satisfy the divisibility rule, the sum of the digits of $63\\underline{\\hphantom{0}}$ must be a multiple of 3. We know that $6+3 = 9$, which is already a multiple of 3. Therefore, the units digit can be any number that keeps the sum of the digits a multiple of 3.\n\n## Step 3: List possible values for the units digit\nThe sum of the digits is 9, and we can add 0, 3, 6, or 9 to 9 without changing the divisibility by 3.\n\n## Step 4: Calculate the possible values of the units digit\nPossible values are 0, 3, 6, and 9.\n\n## Step 5: Determine the greatest possible difference between two of the possibilities\nThe greatest possible difference is $9-0 = 9$.\n\nThe final answer is: $\\boxed{9}$',
        'The three-digit integer $63\\underline{\\hphantom{0}}$ is a multiple of 3. What is the greatest possible difference between two of the possibilities for the units digit? $9$ is the greatest possible difference.\nas $\\boxed{\\text{(E)}}$.\n## Step 1: Recall the divisibility rule for 3\nA number is divisible by 3 if the sum of its digits is divisible by 3. We are looking for a number that is a multiple of 3.\n\n## Step 2: Apply the divisibility rule for 3 to the given number\nThe given number is 630. The sum of its digits is 6 + 3 + 0 = 9, which is divisible by 3.\n\n## Step 3: Determine the possible values for the units digit\nTo maintain divisibility by 3, the units digit must be chosen such that the total sum of digits remains divisible by 3. Since 630 is divisible by 3, the units digit can range from 0 to 9 while keeping the divisibility by 3.\n\n## Step 4: List all possible values for the units digit\nPossible values for the units digit are 0, 3, 6, 9.\n\n## Step 5: Calculate the differences between the maximum and minimum possible values for the units digit\nThe maximum value is 9 and the minimum value is 0. The greatest possible difference is 9 - 0 = 9.\n\nThe final answer is: $\\boxed{9}$'
    ]
    
    print(reward_func(queries, [], []))