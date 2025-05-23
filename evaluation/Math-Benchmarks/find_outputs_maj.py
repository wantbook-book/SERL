import argparse
import pandas as pd
from typing import Any
from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig, parse
# from math_verify.grader import verify
import sympy
from pathlib import Path
from typing import Iterable, Union
import json
import os
def parse_args():
    parser = argparse.ArgumentParser(description='Extract and evaluate answers using sympy')
    parser.add_argument('--input_jsonl', type=str, required=True, help='Path to input josnl file containing model outputs')
    parser.add_argument('--output_jsonl', type=str, required=True, help='Path to output jsonl file for extracted answers')
    parser.add_argument('--gold_is_latex', action='store_true', help='Use basic latex normalization', default=True)
    return parser.parse_args()


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception as e:
                print("Error in loading:", line)
                exit()


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


def serialize_sympy_object(obj: Any) -> str:
    """Convert sympy object to string representation."""
    if obj is None:
        return ""
    try:
        if isinstance(obj, (list, tuple)):
            return ", ".join(str(x) if x is not None else "" for x in obj)
        return str(obj)
    except Exception as e:
        return f"Error: {str(e)}"

def compare_answers(extracted: Any, gold: Any) -> bool:
    """Compare extracted answer with gold answer."""
    if extracted is None or gold is None:
        return False
    try:
        # Handle lists/tuples of expressions
        if isinstance(extracted, (list, tuple)) and isinstance(gold, (list, tuple)):
            if len(extracted) != len(gold):
                return False
            return all(sympy.simplify(a - b) == 0 for a, b in zip(extracted, gold))
        
        # Handle single expressions
        return sympy.simplify(extracted - gold) == 0
    except Exception:
        # If comparison fails (e.g. different types), return False
        return False

def process_answers(input_list: list[dict], gold_is_latex: bool) -> Union[list[dict], dict]:
    """Process each answer through the sympy extraction workflow and compare with gold using math_verify."""
    results = []
    
    
    correct_count = 0
    total_count = 0
    
    # Create the verification function
    verify_func = math_metric(
        # gold_extraction_target=(LatexExtractionConfig() if gold_is_latex else ExprExtractionConfig(),),
        gold_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        aggregation_function=max,
        precision=6
    )
    output_list = []
    # extraction_target = (ExprExtractionConfig(), LatexExtractionConfig())
    for i, item in enumerate(input_list):
        print('='*20, i, '='*20)
        # extracted_answers = None
        # gold_answers = None
        grade = 0
        codes = item['code']
        maj_count = [0]*len(codes)
        maj_index = 0
        try:
            # Use the verification function
            # extracted_answers = []
            # for code in codes:
            #     # extracted = parse(code, extraction_config=extraction_target)
            #     # extracted_answers.append(serialize_sympy_object(extracted[0]))
            #     extracted_answers.append(parse(code))

            # item['preds'] = extracted_answers

            # majority voting
            correct_matrix = [[False]*len(codes) for _ in range(len(codes))]
            preds = [None]*len(codes)
            for i in range(len(codes)):
                for j in range(i, len(codes)):
                    if i == j and i != len(codes)-1:
                        correct_matrix[i][j] = True
                    else:
                        is_true = False
                        try:
                            # is_true = verify(preds[i], preds[j])
                            # grade, _ = verify_func(extracted_answers[i], extracted_answers[j])
                            grade, extracted_answers = verify_func([codes[i]], [codes[j]])
                            is_true = grade == 1.0
                            if extracted_answers is None:
                                if preds[i] is None:
                                    preds[i] = ['']
                            else:
                                if preds[i] is None:
                                    preds[i] = extracted_answers[0]
                            if not is_true:
                                grade, extracted_answers = verify_func([codes[j]], [codes[i]])
                                is_true = grade == 1.0
                                if extracted_answers is None:
                                    if preds[i] is None:
                                        preds[i] = ['']
                                else:
                                    if preds[i] is None:
                                        preds[i] = extracted_answers[1]
                        except:
                            # print(f"maj reward: Error verifying answers: {e}")
                            is_true = False
                        correct_matrix[i][j] = is_true
                        correct_matrix[j][i] = is_true
            for i in range(len(codes)):
                maj_count[i] = sum(correct_matrix[i])
            maj_index = maj_count.index(max(maj_count))

            preds_group_idx = [-1]*len(codes)
            # Assign answer IDs starting from 1 and incrementing consecutively
            for row_i, correct_row in enumerate(correct_matrix):
                for col_i, is_correct in enumerate(correct_row):
                    if is_correct and preds_group_idx[col_i] == -1:
                        if preds_group_idx[row_i] == -1:
                            preds_group_idx[col_i] = row_i
                        else:
                            preds_group_idx[col_i] = preds_group_idx[row_i]


            # grade, extracted_answers = verify_func([item['gt_cot']], [item['code'][maj_index]])
            
            # if extracted_answers is None:
            #     extracted_answers = None
            #     gold_answers = None
            # else:
            #     gold_answers = extracted_answers[0]
            #     extracted_answers = extracted_answers[1]

            # total_count += 1
            # if grade == 1:
            #     correct_count += 1
            # if maj_count[maj_index] <= l_maj_c or maj_count[maj_index] >= r_maj_c:
            #     continue
            question = item.get('question', '')
            if question == '':
                question = item.get('problem', '')
            assert question != '', 'question and problem are both empty'
            new_item = {
                "idx": item['idx'],
                "problem": question,
                "answer": item['answer'],
                # "solution": item['solution'],
                "maj_num": maj_count[maj_index],
                "total": len(codes),
                "code": codes,
                "preds": preds,
                "preds_group_idx": preds_group_idx,
            }
            output_list.append(new_item)
            
            # item['maj_extracted_answer'] = [extracted_answers]
            # item['extracted_gold'] = [gold_answers]
            # item['maj_is_correct'] = grade == 1
            # item['maj_count'] = maj_count[maj_index]
            # output_list.append(item)
            
        except Exception as e:
            print(e)
            # item['maj_extracted_answer'] = [extracted_answers]
            # item['extracted_gold'] = [gold_answers]
            # item['maj_is_correct'] = grade == 1
            # item['error'] = str(e)
            # item['maj_count'] = maj_count[maj_index]
            # output_list.append(item)

    
    
    # Calculate accuracy
    # accuracy = correct_count / total_count if total_count > 0 else 0
    # print(f"\nEvaluation Results:")
    # print(f"Total examples: {total_count}")
    # print(f"Correct answers: {correct_count}")
    # print(f"Accuracy: {accuracy:.2%}")
    
    # Add summary stats to the dataframe
    # result = {
    #     'accuracy': accuracy,
    #     'total_count': total_count,
    #     'correct_count': correct_count
    # }
    result = {}
    return output_list, result

def main():
    args = parse_args()
    
    # Load input CSV
    input_list = load_jsonl(args.input_jsonl)
    
    # Process answers and extract sympy objects
    output_list, result_json = process_answers(input_list, args.gold_is_latex)
    
    # Save results to output CSV
    # results_df.to_csv(args.output_csv, index=False)
    save_jsonl(output_list, args.output_jsonl)
    # with open(
    #     args.output_jsonl.replace(".jsonl", f"_metrics.json"), "w"
    # ) as f:
    #     json.dump(result_json, f, indent=4)
    print('filtered data size:', len(output_list))
    print(f"\nResults saved to {args.output_jsonl}")

if __name__ == "__main__":
    main()


