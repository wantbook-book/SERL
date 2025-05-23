import argparse
import pandas as pd
from typing import Any
from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig, parse
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

def load_csv_data(csv_path: str) -> pd.DataFrame:
    """Load and validate CSV data."""
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['answer', 'gold']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        return df
    except Exception as e:
        raise Exception(f"Error loading CSV file: {str(e)}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
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
    for item in input_list:
        extracted_answers = None
        gold_answers = None
        grade = 0
        try:
            # Use the verification function
            # grade, extracted_answers = verify_func(["\\boxed{"+item['answer']+"}"], [item['code'][0]])
            grade, extracted_answers = verify_func([item['answer']], [item['code'][0]])
            # grade, extracted_answers = verify_func([item['solution']], [item['code'][0]])
            if extracted_answers is None:
                extracted_answers = None
                gold_answers = None
            else:
                gold_answers = extracted_answers[0]
                extracted_answers = extracted_answers[1]
            
            if grade != 1:
                grade, extracted_answers = verify_func([item['code'][0]], [item['answer']])
            if extracted_answers is None:
                extracted_answers = None
                gold_answers = None
            else:
                gold_answers = extracted_answers[1]
                extracted_answers = extracted_answers[0]

            total_count += 1
            if grade == 1:
                correct_count += 1
            
            item['extracted_answer'] = [extracted_answers]
            item['extracted_gold'] = [gold_answers]
            item['is_correct'] = grade == 1
            output_list.append(item)
            
        except Exception as e:
            item['extracted_answer'] = [extracted_answers]
            item['extracted_gold'] = [gold_answers]
            item['is_correct'] = grade == 1
            item['error'] = str(e)
            output_list.append(item)

    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"\nEvaluation Results:")
    print(f"Total examples: {total_count}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Add summary stats to the dataframe
    result = {
        'accuracy': accuracy,
        'total_count': total_count,
        'correct_count': correct_count
    }
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
    with open(
        args.output_jsonl.replace(".jsonl", f"_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    print(f"\nResults saved to {args.output_jsonl}")

if __name__ == "__main__":
    main()


