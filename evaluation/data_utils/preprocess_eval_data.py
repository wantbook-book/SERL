import re
import json
from collections import defaultdict, Counter
import random
import regex
def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def convert_word_number(text: str) -> str:
    try:
        text = str(w2n.word_to_num(text))
    except:
        pass
    return text


# units mainly from MathQA
unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]

unit_texts.extend([t + "s" for t in unit_texts])
def strip_string(string, skip_unit=False):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    # replace \\ with \
    string = string.replace("\\!", "")
    # string = string.replace("\\ ", "")
    # string = string.replace("\\\\", "\\")

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    if not skip_unit:
        # Remove unit: texts
        for _ in range(2):
            for unit_text in unit_texts:
                # use regex, the prefix should be either the start of the string or a non-alphanumeric character
                # the suffix should be either the end of the string or a non-alphanumeric character
                _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
                if _string != "":
                    string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")

    # convert word number to digit
    string = convert_word_number(string)

    # replace "\\text{...}" to "..."
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    # string = string.replace("\\cdot", "")
    if (
        string.startswith("{")
        and string.endswith("}")
        and string.isalnum()
        or string.startswith("(")
        and string.endswith(")")
        and string.isalnum()
        or string.startswith("[")
        and string.endswith("]")
        and string.isalnum()
    ):
        string = string[1:-1]

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

direct_answer_trigger_for_fewshot = ("choice is", "answer is")

def choice_answer_clean(pred: str):
    pred = pred.strip("\n")

    # Determine if this is ICL, if so, use \n\n to split the first chunk.
    # ----------------------------------------------
    # ICL = False
    # for trigger in direct_answer_trigger_for_fewshot:
    #     if pred.count(trigger) > 1:
    #         ICL = True
    # if ICL:
    #     pred = pred.split("\n\n")[0]
    # ----------------------------------------------

    # Split the trigger to find the answer.
    preds = re.split("|".join(direct_answer_trigger_for_fewshot), pred)
    if len(preds) > 1:
        answer_flag = True
        pred = preds[-1]
    else:
        answer_flag = False

    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")

    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]

    if len(pred) == 0:
        pred = ""
    else:
        if answer_flag:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last e
            pred = pred[-1]

    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")

    return pred
def extract_answer(pred_str, data_name, use_last_number=True):
    pred_str = pred_str.replace("\u043a\u0438", "")
    if data_name in ["mmlu_stem", "sat_math", "aqua", "gaokao2023"]:
        # TODO check multiple choice
        return choice_answer_clean(pred_str)

    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        # minerva_math
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
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
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    elif "答案是" in pred_str:
        # Handle Chinese few-shot multiple choice problem answer extraction
        pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    else:  # use the last number
        if use_last_number:
            pattern = "-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ""
        else:
            pred = ""

    # choice answer
    if (
        data_name in ["sat_math", "aqua"]
        or "mmlu" in data_name
    ):
        tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
        if tmp:
            pred = tmp[-1]
        else:
            pred = pred.strip().strip(".")

    # multiple line
    # pred = pred.split("\n")[0]
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])
    return pred

def extract_gsm8k_answer(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution
    exit()

def preprocess_gsm8k(file_path, output_path):
    """
    Preprocess the GSM8K dataset to extract the final answer from the solution string.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Extract the final answer from the solution string
            entry['answer'] = extract_gsm8k_answer(entry['answer'])
            if "boxed" not in entry['answer']:
                entry['answer'] = "\\boxed{" + entry['answer'] + "}"
            data.append(entry)

    # Save the preprocessed data to a new JSON file
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    exit()

def preprocess_amc23(file_path, output_path):
    """
    Preprocess the AMC23 dataset to extract the final answer from the solution string.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Extract the final answer from the solution string
            entry['answer'] = str(entry['answer'])
            if "boxed" not in entry['answer']:
                entry['answer'] = "\\boxed{" + entry['answer'] + "}"
            data.append(entry)

    # Save the preprocessed data to a new JSON file
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    exit()

def preprocess_math_500(file_path, output_path):
    """
    Preprocess the Math Hard dataset to extract the final answer from the solution string.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Extract the final answer from the solution string
            if "boxed" not in entry['answer']:
                entry['answer'] = "\\boxed{" + entry['answer'] + "}"
            del entry['solution']

            data.append(entry)

    # Save the preprocessed data to a new JSON file
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    exit()

def preprocess_aqua(file_path, output_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            options = entry["options"]
            choice = "(" + "(".join(options)
            choice = choice.replace("(", " (").replace(")", ") ").strip()
            choice = "\nAnswer Choices: " + choice
            question = entry["question"].strip() + choice
            entry['question'] = question

            answer = entry['correct']
            # Extract the final answer from the solution string
            if "boxed" not in answer:
                answer = "\\boxed{" + answer + "}"
            entry['answer'] = answer
            data.append(entry)

    # Save the preprocessed data to a new JSON file
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    exit()

def preprocess_asdiv(file_path, output_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Extract the final answer from the solution string
            question = f"{entry['body'].strip()} {entry['question'].strip()}"
            del entry['body']
            entry['question'] = question
            answer = re.sub(r"\(.*?\)", "", entry["answer"]).strip()
            if "boxed" not in answer:
                answer = "\\boxed{" + answer + "}"
            entry['answer'] = answer
            data.append(entry)

    # Save the preprocessed data to a new JSON file
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    exit()

def preprocess_carp_en(file_path, output_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Extract the final answer from the solution string
            entry['question'] = entry['content']
            del entry['content']
            answer = entry['answer']
            if "boxed" not in answer:
                answer = "\\boxed{" + answer + "}"
            entry['answer'] = answer
            data.append(entry)

    # Save the preprocessed data to a new JSON file
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    exit()

def preprocess_college_math(file_path, output_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            entry['answer'] = entry["answer"].replace("$", "").strip()
            if "boxed" not in entry['answer']:
                entry['answer'] = "\\boxed{" + entry['answer'] + "}"
            data.append(entry)

    # Save the preprocessed data to a new JSON file
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    exit()

def preprocess_mawps(file_path, output_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Extract the final answer from the solution string
            answer = str(entry["target"])
            if "boxed" not in answer:
                answer = "\\boxed{" + answer + "}"
            entry['answer'] = answer
            del entry['target']
            entry['question'] = entry['input']
            del entry['input']
            data.append(entry)

    # Save the preprocessed data to a new JSON file
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    exit()

def preprocess_minerva_math(file_path, output_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            solution = entry['solution']
            answer = extract_answer(solution, "minerva_math")
            # Extract the final answer from the solution string
            if "boxed" not in answer:
                answer = "\\boxed{" + answer + "}"
            entry['answer'] = answer
            del entry['solution']
            entry['question'] = entry['problem']
            del entry['problem']
            data.append(entry)

    # Save the preprocessed data to a new JSON file
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    exit()

def preprocess_mmlu_stem(file_path, output_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Extract the final answer from the solution string
            abcd = "ABCD"
            answer = abcd[entry["answer"]]
            if "boxed" not in answer:
                answer = "\\boxed{" + answer + "}"
            entry['answer'] = answer

            options = entry["choices"]
            assert len(options) == 4
            for i, (label, option) in enumerate(zip("ABCD", options)):
                options[i] = f"({label}) {str(option).strip()}"
            options = " ".join(options)
            # question = f"{example['question'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{options}"
            question = f"{entry['question'].strip()}\nAnswer Choices: {options}"
            entry['question'] = question
            data.append(entry)

    # Save the preprocessed data to a new JSON file
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    exit()

def preprocess_olympiadbench(file_path, output_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Extract the final answer from the solution string
            answer = entry["final_answer"][0].strip("$")
            if "boxed" not in answer:
                answer = "\\boxed{" + answer + "}"
            entry['answer'] = answer

            data.append(entry)

    # Save the preprocessed data to a new JSON file
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    exit()

def preprocess_sat_math(file_path, output_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            options = entry["options"].strip()
            assert "A" == options[0]
            options = "(" + options
            for ch in "BCD":
                if f" {ch}) " in options:
                    options = regex.sub(f" {ch}\) ", f" ({ch}) ", options)
            question = f"{entry['question'].strip()}\nAnswer Choices: {options}"
            entry['question'] = question
            # Extract the final answer from the solution string
            answer = entry["Answer"]
            if "boxed" not in answer:
                answer = "\\boxed{" + answer + "}"
            entry['answer'] = answer
            del entry["Answer"]

            data.append(entry)

    # Save the preprocessed data to a new JSON file
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    exit()

def preprocess_svamp(file_path, output_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            body = entry["Body"].strip()
            if not body.endswith("."):
                body = body + "."
            question = f'{body} {entry["Question"].strip()}'
            entry['question'] = question
            del entry["Question"]
            del entry["Body"]
            # Extract the final answer from the solution string
            answer = str(entry["Answer"])
            if "boxed" not in answer:
                answer = "\\boxed{" + answer + "}"
            entry['answer'] = answer
            del entry["Answer"]

            data.append(entry)

    # Save the preprocessed data to a new JSON file
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    exit()

def preprocess_tabmwp(file_path, output_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Extract the final answer from the solution string
            gt_ans = entry["answer"]
            if entry["ans_type"] in ["integer_number", "decimal_number"]:
                if "/" in gt_ans:
                    gt_ans = int(gt_ans.split("/")[0]) / int(gt_ans.split("/")[1])
                elif "," in gt_ans:
                    gt_ans = float(gt_ans.replace(",", ""))
                elif "%" in gt_ans:
                    gt_ans = float(gt_ans.split("%")[0]) / 100
                else:
                    gt_ans = float(gt_ans)
            answer = str(gt_ans)
            if "boxed" not in answer:
                answer = "\\boxed{" + answer + "}"
            entry['answer'] = answer

            title_str = (
                f'regarding "{entry["table_title"]}" ' if entry["table_title"] else ""
            )
            question = f"Read the following table {title_str}and answer a question:\n"
            question += f'{entry["table"]}\n{entry["question"]}'
            if entry["choices"]:
                question += (
                    f' Please select from the following options: {entry["choices"]}'
                )
            entry['question'] = question
            del entry["table_title"]
            del entry["table"]
            data.append(entry)

    # Save the preprocessed data to a new JSON file
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    exit()

if __name__ == "__main__":
    # file_path = 'tabmwp/test_raw.jsonl'
    # output_path = 'tabmwp/test.jsonl'
    # preprocess_tabmwp(file_path, output_path)

    # file_path = 'svamp/test_raw.jsonl'
    # output_path = 'svamp/test.jsonl'
    # preprocess_svamp(file_path, output_path)

    # file_path = 'sat_math/test_raw.jsonl'
    # output_path = 'sat_math/test.jsonl'
    # preprocess_sat_math(file_path, output_path)

    # file_path = 'olympiadbench/test_raw.jsonl'
    # output_path = 'olympiadbench/test.jsonl'
    # preprocess_olympiadbench(file_path, output_path)

    # file_path = 'mmlu_stem/test_raw.jsonl'
    # output_path = 'mmlu_stem/test.jsonl'
    # preprocess_mmlu_stem(file_path, output_path)

    # file_path = 'minerva_math/test_raw.jsonl'
    # output_path = 'minerva_math/test.jsonl'
    # preprocess_minerva_math(file_path, output_path)
    
    file_path = 'mawps/test_raw.jsonl'
    output_path = 'mawps/test.jsonl'
    preprocess_mawps(file_path, output_path)

    # file_path = 'gaokao2023en/test_raw.jsonl'
    # output_path = 'gaokao2023en/test.jsonl'
    # preprocess_college_math(file_path, output_path)

    # file_path = 'college_math/test_raw.jsonl'
    # output_path = 'college_math/test.jsonl'
    # preprocess_college_math(file_path, output_path)

    # file_path = 'carp_en/test_raw.jsonl'
    # output_path = 'carp_en/test.jsonl'
    # preprocess_carp_en(file_path, output_path)

    # file_path = 'asdiv/test_raw.jsonl'
    # output_path = 'asdiv/test.jsonl'
    # preprocess_asdiv(file_path, output_path)

    # file_path = 'aqua/test_raw.jsonl'
    # output_path = 'aqua/test.jsonl' 
    # preprocess_aqua(file_path, output_path)

    # file_path = 'gsm8k/test_raw.jsonl'
    # # file_path = '/xxx/Math-Verify/outputs/xxx/orlhf_checkpoints/checkpoint/llama3-3b-500seed_gen7500_filtered_32maj8_random_bon_maj_bs16/global_step100_hf/math_eval/gsm8k/test_pure_-1_seed0_t1.0_s0_e-1.jsonl'
    # output_path = 'gsm8k/test.jsonl'
    # # output_path = '/xxx/Math-Verify/outputs/xxx/orlhf_checkpoints/checkpoint/llama3-3b-500seed_gen7500_filtered_32maj8_random_bon_maj_bs16/global_step100_hf/math_eval/gsm8k/test_pure_-1_seed0_t1.0_s0_e-1.jsonl'
    # preprocess_gsm8k(file_path, output_path)

    # file_path = 'amc23/test_raw.jsonl'
    # # file_path = '/xxx/Math-Verify/outputs/xxx/orlhf_checkpoints/checkpoint/llama3-3b-500seed_gen7500_filtered_32maj8_random_bon_maj_bs16/global_step100_hf/math_eval/amc23/test_pure_-1_seed0_t1.0_s0_e-1.jsonl'
    # output_path = 'amc23/test.jsonl'
    # # output_path = '/xxx/Math-Verify/outputs/xxx/orlhf_checkpoints/checkpoint/llama3-3b-500seed_gen7500_filtered_32maj8_random_bon_maj_bs16/global_step100_hf/math_eval/amc23/test_pure_-1_seed0_t1.0_s0_e-1.jsonl'
    # preprocess_amc23(file_path, output_path)

    # file_path = 'aime24/test_raw.jsonl'
    # output_path = 'aime24/test.jsonl'
    # preprocess_amc23(file_path, output_path)

    # file_path = 'math_hard/test_raw.jsonl'
    # output_path = 'math_hard/test.jsonl'
    # preprocess_math_hard(file_path, output_path)

    # file_path = 'math_500/test_raw.jsonl'
    # file_path = "/xxx/Math-Verify/outputs/xxx/share/LLMAgent/model/Llama-3.2-3B-Instruct/math_eval_bon_32/math/train_with_idx_pure_-1_seed0_t1.0_s0_e-1_raw.jsonl"
    # output_path = '/xxx/Math-Verify/outputs/xxx/share/LLMAgent/model/Llama-3.2-3B-Instruct/math_eval_bon_32/math/train_with_idx_pure_-1_seed0_t1.0_s0_e-1.jsonl'
    # preprocess_math_500(file_path, output_path)