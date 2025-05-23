from math_verify import math_metric, LatexExtractionConfig, ExprExtractionConfig
import torch

def reward_func(queries: list[str], prompts: list[str], labels: list[str])->torch.Tensor:
    """
    queries: list of strings, each string is a question and response
    """

    # Create the verification function
    verify_func = math_metric(
        gold_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        aggregation_function=max,
        precision=6
    )

    reward_list = []

    # Iterate over each question-response pair
    for query, label in zip(queries, labels):
        # Use the verify function to validate the answers
        is_true = False
        try:
            # is_true = verify(extracted_answer, gt_answer)
            grade, extracted_answers = verify_func([label], [query])
            is_true = grade == 1
            if not is_true:
                grade, extracted_answers = verify_func([query], [label])
                is_true = grade == 1

            print(f"resp: {extracted_answers[0][0]}; gt: {extracted_answers[1][0]}; is_true: {is_true}")
        except:
            # print(f"Error verifying answers: {e}")
            print("="*20+"reward error"+"="*20)
            print(query)
            print(label)
            is_true = False
        reward = 1.0 if is_true else 0.0

        reward_list.append(reward)

    return torch.tensor(reward_list)



if __name__ == '__main__':
    queries = [

    ]
    labels = [

    ]

    item = {"idx": 0, "prompt": "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$", "label": "We have that $r = \\sqrt{0^2 + 3^2} = 3.$  Also, if we draw the line connecting the origin and $(0,3),$ this line makes an angle of $\\frac{\\pi}{2}$ with the positive $x$-axis.\n\n[asy]\nunitsize(0.8 cm);\n\ndraw((-0.5,0)--(3.5,0));\ndraw((0,-0.5)--(0,3.5));\ndraw(arc((0,0),3,0,90),red,Arrow(6));\n\ndot((0,3), red);\nlabel(\"$(0,3)$\", (0,3), W);\ndot((3,0), red);\n[/asy]\n\nTherefore, the polar coordinates are $\\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}.$", "datasource": "default", "level": 2, "responses": ["Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$\n## Step 1: Recall the conversion formulas from rectangular to polar coordinates.\nThe conversion from rectangular coordinates $(x, y)$ to polar coordinates $(r, \\theta)$ can be done using the formulas $r = \\sqrt{x^2 + y^2}$ for the radial coordinate and $\\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)$ for the angular coordinate.\n\n## Step 2: Apply the conversion formulas to the given point $(0,3)$.\nGiven the point $(0,3)$, we can substitute $x = 0$ and $y = 3$ into the formulas. For $r$, we have $r = \\sqrt{0^2 + 3^2} = \\sqrt{0 + 9} = \\sqrt{9} = 3$. For $\\theta$, we have $\\theta = \\tan^{-1}\\left(\\frac{3}{0}\\right)$. However, we must be careful with the value of $x$ because $\\tan^{-1}\\left(\\frac{3}{0}\\right)$ is undefined in the classical sense, but we know that the point $(0,3)$ lies on the positive $y$-axis, which corresponds to an angle of $\\frac{\\pi}{2}$ radians.\n\n## Step 3: Determine the correct polar coordinate representation.\nSince $r = 3$ and $\\theta = \\frac{\\pi}{2}$, the polar coordinate representation of the point $(0,3)$ is $\\left(3, \\frac{\\pi}{2}\\right)$.\n\nThe final answer is: $\\boxed{\\left(3, \\frac{\\pi}{2}\\right)}$"], "reward": [0.0]}
    queries.append(item["responses"][0])
    labels.append(item["label"])
    print(reward_func(queries, [], labels))