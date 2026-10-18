from time import sleep
import os
import json
import argparse
from autogen import OpenAIWrapper

from autogen.math_utils import eval_math_responses, get_answer
from autogen.code_utils import execute_code
from utils import (
    load_local_dataset,
    math_type_mapping,
    write_json,
    remove_asy_sections,
    mylogger,
    load_config,
    setup_proxy,
)
from tool import synthesize_program

parser = argparse.ArgumentParser()
parser.add_argument("--dry_run", default=False, action="store_true")
parser.add_argument("--folder", "-f", dest="folder", help="saving folder", default="./PoT", type=str)
parser.add_argument("--cache_folder", "-c", dest="cache_folder", default=".cache/PoT", help="cache folder")
parser.add_argument("--samples_per_category", "-s", help="samples per category", default=20, type=int)
parser.add_argument("--temperature", "-t", dest="temperature", help="temperature", default=1, type=float)
parser.add_argument("--seed", dest="seed", help="seed", default=41, type=int)
parser.add_argument("--categories", dest="categories", help="categories", default=[0, 1], nargs="+")
parser.add_argument("--sample_all", help="samples per category", default=0, type=int)
parser.add_argument("--dataset_folder", "-d", dest="dataset_folder", default="./math_dataset", help="path to dataset folder")
parser.add_argument("--config", "-cfg", dest="config", default="OAI_CONFIG_LIST.json", help="config file path")
parser.add_argument("--sleep_time", "-st", dest="sleep_time", help="sleep between API calls", default=0, type=int)
args = parser.parse_args()
args.folder = args.folder + "_baseline_PoT_t" + str(args.temperature) + "_seed" + str(args.seed)
if args.sample_all != 0:
    args.folder += "_random_sample"
os.makedirs(args.folder, exist_ok=True)


def PoT_solve(client, problem, max_tokens=None):
    commented_problem = problem["problem"].replace("\n", "\n# ")
    commented_problem = remove_asy_sections(commented_problem)
    full_prompt = f"""
import math
import numpy as np
import sympy as sp

# Question: {commented_problem}
# Answer this question by implementing a solver() function.
def solver():
    # Let's write a Python program step by step, and then return the answer
    # Firstly, we need define the following variable:
"""
    with open(os.path.join(args.folder, "prompt.txt"), "w") as f:
        f.write(full_prompt)
    if args.dry_run:
        print(full_prompt)
        print("=======================")
        return

    config = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt},
        ],
        "n": 1,
    }
    if max_tokens is not None:
        config["max_tokens"] = max_tokens

    raw_responses = client.create(**config)
    responses = client.extract_text_or_completion_object(raw_responses)
    responses = [r if isinstance(r, str) else r.content for r in responses]

    program = synthesize_program(responses[0], full_prompt)
    return_code, ans, _ = execute_code(program, timeout=5, use_docker=False)
    if isinstance(ans, bytes):
        try:
            ans = ans.decode("ascii")
        except Exception:
            try:
                ans = ans.decode("utf-8")
            except Exception:
                ans = "The return cannot be decoded."

    ans = "Error" if return_code != 0 or ans is None else ans
    response_with_ans = "\\boxed{" + str(ans) + "}"
    return {
        "response_with_ans": response_with_ans,
        "program": program,
    }


if __name__ == "__main__":
    setup_proxy()
    config_list = load_config(args.config)
    client = OpenAIWrapper(
        config_list=config_list,
        cache_seed=args.seed,
        timeout=60 * 10,
    )
    logger = mylogger(os.path.join(args.folder, "log.txt"))

    aggre_correct = 0
    problem_sets = load_local_dataset(args.dataset_folder)
    logger.log("problem id: is_correct $ ans $ correct_ans $ accum_acc", verbose=True)

    for problem_set in problem_sets:
        if len(problem_set) == 0:
            continue

        logger.log("Solving " + problem_set[0]["type"], verbose=True)
        saving_folder = os.path.join(args.folder, math_type_mapping[problem_set[0]["type"]])
        os.makedirs(saving_folder, exist_ok=True)

        done_problems = set()
        for f in os.listdir(saving_folder):
            if "json" in f:
                try:
                    problem_num = int(f.split(".")[0].split('_')[-1])
                    done_problems.add(problem_num)
                except ValueError:
                    continue

        correct_counts = 0
        for count, problem in enumerate(problem_set):
            problem_path = os.path.join(saving_folder, problem["problem_id"] + ".json")

            if int(problem["problem_id"].split('_')[-1]) in done_problems:
                problem = json.load(open(problem_path, "r"))
                aggre_correct += problem["is_correct"]
                correct_counts += problem["is_correct"]
                logger.log(
                    f"{count}: {problem['is_correct']} $ {problem['voted_answer']} $ {problem['correct_ans']} $ {round(correct_counts / (count + 1), 4)} (loaded from previous run)",
                    verbose=True,
                )
                continue

            results = PoT_solve(client, problem)
            metrics = eval_math_responses([results["response_with_ans"]], problem["solution"])
            aggre_correct += metrics["success_vote"]
            correct_counts += metrics["success_vote"]

            problem.update(
                {
                    "is_correct": bool(metrics["success_vote"]),
                    "correct_ans": get_answer(problem["solution"]),
                    "voted_answer": get_answer(metrics["voted_answer"]),
                    "program": results["program"],
                }
            )
            write_json(problem, problem_path)
            logger.log(
                f"{count}: {problem['is_correct']} $ {problem['voted_answer']} $ {problem['correct_ans']}",
                verbose=True,
            )
            if args.dry_run:
                break

        logger.log(
            f"{problem_set[0]['type']} acc: {correct_counts}/{len(problem_set)}= {round(correct_counts / len(problem_set), 4)}",
        )
        logger.log("-----------------------------------")
        os.system("tar -czf " + args.folder + ".tar.gz " + args.folder)

    total_num_problem = sum([len(problem_set) for problem_set in problem_sets])
    logger.log(
        f"Total accuracy: {aggre_correct}/{total_num_problem}={round(aggre_correct / total_num_problem, 4)}",
    )
    logger.log("****************************\n\n\n\n")
    os.system("tar -czf " + args.folder + ".tar.gz " + args.folder)