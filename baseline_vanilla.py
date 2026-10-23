import time
from time import sleep
import os
import json
import argparse
from autogen import OpenAIWrapper
from vanilla_prompts import PROMPTS

from autogen.math_utils import eval_math_responses, get_answer
from utils import (
    load_local_dataset,
    math_type_mapping,
    write_json,
    mylogger,
    load_config,
    setup_proxy,
)

parser = argparse.ArgumentParser()
parser.add_argument("--dry_run", default=False, action="store_true")
parser.add_argument("--folder", "-f", dest="folder", help="saving folder", default="./zeroshot", type=str)
parser.add_argument("--cache_folder", "-c", dest="cache_folder", default=".cache/zeroshot", help="cache folder")
parser.add_argument("--samples_per_category", "-s", help="samples per category", default=20, type=int)
parser.add_argument("--categories", dest="categories", help="categories", default=[0, 1], nargs="+")
parser.add_argument("--temperature", "-t", dest="temperature", help="temperature", default=1, type=float)
parser.add_argument("--seed", dest="seed", help="seed", default=41, type=int)
parser.add_argument("--dataset_folder", "-d", dest="dataset_folder", default="./math_dataset", help="path to dataset folder")
parser.add_argument("--config", "-cfg", dest="config", default="OAI_CONFIG_LIST.json", help="config file path")
parser.add_argument("--sleep_time", "-st", dest="sleep_time", help="sleep between API calls", default=0, type=int)
parser.add_argument("--prompt_type", "-ptype", dest="prompt_type", default="baseline_vanilla", type=str,
                    help="prompt type (key in prompts.py)")
parser.add_argument("--thinking", dest="thinking", action="store_true", default=True,
                    help="enable model thinking (default: True)")
parser.add_argument("--no-thinking", dest="thinking", action="store_false",
                    help="disable model thinking")
args = parser.parse_args()
args.folder = (
    args.folder
    + "_baseline_zeroshot"
    + "_" + args.prompt_type
    + "_t" + str(args.temperature)
    + "_seed" + str(args.seed)
    + ("_think" if args.thinking else "_nothink")
)
os.makedirs(args.folder, exist_ok=True)


def zeroshot_solve(client, full_prompt, problem, file_to_be_saved=None, max_tokens=None, thinking=True):
    start_time = time.time()
    full_prompt += problem["problem"]

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
    if not thinking:
        config["reasoning_effort"] = "none"
    if max_tokens is not None:
        config["max_tokens"] = max_tokens

    raw_responses = client.create(**config)
    
    responses = client.extract_text_or_completion_object(raw_responses)
    responses = [r if isinstance(r, str) else r.content for r in responses]
        
    reasoning = ""
    msg = getattr(raw_responses.choices[0], "message", None)
    if msg is not None:
        reasoning = (
            getattr(msg, "reasoning_content", None)
            or getattr(msg, "reasoning", None)
            or ""
        )
        if not reasoning:
            extra = getattr(msg, "model_extra", None) or {}
            reasoning = extra.get("reasoning_content") or extra.get("reasoning") or ""

    solution_time = time.time() - start_time

    if file_to_be_saved is not None:
        separate_line = "\n" + "-" * 40 + "\n"
        with open(file_to_be_saved, "a") as f:
            f.write(f'Problem: {problem["problem"]}\n{separate_line}')
            f.write(f'reasoning: {reasoning}\n{separate_line}')
            f.write(f'assistant: {responses[0]}\n{separate_line}')
            f.write(f'Solution: {problem["solution"]}\n')

    return {
        "reasoning": reasoning,
        "response_with_ans": responses[0],
        "solution_time": solution_time,
        "thinking": thinking
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

    total_correct = 0
    total_time = 0
    total_problems = 0
    problem_sets = load_local_dataset(args.dataset_folder)
    
    full_prompt = PROMPTS[args.prompt_type]
    with open(os.path.join(args.folder, "prompt.txt"), "w") as f:
        f.write(full_prompt)

    logger.log("id : is_correct $ ans $ correct_ans $ time(s)", verbose=True)

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
        category_time = 0
        for count, problem in enumerate(problem_set):
            problem_path = os.path.join(saving_folder, problem["problem_id"] + ".json")

            if int(problem["problem_id"].split('_')[-1]) in done_problems:
                problem = json.load(open(problem_path, "r"))
                total_correct += problem["is_correct"]
                total_problems += 1
                correct_counts += problem["is_correct"]
                
                solution_time = problem.get("solution_time", 0)
                category_time += solution_time
                total_time += solution_time
                
                logger.log(
                    f"{problem['problem_id']} : {bool(problem['is_correct'])} $ {problem['voted_answer']} $ {problem['correct_ans']} $ {round(solution_time, 2)} (from previous run)",
                    verbose=True,
                )
                continue

            results = zeroshot_solve(
                client, 
                full_prompt, 
                problem, 
                file_to_be_saved=os.path.join(saving_folder, problem["problem_id"] + ".txt"),
                thinking=args.thinking
            )
            if results is None:
                break

            metrics = eval_math_responses([results["response_with_ans"]], problem["solution"])
            total_correct += metrics["success_vote"]
            total_problems += 1
            correct_counts += metrics["success_vote"]

            solution_time = results.get("solution_time", 0)
            category_time += solution_time
            total_time += solution_time

            problem.update(
                {
                    "is_correct": bool(metrics["success_vote"]),
                    "correct_ans": get_answer(problem["solution"]),
                    "voted_answer": get_answer(metrics["voted_answer"]),
                    "reasoning": results.get("reasoning", ""),
                    "response": results["response_with_ans"],
                    "solution_time": solution_time,
                    "thinking": args.thinking
                }
            )
            write_json(problem, problem_path)
            
            logger.log(
                f"{problem['problem_id']} : {bool(problem['is_correct'])} $ {problem['voted_answer']} $ {problem['correct_ans']} $ {round(solution_time, 2)}",
                verbose=True,
            )

        logger.log(
            f"{problem_set[0]['type']} acc: {correct_counts}/{len(problem_set)}= {round(correct_counts / len(problem_set), 4)} | Category Time: {round(category_time, 2)}s",
        )
        logger.log("-----------------------------------")
        if args.dry_run:
            break

    logger.log(
        f"Total accuracy: {total_correct}/{total_problems}={round(total_correct / max(1, total_problems), 4)}",
    )
    logger.log(f"Total time: {round(total_time, 2)}s")
    logger.log("****************************\n\n\n\n")
    os.system("tar -czf " + args.folder + ".tar.gz " + args.folder)