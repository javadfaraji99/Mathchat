from time import sleep
import os
import json
import argparse
from autogen import OpenAIWrapper

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
parser.add_argument("--select", action="store_true")
parser.add_argument("--dataset_folder", "-d", dest="dataset_folder", default="./math_dataset", help="path to dataset folder")
parser.add_argument("--config", "-cfg", dest="config", default="OAI_CONFIG_LIST.json", help="config file path")
parser.add_argument("--sleep_time", "-st", dest="sleep_time", help="sleep between API calls", default=0, type=int)
args = parser.parse_args()
args.folder = args.folder + "_baseline_zeroshot_t" + str(args.temperature) + "_seed" + str(args.seed)
os.makedirs(args.folder, exist_ok=True)


def zeroshot_solve(client, problem, max_tokens=None):
    full_prompt = """Solve a math problem carefully. Put the final answer in \\boxed{}.\n\nProblem: """

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

    return {
        "response_with_ans": responses[0],
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

    selected_samples = {
        "Algebra": [108],
    }

    logger.log("problem id: is_correct $ ans $ correct_ans $ accum_acc", verbose=True)

    for problem_set in problem_sets:
        if len(problem_set) == 0:
            continue

        if args.select:
            if problem_set[0]["type"] in selected_samples and len(selected_samples[problem_set[0]["type"]]) > 0:
                problem_set = [problem_set[i] for i in selected_samples[problem_set[0]["type"]]]
                print(problem_set[0]["type"], selected_samples[problem_set[0]["type"]])
            else:
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

            results = zeroshot_solve(client, problem)
            if results is None:
                break

            metrics = eval_math_responses([results["response_with_ans"]], problem["solution"])
            aggre_correct += metrics["success_vote"]
            correct_counts += metrics["success_vote"]

            problem.update(
                {
                    "is_correct": bool(metrics["success_vote"]),
                    "correct_ans": get_answer(problem["solution"]),
                    "voted_answer": get_answer(metrics["voted_answer"]),
                    "response": results["response_with_ans"],
                }
            )
            write_json(problem, problem_path)
            logger.log(
                f"{count}: {problem['is_correct']} $ {problem['voted_answer']} $ {problem['correct_ans']}",
                verbose=True,
            )

        logger.log(
            f"{problem_set[0]['type']} acc: {correct_counts}/{len(problem_set)}= {round(correct_counts / len(problem_set), 4)}",
        )
        logger.log("-----------------------------------")
        if args.dry_run:
            break

    total_num_problem = sum([len(problem_set) for problem_set in problem_sets])
    logger.log(
        f"Total accuracy: {aggre_correct}/{total_num_problem}={round(aggre_correct / total_num_problem, 4)}",
    )
    logger.log("****************************\n\n\n\n")
    os.system("tar -czf " + args.folder + ".tar.gz " + args.folder)