# Copyright © 2023 Abacus.AI. All rights reserved.

import argparse
import os
from tqdm import tqdm

import datasets

from models import add_posembed_args
from utils import maybe_monkey_patch, get_output_dir, load_model, test_lines_one_sample


def longeval_test(model, tokenizer, output_dir, args):

    lines_dataset = datasets.load_dataset('abacusai/LongChat-Lines')
    lines = list(lines_dataset.keys())

    if args.eval_shortest_only:
        lines = [min(lines)]

    for num_lines in lines:
        print(f"************ Start testing {num_lines} lines per LRT prompt ************")

        output_file = os.path.join(output_dir, f"{num_lines}_response.txt")
        num_correct = 0
        avg_length = 0

        test_cases = lines_dataset[num_lines]
        for idx, test_case in tqdm(enumerate(test_cases)):
            correct, prompt_length, _ = test_lines_one_sample(model=model, tokenizer=tokenizer, test_case=test_case,
                                                              output_file=output_file, idx=idx, args=args)
            avg_length += prompt_length / len(test_cases)
            num_correct += correct
        accuracy = num_correct / len(test_cases)

        with open(output_file, "a+") as f:
            f.write(f"Accuracy: {accuracy}")

        print(f"************ Finish testing {num_lines} lines per prompt with average prompt length {avg_length}, accuracy: {accuracy} ************")
        if args.eval_shortest_only:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True, help="model path")
    parser.add_argument("--flash_attn", action='store_true', help="Whether to enable flash attention to save memory, but slower.")
    parser.add_argument("--eval_shortest_only", action='store_true', default=0, help="Only eval the shortest case for illustration purpose")
    parser.add_argument("--peft-model", action='store_true', help="Set to True if this is a peft model")
    parser.add_argument("--base-model", action='store_true', help="Set to True if this is a base model")
    parser = add_posembed_args(parser)
    args = parser.parse_args()

    maybe_monkey_patch(args)
    output_dir = get_output_dir(args)

    model, tokenizer = load_model(args)
    longeval_test(model, tokenizer, output_dir, args)
