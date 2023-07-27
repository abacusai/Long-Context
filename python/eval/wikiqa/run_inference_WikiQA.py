# Copyright © 2023 Abacus.AI. All rights reserved.

from longeval.utils import maybe_monkey_patch, load_model
import argparse
from fastchat.model import get_conversation_template
import json
from tqdm import tqdm

import datasets

from models import add_posembed_args


def main(args):

    if args.task == 'freeform':
        dataset = datasets.load_dataset('abacusai/WikiQA-Free_Form_QA')
    elif args.task == 'numeric':
        dataset = datasets.load_dataset('abacusai/WikiQA-Altered_Numeric_QA')
    else:
        raise ValueError(f'Unsupported task type: {args.task}')

    try:
        data = dataset[args.task_length]
    except KeyError:
        raise ValueError(f'Unsupported task length: {args.task_length}')

    maybe_monkey_patch(args)
    model, tokenizer = load_model(args)

    ret_output_json = []
    for d in tqdm(data):
        conv = get_conversation_template("vicuna")

        conv.append_message(conv.roles[0], d['conversations'][0]['value'])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input = tokenizer(prompt, return_tensors="pt")
        prompt_length = input.input_ids.size()[-1]

        use_cache = False

        output = model.generate(input_ids=input.input_ids.to(model.device), max_new_tokens=50, use_cache=use_cache)[0]
        output = output[prompt_length:]
        output = tokenizer.batch_decode([output], skip_special_tokens=True)

        out_str = "\n======================= Input: ================================\n"
        out_str += d['conversations'][0]['value']
        out_str += "\n======================= Reference: ================================\n"
        out_str += d['conversations'][1]['value']
        out_str += "\n======================= Output: ================================\n"
        out_str += output[0]

        # Append to file.
        with open(args.txt_out_file, 'a+') as f:
            f.write(out_str)

        json_out = {}
        json_out['input'] = d['conversations'][0]['value']
        json_out['ref'] = d['conversations'][1]['value']
        json_out['output'] = output
        ret_output_json.append(json_out)

    return ret_output_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True, help="model path")
    parser.add_argument("--flash_attn", action='store_true', help="Whether to enable flash attention to save memory, but slower.")
    parser.add_argument("--peft-model", action='store_true', help="Set to True if this is a peft model")
    parser.add_argument("--base-model", action='store_true', help="Set to True if this is a base model")
    parser.add_argument("--scale-context", type=float, help='Set context length scaling for interpolation.')
    parser.add_argument("--task", type=str, required=True, help="Which evaluation task to use. Currently support [freeform, numeric]")
    parser.add_argument("--task-length", type=str, required=True, help="What length of task context to use. Currently support [2k, 4k, 8k, 16k]")
    parser.add_argument("--txt-out-file", type=str, required=True, help='Stream inference output to this file.')
    parser.add_argument("--json-out-file", type=str, required=True, help='Save final generations to this file.')
    parser = add_posembed_args(parser)
    args = parser.parse_args()

    maybe_monkey_patch(args)
    ret_output = main(args)

    # Save ret_output to file
    with open(args.json_out_file, 'w') as f:
        json.dump(ret_output, f)
