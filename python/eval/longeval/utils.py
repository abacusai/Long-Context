# Copyright © 2023 Abacus.AI. All rights reserved.

import json
import os
import re

import torch
import transformers
from transformers import logging

from fastchat.model import get_conversation_template

from models.interpolate import ScaledLlamaRotaryEmbedding
from models.xpos import LlamaXPosAttention

logging.set_verbosity_error()


def maybe_monkey_patch(args):
    if args.flash_attn:
        from fastchat.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()


def get_output_dir(args):
    path = args.model_name_or_path

    if path[-1] == "/":
        path = path[:-1]
    name = path.split("/")[-1]

    output_dir = f"predictions/{name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"output to {output_dir}")
    return output_dir


def load_testcases(test_file):
    with open(test_file, 'r') as json_file:
        json_list = list(json_file)

    test_cases = []
    for test_case in json_list:
        test_case = json.loads(test_case)
        test_cases.append(test_case)

    return test_cases


def load_model(args):
    if args.peft_model and args.base_model:
        raise ValueError('Warning! Both peft-model and base-model flags should not be set.')

    if args.peft_model:
        from peft import PeftModel, PeftConfig
        config = PeftConfig.from_pretrained(args.model_name_or_path)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16,
            load_in_8bit=False)
    elif args.base_model:
        config = transformers.AutoConfig.from_pretrained(args.model_name_or_path)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16,
            load_in_8bit=False)

    if args.xpos:
        LlamaXPosAttention.patch(model, config, scale_base=8192)
    else:
        scaling_kwargs = {'truncate': args.truncate_pos, 'randomize': args.randomized}
        if args.scale_context is not None:
            scaling_kwargs['scale'] = args.scale_context
        if args.scale_power is not None:
            scaling_kwargs['scale_power'] = args.scale_power
        ScaledLlamaRotaryEmbedding.patch(model, **scaling_kwargs)

    if args.peft_model:
        model = PeftModel.from_pretrained(model, args.model_name_or_path)

    model = model.cuda()
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="right",
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token

    return model, tokenizer


def test_lines_one_sample(model, tokenizer, test_case, output_file, idx, args):
    prompt = test_case["prompt"]
    expected_number = test_case["expected_number"]

    conv = get_conversation_template("vicuna")
    print(f"Using conversation template: {conv.name}")

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input = tokenizer(prompt, return_tensors="pt")
    prompt_length = input.input_ids.shape[-1]

    output = model.generate(input_ids=input.input_ids.to(model.device), min_new_tokens=5, max_new_tokens=35, use_cache=False)[0]
    output = output[prompt_length:]
    output = tokenizer.batch_decode([output], skip_special_tokens=True)[0]

    # Matching the first digit of the model output.
    response_number = re.findall("\d+", output)
    if response_number is not None and len(response_number) > 0:
        response_number = int(response_number[0])
    else:
        print(f"Got unparsable result")
        response_number = -1

    summary = f"Label: {expected_number}, Predict: {output}, Parsed: {response_number}, prompt length: {prompt_length}".replace('\n', ' ')
    print(summary)
    if idx == 0:
        with open(output_file, "w") as f:
            f.write(summary)
            f.write("\n")
    else:
        with open(output_file, "a+") as f:
            f.write(summary)
            f.write("\n")

    return expected_number == response_number, prompt_length, summary
