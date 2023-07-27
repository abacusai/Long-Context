# Copyright © 2023 Abacus.AI. All rights reserved.

"""
This is the script we used for finetuning the base model on samples
with different context lengths and positional embeddings. The data
has to be supplied as a train set and a val set. There is also support
for including instruct data but generally we separated the stages.

```shell
deepspeed --no_local_rank python/train/finetune_context.py --deepspeed \
    --base-model /path/to/llama/hf/llama-13b \
    --training-data DATASET:/data/siddartha/data/rp_long_train \
    --val-data DATASET:/data/siddartha/data/rp_long_val \
    --instruct-data= \
    --deepspeed_config python/train/deepspeed_large.json \
    --batch-size 32 \
    --micro-batch-size 4 \
    --epochs 1 \
    --context-length 4096 \
    --scale-context 4 \
    --learning-rate 2e-5 \
    --deepspeed
```
"""
import argparse
import logging
import os
from typing import List, Tuple

import datasets
import deepspeed
import numpy as np
import torch
import transformers
from torch import distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from models.interpolate import ScaledLlamaRotaryEmbedding
from models.xpos import XPOS
from models import add_posembed_args


def join_pages(record):
    if '_chunked_text' in record:
        return {'document': '\n\n'.join(record['_chunked_text'])}
    elif 'text' in record:
        return {'document': record['text']}
    else:
        raise NotImplementedError(str(record.keys()))


def chunk(batch: List[dict], chunk_size: int, step_size=None):
    input_id_list, attention_mask_list = batch['input_ids'], batch['attention_mask']
    input_id_parts, attn_mask_parts = [], []

    step_size = step_size or chunk_size // 2
    for input_ids, attn_mask in zip(input_id_list, attention_mask_list):
        padded_size = ((len(input_ids) + (chunk_size - 1)) // chunk_size) * chunk_size
        limit = max(1, len(input_ids) - step_size)
        input_ids_t = np.empty((padded_size,), dtype=np.int32)
        attn_t = np.empty((padded_size,), dtype=np.int32)

        input_ids_t[:len(input_ids)], input_ids_t[len(input_ids):] = input_ids, 0
        attn_t[:len(input_ids)], attn_t[len(input_ids):] = attn_mask, 0

        for s in range(0, limit, step_size):
            input_id_parts.append(input_ids_t[s:s + chunk_size])
            attn_mask_parts.append(attn_t[s:s + chunk_size])

    result = {
        'input_ids': input_id_parts,
        'attention_mask': attn_mask_parts
    }
    return result


def tokenize_and_chunk(doc_corpus: datasets.Dataset, tokenizer: AutoTokenizer, context_length: int):
    tokenized_corpus = doc_corpus.map(
        lambda r: tokenizer(
            r['document'],
            add_special_tokens=False,
            return_token_type_ids=False),
        batched=True, batch_size=10, num_proc=64)
    return tokenized_corpus.select_columns(['input_ids', 'attention_mask']).map(
        lambda r: chunk(r, context_length), batched=True, batch_size=10, num_proc=64)


def load_corpus(args: argparse.Namespace, tokenizer: AutoTokenizer, datapaths: List[str]):
    raw_corpus = datasets.load_dataset('json', data_files=datapaths)
    doc_corpus = raw_corpus.map(join_pages, remove_columns=[c for c in raw_corpus['train'].column_names if c != '_id'])
    return tokenize_and_chunk(doc_corpus['train'], tokenizer, args.context_length)


def load_instructions(args: argparse.Namespace, tokenizer: AutoTokenizer, datapaths: List[str]):
    ds = datasets.load_dataset('json', data_files=datapaths, split=datasets.Split.TRAIN)
    return ds.map(
        generate_and_tokenize_prompt,
        fn_kwargs={'tokenizer': tokenizer, 'context_length': args.context_length, 'masked': args.mask_prompt},
        remove_columns=list(ds.column_names), num_proc=64)


def load_text_dataset(args: argparse.Namespace, tokenizer: AutoTokenizer, datapaths: List[str]):
    ds = datasets.load_from_disk(datapaths[0])['train']
    ds = ds.map(join_pages, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x['document']) < 1_000_000, num_proc=32)
    return tokenize_and_chunk(ds, tokenizer, args.context_length)


def load_training_data(args: argparse.Namespace, tokenizer: AutoTokenizer, datapaths: List[str]):
    parts = []
    for dp in datapaths:
        data_type, datapath = dp.split(':')
        if data_type == 'CORPUS':
            data = load_corpus(args, tokenizer, [datapath])
        elif data_type == 'DATASET':
            data = load_text_dataset(args, tokenizer, [datapath])
        elif data_type == 'INSTRUCTION':
            data = load_instructions(args, tokenizer, [datapath])
        else:
            raise ValueError(f'Unknown data type: {data_type}')
        parts.append(data)
    return datasets.concatenate_datasets(parts)


def filter_long(data: datasets.Dataset, length):
    data = data.filter(lambda x: len(x['input_ids']) <= length, num_proc=32)
    return data


def assemble_data(args: argparse.Namespace) -> Tuple[AutoTokenizer, datasets.Dataset, datasets.Dataset]:
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    data = load_training_data(args, tokenizer, [args.training_data])
    instruct_datasets = []
    requested = args.instruct_data.split(',')
    if 'alpaca' in requested:
        instruct_datasets.append(datasets.load_dataset('yahma/alpaca-cleaned')['train'])

    if 'dolly' in requested:
        dolly_data = datasets.load_dataset('databricks/databricks-dolly-15k')['train']
        dolly_data = (dolly_data
                      .remove_columns('category')
                      .rename_column('context', 'input')
                      .rename_column('response', 'output'))
        instruct_datasets.append(dolly_data)

    if instruct_datasets:
        instruct_data = instruct_datasets[0]
        if len(instruct_datasets) > 1:
            instruct_data = datasets.concatenate_datasets(instruct_datasets)
        instruct_tokenized = instruct_data.map(
            generate_and_tokenize_prompt,
            fn_kwargs={'tokenizer': tokenizer, 'context_length': 4 * args.context_length, 'masked': args.mask_prompt},
            remove_columns=list(instruct_data.column_names), num_proc=16)
        data = datasets.interleave_datasets([data, instruct_tokenized], stopping_strategy='all_exhausted')
    else:
        logging.info('Skipping instruct datasets')

    if args.local_rank == 0:
        logging.info(str(data))
    data = filter_long(data, args.context_length)
    if args.local_rank == 0:
        logging.info(str(data))
    val_data = None
    if args.val_data:
        val_data = load_training_data(args, tokenizer, [args.val_data])
        val_data = filter_long(val_data, args.context_length)
    return tokenizer, data, val_data


class AdjustXPOSCallback(transformers.TrainerCallback):
    def __init__(self, xpos: XPOS, log: bool = False):
        self.xpos = xpos
        self.log = log

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        scale_base = self.xpos.scale_base
        if scale_base > 128:
            self.xpos.set_scale_base(scale_base - 8)
        return super().on_step_end(args, state, control, **kwargs)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.log:
            logging.info(f'Scale base = {self.xpos.scale_base}')
        return super().on_log(args, state, control, **kwargs)


def main(args):
    if not args.xpos:
        try:
            from fastchat.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
            replace_llama_attn_with_flash_attn()
            logging.info('Replaced llama with Flash attn')
        except ImportError:
            pass

    # torch.backends.cuda.enable_flash_sdp(True)
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    world_size = 1
    local_rank = 0
    device_map = 'auto'
    if args.deepspeed:
        deepspeed.init_distributed()
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()
        device_map = {'': local_rank}
        logging.info(f'Set device_map = {device_map} on rank = {local_rank}')

    micro_batch_size = args.micro_batch_size or args.batch_size
    gradient_accumulation_steps = args.batch_size // (micro_batch_size * world_size)

    if args.deepspeed and local_rank > 0:
        dist.barrier()
    tokenizer, train_data, val_data = assemble_data(args)
    train_data = train_data.shuffle(seed=0x5eed)
    VAL_SET_SIZE = 0
    if val_data is not None:
        VAL_SET_SIZE = min(3000, len(val_data))
        np.random.seed(0x5eed)
        val_data = val_data.select(np.random.choice(len(val_data), VAL_SET_SIZE, replace=False))
    if args.deepspeed:
        if local_rank == 0:
            dist.barrier()  # THis lets the group start loading from cache.
        dist.barrier()  # Waits for all others to load cached datasets.

    config = AutoConfig.from_pretrained(args.base_model)
    config.max_position_embeddings = args.context_length
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        torch_dtype=torch.float16,
        load_in_8bit=False,
        device_map=device_map,
    )
    pos_embed = None
    if args.scale_context is not None:
        pos_embed = ScaledLlamaRotaryEmbedding.patch(model, scale=args.scale_context, truncate=args.truncate_pos, randomize=args.randomized)
    elif args.scale_power is not None:
        pos_embed = ScaledLlamaRotaryEmbedding.patch(model, scale_power=args.scale_power, truncate=args.truncate_pos, randomize=args.randomized)
    elif args.xpos:
        from models.xpos import LlamaXPosAttention
        pos_embed = LlamaXPosAttention.patch(model, config, scale_base=8192)

    learning_rate = args.learning_rate
    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        config = LoraConfig(
            r=args.use_lora,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    torch.cuda.empty_cache()

    collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    if args.mask_prompt:
        # I think we can actually always use this one.
        collator = transformers.DataCollatorForTokenClassification(tokenizer)
    model.config.use_cache = False

    callbacks = []
    if args.xpos:
        callbacks.append(AdjustXPOSCallback(pos_embed, log=(local_rank == 0)))

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            optim=('adamw_torch' if not args.deepspeed_config else transformers.TrainingArguments.default_optim),
            warmup_steps=300,
            num_train_epochs=args.epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            evaluation_strategy='steps' if VAL_SET_SIZE > 0 else 'no',
            save_strategy='steps',
            eval_steps=300 if VAL_SET_SIZE > 0 else None,
            save_steps=300,
            # save_steps=500 if VAL_SET_SIZE > 0 else 0,
            output_dir=args.output_dir,
            save_total_limit=2,
            load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
            ddp_find_unused_parameters=False,
            report_to='tensorboard',
            deepspeed=args.deepspeed_config,
        ),
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train()

    if args.use_lora:
        model.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)

    logging.info("\n If there's a warning about missing keys above, please disregard :)")


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
"""


def generate_and_tokenize_prompt(data_point, tokenizer, context_length, masked=False, min_answer=100):
    prompt = generate_prompt(data_point)
    if masked:
        prompt_result = tokenizer(prompt, truncation=False)
        assert len(prompt_result['input_ids']) < context_length - min_answer, 'No room left for output'
        output_result = tokenizer(data_point['output'])
        sample = {
            k: (prompt_result[k][:-1] + output_result[k][1:])[:context_length]
            for k in ('input_ids', 'attention_mask')
        }
        sample['labels'] = (
            ([-100] * (len(prompt_result['input_ids']) - 1)) +
            output_result['input_ids'][1:]
        )[:context_length]
    else:
        result = tokenizer(prompt + data_point['output'])
        sample = {
            'input_ids': result['input_ids'][:context_length],
            'attention_mask': result['attention_mask'][:context_length],
        }

    return sample


def cmdline():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(prog='finetune_llm.py', description='Running finetuned training on a tokenized corpus')
    parser.add_argument('--local_rank', type=int, help='Deepspeed provided local rank')
    parser.add_argument('--training-data', required=True, help='Path to chunked corpus text data')
    parser.add_argument('--val-data', help='Data to be used for validation')
    parser.add_argument('--base-model', required=True, help='Base pretrained model')
    parser.add_argument('--output-dir', default=os.getcwd(), help='Output directory, defaults to current working directory')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train')
    parser.add_argument('--instruct-data', default='alpaca', help='Comma separated list of instruct corpuses [alpaca,dolly].')
    parser.add_argument('--prepare-data', action='store_true', help='Just prepare data and cache for training.')
    parser.add_argument('--mask-prompt', action='store_true', help='Mask the prompt part of instruction samples.')
    parser.add_argument('--use-lora', default=0, type=int, help='If set uses LoRA of the given rank to speedup training.')
    parser.add_argument('--context-length', default=2048, type=int, help='Set the context length of the model')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size for training')
    parser.add_argument('--micro-batch-size', default=None, type=int, help='Micro-batch size for training')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate to use for training.')
    parser = add_posembed_args(parser)
    deepspeed.add_config_arguments(parser)

    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    if args.prepare_data:
        assemble_data(args)
    else:
        main(args)


if __name__ == '__main__':
    cmdline()
