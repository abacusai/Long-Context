# Copyright © 2023 Abacus.AI. All rights reserved.

"""
This is an example script showing how we setup the data for training.
It is very simple but the handling of train and val is a little non standard.

```shell
python train/prepare_data.py togethercomputer/RedPajama-Data-1T-Sample /tmp/rp_long
```
"""
import argparse
import datasets
from pathlib import Path

TRAIN = Path('train')
VAL = Path('val')


def filter_dataset(args):
    source = datasets.load_dataset(args.source)
    text_limit = args.char_len
    filtered = source.filter(lambda x: len(x['text']) > text_limit, num_proc=8)
    # RedPajama only has a train split and this script only uses that split.
    filtered = filtered['train'].shuffle(0x5eed)
    train = filtered.select(range(args.val_size, len(filtered)))
    val = filtered.select(range(args.val_size))
    base = Path(args.dest)
    train.save_to_disk(base / TRAIN)
    val.save_to_disk(base / VAL)


def main():
    parser = argparse.ArgumentParser(
        prog='prepare_data.py',
        description='Create train and validation datasets of long samples.')
    parser.add_argument(
        '--char-len', type=int, default=(5 * 4096),
        help='The minium sample length to retain.')
    parser.add_argument(
        '--val-size', type=int, default=5000,
        help='Number of items to pull for validation.')
    parser.add_argument(
        'source',
        help='The source Huggingface dataset')
    parser.add_argument(
        'dest',
        help='Destination prefix at which to write results.')

    args = parser.parse_args()

    filter_dataset(args)

if __name__ == '__main__':
    main()