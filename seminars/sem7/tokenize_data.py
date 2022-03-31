import argparse
import os
from transformers import GPT2Tokenizer
from tqdm import tqdm
import pickle
import hashlib
import pandas as pd


def add_tokenize_data_args(parser):
    group = parser.add_argument_group('data_tokenization', 'data tokenization')

    group.add_argument('--data_path', type=str, default="dataset_2018.csv",
                       help='path to data file')
    group.add_argument('--cache_path', type=str, default="./cache",
                       help='path to tokenized cache of data file')
    group.add_argument('--tokenizer-path', type=str, default="sberbank-ai/rugpt3small_based_on_gpt2",
                       help='path to model dir')
    group.add_argument('--overwrite_cache', action='store_true',
                       help='is overwrite cache')
    group.add_argument('--bos_token', type=str, default="<s>", help='bos token')
    group.add_argument('--pad_token', type=str, default="<pad>", help='pad token')
    group.add_argument('--eos_token', type=str, default="</s>", help='eos token')
    group.add_argument('--block_size', type=int, default=512, help='block size or seq len')

    return parser


def get_args(skip_unknown=False):
    parser = argparse.ArgumentParser(description='Tokenization data arg parser')
    parser = add_tokenize_data_args(parser)
    if skip_unknown:
        args, unknown = parser.parse_known_args()
    else:
        args = parser.parse_args()
    return args


def get_cached_raw_features_file_name(data_path, cache_path, tokenizer):
    tok_hash = hashlib.sha224(str(tokenizer).encode()).hexdigest()
    path_hash = hashlib.sha224(data_path.encode()).hexdigest()
    return os.path.join(cache_path, f"{path_hash}_{tok_hash}.cache")


def load_tokenizer(tokenizer=None, tokenizer_path=None, bos_token="<s>", eos_token="</s>", pad_token="<pad>", **kwargs):
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    if tokenizer.bos_token is None or tokenizer.bos_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({"bos_token": bos_token})
        tokenizer.add_special_tokens({"eos_token": eos_token})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": pad_token})
    return tokenizer


def tokenize_and_load(
        data_path, cache_path, tokenizer=None, tokenizer_path=None,
        bos_token="<s>", eos_token="</s>", pad_token="<pad>", overwrite_cache=False, block_size=512, **kwargs
):
    tokenizer = load_tokenizer(
        tokenizer=tokenizer, tokenizer_path=tokenizer_path,
        bos_token=bos_token, eos_token=eos_token, pad_token=pad_token
    )
    os.makedirs(cache_path, exist_ok=True)
    cached_features_file = get_cached_raw_features_file_name(data_path, cache_path, tokenizer)
    print("Cached features file", cached_features_file)
    if not os.path.exists(cached_features_file) or overwrite_cache:
        print("Start processing file", data_path)
        df = pd.read_csv(data_path)
        text = []
        for doc in tqdm(df.text, total=len(df.text), leave=False, desc="encoding documents"):
            doc = f"{bos_token}{doc}{eos_token}\n"
            doc = tokenizer.encode(doc)
            text.extend(doc)
        texts = []
        for i in range(0, len(text) - block_size + 1, block_size):
                example = text[i:i + block_size]
                if len(example) == block_size:
                    texts.append(example)
        with open(cached_features_file, "wb") as handle:
            pickle.dump(texts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading cached features from file")
        with open(cached_features_file, "rb") as handle:
            texts = pickle.load(handle)
    return texts, tokenizer


def main():
    args = get_args()
    _ = tokenize_and_load(**vars(args))


if __name__ == "__main__":
    main()
