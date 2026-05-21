"""一次性预 tokenize：把 JSONL 转成 .bin 张量文件（使用 HuggingFace 原生 tokenizer，Rust 后端）。"""
import os, sys, json, time
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tokenizers import Tokenizer as HFTokenizer

_worker_tokenizer = None
_worker_max_seq_len = None
_worker_bos = None
_worker_eos = None
_worker_pad = None

def _init_worker(tokenizer_path, max_seq_len):
    global _worker_tokenizer, _worker_max_seq_len, _worker_bos, _worker_eos, _worker_pad
    import json
    _worker_tokenizer = HFTokenizer.from_file(os.path.join(tokenizer_path, 'tokenizer.json'))
    _worker_max_seq_len = max_seq_len
    with open(os.path.join(tokenizer_path, 'tokenizer_config.json')) as f:
        cfg = json.load(f)
    _worker_bos = _worker_tokenizer.token_to_id(cfg['bos_token'])
    _worker_eos = _worker_tokenizer.token_to_id(cfg['eos_token'])
    _worker_pad = _worker_tokenizer.token_to_id(cfg['pad_token'])

def _tokenize_one(args):
    idx, line = args
    data = json.loads(line)
    text = str(data['text'])
    # tokenizers.Tokenizer API: encode() -> Encoding with .ids
    encoded = _worker_tokenizer.encode(text)
    tokens = [_worker_bos] + encoded.ids[:_worker_max_seq_len - 2] + [_worker_eos]
    pad_len = _worker_max_seq_len - len(tokens)
    input_ids = np.array(tokens + [_worker_pad] * pad_len, dtype=np.int16)
    labels = np.array(tokens + [-100] * pad_len, dtype=np.int16)
    return idx, input_ids, labels


def pre_tokenize(data_path, out_path, tokenizer_path='../model', max_seq_len=340, num_workers=None):
    if num_workers is None:
        num_workers = min(cpu_count(), 8)
    print(f"加载数据: {data_path}")
    with open(data_path) as f:
        lines = f.readlines()
    total = len(lines)
    print(f"共 {total:,} 条，{num_workers} 进程并行，tokenizer: HuggingFace PreTrainedTokenizerFast (Rust)")

    t0 = time.time()
    arr = np.zeros((total, 2, max_seq_len), dtype=np.int16)

    with Pool(num_workers, initializer=_init_worker, initargs=(tokenizer_path, max_seq_len)) as pool:
        for idx, input_ids, labels in tqdm(
            pool.imap_unordered(_tokenize_one, enumerate(lines), chunksize=256),
            total=total, desc="Tokenizing"
        ):
            arr[idx, 0, :] = input_ids
            arr[idx, 1, :] = labels
    del lines

    print(f"写入磁盘: {out_path} ...")
    arr.tofile(out_path)
    # 保存形状元数据，供 BinDataset 加载时使用
    meta_path = out_path + '.json'
    with open(meta_path, 'w') as f:
        json.dump({'shape': list(arr.shape), 'dtype': 'int16'}, f)
    elapsed = time.time() - t0
    print(f"完成: {out_path} ({os.path.getsize(out_path)/1e9:.2f} GB), 耗时 {elapsed/60:.1f} 分钟, meta: {meta_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t.jsonl")
    parser.add_argument("--out_path", type=str, default="../dataset/pretrain_t2t.bin")
    parser.add_argument("--max_seq_len", type=int, default=340)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()
    pre_tokenize(args.data_path, args.out_path, max_seq_len=args.max_seq_len, num_workers=args.num_workers)
