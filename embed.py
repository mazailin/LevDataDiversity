#!/usr/bin/env python3
"""
Embedding computation script.
Reads all JSONL files in the input directory, computes embeddings, and preserves the directory structure.
"""

from openai import OpenAI
import struct
import json
from transformers import AutoTokenizer
from pathlib import Path
from multiprocessing import Pool, Manager
from tqdm import tqdm
from argparse import ArgumentParser
import os
import sys


def truncate_texts(texts, tokenizer, max_length=512):
    """Truncate texts to the specified token length."""
    truncated_texts = []
    for text in texts:
        encoded_input = tokenizer(text, return_tensors="pt")
        token_count = len(encoded_input["input_ids"][0])
        truncated_input = text
        if token_count > max_length:
            truncated_input = tokenizer.decode(
                encoded_input["input_ids"][0][:max_length],
                skip_special_tokens=True
            )
        truncated_texts.append(truncated_input)
    return truncated_texts


def compute_embed(model, tokenizer, client, text_list, max_length, fo):
    """Compute embeddings for a batch of texts."""
    truncated_prompts = truncate_texts(texts, tokenizer, max_length)
    outputs = client.embeddings.create(model=model, input=truncated_prompts)
    for e in outputs.data:
        fo.write(struct.pack("f" * len(e.embedding), *e.embedding))


def process_jsonl_file(jsonl_path, embedding_path, batch_size, max_length, client, model, tokenizer, doc_num=None):
    """Process a single JSONL file and generate embeddings."""
    with open(embedding_path, 'wb') as fo:
        with open(jsonl_path, 'r') as f:
            count = 0
            batch_list = []
            for line in f:
                count += 1
                if doc_num is not None and count > doc_num:
                    break
                try:
                    data = json.loads(line.strip())
                    if 'text' in data:
                        text_key = 'text'
                    elif 'tex' in data:
                        text_key = 'tex'
                    else:
                        text_key = 'text'
                    batch_list.append(data[text_key])
                except (json.JSONDecodeError, KeyError):
                    continue

                if len(batch_list) == batch_size:
                    compute_embed(model, tokenizer, client, batch_list, max_length, fo)
                    batch_list = []

            if batch_list:
                compute_embed(model, tokenizer, client, batch_list, max_length, fo)


def get_done_path(jsonl_path, input_dir, log_dir):
    """Generate done marker file path based on JSONL file path."""
    rel_path = str(Path(jsonl_path).relative_to(input_dir))
    done_file = rel_path.replace('/', '_').replace('.jsonl', '.done')
    return log_dir / done_file


def get_embedding_path(jsonl_path, input_dir, output_dir):
    """Generate embedding file path based on JSONL file path, preserving directory structure."""
    rel_path = Path(jsonl_path).relative_to(input_dir)
    return Path(output_dir) / rel_path.with_suffix('.embedding')


def init_worker(mq_, progress_):
    """Initialize worker process."""
    global mq, progress
    mq = mq_
    progress = progress_


def worker(worker_id, api_url, model_name, batch_size, max_length, doc_num, input_dir, log_dir):
    """Worker process function."""
    global mq, progress

    client = OpenAI(base_url=api_url, api_key="EMPTY")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    while True:
        file_info = mq.get()
        if file_info is None:
            break

        jsonl_path, embedding_path = file_info
        done_path = get_done_path(jsonl_path, input_dir, log_dir)

        if done_path.exists():
            progress.put(jsonl_path)
            continue

        try:
            embedding_path.parent.mkdir(parents=True, exist_ok=True)
            process_jsonl_file(jsonl_path, embedding_path, batch_size, max_length, client, model_name, tokenizer, doc_num)
            done_path.touch()
            progress.put(jsonl_path)
        except Exception as e:
            print(f"Error processing {jsonl_path}: {e}")
            progress.put(jsonl_path)


def find_jsonl_files(input_dir):
    """Find all JSONL files in the input directory recursively."""
    jsonl_files = []

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.startswith('.'):
                continue
            if file.endswith('.jsonl'):
                jsonl_files.append(os.path.join(root, file))

    return sorted(jsonl_files)


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Compute embeddings for JSONL files")
    parser.add_argument("--model", default="/models/Qwen3-Embedding-0.6B",
                       help="Path to the embedding model")
    parser.add_argument("--input_dir", required=True,
                       help="Input directory containing JSONL files")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for embedding files")
    parser.add_argument("--log_dir", default=None,
                       help="Log directory for done marker files")
    parser.add_argument("--batch_size", default=2048, type=int,
                       help="Batch size for embedding computation")
    parser.add_argument("--doc_num", default=None, type=int,
                       help="Max number of documents per file (None for all)")
    parser.add_argument("--max_length", default=1500, type=int,
                       help="Maximum token length for truncation")
    parser.add_argument("--workers", default=10, type=int,
                       help="Number of worker processes")
    parser.add_argument("--api_url", default="http://127.0.0.1:8000/v1",
                       help="Embedding API server URL")
    return parser.parse_args()


if __name__ == "__main__":
    """
    Before running this script, start the embedding model server first, e.g.:
    vllm serve /models/Qwen3-Embedding-0.6B -dp 8 --gpu-memory-utilization 0.9 --task embed --enforce-eager --disable-custom-all-reduce
    """
    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    log_dir = Path(args.log_dir) if args.log_dir else Path(output_dir) / "log_embed"

    if not os.path.exists(input_dir):
        print(f"Error: input directory does not exist: {input_dir}")
        sys.exit(1)

    log_dir.mkdir(exist_ok=True, parents=True)
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    model_name = args.model
    batch_size = args.batch_size
    doc_num = args.doc_num
    max_length = args.max_length
    workers = args.workers
    api_url = args.api_url

    print("Scanning for JSONL files...")
    jsonl_files = find_jsonl_files(input_dir)
    print(f"Found {len(jsonl_files)} JSONL files")

    if not jsonl_files:
        print("Error: no JSONL files found in the input directory")
        sys.exit(1)

    for jsonl_file in jsonl_files:
        embed_path = get_embedding_path(jsonl_file, input_dir, output_dir)
        embed_path.parent.mkdir(parents=True, exist_ok=True)

    mq = Manager().Queue()
    progress = Manager().Queue()

    pool = Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(mq, progress),
    )

    pool.starmap_async(worker, [(i, api_url, model_name, batch_size, max_length, doc_num, input_dir, log_dir) for i in range(workers)])

    total = 0
    for jsonl_file in jsonl_files:
        embed_path = get_embedding_path(jsonl_file, input_dir, output_dir)
        done_path = get_done_path(jsonl_file, input_dir, log_dir)

        if not done_path.exists():
            mq.put((jsonl_file, embed_path))
            total += 1

    print(f"Processing {total} files")

    for _ in range(workers):
        mq.put(None)

    with tqdm(total=total, desc="Progress") as pbar:
        for _ in range(total):
            progress.get()
            pbar.update(1)

    pool.close()
    pool.join()

    print("All files processed!")
