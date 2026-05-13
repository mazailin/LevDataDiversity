from multiprocessing import Pool, Manager
from tqdm import tqdm
from pathlib import Path
from itertools import batched
import numpy as np
from collections import defaultdict
from argparse import ArgumentParser
import os
from Lev import LevSelector
import sys
import math
import shutil
import time

def get_sampler(method, embed, index_list, file_id, worker_id, batch_size=1024, step_size=12):
    """Get sampler"""
    sampler = LevSelector(alpha, batch_size, step_size, worker_id % 8)

    if worker_id%8==0:
        pbar = tqdm(total=np.ceil(embed.shape[0]/batch_size).astype(int).astype(object), desc=f'file {file_id}', leave=False)
    else:
        pbar = None

    sampled_ind_list=[[] for _ in range(len(alpha))]
    for ii in range(0, embed.shape[0]//batch_size):
        cur_ind = index_list[ii*batch_size:(ii+1)*batch_size]
        cur_embed = embed[cur_ind]
        subset = sampler.hunger_select(cur_embed)
        for jj in range(len(alpha)):
            sampled_ind_list[jj].append(cur_ind[subset[jj]])
        pbar is not None and pbar.update()
    pbar is not None and pbar.close()
    return [np.concatenate(sampled_ind) for sampled_ind in sampled_ind_list]

def load_embed(file_names, dataset, get_embed=True):
    """Load embedding vectors - adapt to arbitrary directory structure"""
    track_list = []
    file_size_list = [0]

    for file_name in file_names:
        # Build embedding file path based on relative path
        file_path = os.path.join(embed_folder, dataset, file_name)
        file_size = os.path.getsize(file_path) // 4 // embed_dim
        file_size_list.append(file_size_list[-1] + file_size)
        # Use file name as identifier instead of integer ID
        track_list += list(zip([file_name] * file_size, range(file_size)))

    if get_embed:
        embed = np.empty((file_size_list[-1], embed_dim), dtype=np.float32)
        def load_single_file(ii):
            file_name = file_names[ii]
            st, ed = file_size_list[ii], file_size_list[ii+1]
            file_path = os.path.join(embed_folder, dataset, file_name)
            embed[st:ed] = np.fromfile(file_path, dtype=np.float32).reshape(-1, embed_dim)

        for ii in range(len(file_names)):
            load_single_file(ii)
    else:
        embed=None

    return embed, track_list

def write_files(select_pool_list, track_list, file_names, dataset):
    """
    Write selected files - preserve same directory structure as input
    select_pool: selected indices (allowed to repeat)
    track_list: the original pos (file_name, doc_id) of each index
    file_names: current file names
    dataset: dataset relative path
    """
    for i in range(len(alpha)):
        select_pool=select_pool_list[i] # selected indices of alpha[i] selection ratio

        rng = np.random.default_rng(seed=0)
        select_pool = rng.permutation(select_pool)
        select_track = [track_list[x] for x in select_pool] # can repeat
        # output
        grouped = defaultdict(list)
        for k, v in select_track:
            grouped[k].append(v)

        # Create corresponding output directory for each dataset, preserving nested structure
        dataset_output_path = index_folder_list[i] / dataset
        dataset_output_path.mkdir(exist_ok=True, parents=True)

        for file_name in file_names:
            # Replace .embedding with .index as index file name
            output_file = os.path.join(dataset_output_path / file_name.replace('.embedding', '.index'))
            with open(output_file, 'w') as f:
                for item in grouped[file_name]:
                    f.write(f'{item}\n')

def work(file_names, worker_id, dataset):
    """Work function"""
    if strategy == 'lev':
        embed, track_list = load_embed(file_names, dataset, get_embed=True)
        rng = np.random.default_rng(seed=0)
        index_list = rng.permutation(embed.shape[0])
        select_pool = get_sampler(strategy, embed, index_list, file_names[0], worker_id, batch_size=batch_size, step_size=step_size)

    elif strategy=='random':
        embed, track_list = load_embed(file_names, dataset, get_embed=False)
        total = len(track_list)
        rng = np.random.default_rng(seed=0)
        select_pool = [rng.choice(total, size=math.ceil(aa * total / 100), replace=False) for aa in alpha]

    else:
        print(f'Method: {strategy} not registered! Use random sampling!')
        embed, track_list = load_embed(file_names, dataset, get_embed=False)
        total = len(track_list)
        select_num = math.ceil(alpha[0] * total / 100)
        rng = np.random.default_rng(seed=0)
        select_pool = [rng.choice(total, size=select_num, replace=False)]

    write_files(select_pool, track_list, file_names, dataset)

def get_done_path(file_name, dataset):
    """Get done marker file path for a single file"""
    return log_dir / f'{dataset.replace("/","_")}_{file_name}.done'

def init_worker(mq_, progress_):
    """Initialize worker process"""
    global mq, progress
    mq = mq_
    progress = progress_

def worker(worker_id):
    """Worker process function"""
    global mq, progress
    while True:
        task = mq.get()
        if task is None: # No more tasks, terminate
            break

        file_names, dataset = task

        # Check if there are unfinished files
        pending_files = []
        for file_name in file_names:
            done_path = get_done_path(file_name, dataset)
            if not done_path.exists():
                pending_files.append(file_name)

        # If no unfinished files, continue to next task
        if not pending_files:
            progress.put((file_names, dataset))
            continue

        # Only process unfinished files
        work(pending_files, worker_id, dataset)

        # Create done marker for each completed file
        for file_name in pending_files:
            done_path = get_done_path(file_name, dataset)
            done_path.touch()

        progress.put((file_names, dataset))

def get_available_datasets():
    """Use os.walk to get all directories containing embedding files"""
    embed_path = embed_folder
    datasets = set()

    if os.path.exists(embed_path):
        for root, dirs, files in os.walk(embed_path):
            # Check if current directory has embedding files
            embedding_files = [f for f in files if f.endswith('.embedding')]
            if embedding_files:
                # Get relative path to embed_path
                rel_path = os.path.relpath(root, embed_path)
                if rel_path == '.':
                    # If current directory is embed_path itself, use empty string
                    rel_path = ''
                datasets.add(rel_path)

    return sorted(datasets)

def get_embedding_files_for_dataset(dataset):
    """Get list of embedding file names for the specified dataset"""
    if dataset == '':
        dataset_path = embed_folder
    else:
        dataset_path = os.path.join(embed_folder, dataset)

    if not os.path.exists(dataset_path):
        return []

    # Get all embedding files, sorted by file name
    embedding_files = [f for f in os.listdir(dataset_path) if f.endswith('.embedding')]
    return sorted(embedding_files)

if __name__=="__main__":
    # Parse command line arguments
    parser = ArgumentParser(description='Data selection script')
    parser.add_argument('--embed_folder', type=str, required=True,
                       help='Path to the folder containing embedding files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for selected indices')
    parser.add_argument('--embed_dim', type=int, default=1024,
                       help='Embedding dimension (default: 1024)')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Size of the batch to select data within')
    parser.add_argument('--strategy', type=str, default='lev',
                       help='Selection strategy: lev or random (default: lev)')
    parser.add_argument('--selection_ratio', type=str, default='20',
                       help='Selection ratio(s) in percentage, e.g. 40 or 40,50,60 (default: 20)')
    parser.add_argument('--step_size', type=int, default=2,
                       help='Step size for lev strategy (default: 2)')
    parser.add_argument('--chunk_size', type=int, default=1,
                       help='Chunk size for processing (default: 1)')
    parser.add_argument('--num_files', type=int, default=None,
                       help='Num of files to process. If None, process all.')

    args = parser.parse_args()

    strategy = args.strategy
    embed_folder = args.embed_folder
    output_dir = args.output_dir
    embed_dim = args.embed_dim
    batch_size = args.batch_size
    chunk_size = args.chunk_size
    num_files = args.num_files

    # Parse selection ratios
    ratio_str = args.selection_ratio
    if ',' in ratio_str:
        alpha = sorted([float(a) for a in ratio_str.split(',')])
    else:
        alpha = [float(ratio_str)]

    # Set step_size
    if strategy == 'lev':
        step_size = args.step_size
    else:
        step_size = None

    print(f"Embedding folder: {embed_folder}")
    print(f"Output directory: {output_dir}")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Strategy: {strategy}")
    print(f"Selection ratio: {alpha}")
    print(f"Step size: {step_size}")
    print(f"Chunk size: {chunk_size}")

    # Construct method name for output directory
    ratio_part = ','.join([str(int(a)) for a in alpha])
    if strategy == 'lev':
        method = f'{strategy}_{ratio_part}_{step_size}_{batch_size}'
    else:
        method = f'{strategy}_{ratio_part}_{batch_size}'
    print(f'Current method: {method}')

    # Multi-process setup, 8 GPUs by default
    workers = 8

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    index_folder_list=[]
    for aa in alpha:
        aa_ratio = str(int(aa))
        if strategy == 'lev':
            this_dir_name = f'{strategy}_{aa_ratio}_{step_size}_{batch_size}'
        else:
            this_dir_name = f'{strategy}_{aa_ratio}_{batch_size}'
        base_dir = os.path.join(output_dir, this_dir_name)
        if os.path.isdir(base_dir):
            shutil.rmtree(base_dir)
            print(f'delete previous dir: {base_dir}')
        
        index_dir = Path(base_dir) / 'index'
        index_dir.mkdir(exist_ok=True, parents=True)
        index_folder_list.append(index_dir)
    
    log_dir = Path(output_dir)/ f'{method}' / 'log'
    log_dir.mkdir(exist_ok=True, parents=True)

    # Get all available datasets
    datasets = get_available_datasets()
    print(f"Available datasets: {datasets}")

    if not datasets:
        print("No dataset directories with embedding files found")
        sys.exit(1)

    # Create tasks for each dataset
    mq = Manager().Queue()
    progress = Manager().Queue()

    # Initialize worker pool
    pool = Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(mq, progress)
    )
    pool.map_async(worker, range(workers)) # Start workers subprocesses

    # Dispatch tasks for each dataset
    total_tasks = 0
    total_files = 0
    for dataset in datasets:
        # Get all embedding file names for this dataset
        if num_files:
            embedding_files = get_embedding_files_for_dataset(dataset)[:num_files]
        else:
            embedding_files = get_embedding_files_for_dataset(dataset)
        if len(embedding_files) == 0:
            print(f"No embedding files found for dataset: {dataset}")
            continue

        print(f"Processing dataset: {dataset}, files: {len(embedding_files)}")
        total_files+=len(embedding_files)

        # Group embedding files by chunk_size
        task_list = list(batched(embedding_files, chunk_size))

        # Dispatch tasks (always dispatch all tasks, worker checks which files need processing internally)
        for file_names in task_list:
            mq.put((file_names, dataset))
            total_tasks += 1

    # Add termination signals
    for _ in range(workers):
        mq.put(None)

    # Progress bar
    print(f"Total files to process: {total_files}")

    pbar = tqdm(total=total_tasks, desc='  Processed chunks: ')
    for _ in range(total_tasks):
        task = progress.get()
        pbar.update(1)

    pool.close()
    pool.join()
    print("Data selection completed!")