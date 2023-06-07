import numpy as np
import faiss
import time
import math
import os
import re
from tqdm import tqdm
import json
from onmt.utils.parse import ArgumentParser
import onmt.opts as opts


def main(opt):
    """main"""
    if not opt.subdirs:
        all_dirs = [d for d in opt.dstore_dir.split(",") if d.strip()]
    else:
        parent_dirs = [d for d in opt.dstore_dir.split(",") if d.strip()]
        all_dirs = []
        for parent_dir in parent_dirs:
            subdirs = os.listdir(opt.dstore_dir)
            for subdir in subdirs:
                d = os.path.join(parent_dir, subdir)
                if os.path.isdir(d):
                    all_dirs.append(d)
    if opt.subdirs_range:
        start, end = opt.subdirs_range.split(",")
        start = int(start)
        end = int(end)
        valid_all_dirs = []
        for d in all_dirs:
            match_idx = re.match("token_(\d+)", os.path.basename(d))
            if match_idx is None:
                continue
            match_idx = int(match_idx.group(1))
            if start <= match_idx < end:
                valid_all_dirs.append(d)
        all_dirs = valid_all_dirs

    print(f"Select {len(all_dirs)} dir to build indexes")

    for dstore_dir in tqdm(all_dirs):
        print(dstore_dir)
        build(dstore_dir, opt)
        #build('data/PD-rm/train_P_data_stores/token_145', opt)

def get_auto_index_type(dstore_size,chunk_size,hidden_size):
    """we choose index type by https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index"""
    dstore_size = min(dstore_size, chunk_size)
    if dstore_size < 3000:
        return "IDMap,,Flat"
    clusters = min(int(4 * math.sqrt(dstore_size)), dstore_size // 30)
    if dstore_size < 30000:
        return "IDMap,,Flat"
    if dstore_size < 10 ** 6:
        # return f"OPQ128_512,IVF{clusters},PQ128"
        return f"OPQ64_{hidden_size},IVF{clusters},PQ64"
    return f"OPQ64_{hidden_size},IVF{clusters},PQ64"  # 我们只用最多20w数据


def build(dstore_dir,opt):
    info = json.load(open(os.path.join(dstore_dir, "info.json")))
    dstore_size, hidden_size, vocab_size, dstore_fp16, val_size = (
        info["dstore_size"],
        info["hidden_size"],
        info.get("vocab_size", None),
        info.get("dstore_fp16", False),
        info.get("val_size", 2),
    )

    if opt.save_dir == '':
        save_dir = dstore_dir
    else:
        save_dir = opt.save_dir +'/'+ dstore_dir.split('/')[-1]

    max_num=opt.chunk_size
    key_file = os.path.join(dstore_dir, "keys.npy")
    key=np.memmap(key_file,dtype= np.float32,mode='r',shape=(dstore_size, hidden_size))

    if max_num<dstore_size:
        key=key[:max_num]
    key=np.array(key.astype(np.float32))
    use_gpu=True

    index_type = get_auto_index_type(dstore_size, max_num, hidden_size)
    metric=faiss.METRIC_INNER_PRODUCT
    index = faiss.index_factory(hidden_size, index_type, metric)

    if opt.use_gpu:
        print("Using gpu for training")
        # we need only a StandardGpuResources per GPU
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        index = faiss.index_cpu_to_gpu(res, opt.gpu, index, co)
        # index = faiss.index_cpu_to_gpu(res, 1, index)
        # co = faiss.GpuMultipleClonerOptions()
        # co.useFloat16 = True
        # index = faiss.index_cpu_to_all_gpus(index=index,co=co)

    start_time = time.time()
    print('Training Index')
    index.train(key)
    print('Training took {} s'.format(time.time() - start_time))

    if use_gpu:
        index = faiss.index_gpu_to_cpu(index)
    print('Writing index after training')
    trained_file = os.path.join(save_dir, f"faiss_store.trained.cosine")
    faiss.write_index(index, trained_file)
    print('Writing index took {} s'.format(time.time() - start_time))

    index = faiss.read_index(trained_file)
    start = index.ntotal

    print(f"start from {start} line, due to pretrained faiss file {trained_file}")
    faiss_file = os.path.join(save_dir, f"faiss_store.cosine")
    #添加训练数据
    if opt.use_chunk and dstore_size > max_num:
        dstore_size = max_num
    start_time = time.time()
    while start < dstore_size:
        # while start < chunk_size:
        print("add with ids")
        end = min(dstore_size, start + opt.chunk_size)
        to_add = np.array(key[start:end])
        if opt.metric == "cosine":
            norm = np.sqrt(np.sum(to_add ** 2, axis=-1, keepdims=True))
            if (norm == 0).any():
                print(f"found zero norm vector in {dstore_dir}")
                norm = norm + 1e-10
            to_add = to_add / norm

        index.add_with_ids(to_add.astype(np.float32), np.arange(start, end))
        start = end

        print(f'Added {index.ntotal} tokens so far')
        faiss.write_index(index, faiss_file)
    print(f"Adding total {index.ntotal} keys")
    print('Adding took {} s'.format(time.time() - start_time))
    print('Writing Index')
    start = time.time()
    faiss.write_index(index, faiss_file)
    print('Writing index took {} s'.format(time.time() - start))
    print(f"Wrote data to {faiss_file}")


def _get_parser():
    parser = ArgumentParser(description='bulid_ds.py')
    opts.config_opts(parser)
    opts.index_build_opt(parser)
    return parser

def cli_main():
    parser = _get_parser()
    opt = parser.parse_args()
    main(opt)
if __name__ == "__main__":
    cli_main()


'''
当OPQ64时， t.index 存储大小为18.74M， 当OPQ32时候， t.index存储大小为12.64M
t,npu 存储大小为390.63M


CPU训练
Training Index
Training took 474.93776392936707 s
Writing index after training
Writing index took 0.09656310081481934 s

GPU训练
OPQ64
Training Index
Training took 473.99483728408813 s
Writing index after training
Writing index took 0.09955811500549316 s

OPQ32,使用一块GPU
Training Index
Training t ook 405.12881660461426 s
Writing index after training
Writing index took 0.5902245044708252 s

OPQ32,使用两块GPU
Training Index
Training took 163.82580614089966 s
Writing index after training
Writing index took 0.01466059684753418 s



OPQ64,使用两块GPU
Training Index
Training took 298.94354343414307 s
Writing index after training
Writing index took 0.021538257598876953 s


'''