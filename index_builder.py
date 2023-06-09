# encoding: utf-8
"""



@version: 1.0
@file: index_builder

"""

import os
import time

import faiss
import numpy as np
import math

from data_store import DataStore



class IndexBuilder:
    """Read DataStore and build faiss index"""

    def __init__(self, dstore_dir: str, output_dir: str = "", use_gpu=False, metric="l2", suffix="",use_chunk=False,use_memory=False):
        self.dstore = DataStore.from_pretrained(dstore_dir, mode="r", warmup=False,use_memory=use_memory)
        self.output_dir = output_dir or dstore_dir
        self.use_gpu = use_gpu
        self.metric = metric
        self.suffix = suffix
        self.use_chunk=use_chunk

    def exists(self):
        return os.path.exists(self.trained_file) and os.path.exists(self.faiss_file)

    @property
    def trained_file(self):
        """get trained index file path"""
        file_path = os.path.join(self.output_dir, f"faiss_store.trained.{self.metric}{self.suffix}")
        return file_path

    @property
    def faiss_file(self):
        """get final index file path"""
        file_path = os.path.join(self.output_dir, f"faiss_store.{self.metric}{self.suffix}")
        return file_path

    def get_auto_index_type(self,chunk_size):
        """we choose index type by https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index"""
        dstore_size = min(self.dstore.dstore_size,chunk_size)
        if dstore_size < 3000:
            return "IDMap,,Flat"
        clusters = min(int(4 * math.sqrt(dstore_size)), dstore_size // 30)
        if dstore_size < 30000:
            return "IDMap,,Flat"
        # if dstore_size < 10**6:
        #     return f"OPQ128_512,IVF{clusters},"
        # return f"OPQ128_512,IVF{clusters}_HNSW32,PQ128"

        #return f"IDMap,IVF{clusters},Flat"      #可以加快索引训练速度，但是不能降低数据存储

        if dstore_size < 10 ** 6:
            # return f"OPQ128_512,IVF{clusters},PQ128"
            return f"OPQ64_{self.dstore.hidden_size},IVF{clusters},PQ64"  # we use 64 here since faiss does not support >64 in gpu mode
        return f"OPQ64_{self.dstore.hidden_size},IVF{clusters},PQ64"  # 我们只用最多20w数据
        return f"OPQ64_{self.dstore.hidden_size},IVF{clusters}_HNSW32,PQ64"

    def build(self, index_type: str, chunk_size=1000000, seed=None, start: int = 0, overwrite=False):
        """build faiss index"""
        if index_type == "auto":
            index_type = self.get_auto_index_type(chunk_size)

        self.train(index_type=index_type, max_num=chunk_size, seed=seed, overwrite=overwrite)
        print('Adding Keys')
        pretrained_file = self.trained_file
        if os.path.exists(self.faiss_file) and not overwrite:
            pretrained_file = self.faiss_file
            print("faiss idx file exists, use it as pretrain idx")

        index = faiss.read_index(pretrained_file)

        if pretrained_file == self.faiss_file:
            start = index.ntotal
            print(f"start from {start} line, due to pretrained faiss file {self.faiss_file}")

        dstore_size = self.dstore.dstore_size
        if self.use_chunk and dstore_size>chunk_size:
            dstore_size=chunk_size
        start_time = time.time()
        while start < dstore_size:
        #while start < chunk_size:
            print("add with ids")
            end = min(dstore_size, start + chunk_size)
            to_add = np.array(self.dstore.keys[start:end])
            if self.metric == "cosine":
                norm = np.sqrt(np.sum(to_add ** 2, axis=-1, keepdims=True))
                if (norm == 0).any():
                    print(f"found zero norm vector in {self.dstore.dstore_dir}")
                    norm = norm + 1e-10
                to_add = to_add / norm

            index.add_with_ids(to_add.astype(np.float32), np.arange(start, end))
            start = end

            print(f'Added {index.ntotal} tokens so far')
            faiss.write_index(index, self.faiss_file)


        print(f"Adding total {index.ntotal} keys")
        print('Adding took {} s'.format(time.time() - start_time))
        print('Writing Index')
        start = time.time()
        faiss.write_index(index, self.faiss_file)
        print('Writing index took {} s'.format(time.time() - start))
        print(f"Wrote data to {self.faiss_file}")

    def train(self, index_type, max_num=None, seed=None, overwrite=False):
        """train clusters with sampled data"""
        hidden_size, dstore_size = self.dstore.hidden_size, self.dstore.dstore_size
        if os.path.exists(self.trained_file) and not overwrite:
            print("trained file already exists. Use existing file.")
            return

        metric = faiss.METRIC_L2 if self.metric == "l2" else faiss.METRIC_INNER_PRODUCT
        index = faiss.index_factory(hidden_size, index_type, metric)

        if self.use_gpu:
            print("Using gpu for training")
            # we need only a StandardGpuResources per GPU
            # res = faiss.StandardGpuResources()
            # co = faiss.GpuClonerOptions()
            #
            # # here we are using a 64-byte PQ, so we must set the lookup tables to
            # # 16 bit float (this is due to the limited temporary memory).
            # co.useFloat16 = True
            # index = faiss.index_cpu_to_gpu(res, 1, index, co)
            multi_gpu=False
            if multi_gpu:
                co = faiss.GpuMultipleClonerOptions()
                co.useFloat16 = True
                index = faiss.index_cpu_to_all_gpus(index=index, co=co)
            else:
                res = faiss.StandardGpuResources()
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
                index = faiss.index_cpu_to_gpu(res, 0, index, co)
        if self.dstore.dstore_size < max_num:
            sample_keys = np.array(self.dstore.keys.astype(np.float32))
        else:
            np.random.seed(seed)
            max_num = max_num or self.dstore.dstore_size
            # random_sample = np.random.choice(np.arange(dstore_size),
            #                                  size=[min(max_num, dstore_size)],
            #                                  replace=False)
            # sample_keys = self.dstore.keys[random_sample].astype(np.float32).copy()  # [N, d]
            sample_keys = np.array(self.dstore.keys[: max_num].astype(np.float32)) # [N, d]
            if self.metric == "cosine":
                norm = np.sqrt(np.sum(sample_keys ** 2, axis=-1, keepdims=True))
                if (norm == 0).any():
                    print(f"found zero norm vector in {self.dstore.dstore_dir}")
                    norm = norm + 1e-10
                sample_keys = sample_keys / norm
        start = time.time()
        print('Training Index')
        index.train(sample_keys)
        print('Training took {} s'.format(time.time() - start))
        if self.use_gpu:
            index = faiss.index_gpu_to_cpu(index)

        print('Writing index after training')
        start = time.time()
        faiss.write_index(index, self.trained_file)
        print('Writing index took {} s'.format(time.time() - start))

