"""
project: kNN-IME
file: decoder_store
author: JDS
create date: 2021/9/23 17:09
description: 
"""

# encoding: utf-8
"""
@version: 1.0
@file: data_store
"""

import os
import numpy as np
import time
import json
import math
from tqdm import tqdm


def warmup_mmap_file(path, n=1000, verbose=True):
    megabytes = 1024 * 1024
    print(f"Warming up file {path}")
    total = math.floor(os.path.getsize(path)/megabytes)
    pbar = tqdm(total=total, desc=f"Warm up") if verbose else None
    with open(path, 'rb') as stream:
        while stream.read(n * megabytes):
            if pbar is not None:
                update = n
                if update + pbar.n > total:
                    update = total - pbar.n
                pbar.update(update)

class DecoderStore:
    """
    DataStore to save hidden states
    Attributes:
        keys: [dstore_size, hidden_size]
        vals: [dstore_size, 2], sent-idx and token-idx
    """

    def __init__(self, dstore_size: int, hidden_size: int, dstore_dir: str, vocab_size: int = None, mode="r",freq=0,
                 dstore_fp16: bool = False, no_load_keys: bool = False, use_memory: bool = False, warmup: bool = True,
                 val_size: int = 1):
        self.dstore_size = dstore_size
        self.hidden_size = hidden_size
        self.dstore_dir = dstore_dir
        self.vocab_size = vocab_size
        self.no_load_keys = no_load_keys
        self.dstore_fp16 = dstore_fp16
        self.val_size = val_size
        self.freq=freq
        os.makedirs(dstore_dir, exist_ok=True)
        if not no_load_keys:
            key_file = os.path.join(dstore_dir, "decoder_keys.npy")
            if mode == "r" and warmup:
                warmup_mmap_file(key_file, verbose=False)
            self.keys = np.memmap(key_file,
                                  dtype=np.uint8,
                                  mode=mode,
                                  shape=(dstore_size, hidden_size))
        val_file = os.path.join(dstore_dir, "decoder_vals.npy")
        if mode == "r" and warmup:
            warmup_mmap_file(val_file, verbose=False)
        self.vals = np.memmap(val_file,
                              dtype=np.int16 if self.dstore_fp16 else np.int32,
                              mode=mode,
                              shape=(dstore_size, val_size))

        sent_ids_file = os.path.join(dstore_dir, "sent_ids.npy")
        if mode == "r" and warmup:
            warmup_mmap_file(sent_ids_file, verbose=False)
        self.sent_ids = np.memmap(sent_ids_file,
                              dtype=np.int16 if self.dstore_fp16 else np.int32,
                              mode=mode,
                              shape=(dstore_size))

        if use_memory and mode == "r":
            start = time.time()
            if not self.no_load_keys:
                self.memory_keys = np.zeros((self.dstore_size, self.hidden_size), dtype=self.keys.dtype)
                self.memory_keys = self.keys[:]
                self.keys = self.memory_keys
            self.vals = np.array(self.vals)
            print('Loading to memory took {} s'.format(time.time() - start))

    def save_info(self):
        """save information of datastore"""
        json.dump(self.info, open(os.path.join(self.dstore_dir, "decoder_info.json"), "w"),
                  sort_keys=True, indent=4, ensure_ascii=False)

    @staticmethod
    def exists(dstore_dir):
        return (
            os.path.exists(os.path.join(dstore_dir, "decoder_keys.npy")) and
            os.path.exists(os.path.join(dstore_dir, "decoder_vals.npy"))
        )

    def close(self):
        self.keys._mmap.close()
        self.vals._mmap.close()


    @staticmethod
    def remove(dstore_dir):
        if os.path.exists(os.path.join(dstore_dir, "decoder_keys.npy")):
            os.remove(os.path.join(dstore_dir, "decoder_keys.npy"))
        if os.path.exists(os.path.join(dstore_dir, "decoder_vals.npy")):
            os.remove(os.path.join(dstore_dir, "decoder_vals.npy"))
        if os.path.exists(os.path.join(dstore_dir, "sent_ids.npy")):
            os.remove(os.path.join(dstore_dir, "sent_ids.npy"))
        if os.path.exists(os.path.join(dstore_dir, "decoder_info.json")):
            os.remove(os.path.join(dstore_dir, "decoder_info.json"))


    @property
    def info(self):
        """get info"""
        id=min(self.freq,self.dstore_size)
        info = {
            "dstore_size": self.dstore_size,
            "hidden_size": self.hidden_size,
            "vocab_size": str(self.vocab_size),
            "dstore_fp16": self.dstore_fp16,
            "val_size": self.val_size,
            "freq":self.freq,
            "id":id
        }
        return info

    # @classmethod
    # def from_pretrained(cls, dstore_dir: str, no_load_keys=False, use_memory=False, mode="r", warmup=True):
    #     """load DataStore from pretrained file"""
    #     info = json.load(open(os.path.join(dstore_dir, "decoder_info.json")))
    #     dstore_size, hidden_size, vocab_size, dstore_fp16, val_size = (
    #         info["dstore_size"],
    #         info["hidden_size"],
    #         info.get("vocab_size", None),
    #         info.get("dstore_fp16", False),
    #         info.get("val_size", 2),
    #     )
    #     return cls(dstore_size=dstore_size, hidden_size=hidden_size, dstore_dir=dstore_dir, dstore_fp16=dstore_fp16,
    #                vocab_size=vocab_size, no_load_keys=no_load_keys, mode=mode, use_memory=use_memory,
    #                warmup=warmup, val_size=val_size)

