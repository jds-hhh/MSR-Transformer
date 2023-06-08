# encoding: utf-8
"""



@desc: quantize features to make sure that it could fit into RAM

"""

import argparse
import json

import faiss
import numpy as np
from tqdm import tqdm
from onmt.utils.parse import ArgumentParser
import onmt.opts as opts
import os


def feature_path(data_dir, mode, type="encoder", suffix=""):
    assert type in ["encoder", "decoder"]
    return os.path.join(data_dir, f"{mode}-features", f"all.mmap.{type}{suffix}")

def quantizer_path(data_dir, type="encoder", code_size=None,suffix=""):
    assert type in ["encoder", "decoder"]
    # return os.path.join(data_dir, f"quantizer-{type}{suffix}.npy")  # todo remove .py
    return os.path.join(data_dir, f"quantizer-{type}{suffix}{code_size}.new")

def quantized_feature_path(data_dir, mode, type="encoder", suffix=""):
    assert type in ["encoder", "decoder"]
    return os.path.join(data_dir, f"{mode}-features", f"quantized-feature-{type}{suffix}.new.npy")


def main(opt):

    data_dir = opt.data_dir
    prefix = opt.prefix
    lang = opt.lang
    subset = opt.subset
    code_size = opt.code_size
    chunk_size = opt.chunk_size

    src_lang, tgt_lang = prefix.split("-")
    if src_lang == lang:
        langtype = "encoder"
    elif tgt_lang == lang:
        langtype = "decoder"
    else:
        raise ValueError(f"lang {lang} not in any side of prefix {prefix}")
    info = json.load(open(os.path.join(data_dir, f"{subset}-features", f"all.mmap.{langtype}.json")))
    hidden_size = info["hidden_size"]
    total_tokens = info["num_tokens"]

    # load mmap src features
    feature_mmap_file = feature_path(data_dir, subset, type=langtype, suffix=opt.suffix)
    print(f"Loading mmap features at {feature_mmap_file}...")
    feature_mmap = np.memmap(feature_mmap_file, dtype=np.float32, mode='r',
                             shape=(total_tokens, hidden_size))

    print(f"Train quantized codes on first {chunk_size} features")
    train_features = np.array(feature_mmap[: chunk_size])


    quantizer = faiss.index_factory(hidden_size,opt.index)
    print("Training Product Quantizer")
    quantizer.train(train_features)

    save_path = quantizer_path(data_dir, langtype,code_size,suffix=opt.suffix)
    faiss.write_index(quantizer, save_path)
    print(f"Save quantizer to {save_path}")

    quantized_codes = np.zeros([total_tokens, code_size], dtype=np.uint8)

    # encode
    start = 0
    total_error = 0
    pbar = tqdm(total=total_tokens, desc="Computing codes")
    while start < total_tokens:
        end = min(total_tokens, start + chunk_size)
        x = np.array(feature_mmap[start: end])
        codes = quantizer.sa_encode(x)

        if opt.compute_error:
            x2 = quantizer.sa_decode(codes)
            # compute reconstruction error
            avg_relative_error = ((x - x2)**2).sum() / (x ** 2).sum()
            print(f"Reconstruction error: {avg_relative_error}")
            total_error += avg_relative_error * (end-start)

        quantized_codes[start: end] = codes
        pbar.update(end-start)
        start = end

    if opt.compute_error:
       print(f"Avg Reconstruction error: f{total_error/total_tokens}")

    qt_path = quantized_feature_path(data_dir, subset, langtype, suffix=opt.suffix)
    np.save(qt_path, quantized_codes)
    print(f"Save quantized feature to {qt_path}")

def _get_parser():
    parser = ArgumentParser(description='bulid_ds.py')
    opts.config_opts(parser)
    opts.quantize_features_opt(parser)
    return parser


def cli_main():
    parser = _get_parser()
    opt = parser.parse_args()
    main(opt)
if __name__ == "__main__":
    cli_main()
