"""
project: kNN-IME
file: build_ds
author: JDS
create date: 2021/9/10 20:08
description: 
"""
from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
import onmt.inputters as inputters
import torch
import onmt.decoders.ensemble
import numpy as np
from tqdm import tqdm
import json
import os
from onmt.translate.translator import build_translator
import warnings
from data_store import *
from multiprocessing import dummy
from collections import Counter

def max_tok_len(new, count, sofar):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src/tgt tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src and tgt length in the current batch
    global max_src_in_batch  # this is a hack
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        # max_tgt_in_batch = 0
    # Src: [<bos> w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(new.src[0]) + 2)
    # Tgt: [w1 ... wM <eos>]
    src_elements = count * max_src_in_batch
    return src_elements

def cal_dataSize(examples):
    src_size=[]
    tgt_size=[]
    for example in tqdm(examples,desc='calculate sentence length'):
        srclen=len(example.src[0])
        tgtlen=srclen     #tgt的长度要比src要长1，因为在接下来保存的数据中我们要保存</s>
        src_size.append(srclen)
        tgt_size.append(tgtlen)
    return src_size,tgt_size


class Build_DS():
    def __init__(self, model,fields,opt):
        self.model = model
        self.fields = fields
        self.opt=opt
        src_field=dict(self.fields)['src'].base_field
        self._src_vocab=src_field.vocab
        tgt_field = dict(self.fields)["tgt"].base_field
        self._tgt_vocab = tgt_field.vocab
        self._tgt_eos_idx = self._tgt_vocab.stoi[tgt_field.eos_token]
        self._tgt_pad_idx = self._tgt_vocab.stoi[tgt_field.pad_token]
        self._tgt_bos_idx = self._tgt_vocab.stoi[tgt_field.init_token]
        self._tgt_unk_idx = self._tgt_vocab.stoi[tgt_field.unk_token]
        self._tgt_vocab_len = len(self._tgt_vocab)
        self.data_type=opt.data_type
        self._filter_pred=None
        self._use_cuda=True if opt.gpu!=-1 else False
        self._dev = (
            torch.device("cuda", opt.gpu)
            if self._use_cuda
            else torch.device("cpu")
        )
        self.src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
        self.tgt_reader = inputters.str2reader["text"].from_opt(opt)

    def token_len_offsets_path(self, data_dir, mode,textType):
        return os.path.join(data_dir, f"{mode}.{textType}.len_offsets.npy")


    def extract_feature_mmap(self,src,tgt=None,batch_size=None,batch_type="sents",):
        src_data = {"reader": self.src_reader, "data": src}
        tgt_data = {"reader": self.tgt_reader, "data": tgt}
        _readers, _data = inputters.Dataset.config(
            [("src", src_data), ("tgt", tgt_data)]
        )

        self.data = inputters.Dataset(
            self.fields,
            readers=_readers,
            data=_data,
            sort_key=inputters.str2sortkey[self.data_type],
            filter_pred=self._filter_pred,
        )

        self.data_iter = inputters.OrderedIterator(
            dataset=self.data,
            device=self._dev,
            batch_size=batch_size,
            batch_size_fn=max_tok_len if batch_type == "tokens" else None,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False,
        )

        src_len_offsets_file = self.token_len_offsets_path(self.opt.data_dir,self.opt.mode,'src')
        if os.path.exists(src_len_offsets_file):
            self.src_len_offsets=np.load(src_len_offsets_file,allow_pickle=True)
        else:
            src_dataSize,tgt_dataSize = cal_dataSize(self.data.examples)  # 计算每一个句子的长度
            src_len_offsets = np.cumsum(src_dataSize)
            self.src_len_offsets = np.insert(src_len_offsets, 0, 0)
            np.save(src_len_offsets_file,  self.src_len_offsets)

        self.total_src_tokens = self.src_len_offsets[-1].item()  # 求得src的总tokens数目

        tgt_len_offsets_file = self.token_len_offsets_path(self.opt.data_dir, self.opt.mode, 'tgt')
        if os.path.exists(tgt_len_offsets_file):
            self.tgt_len_offsets=np.load(tgt_len_offsets_file,allow_pickle=True)
        else:
            tgt_len_offsets = np.cumsum(tgt_dataSize)
            self.tgt_len_offsets = np.insert(tgt_len_offsets, 0, 0)
            np.save(tgt_len_offsets_file, self.tgt_len_offsets)
        self.total_tgt_tokens = self.tgt_len_offsets[-1].item()


        self.model.eval()
        self.hidden_size = self.model.decoder.embeddings.embedding_size

        store_encoder = self.opt.store_encoder
        store_decoder = self.opt.store_decoder

        features_file=os.path.join(self.opt.data_dir, f"{self.opt.gen_subset}-features")


        if not os.path.exists(features_file):
            if store_encoder:
                os.makedirs(features_file, exist_ok=True)
                src_mmap_file = os.path.join(features_file, f"all.mmap.encoder")
                src_mmap_info_file = os.path.join(features_file, f"all.mmap.encoder.json")
                mode = "r+" if os.path.exists(src_mmap_file) else "w+"
                src_mmap_features = np.memmap(src_mmap_file, dtype=np.float32, mode=mode,
                                              shape=(self.total_src_tokens, self.hidden_size))
                json.dump({"hidden_size": self.hidden_size, "num_tokens": self.total_src_tokens},
                          open(src_mmap_info_file, "w"), indent=4, sort_keys=True)

            if store_decoder:
                os.makedirs(features_file, exist_ok=True)
                tgt_mmap_file = os.path.join(features_file, f"all.mmap.decoder{self.opt.suffix}")
                tgt_mmap_info_file = os.path.join(features_file, f"all.mmap.decoder.json")
                mode = "r+" if os.path.exists(tgt_mmap_file) else "w+"
                tgt_mmap_features = np.memmap(tgt_mmap_file, dtype=np.float32, mode=mode,
                                              shape=(self.total_tgt_tokens, self.hidden_size))
                json.dump({"hidden_size": self.hidden_size, "num_tokens": self.total_tgt_tokens},
                          open(tgt_mmap_info_file, "w"), indent=4, sort_keys=True)

            with torch.no_grad():
                for batch in tqdm(self.data_iter, desc='extract feature to mmap'):
                    src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                        else (batch.src, None)
                    tgt = batch.tgt
                    dec_in=tgt[:-1]
                    encoder_features, memory_bank, lengths = self.model.encoder(src, src_lengths)
                    self.model.decoder.init_state(src, memory_bank, encoder_features)
                    decoder_features, attns = self.model.decoder(dec_in, memory_bank,
                                                  memory_lengths=lengths,with_align=False)
                    encoder_features=memory_bank.cpu().numpy()  #这里的memory在openNMT中是编码器最后编码的状态
                    decoder_features = decoder_features.cpu().numpy()
                    sorted_ids,sorted_idxs=batch.indices.sort()
                    for example_id,i in zip(sorted_ids,sorted_idxs):
                        if store_encoder:
                            src_sent_offset=self.src_len_offsets[example_id]
                            src_len= src_dataSize[example_id]
                            assert src_len == self.src_len_offsets[example_id+1]-src_sent_offset #判断句子长度对齐
                            src_mmap_features[src_sent_offset:src_sent_offset+src_len]=encoder_features[:src_len,i,:]

                        if store_decoder:
                            tgt_sent_offset = self.tgt_len_offsets[example_id]
                            tgt_len = tgt_dataSize[example_id]
                            assert tgt_len == self.tgt_len_offsets[example_id + 1] - tgt_sent_offset  # 判断句子长度对齐
                            tgt_mmap_features[tgt_sent_offset:tgt_sent_offset + tgt_len] = decoder_features[:tgt_len, i, :]

                    # del encoder_features
                    # del decoder_features


    def load_token_2d_offsets(self,data_dir, mode, lang,freq,all=False,max_sent=0):
        """
           build or load cached token 2d offsets
           Returns:
               token_2d_offsets:
                   if all=False, it is a list of token offsets, grouped by token idx.
                   token_2d_offsets[token_idx] is an array of shape [token_freq, 2],
                   which contains the sentence indexes and intra-sentence offsets where token_idx appears in dataset
                   if all = True, it is an array of shape [num_tokens, 2]
           """
        cache_file = os.path.join(data_dir, f"{mode}.P.2d_offsets.npy")
        if os.path.exists(cache_file):
            print(f"Loading token 2d-offsets from {cache_file}")
            token_2d_offsets = np.load(cache_file, allow_pickle=True)
            return token_2d_offsets
        #token_2d_offsets = [np.zeros([freq[key], 2], dtype=np.int32) for key in src_dict]
        token_2d_offsets = [np.zeros([f, 2], dtype=np.int32) for f in freq]
        fill_offsets = np.zeros([sum(freq)], dtype=np.int32)
        offset = 0
        for sent_idx in tqdm(range(max_sent), desc="Gathering token offsets"):
            src_ids = self.data.examples[sent_idx].src[0]
            for intra_offset, token in enumerate(src_ids):
                token_idx=self._src_vocab.stoi[token]
                fill_offset = fill_offsets[token_idx]
                if fill_offset >= freq[token_idx]:
                    print(f"token count of {token_idx} exceeds argument freq {freq[token_idx]}, ignore it")
                    continue
                token_2d_offsets[token_idx][fill_offset][0] = sent_idx
                token_2d_offsets[token_idx][fill_offset][1] = intra_offset
                fill_offsets[token_idx] += 1
            offset += len(src_ids)
        np.save(cache_file, token_2d_offsets)
        print(f"Saved token 2d-offsets to {cache_file}")
        return token_2d_offsets


    def get_freq(self):
        freq = [0 for i in range(len(self._src_vocab.itos))]
        for example in tqdm(self.data.examples, desc='calculate freqs'):
            for s in example.src[0]:
                freq[self._src_vocab.stoi[s]]+=1
        return freq

    def build_token_dstores(self,data_dir, subset="train", prefix="P", src_lang="P",
                            workers=1, token_start=0, token_end=0,
                            offset_start=0, offset_end=0, offset_chunk=0,
                            max_sent=0, use_memory=False, ):
        if not use_memory:
            offset_chunk = 0  # don't need to use chunk if do not use memory to load data
        freq_file = os.path.join(data_dir, subset + '.' + src_lang  + '.freq')

        if os.path.exists(freq_file):
            with open(freq_file,'r',encoding='utf-8') as f:
                freq=eval(f.read())
        else:
            if subset=='train':
                freq=self._src_vocab.freqs
                freq = list(freq.values())
                freq.insert(0, 0)  # 插入"<unk>"频率0
                freq.insert(0, 0)  # 插入"<blank>"频率0
            else:
                freq=self.get_freq()
        src_dict=self._src_vocab.itos
        max_sent = min(max_sent, len(self.data.examples)) if max_sent else len(self.data.examples)
        offset_end = min(offset_end, self.total_src_tokens) if offset_end else self.total_src_tokens
        token_2d_offsets = self.load_token_2d_offsets(data_dir, subset, src_lang,freq,max_sent=max_sent)
        assert all(x.shape[0] == f for x, f in zip(token_2d_offsets, freq)), \
            "offsets shape should be consistent with frequency counts"

        src_mmap_file = os.path.join(data_dir, f"{subset}-features", f"all.mmap.encoder")
        src_mmap_features = np.memmap(src_mmap_file, dtype=np.float32, mode='r',
                                      shape=(self.total_src_tokens, self.hidden_size))

        token_end = min(token_end, len(src_dict)) if token_end else len(src_dict)
        pbar = tqdm(total=self.total_src_tokens, desc="Building Datastore for each token")

        if not os.path.exists(freq_file):
            with open(freq_file,'w',encoding='utf-8') as f:
                f.write(freq.__str__())

        def run(features, o_start, o_end):
            if use_memory:
                print("Loading feature to memory")
                features = np.array(features[o_start: o_end])

            def build_dstore(token_idx):
                """build data store for token_idx"""

                dstore_dir = os.path.join(data_dir,
                                          f"{subset}_{src_lang}_data_stores",
                                          f"token_{token_idx}")
                batch_size = 1024

                sent_ids = token_2d_offsets[token_idx][:, 0]
                if len(sent_ids)==0:
                    return
                token_offsets = token_2d_offsets[token_idx][:, 1]
                positions = self.src_len_offsets[sent_ids] + token_offsets

                mask = np.logical_and(positions >= o_start, positions < o_end)
                dstore_offset = 0
                if not any(mask):  #全部mask为False则return
                    return
                while not mask[dstore_offset]:
                    dstore_offset += 1
                sent_ids = sent_ids[mask]
                token_offsets = token_offsets[mask]
                positions = positions[mask]

                mode = "r+" if DataStore.exists(dstore_dir) else "w+"
                datastore = DataStore(dstore_size=max(freq[token_idx], 1), vocab_size=0,
                                      hidden_size=self.hidden_size,
                                      dstore_dir=dstore_dir,
                                      mode=mode)
                total = positions.shape[0]
                s = 0
                while s < total:
                    end = min(s + batch_size, total)
                    batch_positions = positions[s: end]
                    batch_sent_ids = sent_ids[s: end]
                    datastore.keys[dstore_offset + s: dstore_offset + end, :] = features[batch_positions - o_start]
                    datastore.vals[dstore_offset + s: dstore_offset + end, 0] = batch_sent_ids
                    datastore.vals[dstore_offset + s: dstore_offset + end, 1] = token_offsets[s: end]
                    pbar.update(end - s)
                    s = end

                datastore.save_info()

            if workers <= 1:
                for token_idx in range(token_start, token_end):
                    if token_idx==246:
                        a=1
                    build_dstore(token_idx)

            # multi-threading
            else:
                pool = dummy.Pool(self.opt.workers)
                jobs = []
                for token_idx in range(token_start, token_end):
                    job = pool.apply_async(func=build_dstore,
                                           kwds={"token_idx": token_idx})
                    jobs.append(job)

                pool.close()
                pool.join()

        if offset_chunk == 0:
            run(src_mmap_features, offset_start, offset_end)
        else:
            start = offset_start
            while start < offset_end:
                tmp_end = min(start + offset_chunk, offset_end)
                print(f"Building datastore from offset {start} to {tmp_end}")
                run(src_mmap_features, start, tmp_end)
                start = tmp_end

def main(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size)
    shard_pairs = zip(src_shards, tgt_shards)
    load_test_model = (
        onmt.decoders.ensemble.load_test_model
        if len(opt.models) > 1
        else onmt.model_builder.load_test_model
    )
    fields, model, model_opt = load_test_model(opt)
    build_Ds = Build_DS(model, fields, opt)
    for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
        logger.info("Translating shard %d." % i)
        build_Ds.extract_feature_mmap(
            src=src_shard,
            tgt=tgt_shard,
            batch_size=opt.batch_size,
            batch_type=opt.batch_type,
        )

        build_Ds.build_token_dstores(data_dir=opt.data_dir, prefix=opt.prefix, src_lang=opt.lang, subset=opt.mode,
                            workers=opt.workers, token_start=opt.start, token_end=opt.end,
                            max_sent=opt.max_sent, use_memory=opt.use_memory,
                            offset_start=opt.offset_start, offset_end=opt.offset_end,
                            offset_chunk=opt.offset_chunk)


def _get_parser():
    parser = ArgumentParser(description='bulid_ds.py')
    opts.config_opts(parser)
    opts.translate_opts(parser)
    opts.build_ds_opts(parser)
    return parser

def cli_main():
    parser = _get_parser()
    opt = parser.parse_args()
    main(opt)
if __name__ == "__main__":
    cli_main()

