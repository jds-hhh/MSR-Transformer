"""
project: kNN-IME
file: knn_nmt_dataset
author: JDS
create date: 2021/9/26 20:30
description: 
"""

from onmt.pq_wrapper import TorchPQCodec
from multiprocessing import dummy
from numba import jit
import faiss
import numpy as np
from functools import lru_cache
import os
import math
from tqdm import tqdm
import torch


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


class KNNNMTDataset():

    def __init__(self,opt=None,fields=None,pair_dataset=None,src_len_offsets=None,
                 token_neighbors=None,neighbor_tgt_feature=None,max_neighbors_per_token=None,
            train_dataset=None,nsrc_sent_offsets=None,ntgt_sent_offsets=None,
                 decoder_quantizer=None,src_token_freq=None,sort_key=None):
        super(KNNNMTDataset, self).__init__()
        self.opt=opt
        self.fields=fields
        src_field = dict(self.fields)['src'].base_field
        self._src_vocab = src_field.vocab
        tgt_field = dict(self.fields)["tgt"].base_field
        self._tgt_vocab = tgt_field.vocab
        self._tgt_num=len(self._tgt_vocab.itos)

        self.pair_dataset=pair_dataset
        self.src_len_offsets=src_len_offsets
        self.token_neighbors=token_neighbors
        self.neighbor_tgt_feature=neighbor_tgt_feature
        self.max_neighbors_per_token=max_neighbors_per_token
        self.train_dataset=train_dataset
        self.decoder_quantizer=decoder_quantizer
        self.gpu=self.opt.gpu
        if self.gpu!=-1 and self.decoder_quantizer is not None:
            self.decoder_quantizer.cuda(self.gpu)
        self.src_token_freq=src_token_freq
        self.nsrc_sent_offsets=nsrc_sent_offsets
        self.ntgt_sent_offsets=ntgt_sent_offsets

        self.hidden_size=self.neighbor_tgt_feature.shape[-1]
        self.hidden_dtype=self.neighbor_tgt_feature.dtype
        self.src_token_freq=src_token_freq
        self.sort_key=sort_key
        self.beam_size=self.opt.beam_size

        #knn参数配置
        self.topk=self.opt.topk
        self.link_temperature = self.opt.link_temperature
        self.link_ratio = self.opt.link_ratio
        self.sim_metric = self.opt.sim_metric

        if self.opt.quantizer_path:
            self.quantizer = TorchPQCodec(index=faiss.read_index(self.opt.quantizer_path))
        else:
            self.quantizer = None




    def find_step_token_neighbors_features(self,worker,step,src_len_offsets):
        # 测试集src的邻居，邻居来自训练集的src,这里可以并行处理
        #开启多进程会更慢？
        src_token_neighbors = [
            tuple(x) for x in
            self.token_neighbors[src_len_offsets + step][: self.max_neighbors_per_token].tolist()]
        tgt_token_neighbors = []
        for sent_idx, token_idx in src_token_neighbors:
            # ignore self as neighbor
            if self.pair_dataset == self.token_neighbors and sent_idx == step:
                continue
            align = self.get_tgt_align(sent_idx=sent_idx, src_idx=token_idx)
            tgt_token_neighbors.append((sent_idx, align))

        num_ntgts = len(tgt_token_neighbors)
        ntgt_step_feats = np.zeros((num_ntgts, self.hidden_size), dtype=self.hidden_dtype)
        ntgt_step_labels = np.zeros(num_ntgts, dtype=np.int64)
        for idx, (sent_idx, tgt_idx) in enumerate(tgt_token_neighbors):
            offset = self.ntgt_sent_offsets[sent_idx] + tgt_idx
            ntgt_step_feats[idx] = self.neighbor_tgt_feature[offset]
            ntgt_step_labels[idx] = self.get_neighbor_dataset_tgt(sent_idx)[tgt_idx]

        ntgt_step_feats = torch.from_numpy(ntgt_step_feats)
        ntgt_step_labels = torch.from_numpy(ntgt_step_labels)
        if self.gpu!=-1:
            ntgt_step_feats=ntgt_step_feats.cuda(self.gpu)
            ntgt_step_labels=ntgt_step_labels.cuda(self.gpu)
        if self.decoder_quantizer is not None:
            ntgt_step_feats = self.decoder_quantizer.decode(ntgt_step_feats)
        return ntgt_step_feats,ntgt_step_labels


    def get_token_neighbors_features(self,batch,model_hidden):
        #使用多进程
        src, src_lengths = (
            batch.src if isinstance(batch.src, tuple) else (batch.src, None)
        )
        maxlen=max(src_lengths)     #取batch里面句子中最长的长度
        max_neighbors_per_token = self.max_neighbors_per_token

        sent_num=len(src_lengths)
        ntgt_feats = torch.zeros((sent_num,maxlen,max_neighbors_per_token, model_hidden), dtype=torch.float32)
        ntgt_labels = torch.zeros((sent_num,maxlen,max_neighbors_per_token), dtype=torch.int64)
        indices=batch.indices
        for i,indice in enumerate(indices):
            src_length=src_lengths[i]
            pool = dummy.Pool(src_length.item())
            jobs = []
            for worker,step in enumerate(range(src_length)):
                src_len_offsets = self.src_len_offsets[indice]  # 测试集句子偏置
                # job = pool.apply_async(func=self.find_step_token_neighbors_features,
                #                        kwds={"step": step,"src_len_offsets":src_len_offsets})
                job = pool.apply_async(func=self.find_step_token_neighbors_features,
                                       args=(worker,step,src_len_offsets))
                jobs.append(job)
            for job in jobs:
                job.wait()
            for step,job in enumerate(jobs):
                if job.ready():  # 进程函数是否已经启动
                    if job.successful():
                        ntgt_step_feats,ntgt_step_labels = job.get()
                        ntgt_feats[i,step,:,:]=ntgt_step_feats
                        ntgt_labels[i,step,:]=ntgt_step_labels
                    else:
                        raise BaseException("Failed to find the neighbor of the sentence")
            pool.close()
            pool.join()
        if self.gpu != -1:
            ntgt_feats = ntgt_feats.cuda(self.gpu)
            ntgt_labels = ntgt_labels.cuda(self.gpu)
        return ntgt_feats,ntgt_labels

    # def get_token_neighbors_features(self, batch, model_hidden):
    #     # 使用多进程
    #     src, src_lengths = (
    #         batch.src if isinstance(batch.src, tuple) else (batch.src, None)
    #     )
    #     maxlen = max(src_lengths)  # 取batch里面句子中最长的长度
    #     max_neighbors_per_token = self.max_neighbors_per_token
    #
    #     sent_num = len(src_lengths)
    #     ntgt_feats = torch.zeros((sent_num, maxlen, max_neighbors_per_token, model_hidden), dtype=torch.float32)
    #     ntgt_labels = torch.zeros((sent_num, maxlen, max_neighbors_per_token), dtype=torch.int64)
    #     indices = batch.indices
    #     for i, indice in enumerate(indices):
    #         src_length = src_lengths[i]
    #         thread_list = []
    #
    #         for worker, step in enumerate(range(src_length)):
    #             src_len_offsets = self.src_len_offsets[indice]  # 测试集句子偏置
    #             # job = pool.apply_async(func=self.find_step_token_neighbors_features,
    #             #                        kwds={"step": step,"src_len_offsets":src_len_offsets})
    #             t = MyThread(self.find_step_token_neighbors_features, (worker,step,src_len_offsets), self.find_step_token_neighbors_features.__name__)
    #             thread_list.append(t)
    #         for j in range(len(thread_list)):  # 开始线程
    #             thread_list[j].start()
    #         for j in range(len(thread_list)):  # 结束线程
    #             thread_list[j].join()
    #         for step, thread in enumerate(thread_list):
    #             ntgt_step_feats, ntgt_step_labels = thread.result
    #             ntgt_feats[i, step, :, :] = ntgt_step_feats
    #             ntgt_labels[i, step, :] = ntgt_step_labels
    #     if self.gpu != -1:
    #         ntgt_feats = ntgt_feats.cuda(self.gpu)
    #         ntgt_labels = ntgt_labels.cuda(self.gpu)
    #     return ntgt_feats, ntgt_labels

    def get_knn_scores(self,batch,step,dec_out,cls_probs,ntgt_feats,ntgt_labels):
        src, src_lengths = (
            batch.src if isinstance(batch.src, tuple) else (batch.src, None)
        )
        mask = src_lengths >= step
        indices = batch.indices[mask]  # 选择的句子编号
        dec_out = dec_out.permute(1, 0, 2)  # [beam_size,1,hidden]
        beam_size = self.beam_size
        link_probs = torch.zeros_like(cls_probs)
        beam_range = np.arange(0, dec_out.shape[0], beam_size)
        for idx, (indice, beam) in enumerate(zip(indices, beam_range)):
            src_length = src_lengths[idx]
            if step>=src_length:    #句子后面全是</s>
                output = torch.zeros_like(dec_out[beam:beam + beam_size, :, 0]).unsqueeze(-1).repeat(
                    [1, 1, self._tgt_num])
                output[:] = 1e-10
                if self.gpu==-1:
                    labels = torch.full([beam_size, 1, 1], self._tgt_vocab['</s>'], dtype=torch.int64)
                    sim_probs = torch.full([beam_size, 1, 1],1.0, dtype=torch.float32)
                else:
                    labels = torch.full([beam_size, 1, 1], self._tgt_vocab['</s>'], dtype=torch.int64).cuda(self.gpu)
                    sim_probs = torch.full([beam_size, 1, 1], 1.0, dtype=torch.float32).cuda(self.gpu)
                link_prob = output.scatter_add(dim=2, index=labels, src=sim_probs)
                link_probs[beam:beam + beam_size, :] = link_prob.squeeze(1)
            else:
                ntgt_step_feats=ntgt_feats[idx,step,:,:]
                ntgt_step_labels=ntgt_labels[idx,step,:]
                ntgt_step_feats = ntgt_step_feats.unsqueeze(0).repeat([beam_size, 1, 1])
                ntgt_step_labels = ntgt_step_labels.unsqueeze(0).repeat([beam_size, 1])
                features = dec_out[beam:beam + beam_size, :, :]
                link_prob = self.knn_output_layer(
                    features=features,
                    knn_feats=ntgt_step_feats,
                    knn_labels=ntgt_step_labels,
                )
                link_prob = link_prob.squeeze(1)
                link_probs[beam:beam + beam_size, :] = link_prob
        x = torch.log(cls_probs * (1 - self.link_ratio) + link_probs * self.link_ratio + 1e-8)
        return x

    def knn_scores(self,batch,step,dec_out,cls_probs):
        src, src_lengths = (
            batch.src if isinstance(batch.src, tuple) else (batch.src, None)
        )

        mask=src_lengths>=step

        indices=batch.indices[mask]  #选择的句子编号
        max_neighbors_per_token = self.max_neighbors_per_token

        dec_out=dec_out.permute(1,0,2)  #[beam_size,1,hidden]
        #cls_probs=cls_probs.cpu()

        beam_size=self.beam_size
        link_probs=torch.zeros_like(cls_probs)
        beam_range=np.arange(0,dec_out.shape[0],beam_size)
        if self.link_ratio>0.0:
            for idx,(indice,beam) in enumerate(zip(indices,beam_range)):
                src_length=src_lengths[idx]
                if step>=src_length:    #句子后面全是</s>
                    output = torch.zeros_like(dec_out[beam:beam + beam_size, :, 0]).unsqueeze(-1).repeat(
                        [1, 1, self._tgt_num])
                    output[:] = 1e-10
                    if self.gpu==-1:
                        labels = torch.full([beam_size, 1, 1], self._tgt_vocab['</s>'], dtype=torch.int64)
                        sim_probs = torch.full([beam_size, 1, 1],1.0, dtype=torch.float32)
                    else:
                        labels = torch.full([beam_size, 1, 1], self._tgt_vocab['</s>'], dtype=torch.int64).cuda(self.gpu)
                        sim_probs = torch.full([beam_size, 1, 1], 1.0, dtype=torch.float32).cuda(self.gpu)
                    link_prob = output.scatter_add(dim=2, index=labels, src=sim_probs)
                    link_probs[beam:beam + beam_size, :] = link_prob.squeeze(1)
                else:
                    src_len_offsets = self.src_len_offsets[indice]  # 测试集句子偏置
                    # 测试集src的邻居，邻居来自训练集的src
                    src_token_neighbors = [
                        tuple(x) for x in self.token_neighbors[src_len_offsets + step][: max_neighbors_per_token].tolist()]
                    tgt_token_neighbors = []
                    for sent_idx, token_idx in src_token_neighbors:
                        # ignore self as neighbor
                        if self.pair_dataset == self.token_neighbors and sent_idx == step:
                            continue
                        align = self.get_tgt_align(sent_idx=sent_idx, src_idx=token_idx)
                        tgt_token_neighbors.append((sent_idx, align))

                    num_ntgts = len(tgt_token_neighbors)
                    ntgt_feats = np.zeros((num_ntgts, self.hidden_size), dtype=self.hidden_dtype)
                    ntgt_labels = np.zeros(num_ntgts, dtype=np.int64)
                    # ntgt_feats = torch.zeros((num_ntgts, self.hidden_size), dtype=torch.float32)
                    # ntgt_labels = torch.zeros(num_ntgts, dtype=torch.int64)

                    try:
                        for idx, (sent_idx, tgt_idx) in enumerate(tgt_token_neighbors):
                            offset = self.ntgt_sent_offsets[sent_idx] + tgt_idx
                            ntgt_feats[idx] = self.neighbor_tgt_feature[offset]
                            ntgt_labels[idx] = self.get_neighbor_dataset_tgt(sent_idx)[tgt_idx]
                    except:
                        print(sent_idx,tgt_idx)
                        print(self.get_neighbor_dataset_tgt(sent_idx))
                    ntgt_feats = torch.from_numpy(ntgt_feats)
                    ntgt_labels = torch.from_numpy(ntgt_labels)
                    if self.gpu != -1:
                        ntgt_feats = ntgt_feats.cuda(self.gpu)
                        ntgt_labels = ntgt_labels.cuda(self.gpu)
                    if self.decoder_quantizer is not None:
                        ntgt_feats = self.decoder_quantizer.decode(ntgt_feats)
                    ntgt_feats = ntgt_feats.unsqueeze(0).repeat([beam_size, 1, 1])
                    ntgt_labels = ntgt_labels.unsqueeze(0).repeat([beam_size, 1])
                    features=dec_out[beam:beam + beam_size, :,:]
                    #continue
                    link_prob = self.knn_output_layer(
                        features=features,
                        knn_feats=ntgt_feats,
                        knn_labels=ntgt_labels,
                    )
                    link_prob = link_prob.squeeze(1)
                    link_probs[beam:beam + beam_size, :] = link_prob
            #return torch.log(cls_probs).cuda(self.opt.gpu)

            x = torch.log(cls_probs * (1 - self.link_ratio) + link_probs * self.link_ratio + 1e-8)
        else:
            x = torch.log(cls_probs)
        return x


    def knn_output_layer(self, features, knn_feats, knn_labels):
        """
        compute knn-based prob
        Args:
            features: [bsz, tgt_len, h]
            knn-feats: [bsz, knn_num, h]
            knn_labels: [bsz, knn_num]
        Returns:
            knn_probs: [bsz, tgt_len, V]
        """
        knn_num = knn_feats.shape[1]
        tgt_len = features.shape[1]

        # todo support l2
        if self.sim_metric == "cosine":
            knn_feats = knn_feats.transpose(1, 2)  # [bsz, h, knn_num]
            sim = torch.bmm(features, knn_feats)  # [bsz, tgt_len, knn_num]
            norm1 = (knn_feats ** 2).sum(dim=1, keepdim=True).sqrt()  # [bsz, 1, knn_num]
            norm2 = (features ** 2).sum(dim=2, keepdim=True).sqrt()  # [bsz, tgt_len, 1]
            scores = sim / (norm1 + 1e-10) / (norm2 + 1e-10)  # [bsz, tgt_len, knn_num]
        elif self.sim_metric == "l2":
            features = features.unsqueeze(-2)  # [bsz, tgt_len, 1, h]
            knn_feats = knn_feats.unsqueeze(1)  # [bsz, 1, knn_num, h]
            scores = -((features - knn_feats) ** 2).sum(-1)  # todo memory concern: put them in chunk
        elif self.sim_metric == "ip":
            knn_feats = knn_feats.transpose(1, 2)  # [bsz, h, knn_num]
            scores = torch.bmm(features, knn_feats)  # [bsz, tgt_len, knn_num]
        elif self.sim_metric == "biaf":
            norm1 = (knn_feats ** 2).sum(dim=1, keepdim=True).sqrt()  # [bsz, 1, knn_num]
            norm2 = (features ** 2).sum(dim=2, keepdim=True).sqrt()  # [bsz, tgt_len, 1]
            knn_feats = knn_feats / norm1  # [bsz, knn_num, h]
            features = features / norm2  # [bsz, tgt_len, h]
            features = self.biaf_fc(features)  # [bsz, tgt_len, h]
            knn_feats = knn_feats.transpose(1, 2)  # [bsz, h, knn_num]
            scores = torch.bmm(features, knn_feats)  # [bsz, tgt_len, knn_num]
        else:
             raise ValueError(f"Does not support sim_metric {self.sim_metric}")
        mask = (knn_labels == self._tgt_vocab.stoi['<blank>']).unsqueeze(1)  # [bsz, 1, knn_num]
        scores[mask.expand(-1, tgt_len, -1)] -= 1e10
        knn_labels = knn_labels.unsqueeze(1).expand(-1, tgt_len, -1)  # [bsz, tgt_len, knn_num]
        if knn_num > self.topk > 0:
            topk_scores, topk_idxs = torch.topk(scores, dim=-1, k=self.topk)  # [bsz, tgt_len, topk]
            scores = topk_scores
            knn_labels = knn_labels.gather(dim=-1, index=topk_idxs)  # [bsz, tgt_len, topk]

        sim_probs = torch.softmax(scores / self.link_temperature, dim=-1)  # [bsz, tgt_len, knn_num]
        output = torch.zeros_like(sim_probs[:, :, 0]).unsqueeze(-1).repeat(
            [1, 1, self._tgt_num])  # [bsz, tgt_len, V]
        # output[b][t][knn_labels[b][t][k]] += link_probs[b][t][k]
        output = output.scatter_add(dim=2, index=knn_labels, src=sim_probs)
        return output

    def __getitem__(self, i):
        sample=self.pair_dataset[i]
        sample = self.find_knn(sample,i)
        # sample["knn_feats"] = knn_feats
        # sample["knn_labels"] = knn_labels
        return sample

    def find_knn(self,example,i):
        sample_id=i
        src_ids = [self._src_vocab.stoi[s] for s in example.src[0]]
        src_len_offsets = self.src_len_offsets[sample_id]
        max_neighbors_per_token = self.max_neighbors_per_token
        src_token_neighbors = [
            set(tuple(x) for x in self.token_neighbors[src_len_offsets + idx][: max_neighbors_per_token].tolist())
            for idx in range(len(src_ids))]

        #if self.extend_ngram
        all_src_token_neighbors=set(x for ns in src_token_neighbors for x in ns)
        all_tgt_token_neighbors=set()
        for sent_idx, token_idx in all_src_token_neighbors:
            if sent_idx==sample_id:
                continue
            align=self.get_tgt_align(sent_idx=sent_idx,src_idx=token_idx)
            all_tgt_token_neighbors.add((sent_idx, align))

        num_ntgts = len(all_tgt_token_neighbors)
        ntgt_feats = np.zeros((num_ntgts, self.hidden_size), dtype=self.hidden_dtype)
        ntgt_labels = np.zeros(num_ntgts, dtype=np.int32)

        for idx,(sent_idx,tgt_idx) in enumerate(all_tgt_token_neighbors):
            offset = self.ntgt_sent_offsets[sent_idx] + tgt_idx
            ntgt_feats[idx] = self.neighbor_tgt_feature[offset]
            ntgt_labels[idx] = self.get_neighbor_dataset_tgt(sent_idx)[tgt_idx]


        if self.decoder_quantizer is not None:
            ntgt_feats = self.decoder_quantizer.decode(ntgt_feats)
        ntgt_feats = torch.from_numpy(ntgt_feats)
        ntgt_labels = torch.from_numpy(ntgt_labels)
        sample={}
        sample['src']=[src_ids]
        sample['knn_feats']=[ntgt_feats]
        sample["knn_labels"] = [ntgt_labels]
        return sample



    def get_tgt_align(self,sent_idx,src_idx,offset=None):
        # if offset is None:
        #     offset = self.nsrc_sent_offsets[sent_idx] + src_idx
        #result=self.ntgt_sent_offsets[sent_idx]+src_idx     #拼音与汉字字符一一对应
        result=src_idx  #拼音与汉字字符一一对应
        return result


    @lru_cache(maxsize=100000)
    def get_neighbor_dataset_tgt(self, sent_idx):
        """
        extend_neighbors acquires neighbor dataset very frequently,
        we use cache to prevent reading from mmap dataset frequently
        """
        tgt_ids = [self._tgt_vocab.stoi[t] for t in self.train_dataset[sent_idx].tgt[0]]
        return tgt_ids


    # def __getattr__(self, attr):
    #     # avoid infinite recursion when fields isn't defined
    #     if 'fields' not in vars(self):
    #         raise AttributeError
    #     if attr in self.fields:
    #         return (getattr(x, attr) for x in self.examples)
    #     else:
    #         raise AttributeError
    #
    # def save(self, path, remove_fields=True):
    #     if remove_fields:
    #         self.fields = []
    #     torch.save(self, path)
    #
    # @staticmethod
    # def config(fields):
    #     readers, data = [], []
    #     for name, field in fields:
    #         if field["data"] is not None:
    #             readers.append(field["reader"])
    #             data.append((name, field["data"]))
    #     return readers, data


