"""
project: kNN-IME
file: traceing_word
author: JDS
create date: 2021/9/22 14:25
description: 
"""

import torch
import faiss
import numpy as np
import os
from tqdm import tqdm
from onmt.pq_wrapper import TorchPQCodec
from onmt.plasma_utils import PlasmaArray
#from knn_model import *
import json
from multiprocessing import dummy
from onmt import inputters
import os
import gc

class WordTracker():
    def __init__(self,opt=None,fields=None,pretext_dataset=None,batch_size=None,model=None,efsearch=8):
        self.opt = opt
        self.data_dir=self.opt.data_dir
        self.dstore_dir = self.opt.dstore_dir
        self.gpu=self.opt.gpu
        #self.threshold=self.opt.threshold   #设定一个阈值，当相似度小于这个阈值的时候丢弃
        self.topk=self.opt.topk
        self.link_temperature=self.opt.link_temperature
        self.link_ratio=self.opt.link_ratio
        self.sim_metric=self.opt.sim_metric
        self.max_neighbors=self.opt.max_neighbors
        self.nprobe=self.opt.nprobe
        self.efsearch=efsearch
        self.beam_size = self.opt.beam_size

        self.batch_size=batch_size

        src_field = dict(fields)['src'].base_field
        self._src_vocab = src_field.vocab
        self._src_num=len(self._src_vocab.itos)
        tgt_field = dict(fields)["tgt"].base_field
        self._tgt_vocab = tgt_field.vocab
        self._tgt_bos_idx = self._tgt_vocab.stoi[tgt_field.init_token]
        self._tgt_eos_idx = self._tgt_vocab.stoi[tgt_field.eos_token]
        self._tgt_num=len(self._tgt_vocab.itos)
        self.cache_size = self.opt.cache_size
        self.src_cache = {}
        self.tgt_cache = {}
        self.val_cache = {}
        self.sent_ids_cache={}     #记录句子编号id，用于基于tgt的token属于那一句话，在实际使用中，这个并没有什么用，只是方便做分析
        self.ids=[[0,0] for i in range(self._src_num)]  #用于增删的编号和保存的容量大小
        self.freq=[0 for i in range(self._src_num)]
        self.hidden = None
        self.use_sent_ids=opt.use_sent_ids

        self.src_index={}
        self.tgt_features={}
        self.vals={}
        self.sent_ids = {}

        for key in range(len(self._src_vocab.itos)):
            self.src_index[key]=None
            self.tgt_features[key]=None
            self.sent_ids[key] = None


        self.init_remove=self.opt.init_remove   #初始化的时候移除所有数据

        self.read_faiss_index()
        self.load_tgt_features()
        self.decoder_quantizer = TorchPQCodec(index=faiss.read_index(self.opt.quantizer_path))

        # 建立缓存机制，这样可以避免过多的对磁盘和索引操作
        if self.gpu != -1 and self.decoder_quantizer is not None:
            self.decoder_quantizer.cuda(self.gpu)

        self.pretext_dataset = pretext_dataset
        self.model = model
        self.save_update=self.opt.save_update
        self.save_dir=self.opt.save_dir

        if self.pretext_dataset is not None:
            self.MIUs_to_features()

        for token_idx in self.src_index.keys():
            # 初始化数据
            self.src_cache[token_idx] = []
            self.tgt_cache[token_idx] = []
            self.val_cache[token_idx] = []
            self.sent_ids_cache[token_idx] = []


    def index_file_path(self,token_idx):
        index_file = os.path.join(self.data_dir, self.dstore_dir,
                                  f"token_{token_idx}",
                                  f"faiss_store.{self.opt.sim_metric}")
        return index_file

    def feature_file_path(self,token_idx):
        feature_file = os.path.join(self.data_dir, self.dstore_dir,
                                  f"token_{token_idx}",
                                  f"decoder_keys.npy")
        return feature_file
    def vals_file_path(self,token_idx):
        vals_file = os.path.join(self.data_dir, self.dstore_dir,
                                  f"token_{token_idx}",
                                  f"decoder_vals.npy")
        return vals_file

    def sent_ids_file_path(self,token_idx):
        sent_ids_file = os.path.join(self.data_dir, self.dstore_dir,
                                  f"token_{token_idx}",
                                  f"sent_ids.npy")
        return sent_ids_file


    def info_file_path(self, token_idx):
        info_file = os.path.join(self.data_dir, self.dstore_dir,
                                 f"token_{token_idx}",
                                 f"decoder_info.json")
        return info_file


    def read_faiss_index(self):
        '''
        读取faiss-index
        将源的token编号作为键，对应的faiss-index作为值
        :return:
        '''
        # if self.gpu != -1:        #faiss目前在gpu下不支持动态删除索引
        #     res = faiss.StandardGpuResources()
        #     co = faiss.GpuClonerOptions()
        #     co.useFloat16 = True
        for token_idx in self.src_index.keys():
            index_file = self.index_file_path(token_idx)
            if os.path.exists(index_file):
                # print(f'loading {index_file}')
                index = faiss.read_index(index_file, faiss.IO_FLAG_ONDISK_SAME_DIR)
                try:
                    faiss.ParameterSpace().set_index_parameter(index, "nprobe", self.nprobe)
                    #faiss.ParameterSpace().set_index_parameter(index, "quantizer_efSearch", self.efsearch)
                except Exception as e:
                    #print(f"faiss index {index_file} does not have parameter nprobe or efSearch")
                    pass
                    # print(f"faiss index {index_file} does not have parameter nprobe")
                # if self.gpu!=-1:    #faiss目前在gpu下不支持动态删除索引
                #     # try:
                #         index=faiss.index_cpu_to_gpu(res,self.gpu,index,co)
                #         # index = faiss.index_cpu_to_gpu(res, self.gpu, index)
                #         print('token_idx',token_idx)
                    # except:
                    #     print(f"index {index_file} does not support GPU")
                if self.init_remove:        #初始化的时候移除所有索引里面的数据
                    index.remove_ids(np.arange(0,index.ntotal))
                self.src_index[token_idx]=index

    def load_tgt_features(self):
        '''
        加载特征
        self.tgt_features的键为token编号，值为decoder编码的特征向量
        self.vals的键为字符索引编号，值为目标对应的token编号
        :return:
        '''
        for token_idx in self.src_index.keys():
            feature_file = self.feature_file_path(token_idx)
            if os.path.exists(feature_file):
                info_file=self.info_file_path(token_idx)
                #print(f'loading {info_file}')
                info = json.load(open(info_file))
                dstore_size, hidden_size, freq,id = (
                    info["dstore_size"],
                    info["hidden_size"],
                    info["freq"],
                    info['id']
                )
                if self.init_remove:
                    tgt_feature=np.zeros((dstore_size, hidden_size),dtype=np.uint8)
                else:
                    tgt_feature= np.memmap(feature_file, dtype=np.uint8, mode='r',
                             shape=(dstore_size, hidden_size))

                if self.gpu==-1:
                    self.tgt_features[token_idx]=torch.from_numpy(tgt_feature)
                else:
                    # 移到gpu
                    self.tgt_features[token_idx] = torch.from_numpy(tgt_feature).cuda(self.gpu)
                if self.hidden is None:
                    self.hidden=hidden_size

                val_file=self.vals_file_path(token_idx)
                if os.path.exists(val_file):
                    if self.init_remove:
                        val = np.zeros((dstore_size, 1), dtype=np.int32)
                    else:
                        val=np.memmap(val_file, dtype=np.int32, mode='r',
                                 shape=(dstore_size, 1))
                        val=np.array(val)
                    if self.gpu==-1:
                        self.vals[token_idx]=torch.from_numpy(val)
                    else:
                        self.vals[token_idx] = torch.from_numpy(val).cuda(self.gpu)
                    if self.init_remove:
                        self.ids[token_idx][0] = 0  # 编号初始化为0
                        self.ids[token_idx][1] = dstore_size  # 存储空间大小
                        self.freq[token_idx] = 0
                    else:
                        self.ids[token_idx][0] = id #索引编号
                        self.ids[token_idx][1] = dstore_size  # 存储空间大小
                        self.freq[token_idx]=freq
                if self.use_sent_ids:   #使用句子id
                    sent_ids_file=self.sent_ids_file_path(token_idx)
                    if os.path.exists(sent_ids_file):
                        if self.init_remove:
                            sent_ids=np.zeros((dstore_size), dtype=np.int32)
                        else:
                            sent_ids=np.memmap(sent_ids_file, dtype=np.int32, mode='r',
                                     shape=(dstore_size))
                        sent_ids=np.array(sent_ids)
                        self.sent_ids[token_idx]=sent_ids
                    elif self.init_remove:
                        sent_ids = np.zeros((dstore_size), dtype=np.int32)
                        sent_ids = np.array(sent_ids)
                        self.sent_ids[token_idx] = sent_ids

    def MIUs_to_features(self):
        _use_cuda = self.gpu > -1
        _dev = (
            torch.device("cuda", self.gpu)
            if _use_cuda
            else torch.device("cpu")
        )
        batch_size=64
        data_iter = inputters.OrderedIterator(
            dataset=self.pretext_dataset,
            device=_dev,
            batch_size=batch_size,
            batch_size_fn=None,
            train=False,
            sort=False,
            sort_within_batch=False,
            shuffle=False,
        )

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(data_iter,desc='MIUs to features'):
                src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                    else (batch.src, None)
                tgt = batch.tgt
                dec_in = tgt[:-1]
                encoder_features, memory_bank, lengths = self.model.encoder(src, src_lengths)
                self.model.decoder.init_state(src, memory_bank, encoder_features)
                decoder_features, attns = self.model.decoder(dec_in, memory_bank,
                                                             memory_lengths=lengths, with_align=False)
                # enc_out = memory_bank  # 这里的memory在openNMT中是编码器最后编码的状态
                # dec_out = decoder_features
                srcs, _ = (
                    batch.src if isinstance(batch.src, tuple) else (batch.src, None)
                )
                tgts, _ = (
                    batch.tgt if isinstance(batch.tgt, tuple) else (batch.tgt, None)
                )

                indices=batch.indices

                srcs=srcs[:,:,0]
                tgts=tgts[1:-1,:,0]
                src_lengths=src_lengths.tolist()
                for idx in range(batch.batch_size):
                    enc_out=memory_bank[:,idx,:].unsqueeze(1)
                    dec_out=decoder_features[:,idx,:].unsqueeze(1)
                    src=srcs[:,idx].tolist()
                    tgt=tgts[:,idx].tolist()
                    length=src_lengths[idx]
                    src=src[:length]    #去掉<pad>
                    tgt=tgt[:length]
                    indice=indices[idx]
                    self.update(src, tgt, enc_out, dec_out,indice)
                # del memory_bank
                # del enc_out
                # del dec_out
                # del encoder_features
                # del decoder_features
                # del attns
                # del dec_in
                # torch.cuda.empty_cache()
            if self.save_update:
                self.store_update()

    def store_update(self):
        datastore_dir = os.path.join(self.save_dir, self.dstore_dir)
        for key in tqdm(self.src_index,desc='save update...'):
            #强制将buffer数据全部写入到对应的文件中
            if self.src_index[key] is None:
                continue
            cache_len = len(self.src_cache[key])
            if cache_len>0:
                self.write_data(key)
            #写入文件index
            index=self.src_index[key]
            faiss_file = os.path.join(datastore_dir,f"token_{key}",f"faiss_store.cosine")

            os.makedirs(os.path.join(datastore_dir,f"token_{key}"), exist_ok=True)
            faiss.write_index(index, faiss_file)

            #写入文件decoder_feature
            tgt_feature=self.tgt_features[key]
            tgt_feature_file = os.path.join(datastore_dir,f"token_{key}", f"decoder_keys.npy")
            dstore_size, tgt_hidden_size=tgt_feature.shape
            tgt_mmap_feature = np.memmap(tgt_feature_file, dtype=np.uint8, mode='w+',
                                    shape=(dstore_size, tgt_hidden_size))
            tgt_mmap_feature[:]=tgt_feature.cpu().numpy()

            # 写入文件vals
            val=self.vals[key]
            val_file = os.path.join(datastore_dir,f"token_{key}", f"decoder_vals.npy")
            dstore_size,val_hidden_size=val.shape
            val_mmap = np.memmap(val_file,dtype=np.int32, mode='w+',
                                         shape=(dstore_size, val_hidden_size))
            val_mmap[:] = val.cpu().numpy()

            if self.use_sent_ids:
                # 写入文件sent_ids
                sent_ids = self.sent_ids[key]
                sent_ids_file = os.path.join(datastore_dir, f"token_{key}", f"sent_ids.npy")
                sent_ids_mmap = np.memmap(sent_ids_file, dtype=np.int32, mode='w+',
                                     shape=(dstore_size))
                sent_ids_mmap[:]=sent_ids

            (id, dstore_size) = self.ids[key]
            freq=self.freq[key]
            #写入info
            info = {
                "dstore_size": dstore_size,
                "hidden_size": tgt_hidden_size,
                "vocab_size": "0",
                "dstore_fp16": False,
                "val_size": 1,
                "freq": freq,
                "id":id
            }
            json.dump(info, open(os.path.join(datastore_dir,f"token_{key}", "decoder_info.json"), "w"),
                      sort_keys=True, indent=4, ensure_ascii=False)


    def get_knns(self,index,queries, k):
        """
        get distances and knns from queries
        Args:
            queries: Tensor of shape [num, hidden]
            k: number of k, default value is self.k
        Returns:
            dists: knn dists. np.array of shape [num, k]
            knns: knn ids. np.array of shape [num, k]
        """
        if isinstance(queries, torch.Tensor):
            queries = queries.detach().cpu().float().data.numpy()
        dists, knns = index.search(queries, k=k)
        return dists, knns


    def find_token_neighbors_features(self,batch,enc_outs):
        '''
        beam_size为1
        :return:
        '''
        src, src_lengths = (
            batch.src if isinstance(batch.src, tuple) else (batch.src, None)
        )
        src=src[:,0,0].tolist()
        #enc_outs=enc_outs.squeeze(1)
        ntgt_feats= torch.zeros((src_lengths[0],self.max_neighbors,self.hidden), dtype=torch.uint8)
        ntgt_labels=torch.zeros((src_lengths[0],self.max_neighbors), dtype=torch.int64)
        norm = torch.sqrt(torch.sum(enc_outs ** 2, axis=-1, keepdims=True))
        enc_outs = enc_outs / norm
        if self.gpu!=-1:
            ntgt_feats=ntgt_feats.cuda(self.gpu)
            ntgt_labels=ntgt_labels.cuda(self.gpu)
        for id,s in enumerate(src):
            enc_out=enc_outs[id]
            index=self.src_index[s]
            if index!=None:
                knn_dists,knns=self.get_knns(index=index,queries=enc_out,k=self.max_neighbors)    #获取源的邻居
                # 计算buffer与enc_out的相似度
                if len(self.src_cache[s]) > 0:
                    tgt_buffer=torch.stack(self.tgt_cache[s])
                    vals=torch.stack(self.val_cache[s])
                    knn_feats=torch.cat((tgt_buffer,self.tgt_features[s][knns]))    #将目标特征与标签分别堆叠
                    knn_labels=torch.cat((vals,self.vals[s][knns]))
                    knn_feats=knn_feats[0:self.max_neighbors]    #选取最近特征和标签
                    knn_labels=knn_labels[0:self.max_neighbors]
                else:
                    #根据源的邻居找到对应的tgt_features的特征与vals
                    knn_feats=self.tgt_features[s][knns]
                    knn_labels=self.vals[s][knns]
            else:
                knn_feats = torch.ones((self.max_neighbors,self.hidden),dtype=torch.uint8)
                knn_labels = torch.full((self.max_neighbors,self.hidden),s,dtype=torch.int32)
            ntgt_feats[id,:,:]=knn_feats
            ntgt_labels[id,:]=knn_labels[:,0]
        if self.decoder_quantizer is not None:
            ntgt_feats=ntgt_feats.view(-1,self.hidden)
            ntgt_feats = self.decoder_quantizer.decode(ntgt_feats)
            ntgt_feats = ntgt_feats.view(-1,self.max_neighbors, 512)

        return ntgt_feats,ntgt_labels

    def token_step_neighbors_features(self,id,s,enc_outs):
        enc_out = enc_outs[id]
        index = self.src_index[s]
        if index!=None:
            knn_dists, knns = self.get_knns(index=index, queries=enc_out, k=self.max_neighbors)
            if len(self.src_cache[s]) > 0:
                tgt_buffer = torch.stack(self.tgt_cache[s])
                vals = torch.stack(self.val_cache[s])
                knn_feats = torch.cat((tgt_buffer, self.tgt_features[s][knns]))  # 将目标特征与标签分别堆叠
                knn_labels = torch.cat((vals, self.vals[s][knns]))
                knn_feats = knn_feats[0:self.max_neighbors]  # 选取最近特征和标签
                knn_labels = knn_labels[0:self.max_neighbors]
            else:
                # 根据源的邻居找到对应的tgt_features的特征与vals
                knn_feats = self.tgt_features[s][knns]
                knn_labels = self.vals[s][knns]
        else:
            knn_feats = torch.ones((self.max_neighbors, self.hidden), dtype=torch.uint8)
            knn_labels = torch.full((self.max_neighbors, self.hidden), s, dtype=torch.int32)
        return knn_feats,knn_labels


    def multipro_find_token_neighbors_features(self, batch, enc_outs):
        # 多进程运算
        src, src_lengths = (
            batch.src if isinstance(batch.src, tuple) else (batch.src, None)
        )
        src = src[:, 0, 0].tolist()
        # enc_outs=enc_outs.squeeze(1)
        src_length=src_lengths[0]
        ntgt_feats = torch.zeros((src_length, self.max_neighbors, self.hidden), dtype=torch.uint8)
        ntgt_labels = torch.zeros((src_length, self.max_neighbors), dtype=torch.int64)
        if self.gpu!=-1:
            ntgt_feats=ntgt_feats.cuda(self.gpu)
            ntgt_labels=ntgt_labels.cuda(self.gpu)

        pool = dummy.Pool(src_length)
        jobs = []
        norm = torch.sqrt(torch.sum(enc_outs ** 2, axis=-1, keepdims=True))
        enc_outs = enc_outs / norm
        for id,s in enumerate(src):
            job = pool.apply_async(func=self.token_step_neighbors_features,
                                   args=(id,s,enc_outs))
            jobs.append(job)
        for job in jobs:
            job.wait()
        for id,job in enumerate(jobs):
            if job.ready():  # 进程函数是否已经启动
                if job.successful():
                    knn_feats,knn_labels = job.get()
                    ntgt_feats[id, :, :] = knn_feats
                    ntgt_labels[id, :] = knn_labels[:, 0]
                else:
                    print(src[id],self._src_vocab.itos[src[id]])
                    raise BaseException("Failed to find the neighbor of the token")
        pool.close()    #释放资源
        pool.join()
        if self.decoder_quantizer is not None:
            ntgt_feats=ntgt_feats.view(-1,self.hidden)
            ntgt_feats = self.decoder_quantizer.decode(ntgt_feats)
            ntgt_feats = ntgt_feats.view(-1,self.max_neighbors, 512)
        return ntgt_feats, ntgt_labels

    def get_knn_scores(self,batch,dec_out,step,cls_probs,ntgt_feats,ntgt_labels):
        src, src_lengths = (
            batch.src if isinstance(batch.src, tuple) else (batch.src, None)
        )
        src_length=src_lengths[0]
        beam_size=self.beam_size
        if step >= src_length:  #超过src_length，全部视为</s>
            tgt_len = dec_out.shape[0]
            output = torch.zeros([beam_size, tgt_len, self._tgt_num],dtype=torch.float32)  # [bsz, tgt_len, V]
            #output[:] = 1e-10
            if self.gpu == -1:
                labels = torch.full([beam_size, 1, 1], self._tgt_vocab['</s>'], dtype=torch.int64)
                sim_probs = torch.full([beam_size, 1, 1], 1.0, dtype=torch.float32)
            else:
                output=output.cuda(self.gpu)
                labels = torch.full([beam_size, 1, 1], self._tgt_vocab['</s>'], dtype=torch.int64).cuda(self.gpu)
                sim_probs = torch.full([beam_size, 1, 1], 1.0, dtype=torch.float32).cuda(self.gpu)
            output = output.scatter_add(dim=2, index=labels, src=sim_probs)
        else:
            knn_feats = ntgt_feats[step]
            knn_labels = ntgt_labels[step]
            feature = dec_out.permute(1, 0, 2)  # [beam_size,1,hidden]
            # 计算当前解码数据与邻居对应的tgt_features的相似度
            knn_feats = knn_feats.unsqueeze(0).repeat([beam_size, 1, 1]).transpose(1, 2)  # [bsz, h, knn_num]
            sim = torch.bmm(feature, knn_feats)  # [bsz, tgt_len, knn_num]
            norm1 = (knn_feats ** 2).sum(dim=1, keepdim=True).sqrt()  # [bsz, 1, knn_num]
            norm2 = (feature ** 2).sum(dim=2, keepdim=True).sqrt()  # [bsz, tgt_len, 1]
            scores = sim / (norm1 + 1e-10) / (norm2 + 1e-10)  # [bsz, tgt_len, knn_num] #计算出相似度
            knn_num = knn_feats.shape[1]
            # mask = ~(scores > self.threshold)  # 去除相似度低于阈值的邻居
            # scores[mask] -= 1e10
            knn_labels = knn_labels.unsqueeze(0).repeat([self.beam_size,1]).unsqueeze(1)  # [bsz, tgt_len, knn_num]
            if knn_num > self.topk > 0:
                topk_scores, topk_idxs = torch.topk(scores, dim=-1, k=self.topk)  # [bsz, tgt_len, topk]
                scores = topk_scores
                knn_labels = knn_labels.gather(dim=-1, index=topk_idxs)  # [bsz, tgt_len, topk]
            sim_probs = torch.softmax(scores / self.link_temperature, dim=-1)  # [bsz, tgt_len, knn_num]
            output = torch.zeros_like(sim_probs[:, :, 0]).unsqueeze(-1).repeat(
                [1, 1, self._tgt_num])  # [bsz, tgt_len, V]

            #先聚合，再标准化
            # output = torch.zeros_like(scores[:, :, 0]).unsqueeze(-1).repeat(
            #     [1, 1, self._tgt_num])  # [bsz, tgt_len, V]
            # output = output.scatter_add(dim=2, index=knn_labels, src=scores)
            # output = torch.softmax(output / self.link_temperature, dim=-1)

            output = output.scatter_add(dim=2, index=knn_labels, src=sim_probs)
            output[:, :, 0] = 0     #强制把未知符号的概率设为0
        link_prob = output.squeeze(1)
        x = torch.log(cls_probs * (1 - self.link_ratio) + link_prob * self.link_ratio + 1e-8)
        return x


    def write_data(self,key):
        (id, dstore_size) = self.ids[key]
        src_cache=self.src_cache[key]
        cache_len=len(src_cache)
        if dstore_size > id + cache_len:
            # 修改src_index,先删除索引，再增加索引
            src_cache = torch.stack(src_cache).cpu().numpy()  # 已标准化过
            self.src_index[key].remove_ids(np.arange(id, id + cache_len))
            self.src_index[key].add_with_ids(src_cache, np.arange(id, id + cache_len))
            # 修改tgt_feature数据
            tgt_cache = self.tgt_cache[key]
            self.tgt_features[key][id:id + cache_len] = torch.stack(tgt_cache)
            # 修改val数据
            val_cache = self.val_cache[key]
            self.vals[key][id:id + cache_len] = torch.stack(val_cache)
            #修改句子编号
            if self.use_sent_ids:
                sent_ids_cache=self.sent_ids_cache[key]
                self.sent_ids[key][id:id + cache_len]=sent_ids_cache
        else:
            # 分段修改
            mid_id = dstore_size - id
            end_id = (id + cache_len) % dstore_size
            if mid_id != 0:  # 防止插入的索引为空，而报错
                self.src_index[key].remove_ids(np.arange(id, dstore_size))
                self.src_index[key].add_with_ids(torch.stack(src_cache[0:mid_id]).cpu().numpy(),
                                                 np.arange(id, dstore_size))
            if mid_id != cache_len:  # 防止插入的索引为空，而报错
                self.src_index[key].remove_ids(np.arange(0, end_id))
                self.src_index[key].add_with_ids(torch.stack(src_cache[mid_id:]).cpu().numpy(), np.arange(0, end_id))
            # 修改tgt_feature数据
            tgt_cache = self.tgt_cache[key]
            # if self.decoder_quantizer is not None:
            #     tgt_buffer = self.decoder_quantizer.encode(torch.stack(tgt_buffer))
            if mid_id != 0:
                self.tgt_features[key][id:dstore_size] = torch.stack(tgt_cache[0:mid_id])
            if mid_id != cache_len:
                self.tgt_features[key][0:end_id] = torch.stack(tgt_cache[mid_id:])

            # 修改val数据
            val_cache = self.val_cache[key]
            if mid_id != 0:
                self.vals[key][id:dstore_size] = torch.stack(val_cache[0:mid_id])
            if mid_id != cache_len:
                self.vals[key][0:end_id] = torch.stack(val_cache[mid_id:])

            # 修改sent_ids数据
            if self.use_sent_ids:
                sent_ids_cache = self.sent_ids_cache[key]
                if mid_id != 0:
                    self.sent_ids[key][id:dstore_size] =sent_ids_cache[0:mid_id]
                if mid_id != cache_len:
                    self.sent_ids[key][0:end_id] =sent_ids_cache[mid_id:]

        # 修改编号和清空buffer
        self.ids[key][0] = (id + cache_len) % dstore_size
        # self.tgt_cache[key] = []
        # self.src_cache[key] = []
        # self.val_cache[key] = []
        # self.sent_ids_cache[key]=[]
        # gc.collect()
        del src_cache
        del tgt_cache
        del val_cache
        del self.tgt_cache[key][:]
        del self.src_cache[key][:]
        del self.val_cache[key][:]
        if self.use_sent_ids:
            del self.sent_ids_cache[key][:]
        gc.collect()
    # #    torch.cuda.empty_cache()
        self.freq[key] = self.freq[key] + cache_len   #更新频率

    def update(self,src,tgt,enc_out,dec_out,indice):
        enc_out=enc_out.squeeze(1)
        norm1 = torch.sqrt(torch.sum(enc_out ** 2, axis=-1, keepdims=True))
        enc_out = enc_out / norm1   #这里对enc_out标准化，之后就不用进行标准化了
        for idx,(s,t) in enumerate(zip(src,tgt)):
            if self.src_index[s] is None:
                continue
            self.src_cache[s].append(enc_out[idx])
            out=dec_out[idx]
            if self.decoder_quantizer is not None:
                out = self.decoder_quantizer.encode(out)
            self.tgt_cache[s].append(out.squeeze(0))
            t = torch.tensor([t], dtype=torch.int64)
            if self.gpu!=-1:
                t=t.cuda(self.gpu)
            self.val_cache[s].append(t)
            if self.use_sent_ids:
                self.sent_ids_cache[s].append(indice)
        #将超过self.buffer_size的数据写入到对应索引文件或磁盘中
        for key in self.src_cache.keys():
            src_buffer=self.src_cache[key]
            buffer_len = len(src_buffer)
            if buffer_len>=self.cache_size:
               self.write_data(key)

    def sent_to_id(self, src, tgt):
        src = src.split()
        src_id = [self._src_vocab.stoi[s] for s in src]
        tgt = tgt.split()
        tgt_id = [self._tgt_bos_idx] + [self._tgt_vocab.stoi[s] for s in tgt] + [self._tgt_eos_idx]
        return src_id, tgt_id

    def search_sent(self, src, tgt, pred_idx):
        src, tgt = self.sent_to_id(src, tgt)
        src_lengths = len(src)
        src = torch.tensor(src).unsqueeze(1).unsqueeze(1)
        tgt = torch.tensor(tgt).unsqueeze(1).unsqueeze(1)
        src_lengths = torch.tensor([src_lengths])
        if self.gpu != -1:
            src_lengths = src_lengths.to(self.gpu)
            src=src.to(self.gpu)
            tgt=tgt.to(self.gpu)
        self.model.eval()
        with torch.no_grad():
            dec_in = tgt[:-1]
            encoder_features, memory_bank, lengths = self.model.encoder(src, src_lengths)
            self.model.decoder.init_state(src, memory_bank, encoder_features)
            decoder_features, attns = self.model.decoder(dec_in, memory_bank,
                                                         memory_lengths=lengths, with_align=False)
            queries = memory_bank[pred_idx]
            feature = decoder_features[pred_idx]  # [beam_size,1,hidden]

            # 查找源
            s = src[pred_idx][0][0].cpu().item()
            index = self.src_index[s]
            dists, knns = self.get_knns(index, queries,self.max_neighbors)

            if len(self.src_cache[s]) > 0:
                tgt_buffer = torch.stack(self.tgt_cache[s])
                knn_feats = torch.cat((tgt_buffer, self.tgt_features[s][knns]))  # 将目标特征与标签分别堆叠
                knn_feats = knn_feats[0:self.max_neighbors]  # 选取最近特征和标签
                vals = torch.stack(self.val_cache[s])
                knn_labels = torch.cat((vals, self.vals[s][knns]))
                knn_labels=knn_labels[0:self.max_neighbors]
                sent_ids = np.append(self.sent_ids_cache, self.sent_ids[s][knns])
            else:
                knn_feats=self.tgt_features[s][knns]
                knn_feats = knn_feats[0:self.max_neighbors]  # 选取最近特征和标签
                knn_labels=self.vals[s][knns]
                knn_labels=knn_labels[0:self.max_neighbors]
                sent_ids=self.sent_ids[s][knns]
            knn_feats = self.decoder_quantizer.decode(knn_feats)
            # 根据源的范围目标相似度进行计算
            sim = torch.mm(feature, knn_feats.permute(1,0))
            norm1 = (knn_feats ** 2).sum(dim=-1, keepdim=True).sqrt().permute(1,0)
            norm2 = (feature ** 2).sum(dim=-1, keepdim=True).sqrt().permute(1,0)
            scores = sim / (norm1 + 1e-10) / (norm2 + 1e-10)   #计算出相似度
            knn_num = knn_feats.shape[1]
            if knn_num > self.topk > 0:
                topk_scores, topk_idxs = torch.topk(scores, dim=-1, k=self.topk)  # [bsz, tgt_len, topk]
                sent_ids = sent_ids[0,:][topk_idxs[0,:].cpu()]
                scores=scores[0,:][topk_idxs[0,:].cpu()]
                knn_labels=knn_labels[:,0][topk_idxs[0,:].cpu()]
            sim_probs = torch.softmax(scores / self.link_temperature, dim=-1)  # [bsz, tgt_len, knn_num]
            return sent_ids,scores,knn_labels,sim_probs

    def sim_token(self,src,tgt,pred_idx):
        src, tgt = self.sent_to_id(src, tgt)
        src_lengths = len(src)
        src = torch.tensor(src).unsqueeze(1).unsqueeze(1)
        tgt = torch.tensor(tgt).unsqueeze(1).unsqueeze(1)
        src_lengths = torch.tensor([src_lengths])
        if self.gpu != -1:
            src_lengths = src_lengths.to(self.gpu)
            src = src.to(self.gpu)
            tgt = tgt.to(self.gpu)
        self.model.eval()
        with torch.no_grad():
            dec_in = tgt[:-1]
            encoder_features, memory_bank, lengths = self.model.encoder(src, src_lengths)
            self.model.decoder.init_state(src, memory_bank, encoder_features)
            decoder_features, attns = self.model.decoder(dec_in, memory_bank,
                                                         memory_lengths=lengths, with_align=False)
            queries = memory_bank[pred_idx]
            feature = decoder_features[pred_idx]  # [beam_size,1,hidden]

            # 查找源
            s = src[pred_idx][0][0].cpu().item()
            index = self.src_index[s]
            dists, knns = self.get_knns(index, queries, self.max_neighbors)

            if len(self.src_cache[s]) > 0:
                tgt_buffer = torch.stack(self.tgt_cache[s])
                knn_feats = torch.cat((tgt_buffer, self.tgt_features[s][knns]))  # 将目标特征与标签分别堆叠
                knn_feats = knn_feats[0:self.max_neighbors]  # 选取最近特征和标签
                vals = torch.stack(self.val_cache[s])
                knn_labels = torch.cat((vals, self.vals[s][knns]))
                knn_labels = knn_labels[0:self.max_neighbors]
                sent_ids = np.append(self.sent_ids_cache, self.sent_ids[s][knns])
            else:
                knn_feats = self.tgt_features[s][knns]
                knn_feats = knn_feats[0:self.max_neighbors]  # 选取最近特征和标签
                knn_labels = self.vals[s][knns]
                knn_labels = knn_labels[0:self.max_neighbors]
                sent_ids = self.sent_ids[s][knns]
            knn_feats = self.decoder_quantizer.decode(knn_feats)
            # 根据源的范围目标相似度进行计算
            sim = torch.mm(feature, knn_feats.permute(1, 0))
            norm1 = (knn_feats ** 2).sum(dim=-1, keepdim=True).sqrt().permute(1, 0)
            norm2 = (feature ** 2).sum(dim=-1, keepdim=True).sqrt().permute(1, 0)
            scores = sim / (norm1 + 1e-10) / (norm2 + 1e-10)  # 计算出相似度
            topk_scores, topk_idxs = torch.topk(scores, dim=-1, k=self.max_neighbors)  # [bsz, tgt_len, topk]
            knn_labels = knn_labels[:, 0][topk_idxs[0, :].cpu()]

            tokens=[]   #用于记录token
            tokens_feature=[]
            topk_idxs=topk_idxs[0,:]
            for topk_idx,label in zip(topk_idxs,knn_labels):
                token=self._tgt_vocab.itos[label]
                if token not in tokens:
                    tokens.append(token)
                    feature=knn_feats[topk_idx.cpu()]
                    tokens_feature.append(feature)
            tokens_feature=torch.stack(tokens_feature)
            decoder_features=decoder_features.squeeze(1)[:-1]
            norm1 = (decoder_features ** 2).sum(dim=-1, keepdim=True).sqrt()
            norm2 = (tokens_feature ** 2).sum(dim=-1, keepdim=True).sqrt().permute(1, 0)
            sim=torch.matmul(decoder_features,tokens_feature.permute(1, 0))/ (norm1 + 1e-10) / (norm2 + 1e-10)  # 计算出tokns相似度
            return tokens,sim