# MSR-Transformer
### 出自论文《Adaptive Chinese Pinyin IME for Most Similar Representation》

### 主要环境配置
* python=3.6
* pytorch=1.7.1
* faiss-cpu=1.7.2
* OpenNMT-py=2.1.2

### 程序运行
构建词汇表  
onmt_build_vocab -config data/PD_model.yaml -n_sample -1

训练模型  
onmt_build_vocab -config data/PD_model.yaml

查看MSR Transformer命令指引
