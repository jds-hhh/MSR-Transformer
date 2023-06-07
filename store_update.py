"""
project: kNN-IME
file: store_update
author: JDS
create date: 2021/9/30 22:33
description: 
"""

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
import onmt.inputters as inputters
import torch
import onmt.decoders.ensemble
from onmt.knn.word_tracker import *


class Store_Update():
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

        self.word_tracker=WordTracker(opt=self.opt,fields=self.fields,pretext_dataset=None,
                                      batch_size=self.opt.batch_size,model=self.model)      #这里我们先将pretext_data设置为空，后面我们再调用


    def update(self,src,tgt=None,batch_size=None,batch_type="sents"):
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

        self.word_tracker.model.eval()
        self.word_tracker.pretext_dataset=self.data #设置pretext_dataset
        self.word_tracker.MIUs_to_features()    #将MIU转为对应特征
        self.word_tracker.store_update()        #存储特征



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
    store_update = Store_Update(model, fields, opt)
    for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
        logger.info("Translating shard %d." % i)
        store_update.update(
            src=src_shard,
            tgt=tgt_shard,
            batch_size=opt.batch_size,
            batch_type=opt.batch_type,
        )


def _get_parser():
    parser = ArgumentParser(description='bulid_ds.py')
    opts.config_opts(parser)
    opts.translate_opts(parser)
    opts.store_update_opts(parser)
    return parser

def cli_main():
    parser = _get_parser()
    opt = parser.parse_args()
    main(opt)
if __name__ == "__main__":
    cli_main()
