"""
project: kNN-IME
file: similarity heatmap
author: JDS
create date: 2021/10/27 19:43
description: 
"""

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
import onmt.decoders.ensemble
from onmt.knn.word_tracker import *


# src_sent = 'jiao tong shi gu dang shi ren wei tao bi fa lv zhui jiu'
# tgt_sent = '交 通 事 故 当 事 人 为 逃 避 法 律 追 究'

src_sent = 'mu qian nan zhi xing dian xian reng yi yao wu kong zhi fa zuo wei zhu'
tgt_sent = '目 前 难 治 性 癫 痫 仍 以 药 物 控 制 发 作 为 主'
pred_idx=6


class Sim():
    def __init__(self, model,fields,opt):
        self.model = model
        self.fields = fields
        self.opt=opt
        self.word_tracker=WordTracker(opt=self.opt,fields=self.fields,pretext_dataset=None,
                                      batch_size=self.opt.batch_size,model=self.model)

    def sim_tokens(self, src, tgt, pred_idx):
        tokens, sim = self.word_tracker.sim_token(src, tgt, pred_idx)  # 求句子的token相似度
        print(tgt)
        print(tokens)
        print(sim)

def main(opt):
    ArgumentParser.validate_translate_opts(opt)

    load_test_model = (
        onmt.decoders.ensemble.load_test_model
        if len(opt.models) > 1
        else onmt.model_builder.load_test_model
    )
    fields, model, model_opt = load_test_model(opt)
    sim= Sim(model, fields, opt)

    sim.sim_tokens(src_sent,tgt_sent,pred_idx)

def _get_parser():
    parser = ArgumentParser(description='Search.py')
    opts.config_opts(parser)
    opts.translate_opts(parser)
    opts.search_opts(parser)
    return parser

def cli_main():
    parser = _get_parser()
    opt = parser.parse_args()
    main(opt)
if __name__ == "__main__":
    cli_main()







