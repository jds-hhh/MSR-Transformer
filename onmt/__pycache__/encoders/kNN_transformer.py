"""
project: kNN-IME
file: kNN_transformer.py
author: JDS
create date: 2021/9/12 20:27
description: 
"""
from onmt.encoders.transformer import *
import faiss
from typing import Any, Dict, List, Optional, NamedTuple
from torch import Tensor
import torch

EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Optional[Tensor]),  # B x T
        ("encoder_embedding", Optional[Tensor]),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("src_tokens", Optional[Tensor]),  # B x T
        ("src_lengths", Optional[Tensor]),  # B
        ("knn_feats", Optional[Tensor]),  # B x N x C
        ("knn_labels", Optional[Tensor]),  # B x N
    ],
)



class KNNTransformerEncoder(TransformerEncoder):
    def __init__(self, opt,num_layers, d_model, heads, d_ff, dropout,
                 attention_dropout, embeddings, max_relative_positions,
                 pos_ffn_activation_fn=ActivationFunction.relu):
        super(KNNTransformerEncoder, self).__init__(opt, num_layers, d_model, heads, d_ff, dropout,
                 attention_dropout, embeddings, max_relative_positions,
                 pos_ffn_activation_fn=ActivationFunction.relu)
        if opt.quantizer_path:
            self.quantizer = faiss.read_index(opt.quantizer_path)
        else:
            self.quantizer = None

    def forward(
            self,
            src,
            lengths,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
            **kwargs
    ):
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        mask = ~sequence_mask(lengths).unsqueeze(1)
        # Run the forward pass of every layer of the tranformer.
        encoder_states = [] if return_all_hiddens else None
        for layer in self.transformer:
            out = layer(out, mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(out)
        out = self.layer_norm(out)

        knn_feats = kwargs["knn_feats"]  # [B, N, h or M]  # todo: 直接在tgt处encode decoder hidden可能更快？
        bsz, n, d = knn_feats.size()
        if self.quantizer is not None and knn_feats.dtype == torch.uint8:
            knn_feats = self.quantizer.decode(knn_feats.view(-1, d)).view(bsz, n, -1)

        return EncoderOut(
            encoder_out=out,  # T x B x C
            encoder_padding_mask=mask,  # B x T
            encoder_embedding=emb,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src,
            src_lengths=None,
            knn_feats=knn_feats,
            knn_labels=kwargs["knn_labels"],
        )


