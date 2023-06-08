"""
project: kNN-IME
file: kNN_transformer
author: JDS
create date: 2021/9/13 8:41
description: 
"""
from onmt.decoders.transformer import *
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


class KNNTransformerDecoder(TransformerDecoder):
    def __init__(
            self,
            num_layers,
            d_model,
            heads,
            d_ff,
            copy_attn,
            self_attn_type,
            dropout,
            attention_dropout,
            embeddings,
            max_relative_positions,
            aan_useffn,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            pos_ffn_activation_fn=ActivationFunction.relu,
    ):
        super(TransformerDecoder, self).__init__(
            d_model, copy_attn, embeddings, alignment_layer
        )

        self.transformer_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                    self_attn_type=self_attn_type,
                    max_relative_positions=max_relative_positions,
                    aan_useffn=aan_useffn,
                    full_context_alignment=full_context_alignment,
                    alignment_heads=alignment_heads,
                    pos_ffn_activation_fn=pos_ffn_activation_fn,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, tgt, memory_bank=None, step=None, **kwargs):
        """Decode, possibly stepwise."""
        if memory_bank is None:
            memory_bank = self.embeddings(tgt)
        if step == 0:
            self._init_cache(memory_bank)

        tgt_words = tgt[:, :, 0].transpose(0, 1)

        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        pad_idx = self.embeddings.word_padding_idx
        src_lens = kwargs["memory_lengths"]
        src_max_len = self.state["src"].shape[0]
        src_pad_mask = ~sequence_mask(src_lens, src_max_len).unsqueeze(1)
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        with_align = kwargs.pop("with_align", False)
        attn_aligns = []

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = (
                self.state["cache"]["layer_{}".format(i)]
                if step is not None
                else None
            )
            output, attn, attn_align = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=layer_cache,
                step=step,
                with_align=with_align,
            )
            if attn_align is not None:
                attn_aligns.append(attn_align)

        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn
        if with_align:
            attns["align"] = attn_aligns[self.alignment_layer]  # `(B, Q, K)`
            # attns["align"] = torch.stack(attn_aligns, 0).mean(0)  # All avg

        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}
            if isinstance(layer.self_attn, AverageAttention):
                layer_cache["prev_g"] = torch.zeros(
                    (batch_size, 1, depth), device=memory_bank.device
                )
            else:
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache

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
        mask = (knn_labels == self.padding_idx).unsqueeze(1)  # [bsz, 1, knn_num]
        scores[mask.expand(-1, tgt_len, -1)] -= 1e10
        knn_labels = knn_labels.unsqueeze(1).expand(-1, tgt_len, -1)  # [bsz, tgt_len, knn_num]
        if knn_num > self.topk > 0:
            topk_scores, topk_idxs = torch.topk(scores, dim=-1, k=self.topk)  # [bsz, tgt_len, topk]
            scores = topk_scores
            knn_labels = knn_labels.gather(dim=-1, index=topk_idxs)  # [bsz, tgt_len, topk]

        sim_probs = torch.softmax(scores / self.link_temperature, dim=-1)  # [bsz, tgt_len, knn_num]
        output = torch.zeros_like(sim_probs[:, :, 0]).unsqueeze(-1).repeat([1, 1, self.num_classes])  # [bsz, tgt_len, V]
        # output[b][t][knn_labels[b][t][k]] += link_probs[b][t][k]
        output = output.scatter_add(dim=2, index=knn_labels, src=sim_probs)
        return output
