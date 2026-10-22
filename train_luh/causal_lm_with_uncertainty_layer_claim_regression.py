from transformers.modeling_utils import PreTrainedModel, PretrainedConfig

import torch
from torch.nn import MSELoss, MarginRankingLoss

from itertools import chain

from causal_lm_with_uncertainty_layer import CausalLMWithUncertaintyOutput
import numpy as np


class CausalLMWithUncertaintyLayerClaimRegression(PreTrainedModel):
    def __init__(
        self,
        base_model,
        ue_head,
        output_attention: bool = False,
        ue_ranking: float = 0.0,
        ue_rank_margin: float = 0.0,
        margin_loss=False,
        legacy_rank_calc=True,
        pairwise=False,
        group_by_dataset=False,
    ):
        super().__init__(PretrainedConfig())

        self.orig_base_model = base_model
        self.ue_head = ue_head
        self._output_attention = output_attention
        self.ue_ranking = ue_ranking
        self.ue_rank_margin = ue_rank_margin
        self.margin_loss = margin_loss
        self.legacy_rank_calc = legacy_rank_calc
        self.pairwise = pairwise
        self.group_by_dataset = group_by_dataset

    def generate(self, *args, **kwargs):
        raise NotImplementedError

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        context_lengths=None,
        claims=None,
        verified=None,
        dataset_name=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = (
            return_dict
            if return_dict is not None
            else self.orig_base_model.config.use_return_dict
        )
        output_hidden_states = True
        output_attentions = self._output_attention

        outputs = self.orig_base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        logits = outputs.logits
        outputs.context_lengths = context_lengths

        ue_head_input = {"input_ids": input_ids, 
                         "attention_mask": attention_mask,
                         "claims": claims}

        uncertainty = self.ue_head(ue_head_input, outputs)

        if verified is not None:
            verified = torch.as_tensor(list(chain(*verified)), device=self.device)#verified.reshape(-1)
            mask = verified != -100
            #uncertainty_raveled = uncertainty[mask]
            uncertainty_raveled = uncertainty.reshape(-1)
            uncertainty_raveled = uncertainty_raveled[uncertainty_raveled != -100]
            uncertainty_raveled = uncertainty_raveled[mask]
            #uncertainty_raveled = torch.cat(uncertainty).reshape(-1)[mask]#uncertainty.reshape(-1)[mask]
            uncertainty_labels = verified[mask]
            loss_fct = MSELoss()
            ranking_loss_fct = MarginRankingLoss(self.ue_rank_margin)
            loss = loss_fct(
                uncertainty_raveled.to(torch.float32),
                uncertainty_labels.to(torch.float32),
            )
            if self.ue_ranking:
                if self.margin_loss:
                    if self.pairwise:
                        if len(set(dataset_name)) == 1 or not(self.group_by_dataset):
                            combinations = torch.combinations(uncertainty_raveled, 2)
                            combinations_target = torch.combinations(uncertainty_labels, 2)
                            lab_1, lab_2 = combinations[:, 0], combinations[:, 1]
                            bin_target = (combinations_target[:, 0] - combinations_target[:, 1]).sign()
                            rank_reg = ranking_loss_fct(lab_1, lab_2, bin_target)
                        else:
                            # group by dataset
                            unique_data = set(dataset_name)
                            rank_reg = None
                            data_count = 0
                            for dat in unique_data:
                                dat_ids = np.where(dataset_name == dat)[0]
                                if len(dat_ids) > 1:
                                    data_count += 1
                                    combinations = torch.combinations(uncertainty_raveled[dat_ids], 2)
                                    combinations_target = torch.combinations(uncertainty_labels[dat_ids], 2)
                                    lab_1, lab_2 = combinations[:, 0], combinations[:, 1]
                                    bin_target = (combinations_target[:, 0] - combinations_target[:, 1]).sign()
                                    if rank_reg is None:
                                        rank_reg = ranking_loss_fct(lab_1, lab_2, bin_target)
                                    else:
                                        rank_reg += ranking_loss_fct(lab_1, lab_2, bin_target)
                            rank_reg = rank_reg / data_count if rank_reg is not None else torch.tensor(0.0).to(uncertainty_raveled.device)
                    else:
                        bin_target = (uncertainty_labels[1:] - uncertainty_labels[:-1]).sign()
                        rank_reg = ranking_loss_fct(uncertainty_raveled[1:], uncertainty_raveled[:-1], bin_target)
                else:
                    if self.pairwise:
                        combinations = torch.combinations(uncertainty_raveled, 2)
                        combinations_target = torch.combinations(uncertainty_labels, 2)
                        lab_1, lab_2 = combinations[:, 0], combinations[:, 1]
                        rank_reg = -(lab_1 - lab_2) * (combinations_target[:, 0] - combinations_target[:, 1]).to(torch.float32)
                    else:
                        rank_reg = -(uncertainty_raveled[1:] - uncertainty_raveled[:-1]) * (uncertainty_labels[1:] - uncertainty_labels[:-1]).to(torch.float32)
                    if self.legacy_rank_calc:
                        rank_reg = torch.relu(rank_reg.sum())
                    else:
                        rank_reg = torch.relu(rank_reg).sum()
                loss = (1 - self.ue_ranking) * loss + self.ue_ranking * rank_reg
        else:
            loss = None

        return CausalLMWithUncertaintyOutput(
            loss=loss,
            logits=logits,
            uncertainty=uncertainty,
        )


def gradient_checkpointing_enable(self, *args, **kwargs):
    return self.base_model.gradient_checkpointing_enable(*args, **kwargs)
