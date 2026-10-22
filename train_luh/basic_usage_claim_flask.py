import os
import json
import argparse
from flask import Flask, request, jsonify, abort
from lcsvec import lccs, lccs_length
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    set_seed,
)
import datasets
import warnings
from luh import AutoUncertaintyHead

from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
import torch
from itertools import chain
from typing import Optional
from dataclasses import dataclass

from scipy.special import expit
import numpy as np
import pandas as pd


@dataclass
class CausalLMWithUncertaintyOutput(CausalLMOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    uncertainty: torch.FloatTensor = None


class CausalLMWithUncertaintyLayerClaim(PreTrainedModel):
    def __init__(
        self, base_model, ue_head, ue_pos_weight: float, output_attention: bool = False
    ):
        super().__init__(PretrainedConfig())

        self.orig_base_model = base_model
        self.ue_head = ue_head
        self._output_attention = output_attention
        self._ue_pos_weight = ue_pos_weight

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
        return_dict=None,
        **kwargs,
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
            **kwargs,
        )
        outputs.logits = outputs.logits.float()
        logits = outputs.logits
        outputs.context_lengths = context_lengths

        ue_head_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "claims": claims,
        }

        uncertainty = self.ue_head(ue_head_input, outputs)
        loss = None

        return CausalLMWithUncertaintyOutput(
            loss=loss,
            logits=logits,
            uncertainty=uncertainty,
        )


def load_tokenizer(pretrained_model_name_or_path):
    padding_side = "right"
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        model_max_length=2400,
        padding_side=padding_side,
        cache_dir=None,
        token=None,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    pretrained_model_name_or_path, ue_layer_path, pos_weight=5.0, device_map="cuda"
):
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=device_map,
        cache_dir=None,
        token=None,
    )

    uq_head = AutoUncertaintyHead.from_pretrained(ue_layer_path, base_model)

    if uq_head.model_type == "claim":
        model = CausalLMWithUncertaintyLayerClaim(
            base_model,
            ue_head=uq_head,
            ue_pos_weight=pos_weight,
            output_attention=uq_head.output_attentions,
        )
        model.to(device_map)
    for name, param in model.named_parameters():
        # use only for infernce
        param.requires_grad = False
    return model


class DataCollatorForLanguageModelingWithUncertaintyClaim(
    DataCollatorForLanguageModeling
):
    def __init__(
        self, tokenizer, label_name="verified", giga_model=True, *args, **kwargs
    ):
        self._tokenizer = tokenizer
        self.label_name = label_name
        self.giga_model = giga_model
        super().__init__(tokenizer, *args, **kwargs)

    def _adjust_claim_positions(self, context_length, input_ids, claim_obj):
        claim_token_positions = claim_obj["aligned_token_ids"]
        mapping = []
        for idx, token_id in enumerate(input_ids[context_length:]):
            if token_id not in self.tokenizer.all_special_ids:
                mapping.append(idx)
        if len(mapping) != len(claim_token_positions):
            warnings.warn(
                "Number of aligned tokens does not match number of generated tokens in claim; please check if special_ids filtered"
            )
            # in case then we have special_ids in claim due to custom claim splitting
            for idx, token_id in enumerate(input_ids[context_length:]):
                if token_id in self.tokenizer.all_special_ids and idx < len(
                    claim_token_positions
                ):
                    del claim_token_positions[idx]
        adjusted_positions = [mapping[i] for i in claim_token_positions]
        return context_length + torch.tensor(adjusted_positions)

    def torch_call(self, examples):
        batch_size = len(examples)
        # Do padding
        batch = super().torch_call([{"input_ids": e["input_ids"]} for e in examples])
        # Construct context lengths
        context_lengths = []
        for i in range(batch_size):
            reply_len = len(examples[i]["input_ids"]) - len(
                examples[i]["prompt_tokens"]
            )
            if self._tokenizer.padding_side == "right":
                # in case of non-inst giga - remove 1 (eos)
                tok_diff = 2 if self.giga_model else 0
                context_lengths.append(len(examples[i]["prompt_tokens"]) - tok_diff)
            else:
                context_lengths.append(batch["input_ids"][i].shape[0] - reply_len)

        batch["context_lengths"] = torch.tensor(context_lengths)
        # Construct claim tensors
        all_claim_tensors = []
        for i in range(len(batch["input_ids"])):
            instance_claims = []
            for claim in examples[i]["claims"]:
                mask = torch.zeros(batch["input_ids"][i].shape, dtype=int)
                claim_token_positions = self._adjust_claim_positions(
                    batch["context_lengths"][i], batch["input_ids"][i], claim
                )
                if any(e > mask.shape[0] for e in claim_token_positions):
                    print("Error")

                mask[claim_token_positions] = 1
                instance_claims.append(mask[1:])  # ignoring <s>

            all_claim_tensors.append(
                torch.stack(instance_claims)
                if len(instance_claims) > 0
                else torch.zeros(0, full_len - 1, dtype=int)
            )

        batch["claims"] = all_claim_tensors
        dict_batch = dict(batch)
        return dict_batch


class UHeadClaimEvaluator:
    def __init__(self, model_name, uhead_name, device, batch_size=8, giga_model=True):
        self.uhead = load_model(model_name, uhead_name, device_map=device)
        self.uhead.eval()
        self.device = device
        self.tokenizer = load_tokenizer(model_name)
        self.collator = DataCollatorForLanguageModelingWithUncertaintyClaim(
            self.tokenizer, mlm=False, giga_model=giga_model
        )
        self.batch_size = batch_size
        self.giga_model = giga_model
        # self.thresholds = {0.3083: "low", 0.7: "medium", 1.0: "high"}
        self.thresholds = {0.37: "low", 0.76: "medium", 1.0: "high"}
        self.thresholds_int = {-1: "low", 0: "medium", 1: "high"}
        self.thr_vectorized = np.vectorize(lambda x: self.thresholds_int[x])

    def prepare_inputs(self, prompts, outputs, claims):
        assert len(prompts) == len(outputs)
        input_dict = {}
        input_dict["question"] = prompts
        input_dict["reply"] = outputs
        # to add - input_ids, greedy_ids
        # TODO: add batched version
        input_dict["input_ids"] = [
            self.tokenizer(el1 + str(el2))["input_ids"][:-1]
            for el1, el2 in zip(input_dict["question"], input_dict["reply"])
        ]
        input_dict["greedy_ids"] = [
            self.tokenizer(str(el2))["input_ids"][1:-1] for el2 in input_dict["reply"]
        ]
        input_dict["prompt_tokens"] = [
            self.tokenizer(el1, padding=False, return_tensors=None)["input_ids"]
            for el1 in input_dict["question"]
        ]
        # now process and add claims
        input_dict["claims"] = []
        for idx, el in enumerate(input_dict["greedy_ids"]):
            buf_claim_data = []
            buf_claims = claims[idx]
            prev_matched_tok_start = 0
            for inner_idx, text_claim in enumerate(buf_claims):
                # strip bos and eos
                claim_tokens = None
                claim_tokens = self.tokenizer(str(text_claim))["input_ids"][1:-1]
                matched_toks = [
                    x
                    for x in range(len(el))[prev_matched_tok_start:]
                    if el[x : x + len(claim_tokens)] == claim_tokens
                ]
                if len(matched_toks) == 0:
                    # try to find longest common continous subsequence
                    print(
                        f"Claim {text_claim} for sample {idx} is not matched! Switching to partial match..."
                    )
                    lccs_ = lccs(
                        np.array(claim_tokens), np.array(el)[prev_matched_tok_start:]
                    )
                    matched_toks = [
                        x
                        for x in range(len(el))[prev_matched_tok_start:]
                        if el[x : x + len(lccs_)] == lccs_
                    ]
                if len(matched_toks) == 0:
                    # if lccs is not found, propose that claim score is low
                    print(f"Cannot find LCCS for claim {text_claim} for sample {idx}!")
                else:
                    matched_toks = [
                        x + matched_toks[0] for x in range(len(claim_tokens))
                    ]
                    prev_matched_tok_start = matched_toks[0] + 1
                    buf_claim_data.append(
                        {
                            "aligned_token_ids": matched_toks,
                            "claim_text": str(text_claim),
                            "sentence": input_dict["reply"][idx],
                        }
                    )
            input_dict["claims"].append(buf_claim_data)
        return datasets.Dataset.from_dict(input_dict)

    def _threshold_scores(self, ue_scores):
        thresholds_keys = list(self.thresholds.keys())
        ue_scores = np.where(ue_scores <= thresholds_keys[0], -1, ue_scores)
        ue_scores = np.where(
            np.logical_and(
                ue_scores > thresholds_keys[0], ue_scores <= thresholds_keys[1]
            ),
            0,
            ue_scores,
        )
        ue_scores = np.where(
            np.logical_and(
                ue_scores > thresholds_keys[1], ue_scores <= thresholds_keys[2]
            ),
            1,
            ue_scores,
        )
        return self.thr_vectorized(ue_scores)

    def __call__(self, prompts, outputs, claims, use_threshold=False):
        """Calculate the confidence scores for each diagnosis (claim).

        Args:
            prompts: list of input prompts as it passed into LLM to obtain prediction.
            outputs: list of raw LLM text generations - the exact output from main LLM model.
            claims: list of lists of extracted diagnoses for each output.
            use_threshold: use or not thresholding on the calculated confidence scores.
        Returns:
            the confidence score for each diagnosis (claim).

        """
        # preprocess all samples
        samples = self.prepare_inputs(prompts, outputs, claims)
        batch_size = min(self.batch_size, len(samples))

        ue_scores_all = []
        with torch.no_grad():
            for start in range(0, len(samples), batch_size):
                indices = np.arange(start, min(start + batch_size, len(samples)))
                batch = samples.select(indices)
                # collate batch
                batch = self.collator.torch_call(batch)
                # move to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                    elif isinstance(value, list):
                        batch[key] = [
                            el.to(self.device) if hasattr(el, "to") else el
                            for el in value
                        ]
                # predict
                preds = self.uhead(**batch)
                # postprocess scores
                normalized_ue_scores = preds.uncertainty.detach().cpu().numpy()
                # replace padding due to unmatched claims or uneven diagnoses with 100 to get 0 confidence
                normalized_ue_scores[normalized_ue_scores == -100] = 100
                ue_scores = 1 - expit(normalized_ue_scores)
                ue_scores_all.append(ue_scores)
        ue_scores_all = np.concatenate(ue_scores_all, axis=0)
        if use_threshold:
            ue_scores_all = self._threshold_scores(ue_scores_all)
        return ue_scores_all.tolist()


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--uhead_path", type=str, required=True)
args = parser.parse_args()
app = Flask(__name__)
model_name = args.model_path
uhead_name = args.uhead_path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
evaluator = UHeadClaimEvaluator(model_name, uhead_name, device)


@app.route("/post", methods=["POST"])
def post():
    if not request.is_json:
        abort(400, description="Content-Type must be application/json")
    try:
        data = request.get_json()  # Use get_json()
        if data is None:
            abort(400, description="Request body cannot be empty")
    except Exception as e:
        # Catches a general exception in case of a malformed JSON body
        abort(400, description=f"Invalid JSON input: {e}")
    # check if json have all keys
    input_keys = ["prompts", "outputs", "claims", "use_threshold"]
    if not (all([key in data.keys() for key in input_keys])):
        abort(400, description="Missing keys in json")
    # TODO: add input data format check
    # print(inputs)
    scores_raw = evaluator(**data)
    return jsonify(scores_raw)


if __name__ == "__main__":
    app.run(debug=True)
