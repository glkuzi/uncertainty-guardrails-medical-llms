import hydra
from pathlib import Path
import os
import json
from scipy.special import expit
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
)
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
import numpy as np
import random
from itertools import chain

import torch
import torch.nn.init as init

from datasets import DatasetDict, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from transformers import logging as transformers_logging

from causal_lm_with_uncertainty_layer import CausalLMWithUncertaintyLayer
from causal_lm_with_uncertainty_layer_claim import CausalLMWithUncertaintyLayerClaim
from causal_lm_with_uncertainty_layer_claim_regression import CausalLMWithUncertaintyLayerClaimRegression
from luh.utils import load_any_dataset
from luh import AutoUncertaintyHead
from lm_polygraph.ue_metrics.pred_rej_area import PredictionRejectionArea
from optuna.samplers import TPESampler
import warnings

import logging

#transformers_logging.set_verbosity_info()
transformers_logging.set_verbosity_warning()
transformers_logging.enable_default_handler()

log = logging.getLogger()
hf_logger = logging.getLogger("transformers")


def load_tokenizer(config):
    padding_side = "right" if ("giga" in config.model.pretrained_model_name_or_path.lower()) or config.get("giga_mode") else "left"
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.pretrained_model_name_or_path,
        model_max_length=2400,
        padding_side=padding_side,
        cache_dir=getattr(config, 'hf_cache', None),
        token=getattr(config, 'hf_token', None),
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def model_with_bfloat(model_name):
    if "gigachat3" in model_name.lower():
        return True
    if "qwen" in model_name.lower():
        return True
    if "gemma" in model_name.lower():
        return True
    return False


def load_model(config):
    config.model.torch_dtype = globals().get(config.model.torch_dtype)

    use_bfloat_16 = model_with_bfloat(config.model.pretrained_model_name_or_path)
    log.info(f"Loading model {config.model.pretrained_model_name_or_path}...")
    if config.get("random_proxy", False):
        from transformers import AutoConfig
        mod_config = AutoConfig.from_pretrained(config.model.pretrained_model_name_or_path)
        base_model = AutoModelForCausalLM.from_config(mod_config)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model.pretrained_model_name_or_path,
            torch_dtype=torch.float16 if not use_bfloat_16 else torch.bfloat16,
            trust_remote_code=True,
            device_map=config.model.device_map,
            cache_dir=getattr(config, 'hf_cache', None),
            token=getattr(config, 'hf_token', None),
        )

    if config.ue_layer.path:
        uq_head = AutoUncertaintyHead.from_pretrained(config.ue_layer.path, base_model)

    else:
        uq_head = AutoUncertaintyHead.from_config(config.ue_layer.head_cfg, base_model)

        def reinitialize_weights(module):
            if hasattr(module, "weight") and module.weight is not None and (not(hasattr(module, "name") and "positional_encoding" in module.name)):
                if module.weight.ndim >= 2:
                    init.xavier_uniform_(module.weight)
                else:
                    init.uniform_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                init.zeros_(module.bias)

        uq_head.apply(reinitialize_weights)

    if uq_head.model_type == "claim":
        model = CausalLMWithUncertaintyLayerClaim(
            base_model,
            ue_head=uq_head,
            ue_pos_weight=config.ue_layer.pos_weight,
            output_attention=uq_head.output_attentions,
        )
    elif uq_head.model_type == "token":
        model = CausalLMWithUncertaintyLayer(
            base_model,
            ue_head=uq_head,
            ue_pos_weight=config.ue_layer.pos_weight,
            output_attention=uq_head.output_attentions,
        )
    elif uq_head.model_type == "claim_regression":
        model = CausalLMWithUncertaintyLayerClaimRegression(
            base_model,
            ue_head=uq_head,
            output_attention=uq_head.output_attentions,
            ue_ranking=config.ue_layer.get("ue_ranking", 0),
            ue_rank_margin=config.ue_layer.get("ue_rank_margin", 0.0),
            margin_loss=config.ue_layer.get("margin_loss", False),
            legacy_rank_calc=config.ue_layer.get("legacy_rank_calc", True),
            pairwise=config.ue_layer.get("pairwise", False),
            group_by_dataset=config.ue_layer.get("group_by_dataset", False),
        )

    for name, param in model.named_parameters():
        if "ue_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


def load_data(config, tokenizer):
    log.info(f"Loading dataset {config.dataset.path}...")
    dataset = load_any_dataset(config.dataset.path, config)

    if type(dataset) is not DatasetDict:
        dataset = DatasetDict({"train": dataset})

    if config.dataset.num_instances:
        dataset["train"] = dataset["train"].select(range(config.dataset.num_instances))

    tokenized_data = dataset["train"]

    if "test" not in dataset:
        log.info("Performing train-test split...")
        tokenized_data = tokenized_data.train_test_split(
            test_size=config.dataset.test_size
        )

    else:
        val_dataset_name = config.dataset.validation if hasattr(config.dataset, "validation") else "eval"
        tokenized_data = DatasetDict({"train": tokenized_data, "test": dataset[val_dataset_name]})

    def prompt_tokens(inst):
        try:
            if config.get("giga_mode") or config.get("skip_chat_template"):
                return {"prompt_tokens": tokenizer(inst["question"], padding=False, return_tensors=None)["input_ids"]}
            else:
                return {"prompt_tokens": tokenizer.apply_chat_template([{"role": "user", "content": inst["question"]}], add_generation_prompt=True)}
        except:
            # in case of non-instruct model
            return {"prompt_tokens": tokenizer(inst["question"], padding=False, return_tensors=None)["input_ids"]}

    tokenized_data = tokenized_data.map(prompt_tokens)
    log.info(f"Length of the training dataset: {len(tokenized_data['train'])}")
    log.info(f"Length of the testing dataset: {len(tokenized_data['test'])}")

    return tokenized_data


# def _add_attention_mask(e):
#     if "attention_mask" not in e.keys() or e["attention_mask"] is None:
#         e["attention_mask"] = [1 for _ in e["input_ids"]]
#     return e


class DataCollatorForLanguageModelingWithUncertainty(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, *args, **kwargs):
        self._tokenizer = tokenizer
        super().__init__(tokenizer, *args, **kwargs)

    def torch_call(self, examples):
        #examples = [_add_attention_mask(e) for e in examples]
        # ex = [{"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]} for e in examples]
        #examples = [_add_attention_mask(e) for e in examples]
        # ex = [
        #     {k: v for k, v in e.items() if k not in [
        #         "uncertainty_labels", "reply"
        #      ]} for e in examples
        # ]
        # batch = super().torch_call(ex)

        batch_size = len(examples)

        # Do padding of input_ids
        batch = super().torch_call([{'input_ids' : e['input_ids']} for e in examples]) 


        # Construct context lengths
        context_lengths = []
        for i in range(batch_size):
            reply_len = len(examples[i]['input_ids']) - len(examples[i]['prompt_tokens'])
            context_lengths.append(batch["input_ids"][i].shape[0] - reply_len)
        
        batch["context_lengths"] = torch.tensor(context_lengths)


        # Do padding of labels
        all_padded_labels = []
        for idx in range(len(examples)):
            uncertainty_labels = examples[idx]["uncertainty_labels"]
            difference = len(batch["input_ids"][0]) - len(uncertainty_labels)

            if self.tokenizer.padding_side == "right":
                # Llama 3.2
                raise Exception("Internal: detected right padding side, but set 'left' before")
                padded_labels = uncertainty_labels + [-100] * difference
            elif self.tokenizer.padding_side == "left":
                # Mistral, Gemma 2
                padded_labels = [-100] * difference + uncertainty_labels
            else:
                raise ValueError(f"Unknown padding side: {self.tokenizer.padding_side}")

            all_padded_labels.append(padded_labels)

        # print("Before:=========", batch["uncertainty_labels"])
        # for i in range(len(batch)):
        #     batch["uncertainty_labels"][i] = torch.tensor(all_padded_labels[i])
        batch["uncertainty_labels"] = torch.tensor(all_padded_labels)


        # context_lengths = []
        # for i in range(len(batch["input_ids"])):
        #     reply = examples[i]["reply"]
        #     input_ids = batch["input_ids"][i]
        #     ctx = 0
        #     pref_len = len(self._tokenizer.decode(input_ids).split(reply)[0])
        #     while ctx < len(input_ids) and len(self._tokenizer.decode(input_ids[:ctx + 1])) <= pref_len:
        #         ctx += 1
        #     context_lengths.append(ctx)
        
        # batch["context_lengths"] = torch.tensor(context_lengths)
        # print("After:=========",  batch["uncertainty_labels"])
        return batch
    

class DataCollatorForLanguageModelingWithUncertaintyClaim(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, label_name="verified", *args, **kwargs):
        self._tokenizer = tokenizer
        self.label_name = label_name
        super().__init__(tokenizer, *args, **kwargs)

    def _adjust_claim_positions(self, context_length, input_ids, claim_obj):
        claim_token_positions = claim_obj['aligned_token_ids']
        mapping = []
        for idx, token_id in enumerate(input_ids[context_length:]):
            if token_id not in self.tokenizer.all_special_ids:
                mapping.append(idx)

        #print(context_length)
        #print(mapping)
        #print(input_ids)
        #print(claim_token_positions)
        #print(self.tokenizer.decode(input_ids))
        #print(self.tokenizer.decode(input_ids[context_length:]))
        if len(mapping) != len(claim_token_positions):
            warnings.warn("Number of aligned tokens does not match number of generated tokens in claim; please check if special_ids filtered")
            # in case then we have special_ids in claim due to custom claim splitting
            for idx, token_id in enumerate(input_ids[context_length:]):
                if token_id in self.tokenizer.all_special_ids and idx < len(claim_token_positions):
                    del claim_token_positions[idx]
        #print(claim_token_positions)
        #print(mapping)
        #print(input_ids)
        #print(input_ids[context_length:])
        #print(self.tokenizer.decode(input_ids))
        #print(self.tokenizer.decode(input_ids[context_length:]))
        adjusted_positions = [mapping[i] for i in claim_token_positions]
        return context_length + torch.tensor(adjusted_positions)
    

    def torch_call(self, examples):
        batch_size = len(examples)

        # Do padding
        batch = super().torch_call([{'input_ids' : e['input_ids']} for e in examples]) 


        # Construct context lengths
        context_lengths = []
        for i in range(batch_size):
            reply_len = len(examples[i]['input_ids']) - len(examples[i]['prompt_tokens'])
            #print(len(examples[i]['prompt_tokens']))
            #print(examples[i]['prompt_tokens'])
            if self._tokenizer.padding_side == "right":
                # in case of non-inst giga - remove 1 (eos)
                # set tok_diff to 2 for new <diag> tags
                tok_diff = 2 if "giga" in self._tokenizer.name_or_path.lower() else 0
                if self._tokenizer.name_or_path == "ai-sage/GigaChat-20B-A3B-instruct-v1.5-bf16":
                    tok_diff = 0
                context_lengths.append(len(examples[i]['prompt_tokens'])-tok_diff)
            else:
                context_lengths.append(batch["input_ids"][i].shape[0] - reply_len)
        
        batch["context_lengths"] = torch.tensor(context_lengths)


        # Construct claim tensors
        all_claim_tensors = []
        for i in range(len(batch["input_ids"])):
            instance_claims = []
            for claim in examples[i]["claims"]:
                mask = torch.zeros(batch["input_ids"][i].shape, dtype=int)
                claim_token_positions = self._adjust_claim_positions(batch["context_lengths"][i], batch["input_ids"][i], claim)
                if any(e > mask.shape[0] for e in claim_token_positions):
                    print("Error")
                
                mask[claim_token_positions] = 1
                #instance_claims.append(mask[1:]) # ignoring <s>
                if self._tokenizer.name_or_path == "ai-sage/GigaChat-20B-A3B-instruct-v1.5-bf16":
                    instance_claims.append(mask[1:])
                else:
                    instance_claims.append(mask[1:])

            all_claim_tensors.append(torch.stack(instance_claims) if len(instance_claims) > 0 else torch.zeros(0, full_len - 1, dtype=int))

        batch["claims"] = all_claim_tensors


        # Construct labels
        all_labels = []
        for i in range(len(examples)):
            uncertainty_labels = examples[i][self.label_name]
            all_labels.append([e if not np.isnan(e) else -100 for e in uncertainty_labels])
        
        batch["verified"] = all_labels
        # optionally add info about sample origin - for ranking losses
        if "dataset_name" in examples[0].keys():
            batch["dataset_name"] = np.array([el["dataset_name"] for el in examples])
        else:
            batch["dataset_name"] = np.array(["empty" for el in examples])

        
        dict_batch = dict(batch)
        return dict_batch


def compute_claim_level_metrics(tokenized_data, logits):
    from itertools import chain
    
    claim_level_targets = list(chain(*tokenized_data["verified"]))

    num_instances = logits.shape[0]
    claim_level_preds = []
    for i in range(num_instances):
        prompt_tokens = tokenized_data[i]["prompt_tokens"] # TODO: this is incorrect due to padding 
        generated_tokens = tokenized_data[i]["input_ids"][len(prompt_tokens):]
        context_length = logits[i].shape[0] - len(generated_tokens)  # To mitigate padding 
        for claim in tokenized_data["claims"][i]:
            # compute ue score

            claim_preds = [logits[i, context_length + token - 1] for token in claim['aligned_token_ids']]
            ue_score = np.mean(claim_preds)
            claim_level_preds.append(ue_score)

    assert len(claim_level_targets) == len(claim_level_preds)

    mask = ~np.isnan(claim_level_targets)
    claim_level_targets = np.array(claim_level_targets)[mask]
    claim_level_preds = np.array(claim_level_preds)[mask]
    precs, recs, _ = precision_recall_curve(claim_level_targets, claim_level_preds)
    pr_auc = auc(recs, precs)
    return {"claim_level_pr_auc": pr_auc}


def compute_metrics(tokenized_data, eval_pred):
    logits, labels = eval_pred
    labels = labels[1]
    labels = labels[:, 1:] # Shifting labels
    logits = logits[1] if type(logits) is tuple else logits

    mask = (labels != -100).reshape(-1)
    labels = labels.reshape(-1)[mask]
    probas = expit(logits.reshape(-1)[mask])
    predictions = (probas > 0.5).astype(int)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    roc_auc = roc_auc_score(labels, probas)
    precs, recs, _ = precision_recall_curve(labels, probas)
    pr_auc = auc(recs, precs)

    neg_probas = 1.0 - probas
    neg_labels = 1.0 - labels
    neg_predictions = (neg_probas > 0.5).astype(int).reshape(-1)
    neg_f1 = f1_score(neg_labels, neg_predictions)
    neg_roc_auc = roc_auc_score(neg_labels, neg_probas)
    neg_precs, neg_recs, _ = precision_recall_curve(neg_labels, neg_probas)
    neg_pr_auc = auc(neg_recs, neg_precs)

    claim_level_metrics = compute_claim_level_metrics(tokenized_data, logits)

    final_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "neg_f1": neg_f1,
        "neg_roc_auc": neg_roc_auc,
        "neg_pr_auc": neg_pr_auc,
    }

    final_metrics.update(claim_level_metrics)
    return final_metrics


def compute_metrics_claims(tokenized_data, eval_pred):
    logits, labels = eval_pred
    
    labels = np.asarray([e if not np.isnan(e) else -100  for e in list(chain(*tokenized_data["verified"]))])


    mask = (labels != -100).reshape(-1)
    labels = labels.reshape(-1)[mask]

    logits = logits.reshape(-1)
    if logits.shape[0] != mask.shape[0]:
        logits = logits[logits != -100]
    probas = expit(logits)[mask]
    predictions = (probas > 0.5).astype(int)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    roc_auc = roc_auc_score(labels, probas)
    precs, recs, _ = precision_recall_curve(labels, probas)
    pr_auc = auc(recs, precs)

    final_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    return final_metrics


def compute_metrics_claims_regression(tokenized_data, eval_pred):
    prr = PredictionRejectionArea()
    logits, labels = eval_pred
    labels = np.asarray([e if not np.isnan(e) else -100  for e in list(chain(*tokenized_data["alignscore"]))])
    mask = (labels != -100).reshape(-1)
    labels = labels.reshape(-1)[mask]

    logits = logits.reshape(-1)
    logits = logits[logits != -100]
    probas = logits#expit(logits)[mask]
    predictions = probas

    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    prr_score = prr(-predictions, labels)

    final_metrics = {
        "mse": mse,
        "mae": mae,
        "prr": prr_score,
    }
    return final_metrics


class LoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            log.info(logs)


class TrainerCustom(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"
    ):
        return super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=["logits"],
            metric_key_prefix=metric_key_prefix,
        )


def wandb_save_directory(directory_path):
    import wandb

    for file_name in os.listdir(directory_path):
        full_path = os.path.join(directory_path, file_name)
        if os.path.isfile(full_path):  # Make sure it's a file, not a directory
            wandb.save(full_path)


def batched_predict(trainer, test_set, batch_size=100):
    """Call trainer.predict for subsets of test set to avoid OOM
    """
    logits = []
    uncertainty_logits = []
    for idx in range(int(len(test_set) / batch_size)):
        ids = np.arange(idx * batch_size, (idx+1) * batch_size)
        buf_test = test_set.select(ids)
        # override compute_metrics due to changed test set
        if trainer.model.ue_head.model_type == "claim":
            trainer.compute_metrics = lambda eval_pred_: compute_metrics_claims(buf_test, eval_pred_)
        else:
            trainer.compute_metrics = lambda eval_pred_: compute_metrics(buf_test, eval_pred_)
        buf_preds = trainer.predict(buf_test, ignore_keys=["logits"])
        if trainer.model.ue_head.model_type == "claim":
            logits.append(buf_preds[0])
            uncertainty_logits.append(buf_preds[0].squeeze())
        elif trainer.model.ue_head.model_type == "token":
            logits += list(buf_preds[0])
            uncertainty_logits += list(buf_preds[0].squeeze())
        #uncertainty_logits.append(expit(buf_preds[0].squeeze()))
        #print(buf_preds[0][0].shape())
    if trainer.model.ue_head.model_type == "claim":
        logits = np.concatenate(logits, axis=0)
        uncertainty_logits = np.concatenate(uncertainty_logits, axis=0)
    trainer.compute_metrics = lambda eval_pred_: compute_metrics_claims(test_set, eval_pred_)
    return logits, uncertainty_logits


def wrap_labels(example):
    example["alignscore"] = [example["alignscore"]]
    return example


hydra_cfg_path = os.environ.get("HYDRA_CONFIG", None)
hydra_cfg_dir = str(Path(hydra_cfg_path).parent) if hydra_cfg_path is not None else None
hydra_cfg_name = str(Path(hydra_cfg_path).name) if hydra_cfg_path is not None else None


@hydra.main(
    version_base=None,
    config_path=hydra_cfg_dir,
    config_name=hydra_cfg_name,
)
def main(config):
    output_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Output directory: {output_dir}")

    # setup huggingface logger
    hf_logger.handlers = []
    for h in log.handlers:
        hf_logger.addHandler(h)

    if config.report_to == "wandb":
        import wandb

        wandb_cfg = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        config_path_hydra = [
            path["path"]
            for path in HydraConfig.get().runtime.config_sources
            if path["schema"] == "file"
        ][0]
        wandb_cfg["HYDRA_CONFIG"] = (
            Path(config_path_hydra) / HydraConfig.get().job.config_name
        )
        os.environ["WANDB_DIR"] = str(Path(output_dir))
        project = os.environ["WANDB_PROJECT"]
        wandb.init(project=project, dir=output_dir, config=wandb_cfg)
        wandb_save_directory(Path(output_dir) / ".hydra")

    hf_logger.info("Init transformers logger.")

    random_seed = 42
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    set_seed(random_seed)
    np.random.seed(random_seed)

    if os.environ.get("CUDNN_DETERMINISTIC", "0") == "1":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = None
    f_model_init = None
    if config.do_hyperopt:

        def model_init(trial):
            if trial is None:
                return load_model(config)

            model_params = OmegaConf.to_container(config, resolve=True)
            log.info(repr(model_params))
            model_params["ue_layer"]["pos_weight"] = trial.params["pos_weight"]
            model_params["ue_layer"]["head_cfg"]["feature_extractor"][1]["attn_history_sz"] = trial.params["attn_history_sz"]
            if config.ue_layer.head_cfg.head_type == "linear_claim_regression":
                model_params["ue_layer"]["ue_ranking"] = trial.params["ue_ranking"]
                model_params["ue_layer"]["ue_rank_margin"] = trial.params["ue_rank_margin"]
            omega_model_params = OmegaConf.create(model_params)
            log.info(repr(omega_model_params))

            return load_model(omega_model_params)

        f_model_init = model_init

        log.info("Loading model...")
        model = load_model(config)
        log.info("Done.")
        log.info(repr(model))

    else:
        log.info("Loading model...")
        model = load_model(config)
        log.info("Done.")
        log.info(repr(model))

    log.info("Loading tokenizer...")
    tokenizer = load_tokenizer(config)
    log.info("Done.")

    log.info("Loading dataset...")

    tokenized_data = load_data(config, tokenizer)
    log.info("Done.")
    log.info(repr(tokenized_data))
    if model.ue_head.model_type == "claim_regression":
        # map alignscore to transform in to the same foramt as verified
        if isinstance(tokenized_data["train"][0]["alignscore"], float):
            tokenized_data = tokenized_data.map(wrap_labels)

    train_args = TrainingArguments(
        num_train_epochs=config.training_arguments.num_train_epochs,
        per_device_train_batch_size=config.training_arguments.per_device_train_batch_size,
        per_device_eval_batch_size=config.training_arguments.get("per_device_eval_batch_size", config.training_arguments.per_device_train_batch_size), # TODO: add parameter for eval batch size
        gradient_accumulation_steps=config.training_arguments.gradient_accumulation_steps,
        eval_accumulation_steps=4,
        learning_rate=config.training_arguments.learning_rate,
        weight_decay=config.training_arguments.weight_decay,
        max_grad_norm=config.training_arguments.max_grad_norm,
        warmup_ratio=config.training_arguments.warmup_ratio,
        lr_scheduler_type="linear",
        fp16=True,
        fp16_full_eval=False,
        load_best_model_at_end=True if (config.do_save_checkpoints and config.training_arguments.get("load_best_model_at_end", True)) else False,
        metric_for_best_model="pr_auc" if "regression" not in model.ue_head.model_type else "mse",
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch" if config.do_save_checkpoints else "no",
        output_dir=Path(output_dir) / "outputs",
        logging_dir=Path(output_dir) / "transformers_logs",
        report_to=config.report_to if config.report_to else None,
        include_num_input_tokens_seen=True,
        gradient_checkpointing=False,
        dataloader_num_workers=1,
        remove_unused_columns=False
        #label_names=["verified"]
    )

    if "claim" in model.ue_head.model_type:
        def dataset_filter(inst):
            return len(inst['claims']) > 0

        # tokenized_data = {
        #     split: ds.filter(dataset_filter)
        #     for split, ds in tokenized_data.items()
        # }
        tokenized_data = tokenized_data.filter(dataset_filter)
        data_collator = DataCollatorForLanguageModelingWithUncertaintyClaim(tokenizer, label_name="verified", mlm=False)
    elif model.ue_head.model_type == "token":
        data_collator = DataCollatorForLanguageModelingWithUncertainty(tokenizer, mlm=False)
    elif model.ue_head.model_type == "claim_regression":
        data_collator = DataCollatorForLanguageModelingWithUncertaintyClaim(tokenizer, label_name="alignscore", mlm=False)

    callbacks = [LoggerCallback()]
    if config.do_save_checkpoints:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    if model.ue_head.model_type == "claim":
        f_eval = lambda eval_pred_: compute_metrics_claims(tokenized_data["test"], eval_pred_)
    elif model.ue_head.model_type == "token":
        f_eval = lambda eval_pred_: compute_metrics(tokenized_data["test"], eval_pred_)
    elif model.ue_head.model_type == "claim_regression":
        f_eval = lambda eval_pred_: compute_metrics_claims_regression(tokenized_data["test"], eval_pred_)

    if config.do_hyperopt:
        model = None
    trainer = TrainerCustom(
        model=model,
        model_init=f_model_init,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        args=train_args,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=f_eval,
    )

    if config.do_hyperopt:
        # This option is only for hperparameter optimization using optuna
        # For optimization with wandb, use the wandb sweep feature

        log.info(type(list(config.hp.learning_rate)))
        log.info(list(config.hp.learning_rate))
        #print(smth)
        def compute_objective(metrics):
            return metrics[config.hp.eval_metric]

        def hp_space(trial):
            return {
                "training_arguments": {
                    "learning_rate": trial.suggest_categorical(
                        "learning_rate", list(config.hp.learning_rate)
                    ),
                    "warmup_ratio": trial.suggest_categorical(
                        "warmup_ratio", list(config.hp.warmup_ratio)
                    ),
                    "num_train_epochs": trial.suggest_categorical(
                        "num_train_epochs", list(config.hp.num_train_epochs)
                    ),
                },
                "ue_layer": {
                    "attn_history_sz": trial.suggest_categorical(
                        "attn_history_sz", list(config.hp.ue_layer.head_cfg.feature_extractor_1.attn_history_sz)
                    ),
                    "pos_weight": trial.suggest_categorical(
                        "pos_weight", list(config.hp.ue_layer.pos_weight)
                    ),
                    "ue_ranking": trial.suggest_categorical(
                        "ue_ranking", list(config.hp.ue_layer.ue_ranking)
                    ),
                    "ue_rank_margin": trial.suggest_categorical(
                        "ue_rank_margin", list(config.hp.ue_layer.ue_rank_margin)
                    ),
                },
            }

        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=hp_space,
            n_trials=config.hp.n_trials,
            compute_objective=compute_objective,
            sampler=TPESampler(seed=random_seed),
        )

        log.info(f"Best metric: {repr(best_trial.objective)}")
        log.info(f"Best hyperparameters: {repr(str(best_trial.hyperparameters))}")
        with open(Path(output_dir) / "best_hyperparameters.json", "w") as f:
            json.dump(best_trial.hyperparameters, f)

    else:
        if config.do_train:
            trainer.model.orig_base_model.config.use_cache = False

            try:
                trainer.train(ignore_keys_for_eval=["logits"])
            except KeyboardInterrupt:
                log.info("Training interrupted.")
                
            log.info("Done with training.")

            if config.do_save_final_model:
                log.info("Saving model...")
                save_path = Path(output_dir) / "model"
                trainer.model.ue_head.save(save_path)
                # # trainer.save_model(Path(output_dir) / "training_dir" / "final_model")
                # torch.save(
                #     trainer.model.ue_head.state_dict(),
                #     Path(output_dir) / "ue_layer.pth",
                # )
                log.info(f"Saved to: {save_path}")
                if getattr(config, 'save_dir', None) is not None:
                    trainer.model.ue_head.save(Path(config.save_dir))
                    log.info(f"Saved to: {config.save_dir}.")

        if config.do_eval:
            log.info("Evaluating...")
            log.info(trainer.evaluate(ignore_keys=["logits"]))
            log.info("Done with evaluation.")

        if config.do_predict:
            if config.get("batched_predict", 0):
                log.info("Predicting...")
                logits, ue_logits = batched_predict(trainer, tokenized_data["test"], min([config.batched_predict, len(tokenized_data["test"])]))
                log.info("Done with prediction.")
                save_dataset = Dataset.from_dict({
                    "logits" : logits, 
                    "uncertainty_logits" : ue_logits})
            elif config.get("low_memory_predict", False):
                log.info("Predicting...")
                predictions = trainer.predict(tokenized_data["test"], ignore_keys=["logits"])
                log.info("Done with prediction.")
                save_dataset = Dataset.from_dict({
                    "uncertainty_logits" : predictions[0]})
            else:
                log.info("Predicting...")
                predictions = trainer.predict(tokenized_data["test"])
                log.info("Done with prediction.")
                save_dataset = Dataset.from_dict({
                    "logits" : predictions[0][0], 
                    "uncertainty_logits" : predictions[0][1]})
            save_path = Path(output_dir) / "predictions"
            log.info(f"Saving predictions to {save_path}")
            save_dataset.save_to_disk(save_path)

if __name__ == "__main__":
    main()
