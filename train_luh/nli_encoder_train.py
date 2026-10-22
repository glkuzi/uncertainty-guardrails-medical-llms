import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import datasets
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    auc,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass
class Config:
    model_name: str = "answerdotai/ModernBERT-large"
    max_length: int = 1024
    data_path: str = "/home/jovyan/gkuzmin/llm_ue/newly_builded_datasets/med_data/gigachat_big/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split"
    output_dir: str = "/home/jovyan/gkuzmin/llm_ue/med_ue_res/encoder_ue/big_model/"
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 10  # 20
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 50
    eval_steps: int = 100
    seed: int = 42
    seq_level: bool = False


def show_config(cfg: Config) -> Dict:
    cfg_dict = cfg.__dict__
    for k, v in cfg_dict.items():
        print(f"{k}: {v}")
    return cfg_dict


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    precs, recs, _ = precision_recall_curve(labels, preds)
    pr_auc = auc(recs, precs)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an encoder-based NLI model with configurable hyperparameters."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=Config.model_name,
        help="Hugging Face model name or path.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=Config.max_length,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=Config.data_path,
        help="Path to the dataset saved with datasets.save_to_disk.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=Config.output_dir,
        help="Directory to save model, tokenizer, and predictions.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=Config.batch_size,
        help="Per-device batch size for training and evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=Config.gradient_accumulation_steps,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--num_epochs",
        type=float,
        default=Config.num_epochs,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=Config.lr,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=Config.weight_decay,
        help="Weight decay.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=Config.warmup_ratio,
        help="Warmup ratio for the learning rate scheduler.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=Config.logging_steps,
        help="Number of steps between logging events.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=Config.eval_steps,
        help="Number of steps between evaluation runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=Config.seed,
        help="Random seed.",
    )
    parser.add_argument(
        "--seq_level",
        action='store_true',
        default=Config.seq_level,
        help="Use seq level claims or not.",
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = Config(**vars(args))
    show_config(cfg)

    set_seed(cfg.seed)

    raw_ds = datasets.load_from_disk(cfg.data_path)

    # transform dataset - extract claims and flatten
    new_data_dict = {}
    for split in raw_ds.keys():
        buf_data = {"sentence1": [], "sentence2": [], "target": []}
        for el in raw_ds[split]:
            for idx in range(len(el["verified"])):
                buf_data["sentence1"].append(el["question"])
                buf_data["sentence2"].append(el["claims"][idx]["claim_text"])
                buf_data["target"].append(el["verified"][idx])
        new_data_dict[split] = datasets.Dataset.from_dict(buf_data)
    new_data = datasets.DatasetDict(new_data_dict)

    id2label = {0: "non-entailment", 1: "entailment"}
    label2id = {v: k for k, v in id2label.items()}

    # Load tokenizer and model head
    base_tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    def preprocess(batch):
        return base_tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            truncation=True,
            max_length=cfg.max_length,
        )

    tokenized_ds = new_data.map(
        preprocess, batched=True, remove_columns=["sentence1", "sentence2"]
    )

    # Ensure labels are integers 0/1
    for split in tokenized_ds:
        tokenized_ds[split] = tokenized_ds[split].rename_column("target", "labels")

    data_collator = DataCollatorWithPadding(base_tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        logging_steps=cfg.logging_steps,
        save_strategy="steps",
        save_steps=cfg.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",
        greater_is_better=True,
        report_to=["none"],
        seed=cfg.seed,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=base_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    trainer.save_model(cfg.output_dir)
    base_tokenizer.save_pretrained(cfg.output_dir)
    print(train_result)

    metrics = trainer.evaluate(tokenized_ds["validation"])
    print("val", metrics)

    metrics_test = trainer.evaluate(tokenized_ds["test"])
    print("test", metrics_test)
    preds = trainer.predict(tokenized_ds["test"])
    ue_scores = softmax(preds.predictions, axis=-1)[:, -1]

    # reshape ue scores
    import numpy as np
    buf_scores_orig = raw_ds["test"]["verified"]
    buf_scores = (-100 * np.ones((len(raw_ds["test"]["verified"]), 3))).tolist()
    idx = 0
    for score_idx, el in enumerate(buf_scores_orig):
        for inner_idx in range(len(el)):
            buf_scores[score_idx][inner_idx] = ue_scores[idx]
            idx += 1
        #if len(el) < 3 and not(cfg.seq_level):
        #    buf_scores[score_idx] += [-100] * (3 - len(el))
    save_dataset = datasets.Dataset.from_dict({"uncertainty_logits": buf_scores})
    save_path = Path(cfg.output_dir) / "predictions"
    save_dataset.save_to_disk(save_path)


if __name__ == "__main__":
    main()


