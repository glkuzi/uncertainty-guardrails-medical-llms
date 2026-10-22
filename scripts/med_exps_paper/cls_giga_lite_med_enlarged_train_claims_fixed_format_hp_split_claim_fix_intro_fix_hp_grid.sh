#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
# test labeling for 3-5 additional models to increase data diversity
# google/gemma-2-9b-it
# tiiuae/Falcon3-7B-Instruct
# Qwen/Qwen2.5-7B-Instruct
# microsoft/Phi-3-small-8k-instruct
ls;
#cd ../../
eval "$(conda shell.bash hook)"
conda activate /workspace-SR006.nfs1/gkuzmin/llm_ue/conda_llm_ue_env
cd /workspace-SR006.nfs1/gkuzmin/llm_ue/llm-uncertainty-layer/scripts/med_exps_paper/

cd ../../

for pos_weight in 1.0 5.0
do
for attn_history_sz in 5 10
do
for num_train_epochs in 5 10 15
do
for lr in 0.1 0.01 0.001
do
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python train_luh/run_train_luh.py --config-dir=./configs --config-name=run_train_luq_linear_llama8b_tad_fix.yaml model.pretrained_model_name_or_path="/home/jovyan/rvanova/giga1/Lite_v2/hf" dataset.path="./med_data_paper/uhead_datasets/gigachat_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix" report_to=none ue_layer.head_cfg.head_type=linear_claim training_arguments.per_device_train_batch_size=8 training_arguments.per_device_eval_batch_size=8 training_arguments.gradient_accumulation_steps=4 do_predict=True ++low_memory_predict=True training_arguments.learning_rate=${lr} training_arguments.num_train_epochs=${num_train_epochs} ue_layer.pos_weight=${pos_weight} training_arguments.warmup_ratio=0.0 ue_layer.head_cfg.feature_extractor.1.attn_history_sz=${attn_history_sz} output_dir=./med_paper_results/gigachat_lite/hp/enlarged_train/fixed_format_claim_fix_intro_fix/med_data_relaxed_em_target_claim_attn_${attn_history_sz}_pos_${pos_weight}_ep_${num_train_epochs}_lr_${lr} dataset.validation="validation"
done
done
done
done




# examine several models
# alexyalunin/RuBioRoBERTa
# ai-forever/ruRoberta-large
# medicalai/ClinicalBERT
# now train NLI model with base params
CUDA_VISIBLE_DEVICES=0 python ./train_luh/nli_encoder_train.py --model_name "alexyalunin/RuBioRoBERTa" --max_length 512 --data_path "./med_data_paper/uhead_datasets/gigachat_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix" --output_dir "./med_paper_results/gigachat_lite/enlarged_train/fixed_format_claim_fix_intro_fix/nli_model/RuBioRoBERTa";
CUDA_VISIBLE_DEVICES=0 python ./train_luh/nli_encoder_train.py --model_name "ai-forever/ruRoberta-large" --max_length 512 --data_path "./med_data_paper/uhead_datasets/gigachat_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix" --output_dir "./med_paper_results/gigachat_lite/enlarged_train/fixed_format_claim_fix_intro_fix/nli_model/ruRoberta-large";
CUDA_VISIBLE_DEVICES=0 python ./train_luh/nli_encoder_train.py --model_name "medicalai/ClinicalBERT" --max_length 512 --data_path "./med_data_paper/uhead_datasets/gigachat_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix" --output_dir "./med_paper_results/gigachat_lite/enlarged_train/fixed_format_claim_fix_intro_fix/nli_model/ClinicalBERT";

# hp for default model
CUDA_VISIBLE_DEVICES=0 python ./train_luh/nli_encoder_train.py --lr 1e-4 --data_path "./med_data_paper/uhead_datasets/gigachat_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix" --output_dir "./med_paper_results/gigachat_lite/enlarged_train/fixed_format_claim_fix_intro_fix/nli_model/hp/ep10lr1e4/ModernBERT-large";
CUDA_VISIBLE_DEVICES=0 python ./train_luh/nli_encoder_train.py --lr 1e-3 --data_path "./med_data_paper/uhead_datasets/gigachat_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix" --output_dir "./med_paper_results/gigachat_lite/enlarged_train/fixed_format_claim_fix_intro_fix/nli_model/hp/ep10lr1e3/ModernBERT-large";
CUDA_VISIBLE_DEVICES=0 python ./train_luh/nli_encoder_train.py --lr 1e-4 --num_epochs 20 --data_path "./med_data_paper/uhead_datasets/gigachat_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix" --output_dir "./med_paper_results/gigachat_lite/enlarged_train/fixed_format_claim_fix_intro_fix/nli_model/hp/ep20lr1e4/ModernBERT-large";
CUDA_VISIBLE_DEVICES=0 python ./train_luh/nli_encoder_train.py --lr 1e-5 --num_epochs 20 --data_path "./med_data_paper/uhead_datasets/gigachat_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix" --output_dir "./med_paper_results/gigachat_lite/enlarged_train/fixed_format_claim_fix_intro_fix/nli_model/hp/ep20lr1e5/ModernBERT-large";


# hp for best RuRoBERTa model
CUDA_VISIBLE_DEVICES=0 python ./train_luh/nli_encoder_train.py --model_name "ai-forever/ruRoberta-large" --max_length 512 --lr 1e-4 --data_path "./med_data_paper/uhead_datasets/gigachat_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix" --output_dir "./med_paper_results/gigachat_lite/enlarged_train/fixed_format_claim_fix_intro_fix/nli_model/hp/ep10lr1e4/ruRoberta-large";
CUDA_VISIBLE_DEVICES=0 python ./train_luh/nli_encoder_train.py --model_name "ai-forever/ruRoberta-large" --max_length 512 --lr 1e-3 --data_path "./med_data_paper/uhead_datasets/gigachat_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix" --output_dir "./med_paper_results/gigachat_lite/enlarged_train/fixed_format_claim_fix_intro_fix/nli_model/hp/ep10lr1e3/ruRoberta-large";
CUDA_VISIBLE_DEVICES=0 python ./train_luh/nli_encoder_train.py --model_name "ai-forever/ruRoberta-large" --max_length 512 --lr 1e-4 --num_epochs 20 --data_path "./med_data_paper/uhead_datasets/gigachat_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix" --output_dir "./med_paper_results/gigachat_lite/enlarged_train/fixed_format_claim_fix_intro_fix/nli_model/hp/ep20lr1e4/ruRoberta-large";
CUDA_VISIBLE_DEVICES=0 python ./train_luh/nli_encoder_train.py --model_name "ai-forever/ruRoberta-large" --max_length 512 --lr 1e-5 --num_epochs 20 --data_path "./med_data_paper/uhead_datasets/gigachat_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix" --output_dir "./med_paper_results/gigachat_lite/enlarged_train/fixed_format_claim_fix_intro_fix/nli_model/hp/ep20lr1e5/ruRoberta-large";
