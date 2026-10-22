#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
# test labeling for 3-5 additional models to increase data diversity
# google/gemma-2-9b-it
# tiiuae/Falcon3-7B-Instruct
# Qwen/Qwen2.5-7B-Instruct
# microsoft/Phi-3-small-8k-instruct

#ls;
#cd ../../
#eval "$(conda shell.bash hook)"
#conda activate /workspace-SR006.nfs1/gkuzmin/llm_ue/conda_llm_ue_env
#cd /workspace-SR006.nfs1/gkuzmin/llm_ue/llm-uncertainty-layer/scripts/med_exps_paper/

cd ../../

# examine several models
# alexyalunin/RuBioRoBERTa
# ai-forever/ruRoberta-large
# medicalai/ClinicalBERT
# now train NLI model with base params

CUDA_VISIBLE_DEVICES=0 python ./train_luh/nli_encoder_train.py --data_path "./med_data_paper/uhead_datasets/gigachat_big_with_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix_NEW_test_v2" --output_dir "./med_paper_results/gigachat_big_with_lite/enlarged_train/fixed_format_claim_fix_intro_fix/new_test_v2_evals/nli_model/ModernBERT-large";
CUDA_VISIBLE_DEVICES=0 python ./train_luh/nli_encoder_train.py --model_name "alexyalunin/RuBioRoBERTa" --max_length 512 --data_path "./med_data_paper/uhead_datasets/gigachat_big_with_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix_NEW_test_v2" --output_dir "./med_paper_results/gigachat_big_with_lite/enlarged_train/fixed_format_claim_fix_intro_fix/new_test_v2_evals/nli_model/RuBioRoBERTa";
CUDA_VISIBLE_DEVICES=0 python ./train_luh/nli_encoder_train.py --model_name "ai-forever/ruRoberta-large" --max_length 512 --data_path "./med_data_paper/uhead_datasets/gigachat_big_with_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix_NEW_test_v2" --output_dir "./med_paper_results/gigachat_big_with_lite/enlarged_train/fixed_format_claim_fix_intro_fix/new_test_v2_evals/nli_model/ruRoberta-large";
CUDA_VISIBLE_DEVICES=0 python ./train_luh/nli_encoder_train.py --model_name "medicalai/ClinicalBERT" --max_length 512 --data_path "./med_data_paper/uhead_datasets/gigachat_big_with_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix_NEW_test_v2" --output_dir "./med_paper_results/gigachat_big_with_lite/enlarged_train/fixed_format_claim_fix_intro_fix/new_test_v2_evals/nli_model/ClinicalBERT";


"""
CUDA_VISIBLE_DEVICES=0 python ./train_luh/nli_encoder_train.py --data_path "./med_data_paper/uhead_datasets/gigachat_big_with_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix_NEW_test" --output_dir "./med_paper_results/gigachat_big_with_lite/enlarged_train/fixed_format_claim_fix_intro_fix/new_test_evals/nli_model/ModernBERT-large";
CUDA_VISIBLE_DEVICES=0 python ./train_luh/nli_encoder_train.py --model_name "alexyalunin/RuBioRoBERTa" --max_length 512 --data_path "./med_data_paper/uhead_datasets/gigachat_big_with_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix_NEW_test" --output_dir "./med_paper_results/gigachat_big_with_lite/enlarged_train/fixed_format_claim_fix_intro_fix/new_test_evals/nli_model/RuBioRoBERTa";
CUDA_VISIBLE_DEVICES=0 python ./train_luh/nli_encoder_train.py --model_name "ai-forever/ruRoberta-large" --max_length 512 --data_path "./med_data_paper/uhead_datasets/gigachat_big_with_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix_NEW_test" --output_dir "./med_paper_results/gigachat_big_with_lite/enlarged_train/fixed_format_claim_fix_intro_fix/new_test_evals/nli_model/ruRoberta-large";
CUDA_VISIBLE_DEVICES=0 python ./train_luh/nli_encoder_train.py --model_name "medicalai/ClinicalBERT" --max_length 512 --data_path "./med_data_paper/uhead_datasets/gigachat_big_with_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix_NEW_test" --output_dir "./med_paper_results/gigachat_big_with_lite/enlarged_train/fixed_format_claim_fix_intro_fix/new_test_evals/nli_model/ClinicalBERT";
"""