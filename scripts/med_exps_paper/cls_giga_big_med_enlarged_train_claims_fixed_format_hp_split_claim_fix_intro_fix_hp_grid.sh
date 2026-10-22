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
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python train_luh/run_train_luh.py --config-dir=./configs --config-name=run_train_luq_linear_llama8b_tad_fix.yaml model.pretrained_model_name_or_path="/home/jovyan/rvanova/giga1/Lite_v2/hf" dataset.path="./med_data_paper/uhead_datasets/gigachat_big_with_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix" report_to=none ue_layer.head_cfg.head_type=linear_claim training_arguments.per_device_train_batch_size=8 training_arguments.per_device_eval_batch_size=8 training_arguments.gradient_accumulation_steps=4 do_predict=True ++low_memory_predict=True training_arguments.learning_rate=${lr} training_arguments.num_train_epochs=${num_train_epochs} ue_layer.pos_weight=${pos_weight} training_arguments.warmup_ratio=0.0 ue_layer.head_cfg.feature_extractor.1.attn_history_sz=${attn_history_sz} output_dir=./med_paper_results/gigachat_big_with_lite/hp/enlarged_train/fixed_format_claim_fix_intro_fix/med_data_relaxed_em_target_claim_attn_${attn_history_sz}_pos_${pos_weight}_ep_${num_train_epochs}_lr_${lr} dataset.validation="validation"
done
done
done
done

"""

# exps with 1.5k as train and val, and 260 as test - train with all fixes, BB claim-level
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python train_luh/run_train_luh.py --config-dir=./configs --config-name=run_train_luq_linear_llama8b_tad_fix.yaml model.pretrained_model_name_or_path="/home/jovyan/rvanova/giga1/Lite_v2/hf" dataset.path="./med_data_paper/uhead_datasets/gigachat_big_with_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix" report_to=none ue_layer.head_cfg.head_type=linear_claim training_arguments.per_device_train_batch_size=8 training_arguments.per_device_eval_batch_size=8 training_arguments.gradient_accumulation_steps=4 do_predict=True ++low_memory_predict=True training_arguments.learning_rate=0.1 training_arguments.num_train_epochs=10 ue_layer.pos_weight=5.0 training_arguments.warmup_ratio=0.0 ue_layer.head_cfg.feature_extractor.1.attn_history_sz=10 output_dir=./med_paper_results/gigachat_big_with_lite/enlarged_train/fixed_format_claim_fix_intro_fix/med_data_relaxed_em_target_claim


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python train_luh/run_train_luh.py --config-dir=./configs --config-name=run_train_luq_linear_llama8b_tad_fix.yaml model.pretrained_model_name_or_path="/home/jovyan/rvanova/giga1/Lite_v2/hf" dataset.path="./med_data_paper/uhead_datasets/gigachat_big_with_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix" report_to=none ue_layer.head_cfg.head_type=linear_claim training_arguments.per_device_train_batch_size=8 training_arguments.per_device_eval_batch_size=8 training_arguments.gradient_accumulation_steps=4 do_predict=True ++low_memory_predict=True training_arguments.learning_rate=0.1 training_arguments.num_train_epochs=10 ue_layer.pos_weight=0.33 training_arguments.warmup_ratio=0.0 ue_layer.head_cfg.feature_extractor.1.attn_history_sz=10 output_dir=./med_paper_results/gigachat_big_with_lite/enlarged_train/fixed_format_claim_fix_intro_fix/med_data_relaxed_em_target_claim_pos_033


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python train_luh/run_train_luh.py --config-dir=./configs --config-name=run_train_luq_linear_llama8b_tad_fix.yaml model.pretrained_model_name_or_path="/home/jovyan/rvanova/giga1/Lite_v2/hf" dataset.path="./med_data_paper/uhead_datasets/gigachat_big_with_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix" report_to=none ue_layer.head_cfg.head_type=linear_claim training_arguments.per_device_train_batch_size=8 training_arguments.per_device_eval_batch_size=8 training_arguments.gradient_accumulation_steps=4 do_predict=True ++low_memory_predict=True training_arguments.learning_rate=0.1 training_arguments.num_train_epochs=10 ue_layer.pos_weight=5.0 training_arguments.warmup_ratio=0.0 ue_layer.head_cfg.feature_extractor.1.attn_history_sz=10 output_dir=./med_paper_results/gigachat_big_with_lite/enlarged_train/fixed_format_claim_fix_intro_fix/med_data_relaxed_em_target_claim_attn_5_pos_5


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python train_luh/run_train_luh.py --config-dir=./configs --config-name=run_train_luq_linear_llama8b_tad_fix.yaml model.pretrained_model_name_or_path="/home/jovyan/rvanova/giga1/Lite_v2/hf" dataset.path="./med_data_paper/uhead_datasets/gigachat_big_with_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix" report_to=none ue_layer.head_cfg.head_type=linear_claim training_arguments.per_device_train_batch_size=8 training_arguments.per_device_eval_batch_size=8 training_arguments.gradient_accumulation_steps=4 do_predict=True ++low_memory_predict=True training_arguments.learning_rate=0.1 training_arguments.num_train_epochs=10 ue_layer.pos_weight=1.0 training_arguments.warmup_ratio=0.0 ue_layer.head_cfg.feature_extractor.1.attn_history_sz=10 output_dir=./med_paper_results/gigachat_big_with_lite/enlarged_train/fixed_format_claim_fix_intro_fix/med_data_relaxed_em_target_claim_pos_1


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python train_luh/run_train_luh.py --config-dir=./configs --config-name=run_train_luq_linear_llama8b_tad_fix.yaml model.pretrained_model_name_or_path="/home/jovyan/rvanova/giga1/Lite_v2/hf" dataset.path="./med_data_paper/uhead_datasets/gigachat_big_with_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix" report_to=none ue_layer.head_cfg.head_type=linear_claim training_arguments.per_device_train_batch_size=8 training_arguments.per_device_eval_batch_size=8 training_arguments.gradient_accumulation_steps=4 do_predict=True ++low_memory_predict=True training_arguments.learning_rate=0.1 training_arguments.num_train_epochs=13 ue_layer.pos_weight=5.0 training_arguments.warmup_ratio=0.0 ue_layer.head_cfg.feature_extractor.1.attn_history_sz=10 output_dir=./med_paper_results/gigachat_big_with_lite/enlarged_train/fixed_format_claim_fix_intro_fix/med_data_relaxed_em_target_claim_hp_prev

"""
"""
# now run for train on 260 samples

########### base exps with base models on k-fold
for idx in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python train_luh/run_train_luh.py --config-dir=./configs --config-name=run_train_luq_linear_llama8b_tad_fix.yaml model.pretrained_model_name_or_path="/home/jovyan/rvanova/giga1/Lite_v2/hf" dataset.path="../newly_builded_datasets/med_data/gigachat_big/med_data_giga_big_relaxed_em_target_claim_fixed_format_intro_fix_cv_${idx}" report_to=none ue_layer.head_cfg.head_type=linear_claim training_arguments.per_device_train_batch_size=8 training_arguments.per_device_eval_batch_size=8 training_arguments.gradient_accumulation_steps=4 do_predict=True ++low_memory_predict=True training_arguments.learning_rate=0.1 training_arguments.num_train_epochs=10 ue_layer.pos_weight=5.0 training_arguments.warmup_ratio=0.0 ue_layer.head_cfg.feature_extractor.1.attn_history_sz=10 output_dir=./DEBUG_workdir_med_data/gigachat_big/fixed_format_claim_fix_intro_fix/med_data_relaxed_em_target_claim_cv_${idx}

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python train_luh/run_train_luh.py --config-dir=./configs --config-name=run_train_luq_linear_llama8b_tad_fix.yaml model.pretrained_model_name_or_path="/home/jovyan/rvanova/giga1/Lite_v2/hf" dataset.path="../newly_builded_datasets/med_data/gigachat_big/med_data_giga_big_relaxed_em_target_claim_fixed_format_intro_fix_cv_${idx}" report_to=none ue_layer.head_cfg.head_type=linear_claim training_arguments.per_device_train_batch_size=8 training_arguments.per_device_eval_batch_size=8 training_arguments.gradient_accumulation_steps=4 do_predict=True ++low_memory_predict=True training_arguments.learning_rate=0.1 training_arguments.num_train_epochs=10 ue_layer.pos_weight=0.33 training_arguments.warmup_ratio=0.0 ue_layer.head_cfg.feature_extractor.1.attn_history_sz=10 output_dir=./DEBUG_workdir_med_data/gigachat_big/fixed_format_claim_fix_intro_fix/med_data_relaxed_em_target_claim_pos_033_cv_${idx}

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python train_luh/run_train_luh.py --config-dir=./configs --config-name=run_train_luq_linear_llama8b_tad_fix.yaml model.pretrained_model_name_or_path="/home/jovyan/rvanova/giga1/Lite_v2/hf" dataset.path="../newly_builded_datasets/med_data/gigachat_big/med_data_giga_big_relaxed_em_target_claim_fixed_format_intro_fix_cv_${idx}" report_to=none ue_layer.head_cfg.head_type=linear_claim training_arguments.per_device_train_batch_size=8 training_arguments.per_device_eval_batch_size=8 training_arguments.gradient_accumulation_steps=4 do_predict=True ++low_memory_predict=True training_arguments.learning_rate=0.1 training_arguments.num_train_epochs=10 ue_layer.pos_weight=5.0 training_arguments.warmup_ratio=0.0 ue_layer.head_cfg.feature_extractor.1.attn_history_sz=5 output_dir=./DEBUG_workdir_med_data/gigachat_big/fixed_format_claim_fix_intro_fix/med_data_relaxed_em_target_claim_attn_5_pos_5_cv_${idx}

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python train_luh/run_train_luh.py --config-dir=./configs --config-name=run_train_luq_linear_llama8b_tad_fix.yaml model.pretrained_model_name_or_path="/home/jovyan/rvanova/giga1/Lite_v2/hf" dataset.path="../newly_builded_datasets/med_data/gigachat_big/med_data_giga_big_relaxed_em_target_claim_fixed_format_intro_fix_cv_${idx}" report_to=none ue_layer.head_cfg.head_type=linear_claim training_arguments.per_device_train_batch_size=8 training_arguments.per_device_eval_batch_size=8 training_arguments.gradient_accumulation_steps=4 do_predict=True ++low_memory_predict=True training_arguments.learning_rate=0.1 training_arguments.num_train_epochs=10 ue_layer.pos_weight=1.0 training_arguments.warmup_ratio=0.0 ue_layer.head_cfg.feature_extractor.1.attn_history_sz=10 output_dir=./DEBUG_workdir_med_data/gigachat_big/fixed_format_claim_fix_intro_fix/med_data_relaxed_em_target_claim_pos_1_cv_${idx}
done
"""

# full_train exps - train on full 260 samples w/o CV, do not used in paper
"""
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python train_luh/run_train_luh.py --config-dir=./configs --config-name=run_train_luq_linear_llama8b_tad_fix.yaml model.pretrained_model_name_or_path="/home/jovyan/rvanova/giga1/Lite_v2/hf" dataset.path="../newly_builded_datasets/med_data/gigachat_big/med_data_giga_big_relaxed_em_target_claim_fixed_format_intro_fix_full_train" report_to=none ue_layer.head_cfg.head_type=linear_claim training_arguments.per_device_train_batch_size=8 training_arguments.per_device_eval_batch_size=8 training_arguments.gradient_accumulation_steps=4 do_predict=True ++low_memory_predict=True training_arguments.learning_rate=0.1 training_arguments.num_train_epochs=10 ue_layer.pos_weight=5.0 training_arguments.warmup_ratio=0.0 ue_layer.head_cfg.feature_extractor.1.attn_history_sz=10 output_dir=./DEBUG_workdir_med_data/gigachat_big/fixed_format_claim_fix_intro_fix/med_data_relaxed_em_target_claim_full_train

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python train_luh/run_train_luh.py --config-dir=./configs --config-name=run_train_luq_linear_llama8b_tad_fix.yaml model.pretrained_model_name_or_path="/home/jovyan/rvanova/giga1/Lite_v2/hf" dataset.path="../newly_builded_datasets/med_data/gigachat_big/med_data_giga_big_relaxed_em_target_claim_fixed_format_intro_fix_full_train" report_to=none ue_layer.head_cfg.head_type=linear_claim training_arguments.per_device_train_batch_size=8 training_arguments.per_device_eval_batch_size=8 training_arguments.gradient_accumulation_steps=4 do_predict=True ++low_memory_predict=True training_arguments.learning_rate=0.1 training_arguments.num_train_epochs=10 ue_layer.pos_weight=0.33 training_arguments.warmup_ratio=0.0 ue_layer.head_cfg.feature_extractor.1.attn_history_sz=10 output_dir=./DEBUG_workdir_med_data/gigachat_big/fixed_format_claim_fix_intro_fix/med_data_relaxed_em_target_claim_pos_033_full_train

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python train_luh/run_train_luh.py --config-dir=./configs --config-name=run_train_luq_linear_llama8b_tad_fix.yaml model.pretrained_model_name_or_path="/home/jovyan/rvanova/giga1/Lite_v2/hf" dataset.path="../newly_builded_datasets/med_data/gigachat_big/med_data_giga_big_relaxed_em_target_claim_fixed_format_intro_fix_full_train" report_to=none ue_layer.head_cfg.head_type=linear_claim training_arguments.per_device_train_batch_size=8 training_arguments.per_device_eval_batch_size=8 training_arguments.gradient_accumulation_steps=4 do_predict=True ++low_memory_predict=True training_arguments.learning_rate=0.1 training_arguments.num_train_epochs=10 ue_layer.pos_weight=5.0 training_arguments.warmup_ratio=0.0 ue_layer.head_cfg.feature_extractor.1.attn_history_sz=5 output_dir=./DEBUG_workdir_med_data/gigachat_big/fixed_format_claim_fix_intro_fix/med_data_relaxed_em_target_claim_attn_5_pos_5_full_train

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python train_luh/run_train_luh.py --config-dir=./configs --config-name=run_train_luq_linear_llama8b_tad_fix.yaml model.pretrained_model_name_or_path="/home/jovyan/rvanova/giga1/Lite_v2/hf" dataset.path="../newly_builded_datasets/med_data/gigachat_big/med_data_giga_big_relaxed_em_target_claim_fixed_format_intro_fix_full_train" report_to=none ue_layer.head_cfg.head_type=linear_claim training_arguments.per_device_train_batch_size=8 training_arguments.per_device_eval_batch_size=8 training_arguments.gradient_accumulation_steps=4 do_predict=True ++low_memory_predict=True training_arguments.learning_rate=0.1 training_arguments.num_train_epochs=10 ue_layer.pos_weight=1.0 training_arguments.warmup_ratio=0.0 ue_layer.head_cfg.feature_extractor.1.attn_history_sz=10 output_dir=./DEBUG_workdir_med_data/gigachat_big/fixed_format_claim_fix_intro_fix/med_data_relaxed_em_target_claim_pos_1_full_train
"""

