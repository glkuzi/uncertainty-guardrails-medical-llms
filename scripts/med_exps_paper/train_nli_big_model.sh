cd ../../
ls;
#conda init;
#conda activate /home/jovyan/gkuzmin/llm_ue/conda_llm_ue_env;
CUDA_VISIBLE_DEVICES=0;
python --data_path "./med_data_paper/uhead_datasets/gigachat_big_with_lite/med_data_1_5k_full_7b_relaxed_em_target_claim_hp_split_fixed_format_intro_fix" --output_dir "./med_paper_results/gigachat_big_with_lite/enlarged_train/fixed_format_claim_fix_intro_fix/nli_model/ModernBERT-large" ./train_luh/nli_encoder_train.py;