#!/usr/bin/env bash

python train.py \
	--language "ptpt" \
	--dir_model_name "qg_ptpt_ptt5_base_answer-text_question_seed_45_exp" \
	--model_name "unicamp-dl/ptt5-base-portuguese-vocab" \
	--tokenizer_name "unicamp-dl/ptt5-base-portuguese-vocab" \
	--train_path "../../data/FairytaleQA_Dataset/processed_gen_v2_ptpt/train.json" \
	--val_path "../../data/FairytaleQA_Dataset/processed_gen_v2_ptpt/val.json" \
	--test_path "../../data/FairytaleQA_Dataset/processed_gen_v2_ptpt/test.json" \
	--max_len_input 512 \
	--max_len_output 128 \
	--encoder_info "answer_text" \
	--decoder_info "question" \
	--max_epochs 1 \
	--batch_size 16 \
	--patience 2 \
	--optimizer "AdamW" \
	--learning_rate 0.0001 \
	--epsilon 0.000001 \
	--num_gpus 1 \
	--seed_value 45