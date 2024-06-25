#!/usr/bin/env bash

python inference_corpus.py \
    --language "ptpt" \
    --checkpoint_model_path "../../checkpoints/qg_ptpt_ptt5_base_answer-text_question_seed_45_exp/model-epoch=XX-val_loss=YY.ckpt" \
    --predictions_save_path "../../predictions/qg_ptpt_ptt5_base_answer-text_question_seed_45_exp/" \
    --test_path "../../data/FairytaleQA_Dataset/processed_gen_v2_ptpt/test.json" \
    --model_name "unicamp-dl/ptt5-base-portuguese-vocab" \
    --tokenizer_name "unicamp-dl/ptt5-base-portuguese-vocab" \
    --max_len_input 512 \
    --max_len_output 128 \
    --encoder_info "answer_text" \
    --decoder_info "question" \
    --num_beams 5 \
    --num_return_sequences 1 \
    --repetition_penalty 1.0 \
    --length_penalty 1.0 \
    --seed_value 45 \