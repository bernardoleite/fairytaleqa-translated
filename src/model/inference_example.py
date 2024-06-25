from transformers import (
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    T5Tokenizer
)

import argparse
import sys
sys.path.append('../')

from models import T5FineTuner
from utils import currentdate
from utils import find_string_between_two_substring
import time
import os
import json
import torch
import special_tokens

def generate(args, device, qgmodel: T5FineTuner, tokenizer: T5Tokenizer, prompt: str) -> str:

    # enconding info
    source_encoding = tokenizer(
        prompt,
        max_length=args.max_len_input,
        padding='max_length',
        truncation = 'only_second',
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    # Put this in GPU (faster than using cpu)
    input_ids = source_encoding['input_ids'].to(device)
    attention_mask = source_encoding['attention_mask'].to(device)

    generated_ids = qgmodel.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_return_sequences=args.num_return_sequences,
        num_beams=args.num_beams,
        max_length=args.max_len_output,
        repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty,
        early_stopping=True, # defaults to False
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True
    )

    # Extract sequences from the generated output
    generated_ids_list = generated_ids['sequences']

    preds = {
        tokenizer.decode(generated_id, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        for generated_id in generated_ids_list
    }
    generated_text = ''.join(preds)
 
    return generated_text

def run(args, prompt):
    # Load args (needed for model init) and log json
    params_dict = dict(
        language = args.language,
        checkpoint_model_path = args.checkpoint_model_path,
        model_name = args.model_name,
        tokenizer_name = args.tokenizer_name,
        max_len_input = args.max_len_input,
        max_len_output = args.max_len_output,
        num_beams = args.num_beams,
        num_return_sequences = args.num_return_sequences,
        repetition_penalty = args.repetition_penalty,
        length_penalty = args.length_penalty,
        seed_value = args.seed_value
    )
    params = argparse.Namespace(**params_dict)

    # Load Tokenizer
    if "mt5" in args.tokenizer_name:
        t5_tokenizer = MT5Tokenizer.from_pretrained(args.tokenizer_name, model_max_length=512)
    else:
        t5_tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name, model_max_length=512)

    # Add special tokens to tokenizer
    if args.language == 'ptpt' or args.language == 'ptbr':
        t5_tokenizer.add_tokens(special_tokens.special_tokens_pt, special_tokens=True)
    elif args.language == 'es':
        t5_tokenizer.add_tokens(special_tokens.special_tokens_es, special_tokens=True)
    elif args.language == 'fr':
        t5_tokenizer.add_tokens(special_tokens.special_tokens_fr, special_tokens=True)
    elif args.language == 'it':
        t5_tokenizer.add_tokens(special_tokens.special_tokens_it, special_tokens=True)
    else:
        t5_tokenizer.add_tokens(special_tokens.special_tokens_en, special_tokens=True)

    # Load T5 base Model
    if "mt5" in args.model_name:
        t5_model = MT5ForConditionalGeneration.from_pretrained(args.model_name)
    else:
        t5_model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # https://stackoverflow.com/questions/69191305/how-to-add-new-special-token-to-the-tokenizer
    # https://discuss.huggingface.co/t/adding-new-tokens-while-preserving-tokenization-of-adjacent-tokens/12604
    t5_model.resize_token_embeddings(len(t5_tokenizer))

    # Load T5 fine-tuned model for QG
    checkpoint_model_path = args.checkpoint_model_path
    qgmodel = T5FineTuner.load_from_checkpoint(checkpoint_model_path, hparams=params, t5model=t5_model, t5tokenizer=t5_tokenizer)

    # Put model in freeze() and eval() model. Not sure the purpose of freeze
    # Not sure if this should be after or before changing device for inference.
    qgmodel.freeze()
    qgmodel.eval()

    # Put model in gpu (if possible) or cpu (if not possible) for inference purpose
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    qgmodel = qgmodel.to(device)
    print ("Device for inference:", device)

    # Generate questions and append predictions
    start_time_generate = time.time()

    generated_text = generate(args, device, qgmodel, t5_tokenizer, prompt)

    print("Inference completed!")

    end_time_generate = time.time()
    gen_total_time = end_time_generate - start_time_generate
    print("Inference time: ", gen_total_time)

    print("Generated Text:\n")
    print(generated_text)

if __name__ == '__main__':
    
    story_text = "Quando o Gon\u00e7alo foi pela primeira vez com o av\u00f4 ao jardim zool\u00f3gico, a bicharada parecia estar \u00e0 espera da visita. As zebras mostraram os dentes de riso, os le\u00f5es rugiram um grande bocejo, as girafas esticaram ainda mais o pesco\u00e7o, os macacos fingiram uma grande zaragata. Os tigres espregui\u00e7aram-se, espregui\u00e7aram-se, como se fossem um tapete... \u2013 \u00d3 av\u00f4, posso levar um tigre para casa? \u2013 pediu o Gon\u00e7alo. O av\u00f4 explicou-lhe que n\u00e3o era muito conveniente. \u00c9 que os tigres nem sempre est\u00e3o ensonados... \u2013 E se eu levasse um macaco? O av\u00f4 tamb\u00e9m n\u00e3o achou muito conveniente. \u2013 Os macacos gostam muito de saltaricar daqui para ali. S\u00e3o muito irrequietos. Os cortinados l\u00e1 de casa n\u00e3o iam aguentar... O Gon\u00e7alo entristeceu. At\u00e9, talvez, estivesse um pouco amuado... Por fim, quando ele prop\u00f4s que levassem um simp\u00e1tico urso de ar pachorrento, o av\u00f4 pensou um bocadinho e, ao contr\u00e1rio do que o Gon\u00e7alo esperava, s\u00f3 disse: \u2013 Vamos l\u00e1 ver o que \u00e9 que se pode arranjar... E n\u00e3o \u00e9 que, dias depois, um urso de pelo castanho e fofo apareceu l\u00e1 em casa?! Passou a ser a melhor companhia do Gon\u00e7alo. Mal o menino adormecia, o urso, p\u00e9 ante p\u00e9, sa\u00eda do quarto e voltava a ser quem era. Vejam do que um av\u00f4 \u00e9 capaz para ver o neto contente... Ant\u00f3nio Torrado, Gon\u00e7alo e a bicharada... e outra hist\u00f3ria, 1.\u00aa edi\u00e7\u00e3o, Lisboa, Edi\u00e7\u00f5es Asa II S.A., 2016 (adaptado)"

    prompt_text = '<texto>' + story_text
    prompt_answer = '<resposta>' + 'O Gonçalo.'
    prompt = prompt_text + prompt_answer

    # Initialize the Parser
    parser = argparse.ArgumentParser(description = 'Generate questions and save them to json file.')

    # Add arguments...

    # Add language
    parser.add_argument('-lang', '--language', type=str, metavar='', default="ptpt", required=False, help='Language of QG.')

    parser.add_argument('-cmp','--checkpoint_model_path', type=str, metavar='', default="../../checkpoints/ptpt/qg_ptpt_ptt5_v2_base_answer-text_question_seed_45/model-epoch=02-val_loss=1.30.ckpt", required=False, help='Model folder checkpoint path.')

    parser.add_argument('-mn','--model_name', type=str, metavar='', default="unicamp-dl/ptt5-base-portuguese-vocab", required=False, help='Model name.')
    parser.add_argument('-tn','--tokenizer_name', type=str, metavar='', default="unicamp-dl/ptt5-base-portuguese-vocab", required=False, help='Tokenizer name.')

    parser.add_argument('-mli','--max_len_input', type=int, metavar='', default=512, required=False, help='Max len input for encoding.')
    parser.add_argument('-mlo','--max_len_output', type=int, metavar='', default=128, required=False, help='Max len output for encoding.')

    parser.add_argument('-nb','--num_beams', type=int, metavar='', default=5, required=False, help='Number of beams.')
    parser.add_argument('-nrs','--num_return_sequences', type=int, metavar='', default=1, required=False, help='Number of returned sequences.')
    parser.add_argument('-rp','--repetition_penalty', type=float, metavar='', default=1.0, required=False, help='Repetition Penalty.')
    parser.add_argument('-lp','--length_penalty', type=float, metavar='', default=1.0, required=False, help='Length Penalty.')
    parser.add_argument('-sv','--seed_value', type=int, default=45, metavar='', required=False, help='Seed value.')

    # Parse arguments
    args = parser.parse_args()

    # Start tokenization, encoding and generation
    run(args, prompt)