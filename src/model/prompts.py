import sys
import json

def nar_EN_to_PT(NAR_ELEM_EN):
    nar_dict = {'character':'personagem','setting':'cenário','action':'ação','feeling':'sentimento','causal':'causal','outcome':'resultado','prediction':'previsão'}

    nar_pt = nar_dict[NAR_ELEM_EN]

    if nar_pt:
        return nar_pt
    else:
        print("Error during nar finding pair.")
        sys.exit()

def build_encoder_info_pt(encoder_info, data_row):
    if encoder_info == "text":
        input_concat = ' '.join(data_row['sections_texts'])
    elif encoder_info == "answer_text":
        input_concat = '<resposta>' + data_row['answers_reference'][0] + '<texto>' + ' '.join(data_row['sections_texts'])
    elif encoder_info == "nar_text":
        input_concat = '<nar>' + nar_EN_to_PT(data_row['attributes'][0]) + '<texto>' + ' '.join(data_row['sections_texts'])
    elif encoder_info == "question_text": #qa
        input_concat = '<pergunta>' + data_row['questions_reference'][0] + '<texto>' + ' '.join(data_row['sections_texts'])
    else:
        print("Error with encoder_info (portuguese).")
        sys.exit()
    return input_concat

def build_encoder_info_en(encoder_info, data_row):
    if encoder_info == "text":
        input_concat = ' '.join(data_row['sections_texts'])
    elif encoder_info == "answer_text":
        input_concat = '<answer>' + data_row['answers_reference'][0] + '<text>' + ' '.join(data_row['sections_texts'])
    elif encoder_info == "nar_text":
        input_concat = '<nar>' + data_row['attributes'][0] + '<text>' + ' '.join(data_row['sections_texts'])
    elif encoder_info == "question_text": #qa
        input_concat = '<question>' + data_row['questions_reference'][0] + '<text>' + ' '.join(data_row['sections_texts'])
    elif encoder_info == "questiongen_text": #qa2
        input_concat = '<question>' + data_row['gen_question'] + '<text>' + ' '.join(data_row['sections_texts'])
    else:
        print("Error with encoder_info (english).")
        sys.exit()
    return input_concat

def build_encoder_info_es(encoder_info, data_row):
    if encoder_info == "text":
        input_concat = ' '.join(data_row['sections_texts'])
    elif encoder_info == "answer_text":
        input_concat = '<respuesta>' + data_row['answers_reference'][0] + '<texto>' + ' '.join(data_row['sections_texts'])
    elif encoder_info == "question_text": #qa
        input_concat = '<pregunta>' + data_row['questions_reference'][0] + '<texto>' + ' '.join(data_row['sections_texts'])
    else:
        print("Error with encoder_info (spanish).")
        sys.exit()
    return input_concat

def build_encoder_info_fr(encoder_info, data_row):
    if encoder_info == "text":
        input_concat = ' '.join(data_row['sections_texts'])
    elif encoder_info == "answer_text":
        input_concat = '<repondre>' + data_row['answers_reference'][0] + '<texte>' + ' '.join(data_row['sections_texts'])
    elif encoder_info == "question_text": #qa
        input_concat = '<question>' + data_row['questions_reference'][0] + '<texte>' + ' '.join(data_row['sections_texts'])
    else:
        print("Error with encoder_info (french).")
        sys.exit()
    return input_concat

def build_encoder_info_it(encoder_info, data_row):
    if encoder_info == "text":
        input_concat = ' '.join(data_row['sections_texts'])
    elif encoder_info == "answer_text":
        input_concat = '<risposta>' + data_row['answers_reference'][0] + '<testo>' + ' '.join(data_row['sections_texts'])
    elif encoder_info == "question_text": #qa
        input_concat = '<questione>' + data_row['questions_reference'][0] + '<testo>' + ' '.join(data_row['sections_texts'])
    else:
        print("Error with encoder_info (italian).")
        sys.exit()
    return input_concat

def build_encoder_info_ro(encoder_info, data_row):
    if encoder_info == "text":
        input_concat = ' '.join(data_row['sections_texts'])
    elif encoder_info == "answer_text":
        input_concat = '<răspundețila>' + data_row['answers_reference'][0] + '<text>' + ' '.join(data_row['sections_texts'])
    elif encoder_info == "question_text": #qa
        input_concat = '<întrebare>' + data_row['questions_reference'][0] + '<text>' + ' '.join(data_row['sections_texts'])
    else:
        print("Error with encoder_info (Romanian).")
        sys.exit()
    return input_concat

def build_encoder_info(encoder_info, data_row, language):
    if language == "ptpt" or language == "ptbr":
        input_concat = build_encoder_info_pt(encoder_info, data_row)
    elif language == "es":
        input_concat = build_encoder_info_es(encoder_info, data_row)
    elif language == "fr":
        input_concat = build_encoder_info_fr(encoder_info, data_row)
    elif language == "it":
        input_concat = build_encoder_info_it(encoder_info, data_row)
    elif language == "ro":
        input_concat = build_encoder_info_ro(encoder_info, data_row)
    else:
        input_concat = build_encoder_info_en(encoder_info, data_row)

    return input_concat

def build_decoder_info_en(decoder_info, data_row):
    if decoder_info == "question":
        target_concat = data_row['questions_reference'][0]
    elif decoder_info == "question_answer":
        target_concat = '<question>' + data_row['questions_reference'][0] + '<answer>' + data_row['answers_reference'][0]
    elif decoder_info == "answer": #qa
        target_concat = data_row['answers_reference'][0]
    else:
        print("Error with decoder_info.")
        sys.exit()
    
    return target_concat

def build_decoder_info_pt(decoder_info, data_row):
    if decoder_info == "question":
        target_concat = data_row['questions_reference'][0]
    elif decoder_info == "question_answer": # one qa pair
        target_concat = '<pergunta>' + data_row['questions_reference'][0] + '<resposta>' + data_row['answers_reference'][0]
    elif decoder_info == "qas": # multiple qa pairs
        target_concat = ''
        for index, q in enumerate(data_row['questions_reference']): 
            target_concat = target_concat + '<pergunta>' + q + '<resposta>' + data_row['answers_reference'][index]
    elif decoder_info == "answer": #qa
        target_concat = data_row['answers_reference'][0]
    else:
        print("Error with decoder_info.")
        sys.exit()
    
    return target_concat

def build_decoder_info_es(decoder_info, data_row):
    if decoder_info == "question":
        target_concat = data_row['questions_reference'][0]
    elif decoder_info == "question_answer":
        target_concat = '<pregunta>' + data_row['questions_reference'][0] + '<respuesta>' + data_row['answers_reference'][0]
    elif decoder_info == "answer": #qa
        target_concat = data_row['answers_reference'][0]
    else:
        print("Error with decoder_info (spanish).")
        sys.exit()
    
    return target_concat

def build_decoder_info_fr(decoder_info, data_row):
    if decoder_info == "question":
        target_concat = data_row['questions_reference'][0]
    elif decoder_info == "question_answer":
        target_concat = '<question>' + data_row['questions_reference'][0] + '<repondre>' + data_row['answers_reference'][0]
    elif decoder_info == "answer": #qa
        target_concat = data_row['answers_reference'][0]
    else:
        print("Error with decoder_info (french).")
        sys.exit()
    
    return target_concat

def build_decoder_info_it(decoder_info, data_row):
    if decoder_info == "question":
        target_concat = data_row['questions_reference'][0]
    elif decoder_info == "question_answer":
        target_concat = '<questione>' + data_row['questions_reference'][0] + '<risposta>' + data_row['answers_reference'][0]
    elif decoder_info == "answer": #qa
        target_concat = data_row['answers_reference'][0]
    else:
        print("Error with decoder_info (italian).")
        sys.exit()
    
    return target_concat

def build_decoder_info_ro(decoder_info, data_row):
    if decoder_info == "question":
        target_concat = data_row['questions_reference'][0]
    elif decoder_info == "question_answer":
        target_concat = '<întrebare>' + data_row['questions_reference'][0] + '<răspundețila>' + data_row['answers_reference'][0]
    elif decoder_info == "answer": #qa
        target_concat = data_row['answers_reference'][0]
    else:
        print("Error with decoder_info (Romanian).")
        sys.exit()
    
    return target_concat

def build_decoder_info(decoder_info, data_row, language):
    if language == "ptpt" or language == "ptbr":
        target_concat = build_decoder_info_pt(decoder_info, data_row)
    elif language == "es":
        target_concat = build_decoder_info_es(decoder_info, data_row)
    elif language == "fr":
        target_concat = build_decoder_info_fr(decoder_info, data_row)
    elif language == "it":
        target_concat = build_decoder_info_it(decoder_info, data_row)
    elif language == "ro":
        target_concat = build_decoder_info_ro(decoder_info, data_row)
    else:
        target_concat = build_decoder_info_en(decoder_info, data_row)

    return target_concat