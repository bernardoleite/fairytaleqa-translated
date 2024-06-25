import argparse
import json
import sys
sys.path.append('../')
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from rouge_score import rouge_scorer, scoring
from statistics import mean

from evaluate import load

def get_rouge_option_rouge_scorer(references, predictions, lower_case=True, language="english"):
    rougeL_p_scores, rougeL_r_scores, rougeL_f_scores = [],[],[]
    P_INDEX, R_INDEX, F_INDEX = 0,1,2

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for index, question_refs in enumerate(references):
        references = question_refs
        gen = predictions[index]
        if lower_case:
            references = [ref.lower() for ref in references]
            gen = gen.lower()
        scores = scorer.score_multi(references, gen)

        rougeL_p_scores.append(scores['rougeL'][P_INDEX])
        rougeL_r_scores.append(scores['rougeL'][R_INDEX])
        rougeL_f_scores.append(scores['rougeL'][F_INDEX])

    return {"r": round(mean(rougeL_r_scores),5), "p": round(mean(rougeL_p_scores),5), "f": round(mean(rougeL_f_scores),5)}

def get_corpus_bleu(references, predictions, lower_case=False, language="english"):
    list_of_references = []
    hypotheses = []

    for question_refs in references:
        tmp_list_of_refs = []
        for ref in question_refs:
            ref_processed = word_tokenize(ref, language=language) # tokenize
            if lower_case:
                ref_processed = [each_string.lower() for each_string in ref_processed] # lowercase
            tmp_list_of_refs.extend([ref_processed])
        list_of_references.append(tmp_list_of_refs)

    for pred in predictions:
        pred_processed = word_tokenize(pred, language=language) # tokenize
        if lower_case:
            pred_processed = [each_string.lower() for each_string in pred_processed] # lowercase
        hypotheses.append(pred_processed)

    bleu_1 = corpus_bleu(list_of_references, hypotheses, weights = [1,0,0,0])
    bleu_2 = corpus_bleu(list_of_references, hypotheses, weights = [0.5,0.5,0,0])
    bleu_3 = corpus_bleu(list_of_references, hypotheses, weights = [1/3,1/3,1/3,0])
    bleu_4 = corpus_bleu(list_of_references, hypotheses, weights = [0.25,0.25,0.25,0.25])

    return {"Bleu_1": round(bleu_1,4), "Bleu_2": round(bleu_2,4), "Bleu_3": round(bleu_3,4), "Bleu_4": round(bleu_4,4)}

def run(args, preds_path):

    for pred_path in preds_path:
        print(pred_path)

        # Read predictions file
        with open(pred_path + "predictions.json", encoding='utf-8') as file:
            predictions = json.load(file)
        
        if args.encoder_info == "question_text":
            REFS_NAME = "answers_reference"
            PREDS_NAME = "gen_answer"
        elif args.encoder_info == "questiongen_text":
            REFS_NAME = "gen_answer"
            PREDS_NAME = "qa_answer"

        preds_refs_all = get_preds_refs_all(predictions, REFS_NAME, PREDS_NAME)
        preds_refs_ex = get_preds_refs_ex(predictions, REFS_NAME, PREDS_NAME)
        preds_refs_nar = get_preds_refs_nar(predictions, REFS_NAME, PREDS_NAME)
        preds_refs_nar_ex = get_preds_refs_nar_ex(predictions, REFS_NAME, PREDS_NAME)

        compute_qa_scores(preds_refs_all)
        compute_qa_scores(preds_refs_ex)
        #compute_qa_scores(preds_refs_nar)
        #compute_qa_scores(preds_refs_nar_ex)

        print("\n\n")

def get_preds_refs_all(predictions, REFS_NAME, PREDS_NAME):
    # Main change for QA evaluation
    answers_reference_all = [ref[REFS_NAME] for ref in predictions]
    answers_generated_all = [pred[PREDS_NAME] for pred in predictions]

    # https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
    answers_reference_all_flat = [item for sublist in answers_reference_all for item in sublist] # for exact_match

    return [
        {"all": [answers_generated_all, answers_reference_all, answers_reference_all_flat]}
    ]

def get_preds_refs_ex(predictions, REFS_NAME, PREDS_NAME):

    # https://stackoverflow.com/questions/29051573/python-filter-list-of-dictionaries-based-on-key-value
    predictions_explicit = list(filter(lambda d: d['ex-or-im1'] in ["explicit"], predictions))
    predictions_implicit = list(filter(lambda d: d['ex-or-im1'] in ["implicit"], predictions))
    
    answers_reference_explicit = [ref[REFS_NAME] for ref in predictions_explicit]
    answers_generated_explicit = [pred[PREDS_NAME] for pred in predictions_explicit]
    answers_reference_implicit = [ref[REFS_NAME] for ref in predictions_implicit]
    answers_generated_implicit = [pred[PREDS_NAME] for pred in predictions_implicit]

    # https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
    answers_reference_explicit_flat = [item for sublist in answers_reference_explicit for item in sublist] # for exact_match
    answers_reference_implicit_flat = [item for sublist in answers_reference_implicit for item in sublist] # for exact_match

    return [
        {"explicit": [answers_generated_explicit, answers_reference_explicit, answers_reference_explicit_flat]},
        {"implicit": [answers_generated_implicit, answers_reference_implicit, answers_reference_implicit_flat]}
    ]

def get_preds_refs_nar(predictions, REFS_NAME, PREDS_NAME):

    predictions_character = list(filter(lambda d: d['attributes'][0] in ["character"], predictions))
    predictions_setting = list(filter(lambda d: d['attributes'][0] in ["setting"], predictions))
    predictions_action = list(filter(lambda d: d['attributes'][0] in ["action"], predictions))
    predictions_feeling = list(filter(lambda d: d['attributes'][0] in ["feeling"], predictions))
    predictions_causal = list(filter(lambda d: d['attributes'][0] in ["causal"], predictions))
    predictions_outcome = list(filter(lambda d: d['attributes'][0] in ["outcome"], predictions))
    predictions_prediction = list(filter(lambda d: d['attributes'][0] in ["prediction"], predictions))

    answers_reference_character = [ref[REFS_NAME] for ref in predictions_character]
    answers_generated_character = [pred[PREDS_NAME] for pred in predictions_character]
    answers_reference_setting = [ref[REFS_NAME] for ref in predictions_setting]
    answers_generated_setting = [pred[PREDS_NAME] for pred in predictions_setting]
    answers_reference_action = [ref[REFS_NAME] for ref in predictions_action]
    answers_generated_action = [pred[PREDS_NAME] for pred in predictions_action]
    answers_reference_feeling = [ref[REFS_NAME] for ref in predictions_feeling]
    answers_generated_feeling = [pred[PREDS_NAME] for pred in predictions_feeling]
    answers_reference_causal = [ref[REFS_NAME] for ref in predictions_causal]
    answers_generated_causal = [pred[PREDS_NAME] for pred in predictions_causal]
    answers_reference_outcome = [ref[REFS_NAME] for ref in predictions_outcome]
    answers_generated_outcome = [pred[PREDS_NAME] for pred in predictions_outcome]
    answers_reference_prediction = [ref[REFS_NAME] for ref in predictions_prediction]
    answers_generated_prediction = [pred[PREDS_NAME] for pred in predictions_prediction]

    answers_reference_character_flat = [item for sublist in answers_reference_character for item in sublist] # for exact_match
    answers_reference_setting_flat = [item for sublist in answers_reference_setting for item in sublist] # for exact_match
    answers_reference_action_flat = [item for sublist in answers_reference_action for item in sublist] # for exact_match
    answers_reference_feeling_flat = [item for sublist in answers_reference_feeling for item in sublist] # for exact_match
    answers_reference_causal_flat = [item for sublist in answers_reference_causal for item in sublist] # for exact_match
    answers_reference_outcome_flat = [item for sublist in answers_reference_outcome for item in sublist] # for exact_match
    answers_reference_prediction_flat = [item for sublist in answers_reference_prediction for item in sublist] # for exact_match

    return [
        {"character": [answers_generated_character, answers_reference_character, answers_reference_character_flat]},
        {"setting": [answers_generated_setting, answers_reference_setting, answers_reference_setting_flat]},
        {"action": [answers_generated_action, answers_reference_action, answers_reference_action_flat]},
        {"feeling": [answers_generated_feeling, answers_reference_feeling, answers_reference_feeling_flat]},
        {"causal": [answers_generated_causal, answers_reference_causal, answers_reference_causal_flat]},
        {"outcome": [answers_generated_outcome, answers_reference_outcome, answers_reference_outcome_flat]},
        {"prediction": [answers_generated_prediction, answers_reference_prediction, answers_reference_prediction_flat]},
    ]


def get_preds_refs_nar_ex(predictions, REFS_NAME, PREDS_NAME):

    predictions_character = list(filter(lambda d: d['attributes'][0] in ["character"], predictions))
    predictions_character_explicit = list(filter(lambda d: d['ex-or-im1'] in ["explicit"], predictions_character))
    predictions_character_implicit = list(filter(lambda d: d['ex-or-im1'] in ["implicit"], predictions_character))

    predictions_setting = list(filter(lambda d: d['attributes'][0] in ["setting"], predictions))
    predictions_setting_explicit = list(filter(lambda d: d['ex-or-im1'] in ["explicit"], predictions_setting))
    predictions_setting_implicit = list(filter(lambda d: d['ex-or-im1'] in ["implicit"], predictions_setting))

    predictions_action = list(filter(lambda d: d['attributes'][0] in ["action"], predictions))
    predictions_action_explicit = list(filter(lambda d: d['ex-or-im1'] in ["explicit"], predictions_action))
    predictions_action_implicit = list(filter(lambda d: d['ex-or-im1'] in ["implicit"], predictions_action))

    predictions_feeling = list(filter(lambda d: d['attributes'][0] in ["feeling"], predictions))
    predictions_feeling_explicit = list(filter(lambda d: d['ex-or-im1'] in ["explicit"], predictions_feeling))
    predictions_feeling_implicit = list(filter(lambda d: d['ex-or-im1'] in ["implicit"], predictions_feeling))

    predictions_causal = list(filter(lambda d: d['attributes'][0] in ["causal"], predictions))
    predictions_causal_explicit = list(filter(lambda d: d['ex-or-im1'] in ["explicit"], predictions_causal))
    predictions_causal_implicit = list(filter(lambda d: d['ex-or-im1'] in ["implicit"], predictions_causal))

    predictions_outcome = list(filter(lambda d: d['attributes'][0] in ["outcome"], predictions))
    predictions_outcome_explicit = list(filter(lambda d: d['ex-or-im1'] in ["explicit"], predictions_outcome))
    predictions_outcome_implicit = list(filter(lambda d: d['ex-or-im1'] in ["implicit"], predictions_outcome))

    predictions_prediction = list(filter(lambda d: d['attributes'][0] in ["prediction"], predictions))
    predictions_prediction_explicit = list(filter(lambda d: d['ex-or-im1'] in ["explicit"], predictions_prediction))
    predictions_prediction_implicit = list(filter(lambda d: d['ex-or-im1'] in ["implicit"], predictions_prediction))


    answers_reference_character_explicit = [ref[REFS_NAME] for ref in predictions_character_explicit]
    answers_reference_character_implicit = [ref[REFS_NAME] for ref in predictions_character_implicit]
    answers_generated_character_explicit = [pred[PREDS_NAME] for pred in predictions_character_explicit]
    answers_generated_character_implicit = [pred[PREDS_NAME] for pred in predictions_character_implicit]

    answers_reference_setting_explicit = [ref[REFS_NAME] for ref in predictions_setting_explicit]
    answers_reference_setting_implicit = [ref[REFS_NAME] for ref in predictions_setting_implicit]
    answers_generated_setting_explicit = [pred[PREDS_NAME] for pred in predictions_setting_explicit]
    answers_generated_setting_implicit = [pred[PREDS_NAME] for pred in predictions_setting_implicit]

    answers_reference_action_explicit = [ref[REFS_NAME] for ref in predictions_action_explicit]
    answers_reference_action_implicit = [ref[REFS_NAME] for ref in predictions_action_implicit]
    answers_generated_action_explicit = [pred[PREDS_NAME] for pred in predictions_action_explicit]
    answers_generated_action_implicit = [pred[PREDS_NAME] for pred in predictions_action_implicit]

    answers_reference_feeling_explicit = [ref[REFS_NAME] for ref in predictions_feeling_explicit]
    answers_reference_feeling_implicit = [ref[REFS_NAME] for ref in predictions_feeling_implicit]
    answers_generated_feeling_explicit = [pred[PREDS_NAME] for pred in predictions_feeling_explicit]
    answers_generated_feeling_implicit = [pred[PREDS_NAME] for pred in predictions_feeling_implicit]

    answers_reference_causal_explicit = [ref[REFS_NAME] for ref in predictions_causal_explicit]
    answers_reference_causal_implicit = [ref[REFS_NAME] for ref in predictions_causal_implicit]
    answers_generated_causal_explicit = [pred[PREDS_NAME] for pred in predictions_causal_explicit]
    answers_generated_causal_implicit = [pred[PREDS_NAME] for pred in predictions_causal_implicit]

    answers_reference_outcome_explicit = [ref[REFS_NAME] for ref in predictions_outcome_explicit]
    answers_reference_outcome_implicit = [ref[REFS_NAME] for ref in predictions_outcome_implicit]
    answers_generated_outcome_explicit = [pred[PREDS_NAME] for pred in predictions_outcome_explicit]
    answers_generated_outcome_implicit = [pred[PREDS_NAME] for pred in predictions_outcome_implicit]

    answers_reference_prediction_explicit = [ref[REFS_NAME] for ref in predictions_prediction_explicit]
    answers_reference_prediction_implicit = [ref[REFS_NAME] for ref in predictions_prediction_implicit]
    answers_generated_prediction_explicit = [pred[PREDS_NAME] for pred in predictions_prediction_explicit]
    answers_generated_prediction_implicit = [pred[PREDS_NAME] for pred in predictions_prediction_implicit]


    answers_reference_character_flat_explicit = [item for sublist in answers_reference_character_explicit for item in sublist] # for exact_match
    answers_reference_character_flat_implicit = [item for sublist in answers_reference_character_implicit for item in sublist] # for exact_match
    
    answers_reference_setting_flat_explicit = [item for sublist in answers_reference_setting_explicit for item in sublist] # for exact_match
    answers_reference_setting_flat_implicit = [item for sublist in answers_reference_setting_implicit for item in sublist] # for exact_match

    answers_reference_action_flat_explicit = [item for sublist in answers_reference_action_explicit for item in sublist] # for exact_match
    answers_reference_action_flat_implicit = [item for sublist in answers_reference_action_implicit for item in sublist] # for exact_match

    answers_reference_feeling_flat_explicit = [item for sublist in answers_reference_feeling_explicit for item in sublist] # for exact_match
    answers_reference_feeling_flat_implicit = [item for sublist in answers_reference_feeling_implicit for item in sublist] # for exact_match

    answers_reference_causal_flat_explicit = [item for sublist in answers_reference_causal_explicit for item in sublist] # for exact_match
    answers_reference_causal_flat_implicit = [item for sublist in answers_reference_causal_implicit for item in sublist] # for exact_match
    
    answers_reference_outcome_flat_explicit = [item for sublist in answers_reference_outcome_explicit for item in sublist] # for exact_match
    answers_reference_outcome_flat_implicit = [item for sublist in answers_reference_outcome_implicit for item in sublist] # for exact_match

    answers_reference_prediction_flat_explicit = [item for sublist in answers_reference_prediction_explicit for item in sublist] # for exact_match
    answers_reference_prediction_flat_implicit = [item for sublist in answers_reference_prediction_implicit for item in sublist] # for exact_match

    return [
        {"character_explicit": [answers_generated_character_explicit, answers_reference_character_explicit, answers_reference_character_flat_explicit]},
        {"character_implicit": [answers_generated_character_implicit, answers_reference_character_implicit, answers_reference_character_flat_implicit]},

        {"setting_explicit": [answers_generated_setting_explicit, answers_reference_setting_explicit, answers_reference_setting_flat_explicit]},
        {"setting_implicit": [answers_generated_setting_implicit, answers_reference_setting_implicit, answers_reference_setting_flat_implicit]},

        {"action_explicit": [answers_generated_action_explicit, answers_reference_action_explicit, answers_reference_action_flat_explicit]},
        {"action_implicit": [answers_generated_action_implicit, answers_reference_action_implicit, answers_reference_action_flat_implicit]},

        {"feeling_explicit": [answers_generated_feeling_explicit, answers_reference_feeling_explicit, answers_reference_feeling_flat_explicit]},
        {"feeling_implicit": [answers_generated_feeling_implicit, answers_reference_feeling_implicit, answers_reference_feeling_flat_implicit]},

        {"causal_explicit": [answers_generated_causal_explicit, answers_reference_causal_explicit, answers_reference_causal_flat_explicit]},
        {"causal_implicit": [answers_generated_causal_implicit, answers_reference_causal_implicit, answers_reference_causal_flat_implicit]},

        {"outcome_explicit": [answers_generated_outcome_explicit, answers_reference_outcome_explicit, answers_reference_outcome_flat_explicit]},
        {"outcome_implicit": [answers_generated_outcome_implicit, answers_reference_outcome_implicit, answers_reference_outcome_flat_implicit]},

        {"prediction_explicit": [answers_generated_prediction_explicit, answers_reference_prediction_explicit, answers_reference_prediction_flat_explicit]},
        {"prediction_implicit": [answers_generated_prediction_implicit, answers_reference_prediction_implicit, answers_reference_prediction_flat_implicit]}
    ]

def compute_qa_scores(gens_refs):

    for elem in gens_refs:
        attribute_name = list(elem.keys())[0]

        gens_refs = list(elem.values())[0]
        gens = gens_refs[0]

        refs = gens_refs[1]
        refs_flat = gens_refs[2]

        print("Attribute: " + attribute_name)
        print("Sample size: ", len(gens))
        if len(gens) > 0:
            # EM
            exact_match_metric = load("exact_match")
            results = exact_match_metric.compute(predictions=gens, references=refs_flat)
            print("EM: ", round(results["exact_match"], 3))

            # Rouge
            rouge_scores =  get_rouge_option_rouge_scorer(refs, gens, lower_case=True, language=args.language)
            print("ROUGEL-F1: ", round(rouge_scores['f'],3))

        print("\n\n")

if __name__ == '__main__':
    # Initialize the Parser
    parser = argparse.ArgumentParser(description = 'Evaluation script fo QA.')

    preds_path = [
    "../predictions/qg_ptpt_ptt5_base_answer-text_question_seed_45_exp/",
    ]

    # Add arguments
    parser.add_argument('-lg','--language', type=str, metavar='', default="portuguese", required=False, help='Language for tokenize.')
    parser.add_argument('-enci','--encoder_info', type=str, metavar='', default="question_text", required=False, help='Information for encoding.')

    # Parse arguments
    args = parser.parse_args()

    # Start evaluation
    run(args, preds_path)