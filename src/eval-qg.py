import argparse
import json
import sys
sys.path.append('../')
import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer, scoring
from statistics import mean
#from bert_score import score
#from bleurt import score

#from pinc import PINCscore

#from self_bleu import SelfBleu
#from metrics import Metrics

#import evaluate
#rom evaluate import load

#from collections import Counter
from nltk import ngrams

import itertools
import random

#from language_tool_python import LanguageTool

#from nlgeval import NLGEval
#nlgeval = NLGEval()  # loads the models

def get_nlgeval(references, predictions, lower_case=True, language="english"):
    list_of_references = []
    hypotheses = []

    for question_refs in references:
        tmp_list_of_refs = []
        for ref in question_refs:
            ref_processed = ref
            if lower_case:
                ref_processed = ref.lower() # lowercase
            tmp_list_of_refs.extend([ref_processed])
        list_of_references.append(tmp_list_of_refs)

    for pred in predictions:
        pred_processed = pred
        if lower_case:
            pred_processed = pred.lower() # lowercase
        hypotheses.append(pred_processed)

    #metrics_dict = nlgeval.compute_metrics(list_of_references, hypotheses)
    return metrics_dict

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

    return {"Bleu_1": round(bleu_1,5), "Bleu_2": round(bleu_2,5), "Bleu_3": round(bleu_3,5), "Bleu_4": round(bleu_4,5)}


def get_bert_score(references, predictions, lower_case=False, language="english"):
    (P, R, F), hashname = score(predictions, references, lang="en", return_hash=True)
    all_bert_scores_f1 = F.tolist()

    mean_all_bert_score_f1 = round(mean(all_bert_scores_f1),4)

    return mean_all_bert_score_f1

def get_bleurt_score(references, predictions, lower_case=False, language="english"):
    bleurt_checkpoint = "F:/bleurt-master/bleurt-master/bleurt/BLEURT-20" #update this

    list_of_references = []
    scorer = score.BleurtScorer(bleurt_checkpoint)

    #predictions = random.sample(predictions, 5)
    #references = random.sample(references, 5)

    for index, ref_group in enumerate(references):
        refs_bertscore = []
        candidate = predictions[index]
        for ref in ref_group:

            scores = scorer.score(references=[ref], candidates=[candidate])
            assert isinstance(scores, list) and len(scores) == 1
            refs_bertscore.append(scores[0])
        print(index, refs_bertscore)
        best_bertscore_f1 = max(refs_bertscore)
        idx_best_bertscore_f1 = refs_bertscore.index(best_bertscore_f1)
        chosen_ref = ref_group[idx_best_bertscore_f1]
        list_of_references.append(chosen_ref)
    

    all_scores = scorer.score(references=list_of_references, candidates=predictions)
    with open('bleurt_gpt3_ctrl_sk_a.json', 'w') as f:
        json.dump(all_scores, f)

    return mean(all_scores)

def get_ppl_score(predictions, lower_case=False, language="english"):
    perplexity = evaluate.load("perplexity", module_type="metric")
    results_perplexity = perplexity.compute(model_id='gpt2', add_start_token=False, predictions=predictions)
    return results_perplexity

def calc_distinct_n_score(text, n):
    # Split the text into individual words
    words = text.split()

    # Calculate n-grams
    ngrams_list = list(ngrams(words, n))

    # Count the occurrences of each n-gram
    ngrams_count = Counter(ngrams_list)

    # Calculate the number of distinct n-grams
    distinct_ngrams = len(ngrams_count)

    # Calculate the distinct-n score
    distinct_n_score = distinct_ngrams / len(words)

    return distinct_n_score

def get_distinct_n_score(predictions, n):
    all_distinct_scores = []

    for pred in predictions:
        distinct_n_score = calc_distinct_n_score(pred, n)
        all_distinct_scores.append(distinct_n_score)

    with open('dist3_answers_gpt3_ctrl_sk_a.json', 'w') as f:
        json.dump(all_distinct_scores, f)

    return mean(all_distinct_scores)

def get_pinc_score(predictions, max_n_gram=3):
    # Create a PINC scorer with the desired max_n_gram
    pinc_scorer = PINCscore(max_n_gram)

    # Lists to store contexts and questions
    contexts = []
    questions = []

    # Iterate through instances
    for instance in predictions:
        # Concatenate sections_texts to create the context
        context = ' '.join(instance['sections_texts'])

        # Save the generated question
        question = instance['gen_question']

        # Append context and question to their respective lists
        contexts.append(context)
        questions.append(question)

    # Calculate PINC scores
    pinc_scores = pinc_scorer.score(contexts, questions)

    return mean(pinc_scores)

def get_self_bleu_score(predictions, n=3):
    # Write questions to questions_tmp.txt
    with open('questions_tmp.txt', 'w', encoding='utf-8') as txt_file:
        for prediction in predictions:
                txt_file.write(prediction + '\n')

    self_bleu = SelfBleu(test_text='questions_tmp.txt', gram=n)

    self_bleu_score_fast = self_bleu.get_bleu_fast()
    #print("Self-BLEU (Fast):", self_bleu_score_fast)

    #self_bleu_score_parallel = self_bleu.get_bleu_parallel()
    #print("Self-BLEU (Parallel):", self_bleu_score_parallel)

    return self_bleu_score_fast

def get_grammar_error_score(predictions):
    tool = LanguageTool('en-US')  # Specify the language ('en-US' for American English)

    total_errors = 0
    for sentence in predictions:
        matches = tool.check(sentence)  # Check for errors in the sentence
        filtered_matches = [match for match in matches if match.ruleId != 'MORFOLOGIK_RULE_EN_US']
        total_errors += len(filtered_matches)  # Count the number of errors in the sentence

        if len(filtered_matches) > 0:
            print(filtered_matches)
            print(sentence)
            print("\n")

    average_errors = total_errors / len(predictions)

    return average_errors

def compute_rouge_by_nar(predictions):

    predictions_character = list(filter(lambda d: d['attributes'][0] in ["character"], predictions))
    predictions_setting = list(filter(lambda d: d['attributes'][0] in ["setting"], predictions))
    predictions_action = list(filter(lambda d: d['attributes'][0] in ["action"], predictions))
    predictions_feeling = list(filter(lambda d: d['attributes'][0] in ["feeling"], predictions))
    predictions_causal = list(filter(lambda d: d['attributes'][0] in ["causal"], predictions))
    predictions_outcome = list(filter(lambda d: d['attributes'][0] in ["outcome"], predictions))
    predictions_prediction = list(filter(lambda d: d['attributes'][0] in ["prediction"], predictions))

    references_character = [ref['questions_reference'] for ref in predictions_character]
    predictions_character = [pred['gen_question'] for pred in predictions_character]
    rouge_scores =  get_rouge_option_rouge_scorer(references_character, predictions_character, lower_case=True, language=args.language)
    print("Mean rouge_scorer CHARACTER: ", round(rouge_scores['f'],3))

    references_setting = [ref['questions_reference'] for ref in predictions_setting]
    predictions_setting = [pred['gen_question'] for pred in predictions_setting]
    rouge_scores =  get_rouge_option_rouge_scorer(references_setting, predictions_setting, lower_case=True, language=args.language)
    print("Mean rouge_scorer SETTING: ", round(rouge_scores['f'],3))

    references_action = [ref['questions_reference'] for ref in predictions_action]
    predictions_action = [pred['gen_question'] for pred in predictions_action]
    rouge_scores =  get_rouge_option_rouge_scorer(references_action, predictions_action, lower_case=True, language=args.language)
    print("Mean rouge_scorer ACTION: ", round(rouge_scores['f'],3))

    references_feeling = [ref['questions_reference'] for ref in predictions_feeling]
    predictions_feeling = [pred['gen_question'] for pred in predictions_feeling]
    rouge_scores =  get_rouge_option_rouge_scorer(references_feeling, predictions_feeling, lower_case=True, language=args.language)
    print("Mean rouge_scorer FEELING: ", round(rouge_scores['f'],3))

    references_causal = [ref['questions_reference'] for ref in predictions_causal]
    predictions_causal = [pred['gen_question'] for pred in predictions_causal]
    rouge_scores =  get_rouge_option_rouge_scorer(references_causal, predictions_causal, lower_case=True, language=args.language)
    print("Mean rouge_scorer CAUSAL: ", round(rouge_scores['f'],3))

    references_outcome = [ref['questions_reference'] for ref in predictions_outcome]
    predictions_outcome = [pred['gen_question'] for pred in predictions_outcome]
    rouge_scores =  get_rouge_option_rouge_scorer(references_outcome, predictions_outcome, lower_case=True, language=args.language)
    print("Mean rouge_scorer OUTCOME: ", round(rouge_scores['f'],3))

    references_prediction = [ref['questions_reference'] for ref in predictions_prediction]
    predictions_prediction = [pred['gen_question'] for pred in predictions_prediction]
    rouge_scores =  get_rouge_option_rouge_scorer(references_prediction, predictions_prediction, lower_case=True, language=args.language)
    print("Mean rouge_scorer PREDICTION: ", round(rouge_scores['f'],3))

def run(args, preds_path):

    for pred_path in preds_path:

        print(pred_path)

        # Read predictions file
        with open(pred_path + "predictions.json") as file:
            predictions_all = json.load(file)

        #compute_rouge_by_nar(predictions_all)
        
        references = [ref['questions_reference'] for ref in predictions_all]
        predictions = [pred['gen_question'] for pred in predictions_all]
        #predictions_answers = [pred['gen_answer'] for pred in predictions_all]

        #references_flat = list(itertools.chain.from_iterable(references))

        # Get BLEU (results are the same as reported from Du et. al (2017))
        #score_corpus_bleu = get_corpus_bleu(references, predictions, lower_case=False, language=args.language)
        #print("Score Corpus Bleu: ", round(score_corpus_bleu['Bleu_4'],3))

        rouge_scores =  get_rouge_option_rouge_scorer(references, predictions, lower_case=True, language=args.language)
        print("Mean rouge_scorer: ", round(rouge_scores['f'],3))

        #bert_score = get_bert_score(references, predictions, lower_case=True, language=args.language)
        #print("Mean BERT_score: ", bert_score)

        #bleurt_score = get_bleurt_score(references, predictions, lower_case=True, language=args.language)
        #print("Mean BLEURT_score: ", bleurt_score)

        #ppl_score_references = get_ppl_score(references_flat, lower_case=False, language=args.language)
        #print("Mean Perplexity score (references): ", ppl_score_references["mean_perplexity"])

        #
        # Linguistic Quality for QUESTIONS #
        #

        #ppl_score_predictions = get_ppl_score(predictions, lower_case=False, language=args.language)
        #print("Mean Perplexity score (predictions): ", round(ppl_score_predictions["mean_perplexity"],3))

        #distinct_score = get_distinct_n_score(predictions, 3)
        #print("Mean Distinct-3 score: ", round(distinct_score, 3))

        #pinc_score = get_pinc_score(predictions_all, 3)
        #print("Mean PINC score: ", round(pinc_score, 3))

        #self_bleu_score  = get_self_bleu_score(predictions, 3)
        #print("Mean Self-Bleu score: ", round(self_bleu_score, 3))

        #grammar_error_score_preds = get_grammar_error_score(predictions)
        #print("Avg. Grammar Error score (predictions): ", round(grammar_error_score_preds,3))

        #
        # Linguistic Quality for ANSWERS #
        #

        #ppl_score_predictions = get_ppl_score(predictions_answers, lower_case=False, language=args.language)
        #print("Mean Perplexity score (predictions): ", round(ppl_score_predictions["mean_perplexity"],3))

        #distinct_score = get_distinct_n_score(predictions_answers, 3)
        #print("Mean Distinct-3 score: ", round(distinct_score,3))

        #grammar_error_score_preds = get_grammar_error_score(predictions_answers)
        #print("Avg. Grammar Error score (predictions): ", round(grammar_error_score_preds,3))

        print("\n\n")

if __name__ == '__main__':
    
    preds_path = [
    "../predictions/qg_ptpt_ptt5_base_answer-text_question_seed_45_exp/",
    ]

    # Initialize the Parser
    parser = argparse.ArgumentParser(description = 'Evaluation script for QG.')

    # Add arguments
    parser.add_argument('-lg','--language', type=str, metavar='', default="portuguese", required=False, help='Language for tokenize.')

    # Parse arguments
    args = parser.parse_args()

    # Start evaluation
    run(args, preds_path)