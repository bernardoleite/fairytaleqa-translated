# FairytaleQA Translated: Enabling Educational Question and Answer Generation in Less-Resourced Languages
===============

Sample source code, data and models for our [ECTEL 2024](https://ea-tel.eu/ectel2024) accepted paper (preprint): [FairytaleQA Translated: Enabling Educational Question and Answer Generation in Less-Resourced Languages](https://arxiv.org/abs/2406.04233)

**Abstract:** Question Answering (QA) datasets are crucial in assessing reading comprehension skills for both machines and humans. While numerous datasets have been developed in English for this purpose, a noticeable void exists in less-resourced languages. To alleviate this gap, our paper introduces machine-translated versions of FairytaleQA, a renowned QA dataset designed to assess and enhance narrative comprehension skills in young children. By employing fine-tuned, modest-scale models, we establish benchmarks for both Question Generation (QG) and QA tasks within the translated datasets. In addition, we present a case study proposing a model for generating question-answer pairs, with an evaluation incorporating quality metrics such as question well-formedness, answerability, relevance, and children suitability. Our evaluation prioritizes quantifying and describing error cases, along with providing directions for future work. This paper contributes to the advancement of QA and QG research in less-resourced languages, promoting accessibility and inclusivity in the development of these models for reading comprehension.

**Authors:** Bernardo Leite, Tomás Freitas Osório, Henrique Lopes Cardoso

## Illustrative Example (English → Portuguese)
![overview_aied24](https://github.com/bernardoleite/fairytaleqa-translated/assets/22004638/292b0703-16b6-4261-9d7b-4f81edf6154f)

## Main Features
* Machine-Translated Data
* Training, inference and evaluation scripts for Question Answering (QA) & Generation (QG)
* Fine-tuned models for QA & QG

## Prerequisites
```bash
Python 3 (tested with version 3.8.5 on Ubuntu 20.04.1 LTS)
```
