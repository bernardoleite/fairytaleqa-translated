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

## Machine-Translated Data
You can find here the machine-translated versions of FairytaleQA:
* [European Portuguese (pt-PT)](https://huggingface.co/datasets/benjleite/FairytaleQA-translated-ptPT)
* [Brazilian Portuguese (pt-BR)](https://huggingface.co/datasets/benjleite/FairytaleQA-translated-ptBR)
* [Spanish](https://huggingface.co/datasets/benjleite/FairytaleQA-translated-spanish)
* [French](https://huggingface.co/datasets/benjleite/FairytaleQA-translated-french)

We also have included machine-translated datasets for Italian and Romanian, although they were not studied in this research:
* [Italian](https://huggingface.co/datasets/benjleite/FairytaleQA-translated-italian)
* [Romanian](https://huggingface.co/datasets/benjleite/FairytaleQA-translated-romanian)

## Fine-Tuned Models
You can find here the fine-tuned models for **Question Answering** (QA):
* [European Portuguese (pt-PT)](https://huggingface.co/benjleite/ptt5-ptpt-qa)
* [Brazilian Portuguese (pt-BR)](https://huggingface.co/benjleite/ptt5-ptbr-qa)
* [Spanish](https://huggingface.co/benjleite/t5s-spanish-qa)
* [French](https://huggingface.co/benjleite/t5-french-qa)

You can find here the fine-tuned models for **Question Generation** QG:
* [European Portuguese (pt-PT)](https://huggingface.co/benjleite/ptt5-ptpt-qg)
* [Brazilian Portuguese (pt-BR)](https://huggingface.co/benjleite/ptt5-ptbr-qg)
* [Spanish](https://huggingface.co/benjleite/t5s-spanish-qg)
* [French](https://huggingface.co/benjleite/t5-french-qg)

## Prerequisites
Python 3 (tested with version 3.8.5 on Ubuntu 20.04.1 LTS)

## Installation and Configuration
TO DO.
## Usage
TO DO.

## Issues and Usage Q&A
To ask questions, report issues or request features, please use the GitHub Issue Tracker.

## Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks in advance!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
### Project
This code is released under the **MIT** license. For details, please see the file [LICENSE](https://github.com/bernardoleite/fairytaleqa-translated/blob/main/LICENSE) in the root directory. Please refer to machine-translated data and fine-tuned models links for their licenses.


## Acknowledgements
The base code is based on a [previous implementation](https://github.com/bernardoleite/question-generation-control).

## References
If you use this software in your work, please kindly cite our research.

Our paper (preprint - accepted for publication at ECTEL 2024):
```bibtex
@article{leite_fairytaleqa_translated_2024,
        title={FairytaleQA Translated: Enabling Educational Question and Answer Generation in Less-Resourced Languages}, 
        author={Bernardo Leite and Tomás Freitas Osório and Henrique Lopes Cardoso},
        year={2024},
        eprint={2406.04233},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
}
```

Original FairytaleQA paper:
```bibtex
@inproceedings{xu-etal-2022-fantastic,
    title = "Fantastic Questions and Where to Find Them: {F}airytale{QA} {--} An Authentic Dataset for Narrative Comprehension",
    author = "Xu, Ying  and
      Wang, Dakuo  and
      Yu, Mo  and
      Ritchie, Daniel  and
      Yao, Bingsheng  and
      Wu, Tongshuang  and
      Zhang, Zheng  and
      Li, Toby  and
      Bradford, Nora  and
      Sun, Branda  and
      Hoang, Tran  and
      Sang, Yisi  and
      Hou, Yufang  and
      Ma, Xiaojuan  and
      Yang, Diyi  and
      Peng, Nanyun  and
      Yu, Zhou  and
      Warschauer, Mark",
    editor = "Muresan, Smaranda  and
      Nakov, Preslav  and
      Villavicencio, Aline",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.34",
    doi = "10.18653/v1/2022.acl-long.34",
    pages = "447--460"
}
```

T5 model:
```bibtex
@article{raffel_2020_t5,
  author  = {Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu},
  title   = {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  journal = {Journal of Machine Learning Research},
  year    = {2020},
  volume  = {21},
  number  = {140},
  pages   = {1-67},
  url     = {http://jmlr.org/papers/v21/20-074.html},
  note={Model URL: \url{huggingface.co/google-t5/t5-base}}
}
```

## Contact
* Bernardo Leite, bernardo.leite@fe.up.pt
* Tomás Freitas Osório, tomas.s.osorio@gmail.com
* Henrique Lopes Cardoso, hlc@fe.up.pt
