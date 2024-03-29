# dl-challenge-2

## Instructions

- run `pip install -r requirements.txt`

### For tokenization:
`python data/tokenization.py`

### For training:
`python model/train.py -s ../data/source_tokenizer -t ../data/target_tokenizer`

### For predicting:
`python predict.py -i data/example_input.txt -o model_pred.txt`

```
├── data
│   ├── datagen.py
│   ├── data.txt
│   ├── example_input.txt
│   ├── source_tokenizer    # saved tokenizer for the input, containes vocab, tokenizer and pickled tokenized data
│   │   └── vocab.json
│   ├── target_tokenizer    # saved tokenizer for the output, containes vocab, tokenizer and pickled tokenized data
│   │   └── vocab.json
│   ├── test.ipynb          # test notebook for tokenization
│   ├── tokenization.py     # tokenization script
│   ├── util.py
│   └── vars.py
├── Deep Learning Challenge.pdf
├── dl_problem_ahmed_mohamadeen.docx
├── essay_answers_ahmed_mohamadeen.docx
├── LICENSE
├── model
│   ├── eval.py
│   ├── model.py            # model class file
│   ├── predict.py          # predict script
│   └── train.py            # train script
└── README.md
```
