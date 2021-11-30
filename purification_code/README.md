
## What's in here?

This directory contains the code/models used to filter the Atomic10X corpus.

There are two main scripts:

1. `classify.py`: this script is used to train a new purification model
2. `predict.py`: this script is used to run the purification model on a large unlabelled corpus.

`classify.py` outputs two things: a model, and a results file, which
gives the dev/test performance. `predict.py`, in turn, takes in the
model and the results file.

## the data you need to run:

1. The labelled corpus, which can be downloaded from here: [purification_dataset.jsonl.zip](https://storage.googleapis.com/ai2-mosaic-public/projects/symbolic-knowledge-decoding/purification_models/purification_dataset.jsonl.zip).
2. The unlabelled candidates, which can be downloaded from here: [unique_data.jsonl.zip](https://storage.googleapis.com/ai2-mosaic-public/projects/symbolic-knowledge-decoding/purification_models/unique_data.jsonl.zip).

## example commands to run

To train a new classifier with the optimal hyperparameters from the paper:

```
python classify.py purification_dataset.jsonl
```
This will output a model and a results json. To run the model, pass the model and the results output by `classify.py` like:

```
python predict.py model~model=roberta-large-mnli~lr=5e-06~bs=128~dropout=0.10 results~model=roberta-large-mnli~lr=5e-06~bs=128~dropout=0.10.json unique_data.jsonl
```

## pretrained models

we originally trained our model using tensorflow 2.4. Since then,
tensorflow has been updated and saved models might not be compatible
with the new version. We make available the output of `classify.py`
both for the model described in the paper that was used to filter
Atomic10X, and also a re-trained version. They are available here:

- [retrained_new_tf.zip](https://storage.googleapis.com/ai2-mosaic-public/projects/symbolic-knowledge-decoding/purification_models/retrained_new_tf.zip)
- [original_from_paper.zip](https://storage.googleapis.com/ai2-mosaic-public/projects/symbolic-knowledge-decoding/purification_models/original_from_paper.zip)