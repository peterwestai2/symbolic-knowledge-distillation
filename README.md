# Symbolic Knowledge Distillation

This is the repository for the project [Symbolic Knowledge Distillation: from General Language Models to Commonsense Models](https://arxiv.org/abs/2110.07178)

We include our generated commonsense knowledge graph (ATOMIC-10X) and resulting commonsense models (COMET-DISTIL).

Code for repeating our experiments will be available soon.

NOTE: For the time being, please contact the authors directly with any questions about data, models, and code

## Data/Model Release

All [data and models](https://storage.googleapis.com/ai2-mosaic-public/projects/symbolic-knowledge-decoding/symbolic-knowledge-distillation.tar.gz) are available for download.

**ATOMIC10X.jsonl**

Each line contains 1 example (event/relation/inference triple) from the full **ATOMIC-10X** corpus. Variants filtered by the critic can be created using the rec_X fields to filter out data at various thresholds.


Fields:


head: the event for this example (string)

relation: the relation for this example, one of [xNeed, xReact, xEffect, xIntent, xAttr, xWant,HinderdBy]

tail: the inference for this example (string)

split: the dataset split for this example, one of: [train,test,val]

rec_X: whether this example is cutoff by the critic at an expected recall of X. Our high-filtration corpus uses 0.5, medium uses 0.8 (bool)
        
p_valid_model: the score assigned by the critic model (float)



**COMET-Distill models**

These follow a standard huggingface format. For each model, we include a file with .bin and config files. We also include outputs on a test set of inputs. Our human evaluation uses a subset of these.

We also include a tokenizer.

Please contact the authors to learn more about using these models.

