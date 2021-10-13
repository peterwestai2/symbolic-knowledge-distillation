# symbolic-knowledge-distillation


### Files

[data and models](https://console.cloud.google.com/storage/browser/ai2-mosaic-public/projects/symbolic-knowledge-decoding/)

**mosaic-ckd_unique_data.jsonl**

Each line contains 1 example (event/relation/inference triple) from the full ATOMIC-10X corpus

Fields:

head: the event for this example (string)
relation: the relation for this example, one of [xNeed, xReact, xEffect, xIntent, xAttr, xWant,HinderdBy]
tail: the inference for this example (string)
split: the dataset split for this example, one of: [train,test,val]
rec_X: whether this example is cutoff by the critic at an expected recall of X.
        Our high-filtration corpus uses 0.5, medium uses 0.8 (bool)
p_valid_model: the score assigned by the critic model (float)
