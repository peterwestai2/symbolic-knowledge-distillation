# Symbolic Knowledge Distillation

This is the repository for the project [Symbolic Knowledge Distillation: from General Language Models to Commonsense Models](https://arxiv.org/abs/2110.07178)

We include data/model resources from the work: generated commonsense knowledge graph (ATOMIC-10X) and resulting commonsense models (COMET-DISTIL).

We also include a **quick start** section describing how to carry out key steps from the work.

___

# Data/Model Release

All [data and models](https://storage.googleapis.com/ai2-mosaic-public/projects/symbolic-knowledge-decoding/symbolic-knowledge-distillation.tar.gz) are available for download.

**ATOMIC10X.jsonl**

Each line contains 1 example (event/relation/inference triple) from the full **ATOMIC-10X** corpus. Variants filtered by the critic can be created using the rec_X fields to filter out data at various thresholds.


Fields:


- head: the event for this example (string)

- relation: the relation for this example, one of [xNeed, xReact, xEffect, xIntent, xAttr, xWant,HinderdBy]

- tail: the inference for this example (string)

- split: the dataset split for this example, one of: [train,test,val]

- rec_X: whether this example is cutoff by the critic at an expected recall of X. Our high-filtration corpus uses 0.5, medium uses 0.8 (bool)
        
- p_valid_model: the score assigned by the critic model (float)



**COMET-Distill models**

These follow a standard huggingface format. For each model, we include a file with .bin and config files. We also include outputs on a test set of inputs. Our human evaluation uses a subset of these. We also include a tokenizer.


___

# Quick Start

Here, we include simple ipython notebooks to carry out key steps from the work. Specifically, to generate data in the same form as ATOMIC-10X with **event generation** and **inference generation** using few-shot prompting. We also cover generation with the finetuned COMET-Distill models from our code release, for **knowledge based completion**. 

## Setup

**Get Code**

```
git clone https://github.com/peterwestai2/symbolic-knowledge-distillation.git
```

The list of packages and versions we use to run this code (other versions may work but are not guaranteed):

- torch==1.10.0
- transformers==4.11.3
- openai==0.3.0

See purification_code directory for information on running this portion of the pipeline.

**Get data/models**

```
wget https://storage.googleapis.com/ai2-mosaic-public/projects/symbolic-knowledge-decoding/symbolic-knowledge-distillation.tar.gz
tar -xvf symbolic-knowledge-distillation.tar.gz
```

## Event generation

*In this step, we generate seed events for the generated ATOMIC-10X corpus*

Run through generate_event.ipynb one block at a time. Instructions are included there. You will need a set of seed events which are used for few shot prompting. We provide such a file but you can substitute in your own, depending on the application.

Note that you will need an OpenAI API key in order to run our code (i.e. you need access to GPT-3). This is not strictly mandatory to apply Symbolic Knowledge Distillation, however. Simply, one can replace the calls to GPT-3 in our code to call a different powerful language model. We would suggest using T5-11B, GPT-Neo, or GPT-J (which we found produced promising results in early experiments). To incorporate this into our code, you will need to replace the gpt3_completion function with one that returns the generation and log-probabilities of one of these models. While we fully support this direction, we are not currently releasing an implementation. 

**Using our data release to skip this step:**

First, follow instructions above to **get data/models**. Then, run the final block in generate_events.ipynb. This will produce a file of pre-generated events from the Symbolic Knowledge Distillation paper, in the same format as if they had been generated above. 


## Inference Generation

*In this step, we generate the ATOMIC-10X corpus by generating inferences for event+relation pairs*

Note: this step requires a file of generated events, as produced in the **event generation** step. As with head generation, you will run each code block in generate_inferences.ipynb, setting parameters as desired. We include documentation within the notebook, and there are parameters which you can leave as default or change as desired. 

**Using our data release to skip this step:**

Follow instructions above to **get data/models**. No code is required if using pre-generated data, simply use the full dataset included in downloaded/ATOMIC10X.jsonl.


## Purification and training COMET-Distill
We do not include quick start code for these steps, as they are more involved. We will include more detailed instructions for these steps soon. 

## Comet Generation (Knowledge Base Completion)

*In this step, we generate ATOMIC-style commonsense form the COMET-distil model*

Follow the generate_COMET_distill ipynb. this gives a simple example of how to generate with a single COMET-distill model on a single input example. This can easily be extrapolated to a full test set. 

We suggest generating on inputs from the ATOMIC2020 test set [point to this] to see how the resulting models perform on other inputs (this is what we do in the paper). If you choose to do this, be careful to note that there may be overlap between ATOMIC2020 and your generated examples, so consider filtering out any event/relation pairs that appear in both your generated set and the ATOMIC2020 test set.
