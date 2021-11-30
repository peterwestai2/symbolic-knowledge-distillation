'''Inference code for:

Symbolic Knowledge Distillation: from General Language Models to Commonsense Models
Peter West, Chandra Bhagavatula, Jack Hessel, Jena D. Hwang, Liwei Jiang, Ronan Le Bras, Ximing Lu, Sean Welleck, Yejin Choi
https://arxiv.org/abs/2110.07178

This code contains the code for running the purification model described in Sec. 4 of the above paper.

First, you either need to train the model yourself (using classify.py)
or you need to download the pretrained model, which we used for
filtering in the paper.

you can then run over the full dataset by doing:

python predict.py model~model=roberta-large-mnli~lr=5e-06~bs=128~dropout=0.10 results~model=roberta-large-mnli~lr=5e-06~bs=128~dropout=0.10.json full_dataset_unique_examples.jsonl

'''
import argparse
import tensorflow as tf
import json
import numpy as np
import sklearn.metrics

import classify
import transformers
import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('model_results')
    parser.add_argument('to_filter')

    parser.add_argument('--recalls',
                        nargs='+',
                        type=float,
                        default=[.5,.6,.7,.8,.9])

    args = parser.parse_args()
    args.model_type = args.model.split('model=')[1].split('~')[0]

    return args


def main():
    args = parse_args()
    np.random.seed(1)

    all_val_precs, all_test_precs, mean_cutoff, mean_prec = [], [], [], []
    with open(args.model_results) as f:
        data = json.load(f)

        val_labels = data['val_labels']
        val_preds = data['val_preds']

        test_labels = data['test_labels']
        test_preds = data['test_preds']

        val_ps, val_rs, val_thresh = sklearn.metrics.precision_recall_curve(y_true=val_labels, probas_pred=val_preds)
        test_ps, test_rs, test_thresh = sklearn.metrics.precision_recall_curve(y_true=test_labels, probas_pred=test_preds)

        for r in args.recalls:
            idx = 0
            while val_rs[idx] > r:
                idx += 1
            print('Val precision@{:.0f}%: {:.3f}, cutoff={:.5f}'.format(r*100, val_ps[idx], val_thresh[idx]))
            all_val_precs.append(val_ps[idx])
            mean_thresh = val_thresh[idx]

            idx = 0
            while test_rs[idx] > r:
                idx += 1
            print('Test precision@{:.0f}%: {:.3f}, cutoff={:.5f}'.format(r*100, test_ps[idx], test_thresh[idx]))
            all_test_precs.append(test_ps[idx])

            mean_thresh += test_thresh[idx]
            mean_thresh /= 2
            mean_cutoff.append(mean_thresh)

            print('Mean threshold: {:.3f}'.format(mean_thresh))

            print('val prec with mean thresh:')
            pred_idxs = np.where(val_preds > mean_thresh)[0]
            all_binary_preds = np.array(val_preds > mean_thresh).astype(np.float32)
            mean_val_prec = np.mean(all_binary_preds[pred_idxs] == np.array(val_labels)[pred_idxs])
            print(mean_val_prec)

            print('test prec with mean thresh:')
            pred_idxs = np.where(test_preds > mean_thresh)[0]
            all_binary_preds = np.array(test_preds > mean_thresh).astype(np.float32)
            mean_test_prec = np.mean(all_binary_preds[pred_idxs] == np.array(test_labels)[pred_idxs])
            mean_prec.append((mean_val_prec + mean_test_prec) / 2.)
            print(mean_test_prec)

    to_filter = []
    with open(args.to_filter) as f:
        for line in tqdm.tqdm(f):
            c_jsonl = json.loads(line)
            c_jsonl['valid'] = -1
            to_filter.append(c_jsonl)

    print('filtering {}'.format(len(to_filter)))
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_type)
    pred_iter = classify.TextIterator(to_filter, tokenizer, args, batch_size=128)

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # Open a strategy scope.
    with strategy.scope():
        keras_model = tf.keras.models.load_model(args.model)
        keras_model.summary()
        preds = []
        for idx in range(5):
            preds.append(keras_model.predict(pred_iter, verbose=1).flatten())
        preds = np.mean(np.array(preds), axis=0)

    for idx, p in enumerate(preds):
        to_filter[idx]['p_valid_model'] = float(p)

    for recall, cutoff, prec in zip(args.recalls, mean_cutoff, mean_prec):
        fname = 'filtered_{}~estrecall={:.3f}~estprec={:.3f}~cutoff={:.3f}.jsonl'.format(args.to_filter.split('.')[0], recall, prec, cutoff)
        valid_idxs = np.where(preds > cutoff)[0]
        print('{} valid points in {}'.format(len(valid_idxs), fname))
        with open(fname, 'w') as f:
            for idx in valid_idxs:
                f.write(json.dumps(to_filter[idx]))
                if idx != valid_idxs[-1]:
                    f.write('\n')

    with open(args.to_filter.split('.')[0] + '_with_prob_est.json', 'w') as f:
        for idx, d in enumerate(to_filter):
            f.write(json.dumps(d))
            if idx != len(to_filter) - 1:
                f.write('\n')


if __name__ == '__main__':
    main()
