'''Classification code for:

Symbolic Knowledge Distillation: from General Language Models to Commonsense Models
Peter West, Chandra Bhagavatula, Jack Hessel, Jena D. Hwang, Liwei Jiang, Ronan Le Bras, Ximing Lu, Sean Welleck, Yejin Choi
https://arxiv.org/abs/2110.07178

This code contains the code for training the purification model described in Sec. 4 of the above paper.

The best hyperparameter settings are set as defaults for this script:
our search is described in more detail in the paper.

So, to get started, you could just run:

python classify.py purification_dataset.jsonl --model roberta-large-mnli

This script will save a model that can then be applied to unlabelled data with predict.py
'''
import argparse
import tensorflow as tf
import transformers
import json
import numpy as np
import pprint
import math
import names #pip install names
import ftfy

_RELATIONS = {
    'HinderedBy':'can be hindered by',
    'xNeed':'but before, PersonX needed',
    'xWant':'as a result, PersonX wants',
    'xIntent':'because PersonX wanted',
    'xReact':'as a result, PersonX feels',
    'xAttr':'so, PersonX is seen as',
    'xEffect':'as a result, PersonX'
}


class TextIterator(tf.keras.utils.Sequence):
    def __init__(self, texts, tokenizer, args, batch_size=32, shuffle=False):
        self.texts = texts
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.args = args
        self._to_string = self._to_string_main

    def __len__(self):
        return math.ceil(len(self.texts) / self.batch_size)

    def _sep_pair_with_name(self, cur):
        p1 = names.get_first_name()
        p2 = p1
        while p2 == p1:
            p2 = names.get_first_name()
        cur = cur.replace('PersonX', p1)
        cur = cur.replace('PersonY', p2)
        cur = ftfy.fix_text(cur)
        cur = cur.split('**SEP**')
        return cur

    def _to_string_main(self, x):
        ''' NLI flatten, but with names. '''
        cur = '{}**SEP**{} {}'.format(x['head'], _RELATIONS[x['relation']], x['tail'])
        return self._sep_pair_with_name(cur)

    def __getitem__(self, idx):
        batch = self.texts[idx * self.batch_size:(idx + 1) *
                           self.batch_size]
        labels = np.array([1.0 if b['valid'] > 0 else 0.0 for b in batch])
        texts = [self._to_string(b) for b in batch]
        text_X = self.tokenizer(texts, return_tensors='np', padding=True)['input_ids']
        return text_X, labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.texts)


def default(obj):
    '''Utility to write obj'''
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_data')
    parser.add_argument('--model',
                        default='roberta-large-mnli',
                        choices=['roberta-base', 'roberta-large', 'roberta-large-mnli'])

    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='overall batch size (indep of num GPUs.)')

    parser.add_argument('--lr',
                        type=float,
                        default=.000005)

    parser.add_argument('--dropout',
                        type=float,
                        default=.1)

    parser.add_argument('--save_model',
                        type=int,
                        default=1,
                        help='should we also save the model?')

    return parser.parse_args()


class LearningRateLinearIncrease(tf.keras.callbacks.Callback):
    def __init__(self, max_lr, warmup_steps, verbose=0):
        super(LearningRateLinearIncrease, self).__init__()
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.verbose = verbose
        self.cur_step_count = 0

    def on_train_begin(self, logs=None):
        tf.keras.backend.set_value(self.model.optimizer.lr, 0.0)

    def on_batch_begin(self, batch, logs=None):
        if self.cur_step_count >= self.warmup_steps:
            return
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        lr += 1./self.warmup_steps * self.max_lr
        if self.verbose and self.cur_step_count % 50 == 0:
            print('\n new LR = {}\n'.format(lr))
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.cur_step_count += 1


def main():
    args = parse_args()
    np.random.seed(1)

    train, val, test = [], [], []
    with open(args.input_data) as f:
        for line in f:
            data = json.loads(line.strip())
            if data['split'] == 'train':
                train.append(data)
            elif data['split'] == 'val':
                val.append(data)
            else:
                test.append(data)

    print('training mean: {:.2f}'.format(100*np.mean([d['valid'] > 0 for d in train])))

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    print('{}/{}/{}'.format(len(train), len(val), len(test)))
    train = TextIterator(train, tokenizer, args, batch_size=args.batch_size, shuffle=True)
    val = TextIterator(val, tokenizer, args, batch_size=args.batch_size,)
    test = TextIterator(test, tokenizer, args, batch_size=args.batch_size,)

    bias_p = 77./100.
    init_v = -math.log((1 - bias_p) / bias_p)

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():

        if 'roberta' in args.model:
            layer = transformers.TFRobertaModel.from_pretrained(args.model).roberta
        else:
            raise NotImplementedError

        keras_input = tf.keras.layers.Input((None,), dtype=tf.int32)
        h = layer(keras_input).pooler_output
        h = tf.keras.layers.Dropout(args.dropout)(h)
        h = tf.keras.layers.Dense(512, activation=tf.keras.activations.gelu)(h)
        h = tf.keras.layers.Dropout(args.dropout)(h)

        pred = tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=tf.keras.initializers.Constant(
            value=init_v))(h)
        keras_model = tf.keras.models.Model(inputs=keras_input,
                                            outputs=pred)

        keras_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                            optimizer='adam',
                            metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold=0.5)] + [
                                tf.keras.metrics.PrecisionAtRecall(recall=p, name='P@{:.0f}%'.format(p*100), num_thresholds=1000)
                                for p in [.2, .3, .4, .5, .6, .7, .8, .9, .95]])

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=.1, verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_P@80%', mode='max', restore_best_weights=True, verbose=True)
    lr_increase = LearningRateLinearIncrease(args.lr, 1000)
    history = keras_model.fit(train, validation_data=val, callbacks=[reduce_lr, lr_increase, early_stopping], epochs=100)
    keras_model.set_weights(early_stopping.best_weights)

    if args.save_model:
        keras_model.save('model~model={}~lr={}~bs={}~dropout={:.2f}'.format(args.model, args.lr, args.batch_size, args.dropout))

    output = history.history

    # for test time, we will marginalize over any randomness in string flattening.
    for split, iterator in zip(['val', 'test'], [val, test]):
        preds = []
        for idx in range(5):
            preds.append(keras_model.predict(iterator).flatten())
        preds = np.mean(np.array(preds), axis=0)
        output['{}_preds'.format(split)] = preds.tolist()
        output['{}_labels'.format(split)]  = []
        for _, y in iterator:
            output['{}_labels'.format(split)].extend(y.flatten().tolist())

    with open('results~model={}~lr={}~bs={}~dropout={:.2f}.json'.format(args.model, args.lr, args.batch_size, args.dropout), 'w') as f:
        f.write(json.dumps(history.history, default=default))


if __name__ == '__main__':
    main()
