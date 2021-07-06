import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
from hashlib import md5
from .conllu import encode_space, join_text
from tfmiss.ops import tfmiss_ops  # load ops


class KerasLayer(tfhub.KerasLayer):
    def _add_existing_weight(self, weight, trainable=None):
        if weight is None:
            return
        return super(KerasLayer, self)._add_existing_weight(weight, trainable)


class STBD:
    def __init__(self, hub_handle, out_format='conllu', batch_chars=9999):
        self.inputs = tf.Variable(
            initial_value=[],
            trainable=False,
            validate_shape=False,
            shape=tf.TensorShape([None]),
            dtype='string')
        self.model = KerasLayer(hub_handle)

        if out_format not in {'conllu', 'dense', 'raw'}:
            raise ValueError('Expecting "out_format" to be one of "conllu", '
                             '"dense" or "raw". Got {}'.format(out_format))
        self.out_format = out_format
        self.batch_chars = batch_chars

    def __call__(self, texts):
        assert isinstance(texts, list) and all(map(lambda s: isinstance(s, str), texts))

        paragraphs = self._bucket(texts)

        if 'conllu' == self.out_format:
            return [self._conllu(p, t) for p, t in zip(paragraphs, texts)]

        if 'dense' == self.out_format:
            return [self._dense(p, t) for p, t in zip(paragraphs, texts)]

        return paragraphs

    def _bucket(self, texts):
        sorted_indices = np.argsort(list(map(len, texts)))[::-1]
        inverted_indices = np.argsort(sorted_indices)

        pred_texts, curr_texts = [], []
        for si in sorted_indices:
            if len(curr_texts) * len(texts[si]) > self.batch_chars:
                pred_texts.extend(self._predict(curr_texts))
                curr_texts = []
            curr_texts.append(texts[si])

        if curr_texts:
            pred_texts.extend(self._predict(curr_texts))

        return [pred_texts[i] for i in inverted_indices]

    def _predict(self, texts):
        self.inputs.assign([t.strip() + ' ' for t in texts])
        words, spaces, probs = self.model(self.inputs)

        total = len(texts)
        words = np.char.decode(words.numpy().reshape([total, -1]).astype('S'), 'utf-8').tolist()
        spaces = np.char.decode(spaces.numpy().reshape([total, -1]).astype('S'), 'utf-8').tolist()
        classes = np.argmax(probs.numpy(), axis=-1).reshape([total, -1]).tolist()

        return [self._post(w, s, c) for w, s, c in zip(words, spaces, classes)]

    def _post(self, words, spaces, classes):
        if '' in words:
            stop = words.index('')
            words = words[:stop]
            spaces = spaces[:stop]
            classes = classes[:stop]

        if classes:
            classes[0] = 2  # always start with a new sentence

        for i in range(len(classes) - 1):
            if spaces[i] and 0 == classes[i + 1]:
                classes[i + 1] = 1  # always break after space

        paragraph, sentence = [], []
        word, space = '', ''
        for wd, sp, cl in zip(words, spaces, classes):
            if 0 != cl and (word or space):
                sentence.append((word, space))
                word, space = '', ''

            if 2 == cl and sentence:
                paragraph.append(sentence)
                sentence = []

            assert not space
            word += wd
            space += sp

        if word or space:
            sentence.append((word, space))
        if sentence:
            paragraph.append(sentence)

        return paragraph

    def _conllu(self, paragraph, source_text):
        actual_text = ''
        for sentence in paragraph:
            for word, space in sentence:
                actual_text += word + space
        assert source_text.strip() == actual_text.strip()

        result = ['# newpar\n']

        for sentence in paragraph:
            text = ''
            for word, space in sentence:
                text += word + space

            sent_text = join_text([text])
            sent_hash = md5(text.encode('utf-8')).hexdigest()
            result.append('# sent_id = {}\n'.format(sent_hash))
            result.append('# text = {}\n'.format(sent_text))

            for i, (word, space) in enumerate(sentence):
                result.append('{}\t{}\t_\t_\t_\t_\t_\t_\t_\t{}\n'.format(i + 1, word, self._conllu_misc(space)))
            result.append('\n')

        return ''.join(result)

    def _conllu_misc(self, space):
        if not space:
            return 'SpaceAfter=No'
        if ' ' == space:
            return '_'

        return 'SpacesAfter={}'.format(encode_space(space))

    def _dense(self, paragraph, source_text):
        result = []
        for sentence in paragraph:
            line = ''
            for word, space in sentence:
                line += word
                line += ' ' if space else '\xa0'
            result.append(line.strip())
            result.append('\n')

        return ''.join(result)
