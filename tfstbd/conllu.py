from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from collections import OrderedDict
from conllu import TokenList

_C_LIKE_REPLACEMENTS = [(r'\s', ' '), (r'\p', '|')]


def has_text(parsed):
    assert hasattr(parsed, 'metadata')

    return parsed.metadata is not None and 'text' in parsed.metadata and \
           parsed.metadata['text'] is not None and len(parsed.metadata['text'])


def repair_spaces(parsed):
    if not has_text(parsed):
        return parsed

    full_text = parsed.metadata['text']
    current_text = ''

    meaning_i = meaning_tokens(parsed)
    for i, t in enumerate(meaning_i):
        current_token = parsed[t]
        next_token = None if meaning_i[-1] == t else parsed[meaning_i[i + 1]]

        current_text += current_token['form']
        assert full_text.startswith(current_text)

        if next_token is None:
            continue

        space = extract_space(current_token).replace('\n', ' ')
        if full_text.startswith(current_text + space + next_token['form']):
            current_text += space
            continue

        tail_text = full_text[len(current_text):]
        space_stop = tail_text.index(next_token['form'])
        space = tail_text[:space_stop]

        current_text += space
        if 'misc' not in parsed[t] or parsed[t]['misc'] is None:
            parsed[t]['misc'] = OrderedDict()
        parsed[t]['misc']['SpacesAfter'] = encode_space(space)

    for last_i in {-1, meaning_i[-1]}:
        if 'misc' in parsed[last_i] and parsed[last_i]['misc'] is not None:
            if 'SpaceAfter' in parsed[last_i]['misc']:
                del parsed[last_i]['misc']['SpaceAfter']
            if not len(parsed[last_i]['misc']):
                parsed[last_i]['misc'] = None

    text = ''.join(extract_text(parsed))
    parsed.metadata['text'] = text.replace('\n', ' ')

    return parsed


def extract_tokens(parsed, last_space=True):
    tokens = []

    meaning_i = meaning_tokens(parsed)
    for t in meaning_i:
        token = parsed[t]
        form = token['form']
        space = extract_space(token)
        tokens.append((form, space))

    if not last_space and len(tokens):
        tokens[-1] = (tokens[-1][0], tokens[-1][1].strip())

    return tokens


def extract_text(parsed, validate=True):
    tokens = extract_tokens(parsed, last_space=False)
    text = itertools.chain(*tokens)
    text = list(filter(len, text))

    if validate and has_text(parsed):
        actual = ''.join(text).replace('\n', ' ').strip()
        stored = parsed.metadata['text'].strip()
        if actual != stored:
            raise ValueError('Extracted text does not match stored one: {}'.format([actual, stored]))

    return text


def split_sent(parsed, validate=True):
    if validate:
        extract_text(parsed, validate)

    result = []
    tokens = []

    for t in parsed:
        tokens.append(t)
        if 'misc' not in t or t['misc'] is None or 'SentenceBreak' not in t['misc']:
            continue

        assert 'Yes' == t['misc']['SentenceBreak']
        result.append(tokens)
        tokens = []

    if len(tokens):
        result.append(tokens)

    for i in range(len(result)):
        r = result[i][-1]
        if 'misc' not in r or r['misc'] is None or 'SpaceAfter' not in r['misc']:
            continue

        del result[i][-1]['misc']['SpaceAfter']
        if not len(result[i][-1]['misc']):
            result[i][-1]['misc'] = None

    return [TokenList(r) for r in result]


def meaning_tokens(parsed):
    meaning, skip = [], set()
    for i, token in enumerate(parsed):
        if isinstance(token['id'], tuple):
            if '.' in token['id']:
                continue
            assert len(token['id']) == 3
            skip.update(token['id'])
        if token['id'] in skip:
            continue
        if not isinstance(token['id'], int) and '.' in token['id']:
            continue
        meaning.append(i)

    return meaning


def decode_space(spaces):
    if '\\' not in spaces:
        spaces = spaces.encode('unicode_escape').decode('utf-8')
    for search, replace in _C_LIKE_REPLACEMENTS:
        spaces = spaces.replace(search, replace)
    spaces = spaces.encode('utf-8').decode('unicode_escape')

    return spaces


def encode_space(spaces):
    spaces = decode_space(spaces)
    spaces = spaces.encode('unicode_escape').decode('utf-8')

    for replace, search in _C_LIKE_REPLACEMENTS:
        spaces = spaces.replace(search, replace)

    return spaces


def extract_space(token):
    if 'misc' not in token or token['misc'] is None:
        return ' '

    if 'SpaceAfter' not in token['misc'] and 'SpacesAfter' not in token['misc']:
        return ' '

    if 'SpaceAfter' in token['misc']:
        assert 'No' == token['misc']['SpaceAfter'], token['misc']['SpaceAfter']
        return ''

    assert 'SpacesAfter' in token['misc']
    if token['misc']['SpacesAfter'] in {'_', ' ', None}:
        return ' '

    return decode_space(token['misc']['SpacesAfter'])
