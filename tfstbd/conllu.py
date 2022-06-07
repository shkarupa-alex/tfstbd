import itertools
from collections import OrderedDict
from conllu.models import TokenList, Token
from typing import List, Tuple
import unicodedata

_C_LIKE_REPLACEMENTS = [(r'\s', ' '), (r'\p', '|')]


def decode_space(spaces: str) -> str:
    # Decode spaces from CoNLL-U format to python

    if '\\' not in spaces:
        spaces = spaces.encode('unicode_escape').decode('utf-8')
    for search, replace in _C_LIKE_REPLACEMENTS:
        spaces = spaces.replace(search, replace)
    spaces = spaces.encode('utf-8').decode('unicode_escape')

    return spaces


def encode_space(spaces: str) -> str:
    # Encode python spaces to CoNLL-U

    spaces = decode_space(spaces)
    spaces = spaces.encode('unicode_escape').decode('utf-8')

    for replace, search in _C_LIKE_REPLACEMENTS:
        spaces = spaces.replace(search, replace)

    return spaces


def extract_space(token: Token) -> str:
    # Extract space encoded in CoNLL-U token misc attribute

    if 'misc' not in token or token['misc'] is None:
        return ' '

    if 'SpaceAfter' not in token['misc'] and 'SpacesAfter' not in token['misc']:
        return ' '

    if 'SpaceAfter' in token['misc']:
        if 'Yes' == token['misc']['SpaceAfter']:
            return ' '

        assert 'No' == token['misc']['SpaceAfter'], 'Wrong "SpaceAfter" value in {}'.format(token)
        return ''

    assert 'SpacesAfter' in token['misc'], 'Wrong token "misc"'
    if token['misc']['SpacesAfter'] in {'_', ' ', None}:
        return ' '

    return decode_space(token['misc']['SpacesAfter'])


def has_text(parsed: TokenList) -> bool:
    # Check if CoNLL-U sentence has original text

    assert hasattr(parsed, 'metadata'), 'Wrong "parsed" value'

    return parsed.metadata is not None and 'text' in parsed.metadata and \
           parsed.metadata['text'] is not None and len(parsed.metadata['text'])


def meaning_tokens(parsed: TokenList) -> List[int]:
    # Search for meaningful token indices

    meaning, skip = [], set()
    for i, token in enumerate(parsed):
        if isinstance(token['id'], tuple):
            if '.' in token['id']:
                continue

            assert 3 == len(token['id']), 'Unexpected token "id" value'

            skip.update(range(token['id'][0], token['id'][-1] + 1))

        if token['id'] in skip:
            continue
        if not isinstance(token['id'], int) and '.' in token['id']:
            continue

        meaning.append(i)

    return meaning


def extract_tokens(parsed: TokenList, last_space: bool = True) -> List[Tuple[str, str]]:
    # Extract "form" and space encoded in CoNLL-U token

    tokens = []
    for i in meaning_tokens(parsed):
        token = parsed[i]
        form = token['form']
        space = extract_space(token)
        tokens.append((form, space))

    if not last_space and len(tokens):
        tokens[-1] = (tokens[-1][0], '')

    return tokens


def extract_text(parsed: TokenList, validate: bool = True, last_space: bool = True) -> List[str]:
    # Extract text encoded in CoNLL-U tokens
    tokens = extract_tokens(parsed, last_space=last_space)
    text = itertools.chain(*tokens)
    text = filter(len, text)
    text = list(map(str, text))

    if validate and has_text(parsed):
        actual = join_text(text)
        stored = join_text([parsed.metadata['text']])
        assert actual == stored, 'Extracted text does not match stored one: {}'.format([actual, stored])

    return text


def join_text(words: List[str]) -> str:
    return ''.join(words).replace('\r', ' ').replace('\n', ' ').strip(' ')


def split_sents(parsed: TokenList, validate: bool = True) -> List[TokenList]:
    # Split sentences based on custom "SentenceBreak" property

    full_text = extract_text(parsed, validate)
    full_text = join_text(full_text)

    sentences, tokens = [], []
    for token in parsed:
        tokens.append(token)

        if 'misc' not in token or token['misc'] is None or 'SentenceBreak' not in token['misc']:
            continue
        assert 'Yes' == token['misc']['SentenceBreak'], 'Wrong "SentenceBreak" value'

        sentences.append(TokenList(tokens))
        tokens = []

    if len(tokens):
        sentences.append(TokenList(tokens))

    for i, sentence in enumerate(sentences):
        last_token = sentence[-1]
        if 'misc' not in last_token or last_token['misc'] is None:
            continue
        if 'SpaceAfter' in last_token['misc']:
            del last_token['misc']['SpaceAfter']
        if 'SentenceBreak' in last_token['misc']:
            del last_token['misc']['SentenceBreak']
        if not len(last_token['misc']):
            last_token['misc'] = None

        sentences[i][-1] = last_token

    for i, sentence in enumerate(sentences):
        part_text = extract_text(sentence, validate=False)
        sentence.metadata['text'] = join_text(part_text)
        sentences[i] = sentence

    if validate:
        rest_text = str(full_text)
        for sentence in sentences:
            rest_text = join_text([rest_text])
            part_text = join_text([sentence.metadata['text']])

            assert rest_text.startswith(part_text), \
                'Full text should starts with a partial: {}'.format([rest_text, part_text])
            rest_text = rest_text[len(part_text):]

        assert not len(join_text([rest_text])), 'Text before and after splitting is different'

    return sentences


def repair_spaces(parsed: TokenList) -> TokenList:
    # Repair text spaces based on tokens

    if not has_text(parsed):
        return parsed

    full_text = parsed.metadata['text']
    current_text = ''

    meaning = meaning_tokens(parsed)
    for i, meaning_i in enumerate(meaning):
        current_token = parsed[meaning_i]
        next_token = None if meaning[-1] == meaning_i else parsed[meaning[i + 1]]

        current_text += current_token['form']
        assert full_text.startswith(current_text), 'Full text should starts with reconstructed one'

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
        if 'misc' not in parsed[meaning_i] or parsed[meaning_i]['misc'] is None:
            parsed[meaning_i]['misc'] = OrderedDict()
        parsed[meaning_i]['misc']['SpacesAfter'] = encode_space(space)

    for last_i in {-1, meaning[-1]}:
        if 'misc' in parsed[last_i] and parsed[last_i]['misc'] is not None:
            if 'SpaceAfter' in parsed[last_i]['misc']:
                del parsed[last_i]['misc']['SpaceAfter']
            if not len(parsed[last_i]['misc']):
                parsed[last_i]['misc'] = None

    parsed.metadata['text'] = join_text(extract_text(parsed))

    return parsed
