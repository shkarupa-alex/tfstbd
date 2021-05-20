def fix_spaces(words, spaces):
    punkt = {',', ';', ':', '.', '!', '?', '(', ')', '-', '+', '=', '/', '\\'}
    for i, (w, s) in enumerate(zip(words, spaces)):
        if s and punkt.intersection(w):
            spaces[i] = False
        elif s and any([c.isalnum() for c in w]):
            spaces[i] = False
        elif not s and not any([c.isalnum() for c in w]) and any([c.isspace() for c in w]):
            spaces[i] = True

    return spaces


def fix_tokens(spaces, tokens):
    tokens[0] = True

    for i, (s0, s1, t1) in enumerate(zip(spaces[:-1], spaces[1:], tokens[1:])):
        if s0 and s1 and t1:
            tokens[i + 1] = False
        if s0 != s1 and not t1:
            tokens[i + 1] = True

    return tokens


def fix_sents(spaces, tokens, sents):
    sents[0] = True

    bad = [i for i, s in enumerate(sents) if s and spaces[i]]
    for b in bad:
        sents[b] = False
        tail = spaces[b:]
        if False in tail:
            sents[b + tail.index(False)] = True

    bad = [i for i, s in enumerate(sents) if s and not tokens[i]]
    for b in bad:
        sents[b] = False
        head = tokens[:b]
        if True in head:
            sents[b - 1 - head[::-1].index(True)] = True

    return sents


def fix_predictions(words, spaces, tokens, sents):
    assert len(words), 'Sentence is empty'
    assert len(words) == len(spaces) == len(tokens) == len(sents), 'Inconsistent predictions length'
    assert '' not in words, 'Words contain empty sting'

    spaces = fix_spaces(words, spaces)
    tokens = fix_tokens(spaces, tokens)
    sents = fix_sents(spaces, tokens, sents)

    assert len(words) == len(spaces) == len(tokens) == len(sents), 'Inconsistent predictions length'
    assert not spaces[0], 'First space check failed'
    assert tokens[0], 'First token check failed'
    assert sents[0], 'First token check failed'

    for wd, sp in zip(words, spaces):
        assert not sp or not any([c.isalnum() for c in wd]), 'Space should not contain alphanumerics'
        assert sp or not wd.isspace(), 'Token should not contain only spaces'

    for sp0, sp1, tk1 in zip(spaces[:-1], spaces[1:], tokens[1:]):
        assert not (sp0 and sp1) or not tk1, 'Token should not starts at the middle of space'
        assert sp0 == sp1 or tk1, 'Space-token borders should start new token'

    for sp, tk, st in zip(spaces, tokens, sents):
        assert not st or not sp, 'Sentence should not starts at space'
        assert not st or tk, 'Sentence should not starts inside token'

    return words, spaces, tokens, sents
