"""
In this file are all the tidbits of code that were created and used at some
point in the main project. They're no longer needed (and thus I apologise for
their lack of quality), but you might find some of them useful.
"""

def probability_dict(counts, all_words):
    d = defaultdict(lambda: defaultdict(float))
    for word_1 in sorted(all_words):
        total = sum(counts[word_1].values())
        for word_2 in sorted(all_words):
            d[word_1][word_2] = counts[word_1].get(word_2, 0.0) / total + 1 / len(all_words)
    return d

def probability_matrix(counts, all_words):
    m = np.empty((len(all_words), len(all_words)), dtype=float)
    for i, word_1 in enumerate(sorted(all_words)):
        total = sum(counts[word_1].values())
        for j, word_2 in enumerate(sorted(all_words)):
            m[i][j] = counts[word_1].get(word_2, 0.0) / total + 1 / len(all_words)
    return m

def print_d_values(s_p):
    for k, v in sorted(s_p.items(), key=lambda x: x[1])[-20:]:
        print(k, ":", v)

def probability_given_word(given_word, counts, all_words):
    total = sum(counts[given_word].values())
    return {
        word: counts[given_word].get(word, 0.0) / total + 1 / len(all_words)
        for word in sorted(all_words)
    }

def find_sentence(all_messages):
    counts = countify(all_messages)
    word = counts["__BEGIN__"].most_common(1)[0][0]
    print(word)
    while word != "__END__":
        word = counts[word].most_common(1)[0][0]
        print(word)

def print_vector(s_p, words):
    for arg in reversed(s_p.argsort(0)[-10:]):
        arg = arg[0]
        print(words[arg], s_p[arg][0]*100)

def save_data(filename, messages):
    m = numpy_parse(messages)
    np.save(filename, m)

def numpy_multi_parse(messages, mapper):
    # note that entries contain __BEGIN__

    s = len(mapper)
    counts = np.ones(s, dtype=int)

    m = np.zeros((s, s), dtype=float)
    for message in messages:
        for word, next_word in get_pairs(message):
            if word in mapper and next_word in mapper:
                counts[mapper[word]] += 1
                m[mapper[word]][mapper[next_word]] += 1

    #m = m / counts[:,None] #+ 1/s #np.full((s, s), 1/s)
    m /= counts[:, None]
    #del counts
    m += 1/s
    return m

def numpy_parse(messages, mapper):
    # note that entries contain __BEGIN__

    s = len(mapper)
    counts = np.ones(s, dtype=int)

    m = np.zeros((s, s), dtype=float)
    for message in messages:
        for word, next_word in get_pairs(message):
            counts[mapper[word]] += 1
            m[mapper[word]][mapper[next_word]] += 1

    #m = m / counts[:,None] #+ 1/s #np.full((s, s), 1/s)
    m /= counts[:, None]
    m += 1/s
    return m

def parse_messages(messages):
    counts = defaultdict(Counter)
    all_words = set()
    for message in messages:
        all_words.update(re.findall(r"[^\W\d_]+", message.lower(), re.UNICODE))
        for word, next_word in get_pairs(message):
            counts[word][next_word] += 1
    return (counts, all_words)

'''
    for i, word_1 in enumerate(sorted(all_words)):
        total = sum(counts[word_1].values())
        for j, word_2 in enumerate(sorted(all_words)):
            m[i][j] = counts[word_1].get(word_2, 0.0) / total + 1 / len(all_words)

        for word, next_word in get_pairs(message):
            counts[word][next_word] += 1
    return (counts, all_words)
'''

def get_pairs(message):
    #words = message.strip().split()
    # find all unicode words
    words = re.findall(r"[^\W\d_]+", message.lower(), re.UNICODE)
    words.append("__END__")
    x, y = itertools.tee(words)
    yield ("__BEGIN__", next(y))
    yield from zip(x, y)

def countsify(messages):
    c = Counter()
    for message in messages:
        c.update(message)
    return c

def setify(messages):
    s = set(("__BEGIN__", "__END__"))
    for message in messages:
        s.update(message)
    return s

def sorted_words(messages):
    a = set(("__BEGIN__", "__END__"))
    for message in messages:
        a.update(re.findall(r"[^\W\d_]+", message.lower(), re.UNICODE))
    return sorted(a)

def sorted_word_multi(messages):
    c = Counter()
    for message in messages:
        c.update(re.findall(r"[^\W\d_]+", message.lower(), re.UNICODE))
    for k in list(c):
        if c[k] == 1:
            del c[k]
    c.update(("__BEGIN__", "__END__"))
    return sorted(c)

def set_of_words(a):
    return {
        word: i
        for i, word in enumerate(a)
    }

def main():
    #words = sorted_words(messages)
    #words = sorted_word_multi(messages)
    #mapper = set_of_words(words)

    '''
    o = scentence_mapper("is to if i s so", mapper)
    for i in o:
        w_max = m[:,i].argsort()[-3:][::-1]
        for j in w_max:

        #w_max = m[mapper["__BEGIN__"]].argmax()
        #print(w_max, w_max.argmax(), words[w_max.argmax()])
            print(words[j], words[i])
    d = {
        word: w_max[mapper[word]]
        for word in mapper
    }
    print_d_values(d)
    '''
    #for word in mapper:
        #print(word, ":", m[mapper["__BEGIN__"]][mapper[word]])
    #counts, all_words = parse_messages(m.text for m in messages if m.text is not None)
    #s_p = probability_given_word("__BEGIN__", counts, all_words)
    #m = probability_matrix(counts, all_words)
    #pprint(m)
