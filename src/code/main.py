#!/usr/bin/python3.6

import json
import itertools
from collections import Counter, defaultdict
import time
from pprint import pprint
import re
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

def pairify(words):
    x, y = itertools.tee(words)
    yield ("__BEGIN__", next(y))
    yield from zip(x, y)

def countify(messages):
    counts = defaultdict(Counter)
    for message in messages:
        for word, next_word in pairify(message + ["__END__"]):
            counts[word][next_word] += 1
    return counts

def wordify(messages):
    m = []
    for message in messages:
        s = re.findall(
            r"[^\W\d_]+",
            re.sub(
                r"(\w)['’](\w)", r"\1\2",
                message[0].lower(),
                re.UNICODE
                ),
            re.UNICODE
        )
        if s:
            m.append(s)
    return m

def setify_multi(messages):
    c = Counter()
    for message in messages:
        c.update(message)
    for k in list(c):
        if c[k] == 1:
            del c[k]
    #c.update(("__BEGIN__", "__END__"))
    #c["__BEGIN__"] = 1
    return c

def mapify(s):
    return {
        word: i
        for i, word in enumerate(s)
    }

def matrify(messages, mapper):
    # note that entries contain __BEGIN__

    s = len(mapper)
    counts = np.ones(s, dtype=int)

    m = np.zeros((s, s), dtype=float)
    for message in messages:
        for word, next_word in pairify(message):
            if word in mapper and next_word in mapper:
                counts[mapper[word]] += 1
                m[mapper[word]][mapper[next_word]] += 1

    #m = m / counts[:,None] #+ 1/s #np.full((s, s), 1/s)
    m /= counts[:, None]
    #del counts
    m += 1/s
    return m

def simify(messages, mapper):
    # note that entries contain __BEGIN__

    s = len(mapper)
    counts = np.ones(s, dtype=int)

    m = np.zeros((s, s), dtype=float)
    for message in messages:
        for word, next_word in pairify(message):
            if word in mapper and next_word in mapper:
                counts[mapper[word]] += 1
                m[mapper[word]][mapper[next_word]] += 1

    l = cosine_similarity(m.T)
    #m = m / counts[:,None] #+ 1/s #np.full((s, s), 1/s)
    m /= counts[:, None]
    #del counts
    m += 1/s
    return m, l

def bmatrify(messages, mapper):
    # note that entries contain __BEGIN__

    s = len(mapper)
    counts = 0
    b = np.zeros(s, dtype=float)
    for message in messages:
        word = message[0]
        if word in mapper:
            b[mapper[word]] += 1
            counts += 1

    #m = m / counts[:,None] #+ 1/s #np.full((s, s), 1/s)
    b /= counts
    #del counts
    b += 1/s
    return b

def scentence_mapper(scentence, mapper):
    return tuple(mapper[word] for word in scentence.split())

def guesser(d, scentence, mapper, words, m):
    o = scentence_mapper(scentence, mapper)
    n_states = d.Decode_sim(o)
    print(", ".join(words[i] for i in n_states))
    #print(scentence.strip(), words[m[n_states[-1]].argmax()])
    print(scentence.strip(), f"[{words[n_states[-1]]}]")
    return scentence.strip() + " " + words[n_states[-1]]

def predicter(d, scentence, mapper, words, m):
    o = scentence_mapper(scentence, mapper)
    n_states = d.Decode_predict(o)
    print(", ".join(words[i] for i in n_states))
    print(scentence.strip(), words[m[n_states[-1]].argmax()])

def predict(d, scentence, mapper, words, m):
    o = scentence_mapper(scentence, mapper)
    for i in d.Decode(o):
        print(words[i], end=", ")
    else:
        print()
        print(scentence.strip(), words[m[i].argmax()])

class Decoder(object):
    def __init__(self, initialProb, transProb, obsProb):
        self.N = initialProb.shape[0]
        self.initialProb = initialProb
        self.transProb = transProb
        self.obsProb = obsProb
        assert self.initialProb.shape == (self.N, 1)
        assert self.transProb.shape == (self.N, self.N)
        assert self.obsProb.shape[0] == self.N

    def Obs(self, obs):
        return self.obsProb[:, obs, None]

    def Decode(self, obs):
        trellis = np.zeros((self.N, len(obs)))
        backpt = np.ones((self.N, len(obs)), 'int32') * -1

        # initialization
        trellis[:, 0] = np.squeeze(self.initialProb * self.Obs(obs[0]))

        for t in range(1, len(obs)):
            trellis[:, t] = (trellis[:, t-1, None].dot(self.Obs(obs[t]).T) * self.transProb).max(0)
            backpt[:, t] = (np.tile(trellis[:, t-1, None], [1, self.N]) * self.transProb).argmax(0)
        # termination
        tokens = [trellis[:, -1].argmax()]
        for i in reversed(range(1, len(obs))):
            tokens.append(backpt[tokens[-1], i])
        return tokens[::-1]

    def Decode_sim(self, obs):
        trellis = np.zeros((self.N, len(obs) + 1))
        backpt = np.ones((self.N, len(obs) + 1), 'int32') * -1

        # initialization
        trellis[:, 0] = np.squeeze(self.initialProb * self.Obs(obs[0]))

        # maybe instead of joining with the last, just generate a new one
        # without saving it and pass it on
        t = 0
        for t in range(1, len(obs)):
            trellis[:, t] = (trellis[:, t-1, None].dot(self.Obs(obs[t]).T) * self.transProb).max(0)
            backpt[:, t] = (np.tile(trellis[:, t-1, None], [1, self.N]) * self.transProb).argmax(0)
        t += 1
        trellis[:, t] = (trellis[:, t-1, None].dot(self.transProb[:, obs[t-1], None].T) * self.transProb).max(0)
        backpt[:, t] = (np.tile(trellis[:, t-1, None], [1, self.N]) * self.transProb).argmax(0)

        # termination
        tokens = [trellis[:, -1].argmax()]
        for i in reversed(range(1, len(obs) + 1)):
            tokens.append(backpt[tokens[-1], i])
        return tokens[::-1]


    def Decode_shift(self, obs):
        trellis = np.zeros((self.N, len(obs)))
        backpt = np.ones((self.N, len(obs)), 'int32') * -1

        # initialization
        if len(obs) > 1:
            trellis[:, 0] = np.squeeze(self.initialProb * self.Obs(obs[1]))

        t = 0
        for t in range(1, len(obs)-1):
            ran = True
            trellis[:, t] = (trellis[:, t-1, None].dot(self.Obs(obs[t+1]).T) * self.transProb).max(0)
            backpt[:, t] = (np.tile(trellis[:, t-1, None], [1, self.N]) * self.transProb).argmax(0)
        if len(obs) > 1:
            t += 1
            trellis[:, t] = (trellis[:, t-1, None].dot(self.Obs(obs[t]).T) * self.transProb).max(0)
            backpt[:, t] = (np.tile(trellis[:, t-1, None], [1, self.N]) * self.transProb).argmax(0)
        # termination
        tokens = [trellis[:, -1].argmax()]
        for i in reversed(range(1, len(obs))):
            tokens.append(backpt[tokens[-1], i])
        return tokens[::-1]

def interactive():

    all_messages, my_messages, other_messages, words, mapper = load_vars()

    t_m, o_m = simify(all_messages, mapper)
    #t_m = matrify(all_messages, mapper)

    s_p = bmatrify(all_messages, mapper)[:, None]

    d = Decoder(s_p, t_m, t_m)

    line = input("Enter text: ")
    while line:
        guesser(d, line, mapper, words, t_m)
        #predict(d, line, mapper, words, t_m)
        #predicter(d, line, mapper, words, t_m)
        line = input("Enter text: ")

def test_standard(lines, word, mapper, all_messages, t, o):

    t_m = matrify(all_messages, mapper)
    s_p = bmatrify(all_messages, mapper)[:, None]
    d = Decoder(s_p, t_m, t_m)

    with open("tests/test_standard.txt", "a") as fo:
        for line in lines:
            obs = scentence_mapper(line, mapper)
            states = d.Decode(obs)
            last_state = states[-1]
            words = [word[i] for i in states]
            prediction = word[t_m[last_state].argmax()]
            print(f"({', '.join(words)})", file = fo)
            print(line.strip(), f"[{prediction}]", end="\n\n", file = fo)

def test_reversed(lines, word, mapper, all_messages, t, o):
    t_m = matrify(t, mapper)
    o_m = matrify(o, mapper)
    s_p = bmatrify(all_messages, mapper)[:, None]
    d = Decoder(s_p, t_m, o_m.T)

    with open("tests/test_reversed.txt", "a") as fo:
        for line in lines:
            obs = scentence_mapper(line, mapper)
            states = d.Decode(obs)
            last_state = states[-1]
            words = [word[i] for i in states]
            prediction = word[last_state]
            print(f"({', '.join(words)})", file = fo)
            print(line.strip(), f"[{prediction}]", end="\n\n", file = fo)

def test_shifted(lines, word, mapper, all_messages, t, o):
    """shifted word test"""

    t_m = matrify(all_messages, mapper)
    s_p = bmatrify(all_messages, mapper)[:, None]
    d = Decoder(s_p, t_m, t_m)

    with open("tests/test_shifted.txt", "a") as fo:
        for line in lines:
            obs = scentence_mapper(line, mapper)
            states = d.Decode_shift(obs)
            last_state = states[-1]
            words = [word[i] for i in states]
            prediction_1 = word[last_state]
            prediction_2 = word[t_m[last_state].argmax()]
            print(f"({', '.join(words)})", file = fo)
            print(line.strip(), f"[{prediction_2}]", end="\n\n", file = fo)

def test_sim_1(lines, word, mapper,  all_messages, t, o):
    """cosine simularity test 1"""

    t_m, o_m = simify(all_messages, mapper)
    s_p = bmatrify(all_messages, mapper)[:, None]
    d = Decoder(s_p, t_m, o_m)

    with open("tests/test_sim_1.txt", "a") as fo:
        for line in lines:
            obs = scentence_mapper(line, mapper)
            states = d.Decode(obs)
            last_state = states[-1]
            words = [word[i] for i in states]
            prediction_1 = word[last_state]
            prediction_2 = word[t_m[last_state].argmax()]
            print(f"({', '.join(words)})", file = fo)
            print(line.strip(), f"[{prediction_2}]", end="\n\n", file = fo)

def test_sim_2(lines, word, mapper, all_messages, t, o):
    """cosine simularity test 2"""

    t_m, o_m = simify(o, mapper)
    t_m = matrify(t, mapper)
    s_p = bmatrify(all_messages, mapper)[:, None]
    d = Decoder(s_p, t_m, o_m)

    with open("tests/test_sim_2.txt", "a") as fo:
        for line in lines:
            obs = scentence_mapper(line, mapper)
            states = d.Decode_sim(obs)
            last_state = states[-1]
            words = [word[i] for i in states]
            prediction = word[last_state]
            print(f"({', '.join(words)})", file = fo)
            print(line.strip(), f"[{prediction}]", end="\n\n", file = fo)

def test_all():

    all_messages, my_messages, other_messages, word, mapper = load_vars()

    with open("scentences.txt") as f:
        lines = f.readlines()
        """
        test_standard(lines, all_messages, transitional, observational)
        """

        #test_standard(lines, word, mapper, all_messages, all_messages, all_messages)
        #test_standard(lines, word, mapper, all_messages, all_messages, my_messages)
        #test_standard(lines, word, mapper, all_messages, my_messages, all_messages)

        #test_reversed(lines, word, mapper, all_messages, all_messages, all_messages)
        #test_reversed(lines, word, mapper, all_messages, all_messages, my_messages)
        #test_reversed(lines, word, mapper, all_messages, my_messages, all_messages)

        #test_shifted(lines, word, mapper, all_messages, all_messages, all_messages)
        #test_shifted(lines, word, mapper, all_messages, all_messages, my_messages)
        #test_shifted(lines, word, mapper, all_messages, my_messages, all_messages)

        #test_sim_1(lines, word, mapper, all_messages, all_messages, all_messages)
        #test_sim_1(lines, word, mapper, all_messages, all_messages, my_messages)
        #test_sim_1(lines, word, mapper, all_messages, my_messages, all_messages)

        #test_sim_2(lines, word, mapper, all_messages, all_messages, all_messages)
        #test_sim_2(lines, word, mapper, all_messages, all_messages, my_messages)
        #test_sim_2(lines, word, mapper, all_messages, my_messages, all_messages)

def load_filename(filename):
    """
    takes in a json python list encoded file of chat messages containing
    chat messages and returns that object.
    """
    with open(filename, encoding="utf8") as fi:
        return json.load(fi)

def load_vars():

    all_messages_list = load_filename("all.json")
    my_messages_list = load_filename("my.json")
    other_messages_list = load_filename("other.json")

    all_messages = wordify(all_messages_list)
    my_messages = wordify(my_messages_list)
    other_messages = wordify(other_messages_list)

    word = sorted(setify_multi(all_messages))
    mapper = mapify(word)

    return (
        all_messages,
        my_messages,
        other_messages,
        word,
        mapper,
    )

if __name__ == "__main__":
    #test_all()
    interactive()
