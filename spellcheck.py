"""
A simple spelling check scheme from Peter Novig's site:
http://norvig.com/spell-correct.html

Modified by Mattal Cock
https://github.com/mattalcock/blog/blob/master/2012/12/5/python-spell-checker.rst
"""

import re, collections

def words(text):
    return re.findall('[a-z]+', text.lower())

def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

# use a big text file for training
corpus = 'big.txt'
NWORDS = train(words(file(corpus).read()))
alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
    s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [a + b[1:] for a, b in s if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
    replaces   = [a + c + b[1:] for a, b in s for c in alphabet if b]
    inserts    = [a + c + b     for a, b in s for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words): 
    return set(w for w in words if w in NWORDS)


# add a hack list of words as back door
hacklist = 'hackwords.txt'
pairs = file(hacklist).read()
lines = pairs.split('\n')
lines = [line.rstrip() for line in lines]
lines = filter(None, lines) # eliminate empty lines
hackdict = {}
for line in lines:
    key, value = line.split(' | ')
    hackdict[key] = value

def correct(word):
    if word in hackdict.keys(): 
        return hackdict[word]
    else:
        candidates = known([word]) or known(edits1(word)) or  known_edits2(word) or [word]
        return max(candidates, key=NWORDS.get)
        
def simple_correct(word):
    if word in hackdict.keys():
        return hackdict[word]
    else:
        return word
        