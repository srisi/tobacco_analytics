
import enchant
import os
import re
import pickle

import numpy as np

from tobacco.configuration import PATH_TOKENIZED
from tobacco.utilities.databases import Database


def load_vocabulary_trie(ngram, return_type='trie'):

    try:
        from marisa_trie import Trie
        vocabulary = Trie()
        local_path = PATH_TOKENIZED + 'tries/full_vocabulary_{}_grams.trie'.format(ngram)
        vocabulary.load(local_path)
    except ImportError:

        local_path = PATH_TOKENIZED + 'tries/full_vocabulary_{}_grams_set.pickle'.format(ngram)
        vocabulary = pickle.load(open(local_path, 'rb'))
        print("loaded vocabulary set for ngram {}. len: {}".format(ngram, len(vocabulary)))

    # In Py 3.6, vocabulary is already a set. Somehow...
    if return_type == 'set':
        return vocabulary
    return vocabulary

def vocabulary_trie_to_set():


    for ngram in range(1,6):
        vocab = load_vocabulary_trie(ngram)
        vocab_set = set(vocab.keys())
        local_path = PATH_TOKENIZED + 'tries/full_vocabulary_{}_grams_set.pickle'.format(ngram)
        print(local_path)
        pickle.dump(vocab_set, open(local_path, 'wb'))


def get_vocabulary_totals(ngram):

    local_path = PATH_TOKENIZED + 'tries/vocabulary_totals_{}_grams'.format(ngram)


    try:
        vocabulary_totals = np.load(local_path + '.npy')

    except IOError:
        print("No vocabulary totals for {}-grams available. Creating them now.".format(ngram))
        db = Database("TOB_FULL")
        _, cur = db.connect()
        vocabulary = load_vocabulary_trie(1)
        vocabulary_totals = np.zeros(len(vocabulary), dtype=np.int64)

        cur.execute('SELECT id, total FROM tokens WHERE ngram=1 ORDER BY id ASC')

        while True:
            row = cur.fetchone()
            if not row:
                break
            print(row['id'], row['total'], type(row['id']))

            vocabulary_totals[row['id']] = row['total']

        np.save(local_path, vocabulary_totals)

    return vocabulary_totals


CONTRACTIONS = {
    "aren't": "are not",
    "can't": "cannot",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "I had / I would",
    "i'll": "I shall / I will",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it shall / it will",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "might've": "might have",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she shall / she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that has / that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you shall / you will",
    "you're": "you are",
    "you've": "you have"
}

def expand_contractions(document):
    '''
    Expands contractions in a document and returns the document.
    '''


    for contraction in CONTRACTIONS:
        document = document.replace(contraction, CONTRACTIONS[contraction])
    return document


def check_dictionary(word, dictionary=enchant.Dict('en-US')):
    '''
    Checks if the word is in the specified enchant dictionary
    Returns True if the word is in dict and False if not
    Standard dictionary is all english words from enchant
    '''


    try:
        if (dictionary.check(word) or dictionary.check(word.capitalize())): return True
        else:
            if len(word) == 1:
                return True
            else:
                return False
    except enchant.errors.Error:
        print("Enchant error with ", word)
        return False

def add_valid_words():
    #adds all of the valid words not in the enchant dictionary to the enchant dictionary
    # expand path adds /home/<user>

    with open(os.path.expanduser('~/tobacco/code/WordCorrelation/tobacco/data/words_accepted.txt')) as f:
        for word in f:
            enchant.Dict('en-US').add_to_pwl(word.rstrip())
    print("Valid words added to dictionary")

def get_valid_words_list():
    '''
    Returns a list of all valid words
    :return:
    '''

    accepted_words = []
    with open(os.path.expanduser('~/tobacco/code/WordCorrelation/tobacco/data/words_accepted.txt')) as f:
        for word in f:
            accepted_words.append(word.rstrip().lower())

    accepted_words.sort(key=len)

    return accepted_words




if __name__ == "__main__":

#    vocabulary_trie_to_set()
    voc = load_vocabulary_trie(1, return_type='set')