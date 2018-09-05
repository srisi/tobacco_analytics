'''
2016-10-8
Rewrite of create_vocabulary.
The old version still used the mysql extract table. Now using the ocr text files

Requirement: extracted documents in PATH_OCR_FILES

'''

import os
import pickle
import re
import time
from collections import Counter
from marisa_trie import Trie
from multiprocessing import Manager, Process, cpu_count

from tobacco.configuration import PATH_TOKENIZED
from tobacco.utilities.ocr import add_valid_words, check_dictionary, get_valid_words_list
from tobacco.utilities.ocr import expand_contractions
from tobacco.utilities.ocr import load_vocabulary_trie

# multiprocessing globals
FLAG_ALL_DONE = "WORK FINISHED"
FLAG_WORKER_FINISHED_PROCESSING = "WORKER FINISHED PROCESSING"


# Split strings into tokens (includes splitting dashes)
WORD_SPLIT_REGEX = re.compile(r"\b\w+\b")


def create_vocabulary(ngram=1, test=False):
    """ Creates  the vocabulary for ngram level

    :param ngram:
    :param test: If true, only runs through the first 10000 documents.
    :return:

    Steps:
    - Get a set of all tokens
    - Retain only the valid ones
    """

    add_valid_words()
    # get a set of all tokens of the ngram level

    print("here")

    token_set = get_all_tokens_in_docs(ngram, test)
    print("Total tokens before merging: ", len(token_set))

    valid_iterator = valid_ngram_iterator(token_set, ngram)

    vocabulary_trie = Trie(valid_iterator)
    vocabulary_trie.save(PATH_TOKENIZED + 'tries/full_vocabulary_{}_grams.trie'.format(ngram))
    print("Total tokens after merging", len(vocabulary_trie))



def valid_ngram_iterator(token_set, ngram):
    """ Iterates over tokens in the set and yields the valid ones

    Filtering is only necessary for 1grams (2+ grams are filtered out if they appear less than 100 times in a later
    step.

    :param token_set:
    :param ngram:
    :return:
    """

    # for ngram != 1, no filtering is necessary.
    if ngram >= 2:
        for token in token_set:
            yield token

    # for ngram == 1, only return valid tokens
    else:
        # first, add all tokens from the valid word list.
        for token in get_valid_words_list():
            yield token

        # Yield the only valid 1-grams up front
        for i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'i']:
            yield i

        count = 0
        yielded = 0
        for token in token_set:
            count += 1
            if count % 10000 == 0:
                print("Filtering 1grams. {}/{}. Yielded so far: {}".format(count, len(token_set), yielded))
            if check_1gram_validity(token):
                yielded += 1
                yield token


INVALID_SHORT_WORDS = {   u'st', u'se', u'sh', u'sc', u'sp', u'so', u'si', u'', u'sm', u'sq', u'sr', u'sf', u'',
                          u'cr', u'cl', u'cu', u'ce', u'ci', u'cy', u'cb', u'cd', u'cw', u'cg', u'ck', u'pl', u'pu',
                          u'pt', u'pf', u'pk', u'pd', u'pp', u'be', u'br', u'bo', u'bu', u'bl', u'bi', u'bk', u'bx',
                          u'bf', u'ar', u'ac', u'ab', u'av', u'aw', u'ax', u'mi', u'mu', u'mt', u'mk', u'do',
                          u'dz', u'rh', u'ry', u'rd', u'rn', u'rs', u'rm', u'rt', u'rb', u'rx', u'rf', u'tr', u'te',
                          u'to', u'th', u'ty', u'ts', u'tc', u'tl', u'tn', u'tm', u'fa', u'fm', u'fy', u'ff', u'em',
                          u'eb', u'hz', u'ir', u'ia', u'li', u'ly', u'ls', u'lg', u'lr', u'ge', u'go', u'gd', u'gk',
                          u'gs', u'wu', u'wk', u'wm', u'ur', u'um', u'uc', u'ne', u'ni', u'nu', u'nb', u'nd',
                          u'np', u'nr', u'ob', u'ox', u'ow', u'oz', u'vb', u'ki', u'kr', u'kl', u'kw', u'kt', u'jg',
                          u'yd', u'zs', u'zn', u'zr', u'xe', u"'", u'\\', u'ii', u'al', u'iii', u'il', u'', u'amp',
                          u'yb', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u''}

def check_1gram_validity(token):
    """ Checks if a 1gram is valid or not

    :param token:
    :return:
    """

    if len(token) == 1:
        return False
    if token in INVALID_SHORT_WORDS:
        return False
    # should now be split
    if token.find('-') > -1:
        print("Dash found. ", token)
        return False
    if token.startswith('0'):
        return False

    # exclude tokens like 98465st Basically, keep things like 1st or 1960s but not other trash.
    if re.match('[0-9]{5,}[a-zA-Z]+', token):
        return False

    try:
        number = int(token)
        if number > 3000:
            return False
    except ValueError:
        pass

    if not check_dictionary(token):
        return False


    return True



def get_all_tokens_in_docs(ngram, test=False):

    '''
    2016/10/15: this does not work for 1grams right now

    :param ngram:
    :return:
    '''

    ocr_paths = get_ocr_filepath_list()
    token_counter = Counter()

    # Entry queue: all pickled ocr files
    entry_queue = Manager().Queue()
    # results queue: ngrams to yield
    results_queue = Manager().Queue()

    number_of_threads = min(14, cpu_count())

    if test:
        ocr_paths = ocr_paths[:10000]

    for path in ocr_paths:
        entry_queue.put(path)
    for i in range(number_of_threads): entry_queue.put(FLAG_ALL_DONE)
    print("Entry queue size: ", entry_queue.qsize())
    for process_n in range(number_of_threads):
        p = Process(target = get_all_tokens_worker, args=(entry_queue, results_queue, ngram))
        p.start()

    threads_finished = 0
    while True:
        new_result = results_queue.get()
        if new_result == FLAG_WORKER_FINISHED_PROCESSING:
            print(threads_finished)
            threads_finished += 1
            if threads_finished == number_of_threads:
                break
        else:
            token_counter += new_result
            print("Merged. Entry queue: {}. token counter: {}".format(entry_queue.qsize(), len(token_counter)))

    for i in [1,5, 10, 20, 30, 50, 100, 200]:
        count = 0
        for val in token_counter.values():
            if val >= i:
                count += 1

        print("{} {}-grams that appear >= {} times.".format(count, ngram, i))


    tokens = set()
    for key in token_counter.keys():
        if ngram == 1 or token_counter[key] >= 100:
            tokens.add(key)

    return tokens

def get_all_tokens_worker(entry_queue, results_queue, ngram):

    token_counter = Counter()

    if ngram >=2:
        vocabulary = load_vocabulary_trie(ngram-1)

    while True:
        entry = entry_queue.get()
        if entry == FLAG_ALL_DONE:
            results_queue.put(token_counter)
            results_queue.put(FLAG_WORKER_FINISHED_PROCESSING)
            break
        else:
            if results_queue.qsize() > 30 or len(token_counter) > 10000000:
                print("Sleeping. Results qsize: ", results_queue.qsize())
                time.sleep(10)


            if len(token_counter) > 1000000 and results_queue.qsize() < 5:
                results_queue.put(token_counter)
                token_counter = Counter()

            ocr = open(entry, encoding='cp1252', errors='ignore').read()
            for token in ngram_generator(ocr, ngram):
                if ngram == 1 or check_ngram_validity(token, vocabulary, ngram):
                    token_counter[" ".join(token)] += 1




def check_ngram_validity(token_list, vocabulary, ngram):

    if ngram == 2:
        if re.match('[0-9]+ [0-9]+[a-z]*', " ".join(token_list)):
            return False

    if " ".join(token_list[:ngram-1]) in vocabulary and " ".join(token_list[-ngram+1:]) in vocabulary:
        return True

    else:
        return False




def get_ocr_filepath_list():
    '''
    Load or create a list of paths to all ocr filepaths

    :return:
    '''

    try:
        paths = pickle.load(open(PATH_TOKENIZED + 'ocr_filepaths_list.pickle', 'rb'))

    except FileNotFoundError:

        print("No ocr filepath list found. Creating a new one.")
        paths = []

#        for root, dirs, files in os.walk(PATH_OCR_FILES):
        for root, dirs, files in os.walk('/pcie/tobacco/tob_docs/'):
            if len(paths) % 10000 == 0: print(len(paths))
            for file in files:
                if file.endswith(".ocr"):
                     paths.append(os.path.join(root, file))
        for root, dirs, files in os.walk('/home/stephan/tobacco/ocr/tob_docs/'):
            if len(paths) % 10000 == 0: print(len(paths))
            for file in files:
                if file.endswith(".ocr"):
                     paths.append(os.path.join(root, file))

        pickle.dump(paths, open(PATH_TOKENIZED + 'ocr_filepaths_list.pickle', 'wb'))

    print("Loaded full paths of {} ocr files.".format(len(paths)))
    return paths

def test(ngram=1):


    voc = load_vocabulary_trie(ngram)

    file = open('{}grams.txt'.format(ngram), 'w')

    count = 0
    for key in sorted(voc.keys()):
        validity = ""
        if ngram == 1:
            validity = str(check_1gram_validity(key))
        elif ngram == 2:
            if re.match('[0-9]+ [0-9]+[a-z]*', key):
                validity = "False"
            else:
                validity = "True"
        file.write(validity + "\t" + key+"\n")
        if validity == "True":
            count += 1

    print(len(voc), count )


def ngram_generator(document, ngram):
    '''
    Returns all ngrams of a given level for the document.
    Each ngram is a list of tokens.
    :param document:
    :param ngram:
    :param joined: Returns ngram either joined (=True) or as list (=False). e.g. doc = "This is a test", joined = "This is". list = ["This", "is"]
    :return:
    '''

    document = document.lower()
    document = expand_contractions(document)
    document_split = re.findall(WORD_SPLIT_REGEX, document)

    start = 0
    while(start + ngram <= len(document_split)):

        ngram_extract = document_split[start : start + ngram]
        yield ngram_extract
        start += 1


def ngram_iterator_test():

    tests = [
        "This is a simple t'est string h'ee'e''re..",
        "More-test here; with it's harder...for it\t\rtester@here it's 1"
    ]

    for test in tests:
        print(test)
        for i in range(1, 6):
            print(i, [ngram for ngram in ngram_generator(test, i)])

if __name__ == "__main__":

#    for i in range(2, 6):
#        create_vocabulary(i, test=False)

    get_ocr_filepath_list()

    pass