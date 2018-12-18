import re
import hashlib

from pathlib import Path

from tobacco.utilities.databases import Database
from tobacco.utilities.ocr import expand_contractions
from tobacco.configuration import WORD_SPLIT_REGEX, PATH_TOKENS
from tobacco.utilities.ocr import load_vocabulary_trie
import MySQLdb as mdb


SANITIZE_REGEX = re.compile(r'[^a-zA-Z0-9*,\ ]+')
VOCABULARY = load_vocabulary_trie(1)


def parse_search_tokens(search_tokens, mp_queue=None):


    #search_tokens = ",".join(search_tokens)
    error = ''
    try:
        if len(search_tokens) == 0:
            error = "Please enter one or multiple search terms."
            return {}, error
    except AttributeError:
        print("attr error with search tokens: {}".format(search_tokens))

    # make sure the input only contains valid characters


    if SANITIZE_REGEX.search(" ".join(search_tokens)):
        error = "Search terms can only contain letters, numbers, spaces, commas, and asterisks but not '{}'.\n".format(
            " ".join(set(SANITIZE_REGEX.findall(" ".join(search_tokens))))
        )
        process_despite_error = False
        if set(SANITIZE_REGEX.findall(" ".join(search_tokens))) == {'-'}:
            error += 'We have replaced the dash with a space.\n'
            search_tokens = [i.replace('-', ' ') for i in search_tokens]
            process_despite_error = True


        for char in ['\'', '\"']:
            if set(SANITIZE_REGEX.findall(" ".join(search_tokens))) == {char}:
                if " ".join(search_tokens).find(char) > -1:
                    print("search tokens before quotation mark", search_tokens)
                    search_tokens = [token.replace(char, '') for token in search_tokens]
                    print("after", search_tokens)
                    process_despite_error = True
                    error += 'We have removed the quotation marks.\n'

        print("process despite error", process_despite_error)
        if not process_despite_error:
            if mp_queue:
                mp_queue.put(({}, error))
                return
            else:
                return {}, error

    final_tokens = set()
    tokens_not_in_vocabulary = []
    for token in search_tokens:
        if token == '':
            continue
        token = expand_contractions(token)
        if token[0] == '*' or token[-1] == '*':

            if token[0] == token[-1]:
                error = "Wildcard searches can only be done at the beginning or the end of a token but not at both."
                if mp_queue:
                    mp_queue.put(({}, error))
                    return
                else:
                    return {}, error
            else:
                final_tokens = final_tokens.union(wildcard_search(token, ngram=len(token.split())))

        else:
            if check_if_token_in_vocabulary(token):
                final_tokens.add(token)
            else:
                tokens_not_in_vocabulary.append(token)


    if len(tokens_not_in_vocabulary) == 1:
        error += 'The term "{}" does not exist in the vocabulary of tobacco-analytics.\n'.format(", ".join(tokens_not_in_vocabulary))
    elif len(tokens_not_in_vocabulary) > 1:
        error += 'The terms "{}" do not exist in the vocabulary of tobacco-analytics\n'.format(", ".join(tokens_not_in_vocabulary))

    error = error.rstrip('\n')
    print("final cleaned search tokens", search_tokens, error)

    if mp_queue:
        mp_queue.put((final_tokens, error))
    else:
        return final_tokens, error


def wildcard_search(token, ngram=None, number_of_tokens=10, ):
    '''
    Parses wildcard tokens when parsing search tokens

    :return:
    '''
    wildcard_tokens = []

    print("wildcard", token[:-1])

    _, cur = Database("TOB_FULL").connect()


    if ngram:
        ngram_search = ' AND ngram={} '.format(ngram)
    else:
        ngram_search = ''


    if token[-1] == '*':
        cur.execute('SELECT token FROM tokens where token LIKE "{}%" {} order by total DESC LIMIT {}'.format(token[:-1], ngram_search, number_of_tokens+1))
    else:
        cur.execute('SELECT token FROM tokens WHERE token_reversed LIKE "{}%" {} ORDER BY total DESC LIMIT {};'.format(token[1:][::-1], ngram_search, number_of_tokens+1))

    for row in cur.fetchall():

        if len(wildcard_tokens) == number_of_tokens: break
        cur_token = row['token']
        if cur_token == token:
            continue
        else:
            wildcard_tokens.append(row['token'])

    return wildcard_tokens

def check_if_token_in_vocabulary(token):

    '''
    A token exists if the accompanying file exists.
    :param token:
    :return:
    '''

    # for 1-grams, lookup in vocabulary
    if len(token.split()) == 1:
        if token in VOCABULARY:
            return True
        else:
            return False

    # else, check if file exists
    else:
        hash = hashlib.sha256(token.encode()).hexdigest()
        token_path = Path(PATH_TOKENS + '{}/{}/{}/{}/{}.npz'.format(hash[0], hash[1], hash[2], hash[3], hash))
        if token_path.is_file():
            return True
        else:
            return False






if __name__ == "__main__":
#    print(parse_search_tokens('addictive'))
#    print(parse_search_tokens(['addict drugs']))
#     w = wildcard_search('johnston *', 2, 100)
#    w = wildcard_search('smoking is*', 3, 100)
#    for i in w:
#       print(i)
#    print(parse_search_tokens(['self-administration']))
    parse_search_tokens(['addiction'])
