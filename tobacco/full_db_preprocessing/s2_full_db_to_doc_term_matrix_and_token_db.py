'''
Turns the full database into a doc-term matrix (for each ngram level). tokenize_full_db()
Stores the column vector of each token/ngram in the table ngram_vectors
'''

import array
import hashlib
import multiprocessing as mp
import os
import time
from marisa_trie import Trie

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from tobacco.configuration import PATH_TOKENIZED, PATH_TOKENS, ADDED_TOKENS
from tobacco.frequencies_preprocessing.preprocessing_tokens import get_ngram_vector
from tobacco.full_db.document_tokenizer import tokenize_document, tokenize_document_to_sections
from tobacco.utilities.databases import Database
from tobacco.utilities.ocr import load_vocabulary_trie
from tobacco.utilities.sparse_matrices import store_csr_matrix_to_file

# multiprocessing globals
FLAG_ALL_DONE = "WORK FINISHED"
FLAG_WORKER_FINISHED_PROCESSING = "WORKER FINISHED PROCESSING"
NUMBER_OF_PROCESSES = 12



def full_db_to_tokens(ngram=1, use_sections=False, add_new_terms=None):
    """ Stores the full database as a csc doc-term matrix (one for each ngram level)

    :param ngram: 1-5
    :param use_sections: use 200 word sections if true, else use full documents
    :return:

    Step 1: Take a slice of the vocabulary and tokenize all documents to a csr matrix. Then transfrom
            that csr to a csc matrix
    Step 2: Take all the csc slices and stack them next to each other.

    Rationale: Creating the full csr matrix at once and then turning it to a csc matrix uses absurd amounts of memory
    Note to future: yes, you have tried and no, it didn't work.

    """


    # Load vocabulary according to ngram level
    vocabulary = load_vocabulary_trie(ngram)

    # Slice the vocabulary into slices with length n, depending on the ngram level.
#    ngram_to_interval = {1: 130000, 2: 2600000, 3: 3100000, 4:3600000, 5: 2500000}
    ngram_to_interval = {1: 140000, 2: 1000000, 3: 3100000, 4:3600000, 5: 2500000}
    voc_interval = ngram_to_interval[ngram]
    # if just adding new terms, then we don't need to slice the vocabulary
    if add_new_terms: voc_interval = 100000000
    print("{} vocabulary slices to process.".format(len(range(0, len(vocabulary)-1, voc_interval))))

    for voc_idx in range(0, len(vocabulary)-1, voc_interval):
        print("Working on voc_idx {} out of {}".format(voc_idx, len(vocabulary)))

        # 2/1/17: just to make it clear: voc_idx is the vocabulary offset
        vocabulary_offset = voc_idx

        vocabulary_slice = {}
        if add_new_terms:
            vocabulary_slice = {new_term: vocabulary[new_term] for new_term in add_new_terms}
            print(vocabulary_slice)
        else:
            for i in range(voc_idx, voc_idx + voc_interval):
                try: vocabulary_slice[vocabulary.restore_key(i)] = i
                except KeyError: pass


        # Initialize arrays for indices and indptr, add first element to indptr
        data = array.array(str("l"))
        indices = array.array(str("l"))
        indptr = array.array(str("l"))
        indptr.append(0)


        entry_queue = get_entry_queue()
        for i in range(NUMBER_OF_PROCESSES): entry_queue.put(FLAG_ALL_DONE)
        print("entry queue size", entry_queue.qsize())

        results_queue = mp.Queue()

        # Initialize and start processes
        for process_n in range(NUMBER_OF_PROCESSES):
            p = mp.Process(target = tokenize_document_worker, args=(entry_queue, results_queue, ngram, vocabulary_slice, vocabulary_offset, use_sections))
            p.start()

        processors_finished = 0
        # next id to be added to the results
        current_id = 0
        # storage dict for returned but not yet added results
        pending_results = {}

        while True:
            new_result = results_queue.get()
            if new_result == FLAG_WORKER_FINISHED_PROCESSING:
                processors_finished += 1
                if processors_finished == NUMBER_OF_PROCESSES:
                    assert entry_queue.qsize() == 0
                    store_vocabulary_slice(data, indices, indptr, vocabulary_slice, ngram, vocabulary_offset, add_new_terms, use_sections)
                    break

            else:
                # all results first get added to the pending_results dict
                pending_results[new_result['id']] = {'indices': new_result['indices'],
                                                     'data': new_result['data']}

                while True:
                    # then, if the next id to be added is in the result,
                    # the result gets moved from the dict to the indices array
                    if current_id in pending_results:
                        if current_id % 10000 == 0:
                            print("Current id: {}. qsize: {}. Data length: {}.".format(current_id, results_queue.qsize(), len(data)))
                            print(len(indptr))

                        if use_sections:
                            for section_id in range(len(pending_results[current_id]['indices'])):
                                indices += pending_results[current_id]['indices'][section_id]
                                data += pending_results[current_id]['data'][section_id]
                                indptr.append(len(indices))
                        else:
                            indices += pending_results[current_id]['indices']
                            data += pending_results[current_id]['data']
                            indptr.append(len(indices))
                        pending_results.pop(current_id, None)
                        current_id += 1
                    else:
                        break


def tokenize_document_worker(entry_queue, results_queue, ngram, vocabulary, vocabulary_offset, use_sections):


    while True:

        if results_queue.qsize() > 300000:
            time.sleep(5)

        doc_id = entry_queue.get()
        if doc_id == FLAG_ALL_DONE:
            results_queue.put(FLAG_WORKER_FINISHED_PROCESSING)
            break
        else:
            tid = doc_id['tid']
            if use_sections:
                indices, data = tokenize_document_to_sections(tid, vocabulary, ngram, vocabulary_offset)
            else:
                indices, data = tokenize_document(tid, vocabulary, ngram, vocabulary_offset)
            results_queue.put({'id': doc_id['id'],
                               'indices': indices,
                               'data': data})



def store_vocabulary_slice(data, indices, indptr, vocabulary_slice, ngram, vocabulary_offset, add_new_terms, use_sections=False):
    '''
    Iterates through vocabulary processed so far and stores every token
    a) in the tokens table of tob_full (token, token_reversed, id, ngram, total)
    b) as a compressed sparse matrix

    :param data:
    :param indices:
    :param indptr:
    :param vocabulary:
    :param ngram:
    :return:
    '''

    print("finished tokenizing. storing vocabulary slice.")

    # parse to int (may not be necessary)
    data = np.frombuffer(data, dtype=np.int64)
    indices = np.frombuffer(indices, dtype = np.int64)
    indptr = np.frombuffer(indptr, dtype=np.int64)

    # if adding new terms, the temp matrix has to have as many columns as the vocabulary as a whole, not just the
    # current vocabulary slice
    if add_new_terms:
        shape = (len(indptr) - 1, len(load_vocabulary_trie(ngram)))
    else:
        shape = (len(indptr) - 1, len(vocabulary_slice))

    temp_matrix = csr_matrix((data, indices, indptr), shape=shape, dtype= np.int64)

    # get global tfidf weights here
    from IPython import embed
    embed()

    temp_matrix = temp_matrix.tocsc()

    print("temp matrix")
    print("shape", temp_matrix.shape)
    print("indptr, voc slice", len(indptr), len(vocabulary_slice))
    print("nnz", temp_matrix.getnnz())
    print("len, sum of data", len(data), np.sum(data))


    db = Database("TOB_FULL")

    tokens = []

    for token in vocabulary_slice:

        if len(tokens) >= 20000:
            print("Quality control on first token vector")
            test_vector = get_ngram_vector(tokens[0]['token'])
            print("token: ", tokens[0]['token'], " total db: ", tokens[0]['total'], "total vector ", test_vector.sum(), "Shape: ", test_vector.shape, " nnz: ",
                  test_vector.getnnz(), "indptr: ", test_vector.indptr, " data len ",  len(test_vector.data),
                  " indices len ", len(test_vector.indices))

            if not use_sections:
                db.batch_insert('tokens',
                                ['token', 'token_reversed', 'id', 'ngram', 'total'],
                                tokens)
                tokens = []

        id = vocabulary_slice[token]


        # extract, indptr, data, and indices directly instead of forming a column slice first
        # the column slice takes about 3secs per term
        # subtract vocabulary offset to get the correct ids
        indptr_token_start = temp_matrix.indptr[id - vocabulary_offset]
        indptr_token_end = temp_matrix.indptr[id+1 - vocabulary_offset]

        indices_token = temp_matrix.indices[indptr_token_start:indptr_token_end]
        data_token = temp_matrix.data[indptr_token_start:indptr_token_end]
        indptr_token = np.array([0, len(indices_token)], dtype=np.int64)


        # if add_new_terms:
        #     shape = (len(load_vocabulary_trie(ngram)), 1)
        # else:
        shape = (temp_matrix.shape[0], 1)
        token_vector = csc_matrix((data_token, indices_token, indptr_token), shape=shape)

        # to compress directory: tar -c tokens | pv --size `du -csh tokens | grep total | cut -f1` | pigz -9 > tokens.tar.gz
        hash_path = hashlib.sha256(token.encode()).hexdigest()
        if use_sections:
            hash_path += '_sections'
        token_path = PATH_TOKENS + '{}/{}/{}/{}/'.format(hash_path[0], hash_path[1], hash_path[2], hash_path[3])
        if not os.path.exists(token_path): os.makedirs(token_path)

        store_csr_matrix_to_file(token_vector, token_path + hash_path, compressed=True)

        if not use_sections:
            tokens.append({
                'token': token,
                'token_reversed': token[::-1],
                'id': id,
                'ngram': ngram,
                'total': np.sum(data_token)
            })

    if not use_sections:
        db.batch_insert('tokens',
                        ['token', 'token_reversed', 'id', 'ngram', 'total'],
                        tokens)

def add_terms():


    for ngram in range(1,3):
        # update vocabulary trie
        # this messes up the ids but I don't use them anymore because I don't use the doc-term matrices anymore
        start = time.time()
        vocabulary = load_vocabulary_trie(ngram)
        keys = vocabulary.keys() + ADDED_TOKENS[ngram]
        vocabulary_new = Trie(keys)
        vocabulary_new.save(PATH_TOKENIZED + 'tries/full_vocabulary_{}_grams.trie'.format(ngram))

        full_db_to_tokens(ngram, add_new_terms=set(ADDED_TOKENS[ngram]))
        print("adding new tokens for {}-gram took {}.".format(ngram, time.time() - start))


def get_entry_queue(n=None):

    doc_ids = []
    db = Database("TOB_FULL")
    con, cur = db.connect()
    if n:
        print("\n\nWARNING: Only using {} documents! Use this for testing purposes only!\n\n".format(n))
        cur.execute("SELECT id, tid from docs order by id asc limit {};".format(n))
    else:
        cur.execute("SELECT id, tid from docs order by id asc;")

    while True:
        doc_id = cur.fetchone()
        if not doc_id:
            break
        doc_ids.append(doc_id)

    entry_queue = mp.Queue()

    for id in doc_ids:
        entry_queue.put(id)

    return entry_queue


if __name__ == "__main__":

    pass
    #
    # full_db_to_tokens(2)
    # full_db_to_tokens(3)
#    full_db_to_tokens(3, use_sections=True)

    full_db_to_tokens(1)