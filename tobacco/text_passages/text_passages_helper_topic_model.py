import pickle
import time
import array

import numpy as np
from scipy.sparse import csr_matrix

from tobacco.configuration import STOP_WORDS_SKLEARN, PATH_TOKENIZED

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF, SparsePCA, LatentDirichletAllocation

from tobacco.text_passages.text_passages_helper_tokenizer import tokenize_section_cython
from tobacco.utilities.ocr import load_vocabulary_trie


GLOBAL_IDF_WEIGHTS = np.load(PATH_TOKENIZED + 'idf_weights.npy')

'''
8/15/17
So, recap of basic linear algebra w/r/t sparsity:
sklearn's nmf (and presumably others) offer l1 and l2 regularization that can be weighed against each other
(e.g. 50% l1, 50% l2)

l1 regularization: reduce number of non-zero weights
l2 regularization: reduce squared sum of weights

i.e. to enforce sparsity in the terms, use l1 regularization
'''

VOCABULARY = load_vocabulary_trie(1)


def tokenize_sections(output_sections, vocabulary, log_likelihoods, tokenizer_type='count', use_global_idf=True):

    start = time.time()

    sections = [i[7] for i in output_sections]
    # map term to token_id
    vocabulary_dict = {token:idx for idx, token in enumerate(vocabulary)}

    indices = array.array(str("i"))
    indptr = array.array(str("i"))
    indptr.append(0)

    for section in sections:
        # pass matrix arrays to cython section so they don't have to be initialized anew and added
        indices, indptr = tokenize_section_cython(section, vocabulary_dict, indices, indptr)

    data = np.ones(len(indices), dtype=np.int32)
    indices = np.frombuffer(indices, dtype = np.int32)
    indptr = np.frombuffer(indptr, dtype=np.int32)
    shape = (len(indptr) - 1, len(vocabulary))

    dtm = csr_matrix((data, indices, indptr), shape=shape, dtype= np.int64)
    dtm.sum_duplicates()

    if tokenizer_type == 'tf':
        tfidf_transformer = TfidfTransformer(use_idf=False)
        dtm =  tfidf_transformer.fit_transform(dtm)
    elif tokenizer_type == 'tfidf':

        tfidf_transformer = TfidfTransformer(use_idf=True)
        tfidf_transformer.fit(dtm)

        if use_global_idf:
            print("using global idf weights for topic model")
            # make local version of global idf weights
            idf_local = np.zeros(len(vocabulary))
            for term_id, term in enumerate(vocabulary):
                idf_local[term_id] = GLOBAL_IDF_WEIGHTS[VOCABULARY[term]]
            tfidf_transformer._idf_diag.data = idf_local

        dtm = tfidf_transformer.transform(dtm)


    print("tokenizer took: {} for {} sections with {} terms.".format(time.time() - start, len(sections), len(vocabulary)))

    return dtm


def calculate_topic_model(output_sections, section_id_to_year_dict, vocabulary, log_likelihoods, n_components=10,
                          topic_model_type='nmf', terms_per_topic=6):



    start_time = time.time()
    if len(output_sections) < 50:
        return []

    use_global_idf = True
    if topic_model_type == 'nmf_local_idf':
        use_global_idf = False
        topic_model_type = 'nmf'


    if topic_model_type in ['lda']:
        dtm = tokenize_sections(output_sections, vocabulary, log_likelihoods, tokenizer_type='count')
    elif topic_model_type in ['none']:
        dtm = tokenize_sections(output_sections, vocabulary, log_likelihoods, tokenizer_type='tf')
    elif topic_model_type in ['nmf', 'spca', 'dict_learning', 'mini_spca', 'svd', 'nmf_sparse']:
        dtm = tokenize_sections(output_sections, vocabulary, log_likelihoods, tokenizer_type='tfidf',
                                use_global_idf=use_global_idf)
        if topic_model_type in ['spca', 'dict_learning', 'mini_spca']:
            dtm = dtm.toarray()


    if topic_model_type == 'lda':
        model = LatentDirichletAllocation(
            n_topics=n_components,
            max_iter=100,
        )
    elif topic_model_type == 'nmf':
        model = NMF(
            n_components=n_components
        )
    elif topic_model_type == 'nmf_sparse':
        model = NMF(
            n_components = n_components,
            l1_ratio=1,
            alpha=0.2
        )

    model.n_jobs = -1

    doc_topic_matrix = model.fit_transform(dtm)
    topic_term_matrix = model.components_

    print("Creating {} model took {} seconds.".format(topic_model_type, time.time() - start_time))
    if topic_model_type in ['nmf', 'nmf2']:
        print("Reconstruction error: {}. Iterations: {}.".format(model.reconstruction_err_, model.n_iter_))


    print("\ndoc term: {}. topic_term: {}. doc_topic: {}".format(dtm.shape, topic_term_matrix.shape, doc_topic_matrix.shape))


    # create topic_term_matrix with only the top 6 terms per topic
    topic_term_top = np.zeros((topic_term_matrix.shape))
    for i in range(topic_term_matrix.shape[0]):
        for term_id in topic_term_matrix[i, :].argsort()[::-1][:6]:
            # print(features[term_id], topic_term_matrix[i, term_id])
            topic_term_top[i, term_id] = topic_term_matrix[i, term_id]
    topic_term_matrix = topic_term_top
    doc_topic_matrix = (topic_term_matrix * dtm.T).T

    print("new topic term nnz", np.count_nonzero(topic_term_matrix))


    results = prepare_results(topic_term_matrix, doc_topic_matrix, vocabulary, section_id_to_year_dict,
                              terms_per_topic=terms_per_topic, topics_to_select=20)

    return results



def prepare_results(topic_term_matrix, doc_topic_matrix, feature_names, doc_id_to_year_dict,
                    terms_per_topic=6, topics_to_select=5):
    """ Formats results for output on the website

    :param topic_term_matrix:
    :param doc_topic_matrix:
    :param feature_names:
    :param doc_id_to_year_dict:
    :param terms_per_topic:
    :param topics_to_select:
    :return:
    """

    results = {}

    topic_weights = []
    for topic_idx, topic in enumerate(topic_term_matrix):
        topic_weights.append((np.sum(topic), topic_idx))
    topic_weights = sorted(topic_weights, reverse=True)

    for final_topic_idx, t in enumerate(topic_weights[:topics_to_select]):
        docs_by_year_list = 116 * [0]
        topic_weight, topic_idx = t
        topic = topic_term_matrix[topic_idx, :]

        terms = []
        term_weights = []
        if np.sum(topic) > 0:
            sorted_terms = [i for i in topic.argsort()[:-terms_per_topic - 1:-1]]
        else:
            sorted_terms = [i for i in topic.argsort()[:-terms_per_topic - 1]][:terms_per_topic]

        for i in sorted_terms:
            if topic_term_matrix[topic_idx, i] > 0:
                terms.append(feature_names[i])
                term_weights.append(np.round(topic_term_matrix[topic_idx, i], 2))

        doc_ids = []
        doc_weights = []
        for doc_id in doc_topic_matrix[:, topic_idx].argsort()[::-1][:1000]:
            if doc_topic_matrix[doc_id, topic_idx] > 0:
                doc_ids.append(int(doc_id))
                doc_weights.append(round(float(doc_topic_matrix[doc_id, topic_idx]), 3))
                year = doc_id_to_year_dict[doc_id]
                docs_by_year_list[year-1901] += 1

        print(terms)

        results[final_topic_idx] = {
            'terms': terms,
            'term_weights': term_weights,
            'doc_ids': doc_ids,
            'doc_weights': doc_weights,
            'doc_ids_by_year_list': docs_by_year_list
        }

    return results



def example():

    output_sections = pickle.load(open('/tobacco/output_sections_ariel.pickle', 'rb'))
    vocabulary = pickle.load(open('/tobacco/vocabulary_ariel.pickle', 'rb'))
    section_id_to_year_dict_ariel = pickle.load(open('/tobacco/section_id_to_year_dict_ariel.pickle', 'rb'))

    g2 = np.load('/tobacco/g2_ariel.npy')
    calculate_topic_model(output_sections=output_sections, vocabulary=vocabulary,
                          section_id_to_year_dict=section_id_to_year_dict_ariel, log_likelihoods=g2, n_components=20,
                          topic_model_type='nmf_g2')


#    from IPython import embed
#    embed()


#    calculate_topic_model(output_sections, vocabulary, 10, topic_model_type='nmf')
    # calculate_topic_model(output_sections, 30, topic_model_type='nmf')
#    topic_model(output_sections, 5, topic_model_type='lda')


    # 5. Store to disk
#    store_nmf_results_to_disk(doc_topic_matrix, topic_term_matrix, 'example.mat')


if __name__ == '__main__':


    example()


'''
['charles', 'sir', 'ellis', 'dear', 'letter', 'mr']
['effects', 'fig', 'nervous', 'literature', 'device', 'central']
['substance', 'secondary', 'transmitter', 'primary', 'secretion', 'present']
['company', 'cigarette', 'reported', 'tobacco', 'safer', 'patent']
['tobacco', 'smoke', 'smoking', 'research', 'return', 'study']
['filter', 'devices', 'nicotine', 'overseas', 'likely', 'triple']
['63', 'letter', 'dr', 'proposal', 'study', 'report']
['january', 'suggested', '24th', 'experiments', '1963', 'time']
['start', 'research', 'work', 'march', 'continuation', '1st']
['normal', 'satisfactory', 'device', 'form', 'smoking', 'pyrolysis']
['small', 'extractable', 'delivery', 'doing', 'nicotine', 'going']
['memorial', 'research', 'proposal', 'institute', 'battelle', 'route']
['dr', '1965', 'battelle', 'discussed', 'hughes', 'felton']
['present', 'february', '1964', 'dr', 'meeting', 'shall']
['work', 'let', 'course', 'success', 'continued', 'know']
['aim', 'proposed', 'work', 'williamson', 'chemical', 'report']
['inhalers', 'acid', 'free', 'smoke', 'non', 'visit']
['methods', 'results', 'device', 'smoking', 'avoiding', 'tables']
['november', 'draft', 'send', 'return', 'letter', 'application']
['agreed', 'sufficient', 'dr', 'hughes', 'offered', 'coverage']

'''


