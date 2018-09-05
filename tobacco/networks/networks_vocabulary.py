from tobacco.utilities.databases import Database
from tobacco.configuration import STOP_WORDS_SKLEARN, YEAR_COUNT, PATH_TOKENIZED
from tobacco.utilities.ocr import get_vocabulary_totals, load_vocabulary_trie
import numpy as np
import pickle


STOP_WORDS = set(STOP_WORDS_SKLEARN).union(
        {'pgnbr', 'quot', 'amp', 'apos', '0', '1','2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
         '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
         'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
         'w', 'x', 'y', 'z',
         'ocr'

         # BAT additions
         # general
         'british', 'american', 'tobacco',

         # addresses
         'road'

         # letters
         'sending', 'note', 'file',

         # enclosures
         'enc', 'enclosed', 'copies', 'copy'

         # greetings
         'dear', 'kind', 'kindest', 'regards', 'sincerely'

         # titles
         'mr', 'mrs', 'dr', 'esq'

         }).union(set(str(i) for i in range(2000)))


def get_vocabulary_and_totals(main_name):

    try:
        vocabulary_dict = pickle.load(open(PATH_TOKENIZED + 'networks/{}_vocabulary.pickle'.format(main_name), 'rb'))
        totals = np.load(open(PATH_TOKENIZED + 'networks/{}_totals.npy'.format(main_name), 'rb'))
    except IOError:

        from tobacco.frequencies_preprocessing.preprocessing_docs import get_ocr_by_tid


        vocabulary_trie = load_vocabulary_trie(1)


        db = Database("TOB_NETWORKS")
        con, cur = db.connect()

        totals = np.zeros(len(vocabulary_trie), dtype=np.int64)
        vocabulary_dict = {
            'token_to_id': {},
            'id_to_token': {},
            'ordered': []
        }

        cur.execute('SELECT DISTINCT(tid) as tid FROM {}_sections'.format(main_name))
        while True:
            row = cur.fetchone()
            if not row: break
            else:
                text = get_ocr_by_tid(row['tid'], return_bytearray=False)
                for token in text.split():
                    if token in vocabulary_trie:
                        totals[vocabulary_trie[token]] += 1

        token_id = 0
        for i in totals.argsort()[-5000:][::-1]:
            token = vocabulary_trie.restore_key(i)
            if token in STOP_WORDS or totals[i] < 10:
                continue
            else:

                vocabulary_dict['id_to_token'][token_id] = token
                vocabulary_dict['token_to_id'][token] = token_id
                vocabulary_dict['ordered'].append(token)
                assert vocabulary_dict['ordered'][token_id] == vocabulary_dict['id_to_token'][token_id]
                token_id+=1

        assert len(vocabulary_dict['ordered']) == len(vocabulary_dict['id_to_token']) == len(vocabulary_dict['token_to_id'])

        # store totals of main person
        totals_name = np.zeros(len(vocabulary_dict['id_to_token']))

        for token in vocabulary_dict['token_to_id']:
            token_id_global = vocabulary_trie[token]
            token_id_local = vocabulary_dict['token_to_id'][token]
            totals_name[token_id_local] = totals[token_id_global]
        np.save(open(PATH_TOKENIZED + 'networks/{}_totals.npy'.format(main_name), 'wb'), totals_name)


        pickle.dump(vocabulary_dict, open(PATH_TOKENIZED + 'networks/{}_vocabulary.pickle'.format(main_name), 'wb'))

    return vocabulary_dict, totals




if __name__ == '__main__':


    get_vocabulary_and_totals('dunn')



