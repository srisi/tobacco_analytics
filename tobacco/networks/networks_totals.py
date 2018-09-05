import numpy as np


from tobacco.frequencies_preprocessing.preprocessing_docs import get_ocr_by_tid

def get_totals(tids, vocabulary, tf=False):

    len_vocabulary = len(vocabulary['token_to_id'])

    totals = np.zeros(len_vocabulary)

    for tid in tids:
        totals_tid = np.zeros(len_vocabulary)
        text = get_ocr_by_tid(tid, return_bytearray=False)
        for token in text.split():
            if token in vocabulary['token_to_id']:
                totals_tid[vocabulary['token_to_id'][token]] += 1

        if tf:
            totals += totals_tid / len(text.split())
        else:
            totals += totals_tid

    return totals