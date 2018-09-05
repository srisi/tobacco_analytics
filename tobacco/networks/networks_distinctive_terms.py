import numpy as np
import math

from tobacco.networks.networks_vocabulary import get_vocabulary_and_totals
from tobacco.utilities.ocr import get_vocabulary_totals, load_vocabulary_trie
from tobacco.networks.networks_totals import get_totals
from tobacco.utilities.databases import Database

def distinctive_terms_overall(main_name):


    global_totals = get_vocabulary_totals(1)
    vocabulary_trie = load_vocabulary_trie(1)

    local_vocabulary, local_totals = get_vocabulary_and_totals(main_name)

    global_totals_localized = np.zeros(len(local_vocabulary['id_to_token']))
    for token in local_vocabulary['token_to_id']:
        local_token_id = local_vocabulary['token_to_id'][token]
        global_token_id = vocabulary_trie[token]
        global_totals_localized[local_token_id] = global_totals[global_token_id]

    print(len(global_totals), len(local_totals), len(global_totals_localized))

    distinctive_terms = get_distinctive_terms(local_totals, global_totals_localized, local_vocabulary)
    print(distinctive_terms)

    db = Database("TOB_NETWORKS")
    con, cur = db.connect()
    cur.execute('SELECT DISTINCT(tid) as tid from {}_docs'.format(main_name))
    tids = [row['tid'] for row in cur.fetchall()]
    totals2 = get_totals(tids, local_vocabulary)
    dist = get_distinctive_terms(totals2, global_totals_localized, local_vocabulary)
    print("\n", dist)

    totals3 = get_totals(tids, local_vocabulary, tf=True)
    dist = get_distinctive_terms(totals3, global_totals_localized, local_vocabulary)
    print("\n",dist)


def distinctive_term_nodes(main_name):


    local_vocabulary, totals_main = get_vocabulary_and_totals(main_name)
    db = Database("TOB_NETWORKS")
    con, cur = db.connect()
    cur.execute('SELECT DISTINCT(node) as node FROM {}_docs'.format(main_name))
    nodes = [row['node'] for row in cur.fetchall()]

    print(nodes)
    for node in nodes:

        cur.execute('SELECT tid from {}_docs WHERE node = "{}"'.format(main_name, node))
        tids = [row['tid'] for row in cur.fetchall()]
        totals_node = get_totals(tids, local_vocabulary)

        distinctive_terms = get_distinctive_terms(totals_node, totals_main, local_vocabulary)

        print("\n", node)
        print(distinctive_terms)



def get_distinctive_terms(selection, remainder, vocabulary_id_to_token):

    log_likelihoods = np.zeros(len(selection))

    selection_sum = selection.sum()
    remainder_sum = remainder.sum()

    for i in range(len(selection)):

        a = selection[i]
        b = remainder[i]
        if a == 0 or b == 0: continue

        a_plus_b_div_by_totals = (a+b) / (selection_sum+remainder_sum)

        e1 = selection_sum * a_plus_b_div_by_totals
        e2 = remainder_sum * a_plus_b_div_by_totals

        g2 = 2 * ((a * math.log(a / e1)) + b * math.log(b / e2))

        if a/e1 < 1: # equivalent to a * math.log(a/e1) < 0 (log function is 0 at 1)
            g2 = -g2

        log_likelihoods[i] = g2

    max_indices = np.argpartition(log_likelihoods, -50)[-50:]
    max_indices = max_indices[np.argsort(log_likelihoods[max_indices])][::-1]

    distinctive_terms = []
    for i in max_indices:
        term = vocabulary_id_to_token[i]
        distinctive_terms.append((term, selection[i], round(log_likelihoods[i], 2)))

    distinctive_terms = distinctive_terms[:50]


    return distinctive_terms




if __name__ == "__main__":
#
    distinctive_terms_overall('dunn')
    distinctive_term_nodes('dunn')