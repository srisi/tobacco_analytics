import hashlib
import json
import os
import urllib.error
import urllib.request

import numpy as np
from tobacco.configuration import PATH_GOOGLE_TOKENS, GOOGLE_TOTALS, YEAR_START, YEAR_END


def get_z_scores(tokens_list, tobacco_totals, mp_results_queue):
    """ Calculates the z scores for all search terms.

    Original implementation by Claytion Darwin. Used here with arrays of ints (instead of ints) to calculate
    the results for all years at the same time.

    Original:
    def prop_compare(x1,n1,x2,n2):
        "x1=obs_freq,n1=obs_total,x2=norm_freq,n2=norm_total"
        # required: n values > 0
        # required: x values >= 0
        # floats
        n1,n2 = float(n1),float(n2)
        # get proportions
        p1 = x1/n1
        p2 = x2/n2
        p3 = (x1+x2)/(n1+n2)
        # check validity
        valid = 0
        if n1*p3 >= 5 and n1*(1-p3) >= 5 and n2*p3 >= 5 and n2*(1-p3) >= 5:
            valid = 1
        # get standard deviation
        sd = math.sqrt((p3*(1-p3))*((1/n1)+(1/n2)))
        # get z statistic
        z = 0.0
        if sd:
            z = (p1-p2)/sd
        # return z-score and v-score
        return z,valid

    :param tokens_list: list of dicts with keys 'token' (str), 'totals' (np. array)
    :param tobacco_totals: totals (by year) for all selected documents.
    :param mp_results_queue: Multiprocessing results queue
    :return:
    """

    z_scores = []

    for token_dict in tokens_list:
        token_name = token_dict['token']
        tobacco_token_counts = token_dict['counts']
        z_scores.append(get_individual_z_score(token_name, tobacco_token_counts, tobacco_totals))

    mp_results_queue.put(('z_scores', z_scores))


def get_individual_z_score(token_name, tobacco_token_counts, tobacco_totals):

    """ Get the annual z scores for one token.

    :param token_name:
    :param tobacco_token_counts:
    :param tobacco_totals:
    :return:
    """

    google_token_counts = get_absolute_google_counts(token_name)

    p1 = tobacco_token_counts / tobacco_totals
    p2 = google_token_counts / GOOGLE_TOTALS
    p3 = (tobacco_token_counts + google_token_counts) / (tobacco_totals + GOOGLE_TOTALS)
    sd = np.sqrt((p3 * (1 - p3)) * ((1 / tobacco_totals) + (1/GOOGLE_TOTALS)))
    z = (p1-p2) / sd
    z = np.nan_to_num(z)

    return z


def get_absolute_google_counts(token_name):

    """    This function retrieves the absolute counts for a given token from the Google Ngram Viewer.

    It first loads the relative frequencies from the ngram viewer and the absolute counts
    for the corpus from Google's source data.
    Then, it multiplies the absolute number of terms in the corpus for any given year with the
    relative frequency of the search token.


    """

    try:
        # this really doesn't need a hash

        hash_path = hashlib.sha256(token_name.encode()).hexdigest()
        token_path = PATH_GOOGLE_TOKENS + '{}/{}/{}/{}/'.format(hash_path[0], hash_path[1], hash_path[2], hash_path[3])
        absolute_counts = np.load(token_path+hash_path+'.npy')

    except FileNotFoundError:

        corpus_id = 15
        # construct the url, i.e. place the token and other parameters where they belong
        url = 'https://books.google.com/ngrams/interactive_chart?content={}&year_start={}&year_end={}' \
              '&corpus={}&smoothing=0'.format(token_name.replace(' ', '+'), YEAR_START, YEAR_END, corpus_id)

        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                page = response.read().decode('utf-8')

                if page.find('var data = [];') > -1:
                    relative_frequencies = 116 * [0]
                else:

                    start = page.find('var data = [')
                    end = page.find('}];', start)
                    data_dict = json.loads(page[start+11:end+2])[0]
                    relative_frequencies = data_dict['timeseries']
                    relative_frequencies += 8 * [relative_frequencies[-1]]

        except urllib.error.HTTPError:
            relative_frequencies = 116 * [0]

        # if general error, return 0 but don't store
        except:
            temp = 116 * [0]
            return np.array([round(temp[i] * GOOGLE_TOTALS[i]) for i in range(len(temp))], dtype=np.float)


        # Step 3: calculate the absolute number of appearances by multiplying the frequencies with the total number of tokens
        absolute_counts = np.array(
            [round(relative_frequencies[i] * GOOGLE_TOTALS[i]) for i in range(len(relative_frequencies))],
            dtype=np.float)

        hash_path = hashlib.sha256(token_name.encode()).hexdigest()
        token_path = PATH_GOOGLE_TOKENS + '{}/{}/{}/{}/'.format(hash_path[0], hash_path[1], hash_path[2], hash_path[3])
        if not os.path.exists(token_path): os.makedirs(token_path)
        np.save(token_path + hash_path, absolute_counts)

    return absolute_counts



if __name__ == "__main__":

    import time
    s = time.time()
    token = 'addictors'
    t = get_absolute_google_counts(token)
    print(len(t), t)
    print(time.time() - s)