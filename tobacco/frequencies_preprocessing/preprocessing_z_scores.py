import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from tobacco.configuration import PATH_GOOGLE_TOKENS, YEAR_START, YEAR_END
from tobacco.utilities.vector import Vector

GOOGLE_TOTALS = Vector(np.array(
    [1285712637.0, 1311315033.0, 1266236889.0, 1405505328.0, 1351302005.0, 1397090480.0, 1409945274.0,
     1417130893.0, 1283265090.0, 1354824248.0, 1350964981.0, 1431385638.0, 1356693322.0, 1324894757.0,
     1211361619.0, 1175413415.0, 1183132092.0, 1039343103.0, 1136614538.0, 1388696469.0, 1216676110.0,
     1413237707.0, 1151386048.0, 1069007206.0, 1113107246.0, 1053565430.0, 1216023821.0, 1212716430.0,
     1153722574.0, 1244889331.0, 1183806248.0, 1057602772.0, 915956659.0, 1053600093.0, 1157109310.0,
     1199843463.0, 1232280287.0, 1261812592.0, 1249209591.0, 1179404138.0, 1084154164.0, 1045379066.0,
     890214397.0,  812192380.0, 926378706.0, 1203221497.0, 1385834769.0, 1486005621.0, 1641024100.0,
     1644401950.0, 1603394676.0, 1621780754.0, 1590464886.0, 1662160145.0, 1751719755.0, 1817491821.0,
     1952474329.0, 1976098333.0, 2064236476.0, 2341981521.0, 2567977722.0, 2818694749.0, 2955051696.0,
     2931038992.0, 3300623502.0, 3466842517.0, 3658119990.0, 3968752101.0, 3942222509.0, 4086393350.0,
     4058576649.0, 4174172415.0, 4058707895.0, 4045487401.0, 4104379941.0, 4242326406.0, 4314577619.0,
     4365839878.0, 4528331460.0, 4611609946.0, 4627406112.0, 4839530894.0, 4982167985.0, 5309222580.0,
     5475269397.0, 5793946882.0, 5936558026.0, 6191886939.0, 6549339038.0, 7075013106.0, 6895715366.0,
     7596808027.0, 7492130348.0, 8027353540.0, 8276258599.0, 8745049453.0, 8979708108.0, 9406708249.0,
     9997156197.0, 11190986329.0, 11349375656.0, 12519922882.0, 13632028136.0, 14705541576.0, 14425183957.0,
     15310495914.0, 16206118071.0, 19482936409.0, 19482936409.0, 19482936409.0, 19482936409.0, 19482936409.0,
     19482936409.0, 19482936409.0, 19482936409.0, 19482936409.0], dtype=np.float64))



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


def get_individual_z_score(token_name: str, tobacco_token_counts: Vector,
                           tobacco_totals: Vector):

    """ Get the annual z scores for one token.
    This would be better with using Vectors everywhere but that would require implementing more
    mathematical operations and I don't think it's currently worth the trouble.

    >>> from tobacco.frequencies_preprocessing.preprocessing_years_cython import transform_doc_to_year_array
    >>> from tobacco.frequencies_preprocessing.preprocessing_totals import get_totals_vector
    >>> addiction_counts = Vector().load_token_vector('addiction', return_type='np_int32')
    >>> addiction_counts_yearly = addiction_counts.convert_to_year_array()
    >>> total_counts_yearly = get_totals_vector('docs', 'np_int32').convert_to_year_array()
    >>> z_scores = get_individual_z_score('addiction', addiction_counts_yearly, total_counts_yearly)
    >>> print(f'Z-score for 1950: {z_scores[50]}')
    Z-score for 1950: 2.4172451505755848

    :param token_name: str
    :param tobacco_token_counts: np.ndarray[int], len 116
    :param tobacco_totals: np.ndarray[int], len 116
    :return:
    """

    google_token_counts = get_absolute_google_counts(token_name)

    p1 = tobacco_token_counts.vector / tobacco_totals.vector
    p2 = google_token_counts.vector / GOOGLE_TOTALS.vector
    p3 = (tobacco_token_counts.vector + google_token_counts.vector) / \
         (tobacco_totals.vector + GOOGLE_TOTALS.vector)
    sd = np.sqrt((p3 * (1 - p3)) * ((1 / tobacco_totals.vector) + (1/GOOGLE_TOTALS.vector)))
    z = (p1-p2) / sd
    z = np.nan_to_num(z)

    return z


def get_absolute_google_counts(token_name: str) -> np.ndarray:

    """    This function retrieves the absolute counts for a given token from the Google Ngram Viewer.

    It first loads the relative frequencies from the ngram viewer and the absolute counts
    for the corpus from Google's source data.
    Then, it multiplies the absolute number of terms in the corpus for any given year with the
    relative frequency of the search token.

    >>> google_counts = get_absolute_google_counts('addiction')
    >>> print(f'Google counts for addiction in 1950: {google_counts[50]}')
    Google counts for addiction in 1950: 2482.0

    >>> type(google_counts)
    <class 'tobacco.utilities.vector.Vector'>

    """

    hash = hashlib.sha256(token_name.encode()).hexdigest()
    file_path = Path(PATH_GOOGLE_TOKENS, hash[0], hash[1], hash[2], hash[3], f'{hash}.npy')

    try:
        # this really doesn't need a hash
        absolute_counts = Vector().load_from_disk(file_path, return_type='np_int32')
#        absolute_counts = np.load(token_path+hash_path+'.npy')

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
        absolute_counts = [round(relative_frequencies[i] * GOOGLE_TOTALS[i])
                           for i in range(len(relative_frequencies))]
        absolute_counts = Vector(np.array(absolute_counts, dtype=np.float64))
        absolute_counts.save_to_disk(file_path)



#        hash = hashlib.sha256(token_name.encode()).hexdigest()
#        file_path = Path(PATH_GOOGLE_TOKENS, hash[0], hash[1], hash[2], hash[3])

#        token_path = PATH_GOOGLE_TOKENS + '{}/{}/{}/{}/'.format(hash_path[0], hash_path[1], hash_path[2], hash_path[3])

 #       if not Path.exists(file_path):
 #           print(file_path)
 #           Path.mkdir(file_path, parents=True)
#        np.save(token_path + hash_path, absolute_counts)

    return absolute_counts



if __name__ == "__main__":

    from tobacco.frequencies_preprocessing.preprocessing_years_cython import transform_doc_to_year_array
    from tobacco.frequencies_preprocessing.preprocessing_totals import get_totals_vector
    addiction_counts = Vector().load_token_vector('addiction', return_type='np_int32')
    addiction_counts_yearly = addiction_counts.convert_to_year_array()
    total_counts_yearly = get_totals_vector('docs', 'np_int32').convert_to_year_array()
    z_scores = get_individual_z_score('addiction', addiction_counts_yearly, total_counts_yearly)
    print(f'Z-score for 1950: {z_scores[50]}')