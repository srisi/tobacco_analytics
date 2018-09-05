import socket
import re

import numpy as np

from tobacco.secrets import get_secret


# Split strings into tokens (includes splitting dashes)
WORD_SPLIT_PATTERN = r"\b\w+\b"
WORD_SPLIT_REGEX = re.compile(WORD_SPLIT_PATTERN)

DOC_COUNT = 11303161
SECTION_LENGTH = 200
SECTION_COUNT = 89483683

YEAR_START = 1901
YEAR_END = 2016
YEAR_COUNT = 116


# Determine if the current machine is local or an ec2 instance
if socket.gethostname() == get_secret('local_machine_name'):
    CURRENT_MACHINE = 'local'
else:
    CURRENT_MACHINE = 'aws'

# Which results db to use--local or RDS
# to switch to local rds server, set results_db_name to RESULTS_LOCAL
# this would make more sense to change to REDIS_DB_NAME but then I'd have to track them down...
RESULTS_DB_NAME = 'RESULTS_RDS'
if RESULTS_DB_NAME == 'RESULTS_LOCAL' and CURRENT_MACHINE == 'local':
    REDIS_HOST = 'localhost'
else:
    REDIS_HOST = get_secret('aws_redis_host')


if CURRENT_MACHINE == 'local':
        PATH_TOKENIZED = '/pcie/tobacco/tokenized/'
        PATH_OCR_FILES = '/home/stephan/tobacco/ocr/tob_docs/'
        PATH_TOKENS = '/tobacco/tokens/'
        PATH_GOOGLE_TOKENS = '/tobacco/google_tokens/'
else:
        # 6/18/17 something went very wrong when taring the archives so they reproduce the original structure on
        # my machine. Don't ask.
        PATH_TOKENIZED = '/tobacco/tokenized/'
        PATH_OCR_FILES = '/tobacco/docs/home/stephan/tobacco/ocr/tob_docs/'
        PATH_TOKENS = '/tobacco/tobacco/tokens/'
        PATH_GOOGLE_TOKENS = '/tobacco/google_tokens/'

# list of totals of the google ngram corpus.
# this would make more sense to put somewhere else...
GOOGLE_TOTALS = np.array([1285712637.0, 1311315033.0, 1266236889.0, 1405505328.0, 1351302005.0, 1397090480.0, 1409945274.0, 1417130893.0, 1283265090.0, 1354824248.0, 1350964981.0, 1431385638.0, 1356693322.0, 1324894757.0, 1211361619.0, 1175413415.0, 1183132092.0, 1039343103.0, 1136614538.0, 1388696469.0, 1216676110.0, 1413237707.0, 1151386048.0, 1069007206.0, 1113107246.0, 1053565430.0, 1216023821.0, 1212716430.0, 1153722574.0, 1244889331.0, 1183806248.0, 1057602772.0, 915956659.0, 1053600093.0, 1157109310.0, 1199843463.0, 1232280287.0, 1261812592.0, 1249209591.0, 1179404138.0, 1084154164.0, 1045379066.0, 890214397.0, 812192380.0, 926378706.0, 1203221497.0, 1385834769.0, 1486005621.0, 1641024100.0, 1644401950.0, 1603394676.0, 1621780754.0, 1590464886.0, 1662160145.0, 1751719755.0, 1817491821.0, 1952474329.0, 1976098333.0, 2064236476.0, 2341981521.0, 2567977722.0, 2818694749.0, 2955051696.0, 2931038992.0, 3300623502.0, 3466842517.0, 3658119990.0, 3968752101.0, 3942222509.0, 4086393350.0, 4058576649.0, 4174172415.0, 4058707895.0, 4045487401.0, 4104379941.0, 4242326406.0, 4314577619.0, 4365839878.0, 4528331460.0, 4611609946.0, 4627406112.0, 4839530894.0, 4982167985.0, 5309222580.0, 5475269397.0, 5793946882.0, 5936558026.0, 6191886939.0, 6549339038.0, 7075013106.0, 6895715366.0, 7596808027.0, 7492130348.0, 8027353540.0, 8276258599.0, 8745049453.0, 8979708108.0, 9406708249.0, 9997156197.0, 11190986329.0, 11349375656.0, 12519922882.0, 13632028136.0, 14705541576.0, 14425183957.0, 15310495914.0, 16206118071.0, 19482936409.0, 19482936409.0, 19482936409.0, 19482936409.0, 19482936409.0, 19482936409.0, 19482936409.0, 19482936409.0, 19482936409.0], dtype=np.float)

# list of all collections included in tobacco analytics, check with TTID to see which collections this includes
VALID_COLLECTIONS = {
        1:  'ucsf_brown & williamson collection',
        2:  'joe camel litigation collection',
        3:  'datta',
        5:  'pm',
        6:  'rj',
        7:  'll',
        8:  'bw',
        9:  'at',
        10: 'ti',
        11: 'ct',
        13: 'liggett & myers records',
        15: 'ba',
        16: 'us tobacco records on smokeless tobacco',
        17: 'richard w. pollay cigarette ads collection',
        18: 'research',
        19: 'canada',
        20: 'tobacco control web archives',
        21: 'gallaher records',
        22: 'trinkets & trash marketing collection',
        37: 'secondhand smoke litigation collection',
        38: 'health warning labels collection',
        41: 'indian tobacco industry collection',
        42: 'e-cigarette marketing web archives'
}

# set of removed document types (currently only 1
REMOVED_DOC_TYPES= {
        'trial list'    # trial lists are invariably compilations of documents from different years.
                        # hence, they trip up the frequency graphs
}

# tokens added to include in future processing of the tobacco docs.
ADDED_TOKENS = {
        1: ['inbifo', 'icosi', 'blalock',
            'am', 'we', 'so', 'do', 'go',
            'mcv', 'vcu', 'omas', 'depo', 'ebay', 'esty', 'lanza', 'ludmerer',
            'denialist'],
        2: ['compound w', 'compound wm', 'compound ws']
}


# Set of stopwords used by sklearn. Unsure if I added any new ones.
STOP_WORDS_SKLEARN = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves", "s"])
