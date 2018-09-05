
from tobacco.configuration import STOP_WORDS_SKLEARN, YEAR_COUNT, PATH_TOKENIZED
SECTION_LENGTH = 100

NETWORK_CONFIGS ={
    'dunn':{
        'name': 'DUNN,WL',
        'start_year': 1960,
        'end_year': 1980
    },
    'green': {
        'name': 'Green, SJ',
        'start_year': 1950,
        'end_year': 1980
    },
}

STOP_WORDS_NETWORKS = set(STOP_WORDS_SKLEARN).union(
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