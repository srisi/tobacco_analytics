from tobacco.text_passages.find_text_passages import find_text_passages
from tobacco.frequencies_preprocessing.preprocessing_globals_loader import get_globals

import pickle

from IPython import embed


EMOTIONS = [
    'positive', 'negative',
    'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust',
    'valence', 'dominance', 'arousal',
    'anger_int', 'fear_int', 'sadness_int', 'joy_int'
]

def sentiment(search_term):

    try:
        sections = pickle.load(open(f'{search_term}a.pickle', 'rb'))
    except FileNotFoundError:

        globals = get_globals(globals_type='passages')
        active_filters = {'doc_type': ['internal communication'], 'collection': [],
                          'availability': [], 'term': []}

        results = find_text_passages([search_term], active_filters=active_filters,
                                     years_to_process=[i for i in range(1950, 1990)],
                                     globals=globals, passage_length=200, logging=True)

        embed()

        sections = []
        for year in results['sections']:
            if not results['sections'][year]:
                continue

            for section in results['sections'][year]:
                s = {
                    'text': section[7].replace('<b>', '').replace('</b>', ''),
                    'date': section[4],
                    'id': section[0]
                }
                sections.append(s)

        pickle.dump(sections, open(f'{search_term}.pickle', mode='wb'))

    lexicon = load_emotion_lexicon()
    for section in sections:
        for emotion in EMOTIONS:
            section[emotion] = 0
            section[f'{emotion}_terms'] = []

        for word in section['text'].split():
            if word in lexicon:
                for emotion in EMOTIONS:
                    try:
                        section[emotion] += lexicon[word][emotion]
                        if lexicon[word][emotion] > 0:
                            section[f'{emotion}_terms'].append(word)
                    except KeyError:
                        pass

    for emotion in EMOTIONS:
        print("")
        for section in sorted(sections, key=lambda x:x[emotion])[::-1][:5]:
            print('{:8s} {:9s}. {:.1f}. {:12s}. {}. ({})'.format(section['id'], section['date'],
                                           section[emotion], emotion, section['text'],
                                                         " ".join(section[f'{emotion}_terms']))
                  )

    embed()

POSITIVE = [
    'happily', 'luckily', 'fortunately'
]


def load_emotion_lexicon():

    try:
        return pickle.load(open('emotion_lexicon.pickle', 'rb'))
    except FileNotFoundError:
        lexicon = {'luckily': {}, 'fortunately': {}}

        with open ('NRC-VAD-Lexicon.txt') as f:
            for line in f.readlines()[1:]:
                token, valence, arousal, dominance = line.split('\t')
                lexicon[token] = {
                    'valence': (float(valence) -0.5) * 2,
                    'arousal': (float(arousal) - 0.5) *2,
                    'dominance': (float(dominance) - 0.5) * 2
                }

        with open('emotion_lexicon.txt') as f:
            for line in f.readlines():
                token, emotion, value = line.split('\t')
                if token in lexicon:
                    lexicon[token][emotion] = float(value)
                else:
                    lexicon[token] = {
                        emotion: float(value)
                    }

        with open('affect_intensity.txt') as f:
            for line in f.readlines():
                token, value, emotion = line.split('\t')
                emotion = f'{emotion.strip()}_int'
                try:
                    lexicon[token][emotion] = (float(value) - 0.5) * 2
                except KeyError:
                    lexicon[token] = {emotion: (float(value) - 0.5) * 2 }


        for term in POSITIVE:
            lexicon[term]['sm_positive'] = 1

        pickle.dump(lexicon, open('emotion_lexicon.pickle', 'wb'))
        return lexicon




if __name__ == '__main__':
    sentiment('addiction')
#    load_valence_lexicon()