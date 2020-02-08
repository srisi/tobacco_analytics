
from tobacco.utilities.databases import Database
import random

from IPython import embed

def markov(seed, no_token_to_generate=100, tokens_to_select=1):

    db =  Database('TOB_FULL')
    con, cur = db.connect()

    seed_length = len(seed.split())

    output = seed


    for i in range(no_token_to_generate):
        cur.execute('''SELECT token, total 
                       from tokens 
                       where token like "{} %" and ngram={}
                       order by total desc;'''.format(seed, seed_length+tokens_to_select))



        continuations = cur.fetchall()
        possible_selections = []
        for continuation in continuations:
            token = " ".join(continuation['token'].split()[seed_length:][-tokens_to_select:])
            for i in range(continuation['total']):
                possible_selections.append(token)

        selection = random.choice(possible_selections)
        output += " " + selection
        seed = " ".join(output.split()[-2:])


        print(output)



if __name__ == "__main__":
    markov('conclusive evidence that', tokens_to_select=2, no_token_to_generate=20)