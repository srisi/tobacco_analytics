import re
import pickle
import array
import math
import json

import networkx as nx

from collections import Counter, defaultdict

from scipy.sparse import csc_matrix, csr_matrix

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from tobacco.configuration import PATH_TOKENIZED, WORD_SPLIT_REGEX
from tobacco.utilities.databases import Database
from tobacco.utilities.sparse_matrices import load_csc_matrix_from_file, store_csr_matrix_to_file

from tobacco.networks.networks_config import NETWORK_CONFIGS, SECTION_LENGTH, STOP_WORDS_NETWORKS
from tobacco.networks.networks_distinctive_terms import get_distinctive_terms
from tobacco.utilities.ocr import get_vocabulary_totals, load_vocabulary_trie



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from IPython import embed
english_name_regex = re.compile('[A-Z][a-zA-Z]+[, -]+[A-Z]{1,3}')


class Networks_Selector():

    '''
    12/10/17 What exactly does this do


    '''

    def __init__(self, main_name):

        self.main_name = main_name
        self.__config = NETWORK_CONFIGS[main_name]

        # alphabetically sorted list of all nodes
        nodes_init = self.__get_nodes_init()

        # create table of all sections at mainname_sections if not exists
        self.__init_sections_table()

        # store node and edge data
        # each one has name, id, no_docs, no_docs_weighted, no_tokens, no_tokens_weighted
        self.nodes, self.edges = self.__get_node_and_edge_data(nodes_init)

        # dict of main dtms, contains 'count', 'tf', and 'tfidf'
        self.dtm = self.__get_dtms()

        # dict of 'node', 'edge', and 'year' filters
        self.filters = self.__get_filters()

        # totals is np array of totals
        # vocabulary is a dict of (dict)'token_to_id', (dict)'id_to_token', and (list)'ordered'), which is an alphabetical list of all terms
        self.vocabulary, self.totals = self.__get_vocabulary_and_totals()





        self.edge_weights = self.__get_edge_weights()

        self.node_positions = self.__node_positions()

        # self.node_positions = self.__get_node_positions()

        # distinctive terms is a dict of 'overall' (network vs. legacy), 'node' (node vs. network), and 'edge' (edge vs. network)
        self.distinctive_terms = self.__get_distinctive_terms()

        # self.term_counts_by_year_and_edge = self.__get_term_counts_by_year_and_edge()

        # self.edge_term_weights = self.__get_edge_term_weights()

        self.__generate_json()

    def __generate_json(self):

        output = {
            'nodes': [],
            'links': []
        }

        for node in self.get_nodes(sorting='no_docs_weighted', full_data=True):
            output['nodes'].append(node)
        for edge in self.get_edges(sorting='no_docs_weighted', full_data=True):
            if edge['no_docs'] > 10:
                edge['source'] = edge['node1']
                edge['target'] = edge['node2']
                output['links'].append(edge)


        print(len(output['links']))

        with open('test_data.json', 'w') as f:
            json.dump(output, f)




    def __node_positions(self):

        plt.figure(3,figsize=(18,18))

        G = nx.Graph()
        local_nodes = []
        for node in self.get_nodes(sorting='no_docs_weighted')[:25]:
            print(node)
            local_nodes.append(node)
            G.add_node(node)

        for edge in self.get_edges(sorting='no_docs_weighted')[:200]:
            if self.edge_weights['all'][edge] > 0.001 and edge[0] in local_nodes and edge[1] in local_nodes:
                # print(edge[0], edge[1], self.edge_weights['all'][edge])
                G.add_edge(edge[0], edge[1], {'weight': self.edges[edge]['no_docs_weighted']*0.01})
                # G.add_edge(edge[0], edge[1], {'weight': self.edge_weights['all'][edge]*100})

        pos = nx.spring_layout(G, weight='weight')
        node_sizes = [100 for i in range(len(self.nodes))]

        print(G.edges(data=True))

        edge_widths = [i[2]['weight']*1 for i in G.edges(data=True)]

        nx.draw_networkx_nodes(G, pos,
                               nodelist=G.nodes(),
                               node_shape='s',
                               node_color='r',
                               node_size=[node_sizes],
                               alpha=0.8)
        nx.draw_networkx_edges(G, pos,
                               width=edge_widths,
                               alpha=0.5)

        labels = {}
        for idx, name in enumerate(G.nodes()):
            labels[name] = name
        nx.draw_networkx_labels(G, pos, labels)
        plt.axis('off')
        # d = json_d
        plt.savefig("test.png", bbox_inches='tight')


    def __get_term_counts_by_year_and_edge(self):
        '''
        'overall' -> csr matrix [edge, term]


        :return:
        '''


        try:
            term_counts_by_year_and_edge = pickle.load(open(PATH_TOKENIZED + 'networks/{}_term_counts_by_year_and_edge.pickle'.format(self.main_name), 'rb'))
        except IOError:

            term_counts_by_year_and_edge = {}

            for term in self.vocabulary['ordered']:
                term_counts_by_year_and_edge[term] = {}
                for year in self.get_years() + ['overall']:
                    term_counts_by_year_and_edge[term][year] = defaultdict(int)


            # process overall
            for edge_id, edge in enumerate(self.edges):
                print("{}/{}. {}".format(edge_id, len(self.edges), edge))
                filtered_dtm = self.apply_filter(edges=[edge])
                word_counts = np.array(filtered_dtm.sum(axis=0)).flatten()
                for term_id, term_count in enumerate(word_counts):
                    if term_count > 0:
                        term = self.vocabulary['id_to_token'][term_id]
                        term_counts_by_year_and_edge[term]['overall'][edge] = term_count

            # process years
            for year in self.get_years():
                for edge_id, edge in enumerate(self.edges):
                    print("{}: {}/{}. {}".format(year, edge_id, len(self.edges), edge))
                    filtered_dtm = self.apply_filter(edges=[edge], years=[year])
                    word_counts = np.array(filtered_dtm.sum(axis=0)).flatten()
                    for term_id, term_count in enumerate(word_counts):
                        if term_count > 0:
                            term = self.vocabulary['id_to_token'][term_id]
                            term_counts_by_year_and_edge[term][year][edge] = term_count



            pickle.dump(term_counts_by_year_and_edge, open(PATH_TOKENIZED + 'networks/{}_term_counts_by_year_and_edge.pickle'.format(self.main_name), 'wb'))

        return term_counts_by_year_and_edge



    def get_nodes(self, sorting='id', full_data=False):
        '''

        :param sorting: 'id', 'no_docs', 'no_docs_weighted', 'no_tokens', 'no_tokens_weighted'
        :param full_data: include all data or just the name
        :return:
        '''

        reverse=True
        if sorting == 'id': reverse = False


        if full_data:
            node_list = [dict(x[1]) for x in sorted(self.nodes.items(), key= lambda x: x[1][sorting], reverse=reverse)]
        else:
            node_list = [x[0] for x in sorted(self.nodes.items(), key= lambda x: x[1][sorting], reverse=reverse)]
        return node_list

        # return sorted(list(self.filters['node'].keys()))


    def get_edges(self, sorting='id', full_data=False):

        reverse=True
        if sorting == 'id': reverse = False


        if full_data:
            edge_list = [dict(x[1]) for x in sorted(self.edges.items(), key= lambda x: x[1][sorting], reverse=reverse)
                                    if x[1]['no_docs_weighted'] > 0]
        else:
            edge_list = [x[0] for x in sorted(self.edges.items(), key= lambda x: x[1][sorting], reverse=reverse)]

        return edge_list


    def get_years(self):
        return sorted(list(self.filters['year'].keys()))


    def apply_filter(self, years=[], nodes=[], edges=[], main_author_is=[], term='', vectorizer_type='count'):

        '''

        Allows filtering the main dtm for years, nodes, and edges


        12/10/17: main_author_is probably no longer needed

        :param years:
        :param nodes:
        :param main_author_is:
        :param term:
        :param vectorizer_type:
        :return:
        '''


        dtm_filter = self.dtm[vectorizer_type].copy()

        if term:
            col_id = self.vocabulary['token_to_id'][term]
            dtm_filter = dtm_filter[:, col_id]

        if years or nodes or edges or main_author_is:

            valid_rows = set()
            year_rows = set()
            node_rows = set()
            edge_rows = set()
            main_author_is_rows = set()

            for year in years:
                try:
                    year_rows = year_rows.union(self.filters['year'][year])
                except KeyError:
                    pass
            if year_rows and not valid_rows:
                valid_rows = year_rows

            for node in nodes:
                try:
                    node_rows = node_rows.union(self.filters['node'][node])
                except KeyError:
                    print("Node {} not found".format(node))
            if node_rows:
                if valid_rows:
                    valid_rows = valid_rows.intersection(node_rows)
                else:
                    valid_rows = node_rows

            for edge in edges:
                try:
                    edge_rows = edge_rows.union(self.filters['edge'][edge])
                except KeyError:
                    print("Edge {} not found".format(edge))
            if edge_rows:
                if valid_rows:
                    valid_rows = valid_rows.intersection(edge_rows)
                else:
                    valid_rows = edge_rows

            for mas in main_author_is:
                try: main_author_is_rows = main_author_is_rows.union(self.filters['main_author_is'][mas])
                except KeyError: pass

            if main_author_is_rows:
                if valid_rows:
                    valid_rows = valid_rows.intersection(main_author_is_rows)
                else:
                    valid_rows = main_author_is_rows


            csr = dtm_filter.tocsr()

            if vectorizer_type == 'count':
                filter_data = array.array(str("l"))
            else:
                filter_data = array.array(str("d"))

            filter_indices = array.array(str("l"))
            filter_indptr = array.array(str("l"))
            filter_indptr.append(0)


            for row_id in range(csr.shape[0]):
                if row_id in valid_rows:
                    start_id = csr.indptr[row_id]
                    end_id = csr.indptr[row_id+1]
                    for i in range(start_id, end_id):
                        filter_data.append(csr.data[i])
                        filter_indices.append(csr.indices[i])
                filter_indptr.append(len(filter_data))


            filter_indices = np.frombuffer(filter_indices, dtype=np.int64)
            filter_indptr = np.frombuffer(filter_indptr, dtype=np.int64)

            if vectorizer_type == 'count':
                filter_data = np.frombuffer(filter_data, dtype=np.int64)
                dtm_filter = csr_matrix((filter_data, filter_indices, filter_indptr),
                                        shape=self.dtm[vectorizer_type].shape, dtype=np.int64).tocsc()
            else:
                filter_data = np.frombuffer(filter_data, dtype=np.float64)
                dtm_filter = csr_matrix((filter_data, filter_indices, filter_indptr),
                                        shape=self.dtm[vectorizer_type].shape, dtype=np.float64).tocsc()

        return dtm_filter

    def __get_node_and_edge_data(self, nodes_init):
        '''
        create dicts to store data on nodes and edges.
        each node and edge has:
            name
            id
            no_docs
            no_docs_weighted
            no_tokens
            no_tokens_weighted

        :param nodes_init:
        :return:
        '''

        try:
            nodes = pickle.load(open(PATH_TOKENIZED + 'networks/{}_nodes.pickle'.format(self.main_name), 'rb'))
            edges = pickle.load(open(PATH_TOKENIZED + 'networks/{}_edges.pickle'.format(self.main_name), 'rb'))

        except IOError:

            edges_init = [(i,j) for i in nodes_init for j in nodes_init if i !=j and i < j ]

            # initialize a dict for every node and edge
            nodes = {n:defaultdict(float, {'name': n, 'id': n_id}) for n_id, n in enumerate(nodes_init)}
            edges = {e:defaultdict(float, {'node1': e[0], 'node2': e[1], 'id': e_id}) for e_id, e in enumerate(edges_init)}

            db_net = Database("TOB_NETWORKS")
            con_net, cur_net = db_net.connect()

            cur_net.execute('SELECT * FROM {}_sections'.format(self.main_name))

            while True:
                row = cur_net.fetchone()
                if not row: break
                else:

                    node1 = row['node1']
                    node2 = row['node2']
                    tid_section = row['tid_section']
                    weight = row['weight']
                    text = row['text']

                    # no docs
                    if tid_section == 0:
                        nodes[node1]['no_docs'] += 1
                        nodes[node2]['no_docs'] += 1
                        edges[(node1, node2)]['no_docs'] += 1

                        nodes[node1]['no_docs_weighted'] += weight
                        nodes[node2]['no_docs_weighted'] += weight
                        edges[(node1, node2)]['no_docs_weighted'] += weight

                    # no tokens
                    no_tokens = len(text.split())
                    nodes[node1]['no_tokens'] += no_tokens
                    nodes[node2]['no_tokens'] += no_tokens
                    edges[(node1, node2)]['no_tokens'] += no_tokens

                    nodes[node1]['no_tokens_weighted'] += no_tokens * weight
                    nodes[node2]['no_tokens_weighted'] += no_tokens * weight
                    edges[(node1, node2)]['no_tokens_weighted'] += no_tokens * weight

            pickle.dump(nodes, open(PATH_TOKENIZED + 'networks/{}_nodes.pickle'.format(self.main_name), 'wb'))
            pickle.dump(edges, open(PATH_TOKENIZED + 'networks/{}_edges.pickle'.format(self.main_name), 'wb'))

        return nodes, edges






    def __get_nodes_init(self):

        try:
            nodes = pickle.load(open(PATH_TOKENIZED + 'networks/{}_nodes.pickle'.format(self.main_name), 'rb'))

        except IOError:
            print("Top 50 nodes not available. Creating now...")
            # Step 1: Identify top 50 nodes
            node_counter = Counter()

            db = Database("TOB_FULL")
            con, cur = db.connect()


            author_and_recipient_commands = [
               '''SELECT recipients.recipient as node, docs.tid, docs.year, "author" as main_person_is
                                  FROM authors, recipients, docs
                                  WHERE authors.author="{}" and authors.doc_id=recipients.doc_id AND docs.id=authors.doc_id
                                  AND docs.year >= {} AND docs.year <= {};'''.format(
                   NETWORK_CONFIGS[self.main_name]['name'], NETWORK_CONFIGS[self.main_name]['start_year'], NETWORK_CONFIGS[self.main_name]['end_year']),
               '''SELECT authors.author as node, authors.doc_id, docs.tid, docs.year, "recipient" as main_person_is
                                  FROM authors, recipients, docs
                                  WHERE recipients.recipient="{}" and authors.doc_id=recipients.doc_id AND docs.id=authors.doc_id
                                  AND docs.year >= {} AND docs.year <= {};'''.format(
                   NETWORK_CONFIGS[self.main_name]['name'], NETWORK_CONFIGS[self.main_name]['start_year'], NETWORK_CONFIGS[self.main_name]['end_year'])
            ]

            for command in author_and_recipient_commands:
                cur.execute(command)
                while True:
                    row = cur.fetchone()
                    if not row: break
                    else:
                        for person in english_name_regex.findall(row['node']):
                            node_counter[person] += 1

            nodes = sorted([i[0] for i in node_counter.most_common(50)])
            print(nodes)
            pickle.dump(nodes, open(PATH_TOKENIZED + 'networks/{}_nodes.pickle'.format(self.main_name), 'wb'))


        return nodes


    def __get_vocabulary_and_totals(self):

        try:
            vocabulary_dict = pickle.load(open(PATH_TOKENIZED + 'networks/{}_vocabulary.pickle'.format(self.main_name), 'rb'))
            totals = np.load(open(PATH_TOKENIZED + 'networks/{}_totals.npy'.format(self.main_name), 'rb'))
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
            np.save(open(PATH_TOKENIZED + 'networks/{}_totals.npy'.format(self.main_name), 'wb'), totals_name)


            pickle.dump(vocabulary_dict, open(PATH_TOKENIZED + 'networks/{}_vocabulary.pickle'.format(self.main_name), 'wb'))

        return vocabulary_dict, totals


    def __init_sections_table(self):

        '''
        12/10/17
        Adds all sections to the section table

        Add: if len(sections) > 0 (i.e. it has been filled already, return immediately.)


        :return:
        '''


        db_net = Database("TOB_NETWORKS")
        con_net, cur_net = db_net.connect()

        cur_net.execute('SELECT COUNT(*) FROM {}_sections'.format(self.main_name))
        section_count = cur_net.fetchall()[0]['COUNT(*)']
        if section_count > 0:
            return

        else:
            from tobacco.frequencies_preprocessing.preprocessing_docs import get_ocr_by_tid

            db_full = Database("TOB_FULL")
            _, cur_full = db_full.connect()
            section_id= 0


            cur_net.execute('''CREATE TABLE IF NOT EXISTS {}_sections(
                        section_id      int           NOT NULL UNIQUE,
                        node1           varchar(255)  NOT NULL,
                        node2           varchar(255)  NOT NULL,
                        tid             varchar(10)   NOT NULL,
                        tid_section     INT           NOT NULL,
                        weight          float         NOT NULL,
                        year            int           NOT NULL,
                        text            TEXT          NOT NULL,

                        PRIMARY KEY(section_id));'''.format(self.main_name))
            con_net.commit()

            count = 0
            for edge in self.edges:
                count += 1
                print("working on edge {}/{}: {}".format(count, len(self.edges), edge))
                author_and_recipient_commands = [
                   '''SELECT docs.tid as tid, docs.year as year, authors.author as author, recipients.recipient as recipient
                                      FROM authors, recipients, docs
                                      WHERE authors.author LIKE "%{}%"  AND recipients.recipient LIKE "%{}%"
                                            AND authors.doc_id=recipients.doc_id AND docs.id=authors.doc_id
                                            AND docs.year >= {} AND docs.year <= {};'''.format(
                       edge[0], edge[1], self.__config['start_year'], self.__config['end_year']),

                   '''SELECT docs.tid as tid, docs.year as year, authors.author as author, recipients.recipient as recipient
                                      FROM authors, recipients, docs
                                      WHERE authors.author LIKE "%{}%"  AND recipients.recipient LIKE "%{}%"
                                            AND authors.doc_id=recipients.doc_id AND docs.id=authors.doc_id
                                            AND docs.year >= {} AND docs.year <= {};'''.format(
                       edge[1], edge[0], self.__config['start_year'], self.__config['end_year']),
                ]

                for command in author_and_recipient_commands:
                    cur_full.execute(command)
                    for row in cur_full.fetchall():
                        weight = self.__get_tid_weight(row['tid'])
                        if edge[0] != self.main_name and edge[1] != self.main_name and weight < 0.1:
                            continue
                        document = get_ocr_by_tid(row['tid'], return_bytearray=False)
                        document_split = re.findall(WORD_SPLIT_REGEX, document)
                        text_sections = [document_split[i:i+SECTION_LENGTH] for i in range(0, len(document_split), SECTION_LENGTH)]
                        text_sections = [" ".join(text_section) for text_section in text_sections]

                        for tid_section_id, section in enumerate(text_sections):
                            cur_net.execute('''INSERT INTO {}_sections (section_id, node1, node2, tid, tid_section, weight, year, text)
                                    VALUES ({}, "{}", "{}", "{}", {}, {}, {}, "{}")'''.format(
                                      self.main_name, section_id, edge[0], edge[1], row['tid'], tid_section_id, weight, row['year'], section
                                    ))
                            section_id += 1

            con_net.commit()

    def __get_tid_weight(self, tid):

        db = Database('TOB_FULL')
        _, cur = db.connect()
        cur.execute('''select docs.tid, authors.author, recipients.recipient
                          FROM docs, authors, recipients
                          WHERE docs.tid="{}" AND docs.id=authors.doc_id AND authors.doc_id =recipients.doc_id;'''.format(tid))
        authors = set()
        recipients= set()
        for row in cur.fetchall():
            for author in english_name_regex.findall(row['author']):
                authors.add(author)
            for recipient in english_name_regex.findall(row['recipient']):
                recipients.add(recipient)
        weight = 1 / (len(authors) * len(recipients))
        # print(weight, authors, recipients)
        return weight


    def __get_dtms(self):

        '''
        returns dict of dtms
        {
            'count': count_dtm,
            'tf': tf_dtm,
            'tfidf': tfidf_dtm
        }


        :return:
        '''



        try:
            dtm_count = load_csc_matrix_from_file(PATH_TOKENIZED + 'networks/{}_dtm_count'.format(self.main_name))
            dtm_tf = load_csc_matrix_from_file(PATH_TOKENIZED + 'networks/{}_dtm_tf'.format(self.main_name))
            dtm_tfidf = load_csc_matrix_from_file(PATH_TOKENIZED + 'networks/{}_dtm_tfidf'.format(self.main_name))

        except IOError:
            print("DTM for {} not found. Creating now...".format(self.main_name))
            db = Database('TOB_NETWORKS')
            con, cur = db.connect()

            # vocabulary, _ = get_vocabulary_and_totals(self.main_name)
            # vocabulary = vocabulary['ordered']

            count_vectorizer = CountVectorizer(vocabulary=self.vocabulary['ordered'])

            cur.execute('SELECT text FROM {}_sections ORDER BY section_id ASC;'.format(self.main_name))
            docs = [row['text'] for row in cur.fetchall()]
            dtm = count_vectorizer.fit_transform(docs)
            print(dtm.shape, dtm.sum())

            dtm_count = dtm.tocsc()
            store_csr_matrix_to_file(dtm_count, PATH_TOKENIZED + 'networks/{}_dtm_count.npz'.format(self.main_name))

            tf_transformer = TfidfTransformer(use_idf=False)
            dtm_tf = tf_transformer.fit_transform(dtm_count.copy())
            store_csr_matrix_to_file(dtm_tf.tocsc(), PATH_TOKENIZED + 'networks/{}_dtm_tf.npz'.format(self.main_name))
            tfidf_transformer = TfidfTransformer(use_idf=True)
            dtm_tfidf = tfidf_transformer.fit_transform(dtm_count.copy())
            store_csr_matrix_to_file(dtm_tfidf.tocsc(), PATH_TOKENIZED + 'networks/{}_dtm_tfidf.npz'.format(self.main_name))


        dtm = {'count': dtm_count,
               'tf': dtm_tf,
               'tfidf': dtm_tfidf}
        return dtm



    def __get_node_positions(self):

        try:
            X = np.load('X.npy')
        except IOError:

            X = np.zeros((len(self.get_nodes()), len(self.vocabulary['ordered'])))
            print(X.shape)
            for idx, node in enumerate(self.get_nodes()):
                row = np.array(self.apply_filter(nodes=[node], vectorizer_type='tfidf').sum(axis=0)).flatten()
                print(node, row.shape, row.sum())
                row = row / row.sum()
                print(row.sum())
                X[idx] = row

            print(len(X.sum(axis=0)), len(X.sum(axis=1)))
            print(X.sum(axis=0))
            print(X.sum(axis=1))

            # if True:
            #     pca_model = PCA(n_components=2)
            #     pca_fitted = pca_model.fit_transform(X)
            #     plt.scatter(pca_fitted[:, 0], pca_fitted[:, 1])
            #     plt.title(self.main_name)
            #     for idx, n in enumerate(self.get_nodes()):
            #         plt.annotate(n, (pca_fitted[idx, 0], pca_fitted[idx, 1]))
            #     plt.show()

            # np.save('X.npy', X)

        tsne_model = TSNE(n_components=2, learning_rate=250)
        tsne_fitted = tsne_model.fit_transform(X)

        plt.scatter(tsne_fitted[:, 0], tsne_fitted[:, 1])
        for idx, n in enumerate(self.get_nodes()):
            plt.annotate(n, (tsne_fitted[idx, 0], tsne_fitted[idx, 1]))
        plt.title('learning rate: {}'.format(tsne_model.learning_rate))
        plt.show()
        return tsne_model

    def __get_distinctive_terms(self):

        try:
            distinctive_terms = pickle.load(open(PATH_TOKENIZED + 'networks/{}_distinctive_terms.pickle'.format(self.main_name), 'rb'))
        except IOError:

            distinctive_terms = {
                'overall': None,
                'nodes': {},
                'edges': {}
            }


            global_totals = get_vocabulary_totals(1)
            vocabulary_trie = load_vocabulary_trie(1)

            totals_legacy = np.zeros(len(self.vocabulary['ordered']))
            for token_id, token in enumerate(self.vocabulary['ordered']):
                token_id_global = vocabulary_trie[token]
                token_total_global = global_totals[token_id_global]
                totals_legacy[token_id] = token_total_global

            distinctive_terms['overall'] = self.__calculate_distinctive_terms(self.totals, totals_legacy)
            print("overall", distinctive_terms['overall'])
            names_set = set([n.lower() for name in self.nodes for n in name.split(',')])
            for term in distinctive_terms['overall']:
                if term[0] not in names_set:
                    print(term)

            for node in self.nodes:
                node_dtm = self.apply_filter(nodes=[node])
                node_counts = np.array(node_dtm.sum(axis=0)).flatten()
                distinctive_terms['nodes'][node] = self.__calculate_distinctive_terms(node_counts, self.totals)
                print("\n", node, distinctive_terms['nodes'][node])

            for edge in self.edges:
                edge_dtm = self.apply_filter(edges=[edge])
                # skip edges without content
                if edge_dtm.sum() == 0: continue
                edge_counts = np.array(edge_dtm.sum(axis=0)).flatten()
                distinctive_terms['edges'][edge] = self.__calculate_distinctive_terms(edge_counts, self.totals)
                print("\n", edge, distinctive_terms['edges'][edge])

            pickle.dump(distinctive_terms, open(PATH_TOKENIZED + 'networks/{}_distinctive_terms.pickle'.format(self.main_name), 'wb'))

        return distinctive_terms



    def __calculate_distinctive_terms(self, selection, remainder):

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
        # set of all parts of names to eliminate
        names_set = set([n.lower() for name in self.nodes for n in name.split(',')])
        for i in max_indices:
            term = self.vocabulary['id_to_token'][i]
            if not term in names_set and not term in STOP_WORDS_NETWORKS:
                distinctive_terms.append((term, selection[i], round(log_likelihoods[i], 2)))

        distinctive_terms = distinctive_terms[:20]


        return distinctive_terms


    def __get_edge_weights(self):
        '''
        Creates and returns a dataframe of edge weights for all nodes and years
        Also contains a column for 'all' for all years.

        Maybe there should be a scaling term for the individual years? I.e. their weights are lower,
        but not all close to zero?
        The idea would be to convey that overall connection strength changes over time.
        Maybe scale with year of highest overall letter exchange? Or year of highest overall exchange?

        :return:
        '''
        try:
            df = pd.read_pickle(PATH_TOKENIZED + 'networks/{}_edge_weights.pickle'.format(self.main_name))

        except IOError:
            print("Edge weights for {} not available. Creating now...".format(self.main_name))

            # Datframe of years as cols and nodes as rows
            df = pd.DataFrame(columns=['all'] + self.get_years(), index=self.edges)


            dtm_full_count = self.__get_dtms()['count']
            no_docs = dtm_full_count.shape[0]
            no_tokens = dtm_full_count.sum()
            print(no_docs, no_tokens)

            # fill 'all 'column'
            for edge in self.edges:
                dtm_edge = self.apply_filter(edges=[edge])
                no_docs_edge = np.count_nonzero(dtm_edge.sum(axis=1))
                no_tokens_edge = dtm_edge.sum()
                df['all'][edge] = (no_docs_edge/no_docs + no_tokens_edge/no_tokens)/2

            # fill years
            # for year in self.get_years():
            #     print(year)
            #     for edge in self.edges:
            #         dtm_edge = self.apply_filter(edges=[edge], years=[year])
            #         no_docs_edge = np.count_nonzero(dtm_edge.sum(axis=1))
            #         no_tokens_edge = dtm_edge.sum()
            #         df[year][edge] = (no_docs_edge/no_docs + no_tokens_edge/no_tokens)/2

            df.to_pickle(PATH_TOKENIZED + 'networks/{}_edge_weights.pickle'.format(self.main_name))

        return df



    def __get_filters(self):

        '''
        12/10/17
        Creates year, node, and edge filters.
        Each filter consists of a set of section ids that belong to the filter.

        e.g.
        filters['year'][1970] = {213, 215, ..., 544, 579}


        :return:
        '''

        try:
            filters = pickle.load(open(PATH_TOKENIZED + 'networks/{}_filters.pickle'.format(self.main_name), 'rb'))
        except IOError:

            print("Filters not available. Creating now...")

            filters = {
                'year': {},
                'edge': {},
                'node': {}
            }

            db = Database('TOB_NETWORKS')
            con, cur = db.connect()

            # Years
            cur.execute('SELECT DISTINCT(year) as years FROM {}_sections'.format(self.main_name))
            for year in [x['years'] for x in cur.fetchall()]:
                cur.execute('SELECT section_id FROM {}_sections WHERE year={}'.format(self.main_name, year))
                filters['year'][year] = set(row['section_id'] for row in cur.fetchall())

            # Edges
            for edge in self.edges:
                cur.execute('SELECT section_id FROM {}_sections WHERE node1 = "{}" AND node2 = "{}"'.format(
                    self.main_name, edge[0], edge[1]))
                filters['edge'][edge] = set(row['section_id'] for row in cur.fetchall())

            # Nodes
            for node in self.nodes:
                cur.execute('SELECT section_id FROM {}_sections WHERE node1 = "{}" OR node2 = "{}"'.format(
                    self.main_name, node, node))
                filters['node'][node] = set(row['section_id'] for row in cur.fetchall())


            # filters = {
            #     'year': self.__get_column_filters('year'),
            #     'node': self.__get_column_filters('node'),
            #     'main_author_is': self.__get_column_filters(self.main_name, 'main_author_is')
            #     'edge': self.__get_column_filters(self.main_name, 'edge')
            # }
            pickle.dump(filters, open(PATH_TOKENIZED + 'networks/{}_filters.pickle'.format(self.main_name), 'wb'))

        return filters





if __name__ == "__main__":
    n = Networks_Selector('dunn')
    n.get_edges(sorting='id', full_data=False)
    n.get_edges(sorting='id', full_data=True)
    # voc, _ = get_vocabulary_and_totals('dunn')
    #
    # import numpy as np
    #
    # print("total init", n.apply_filter(term='flavor').sum())
    #
    # years = n.get_years()
    # nodes = n.get_nodes()
    # year_node_matrix = np.zeros((len(years), len(nodes)), dtype=np.int32)
    # for year_id, year in enumerate(years):
    #     for node_id, node in enumerate(nodes):
    #
    #         val = n.apply_filter(years=[year], nodes=[node], term='flavor')
    #         print(type(val))
    #         print(val.sum())
    #         year_node_matrix[year_id, node_id] = val.sum()
    #
    # print("total final", year_node_matrix.sum())

#    for year in range(NETWORK_CONFIGS['dunn']['start_year'], NETWORK_CONFIGS['dunn']['end_year'] + 1):
#        for
