import pickle

from tobacco.configuration import PATH_TOKENIZED, YEAR_COUNT
from tobacco.frequencies_preprocessing.preprocessing_sections import get_doc_id_to_section_id_dict


# def get_year_doc_transformation_matrix(docs_or_sections='docs'):
#     """ Returns a csc matrix (M) used to turn a 12 million term count (t) vector into a len 116 year vector (y)
#
#     M * t = y
#
#     M -> One row for each year, One col for each document.
#     M(x,y) = 1 if doc y is from year x. 0 otherwise.
#
#     8/31/18: This was a nice way of doing it but I don't think the matrix is needed anymore because the simpler
#     transform_doc_to_year_array fulfills the same function less elegantly but faster.
#
#     :param docs_or_sections: 'docs' or 'sections'
#     :return:
#     """
#
#     try:
#         year_doc_matrix = load_csc_matrix_from_file(PATH_TOKENIZED + 'year_doc_matrix_{}'.format(docs_or_sections))
#         if not year_doc_matrix.dtype == np.uint8:
#             year_doc_matrix = csc_matrix(year_doc_matrix, dtype=np.uint8)
#
#         # print("year doc", docs_or_sections)
#         # print(type(year_doc_matrix), year_doc_matrix.dtype)
#         # print(year_doc_matrix.sum(), year_doc_matrix.nnz)
#
#     except IOError:
#
#
#         print("Year Doc Transformation matrix not available for {}. Creating now...".format(docs_or_sections))
#
#         doc_id_to_section_id_dict = get_doc_id_to_section_id_dict()
#
#         db = Database("TOB_FULL")
#         con, cur = db.connect()
#
#         n = DOC_COUNT
#         if docs_or_sections == 'sections':
#             n = SECTION_COUNT
#
#         year_doc_matrix = lil_matrix((YEAR_COUNT, n))
#
#         for year in range(YEAR_START, YEAR_END+1):
#             print(year)
#             cur.execute("SELECT MIN(id) FROM docs WHERE year = {}".format(year))
#             min_id = cur.fetchall()[0]['MIN(id)']
#             cur.execute("SELECT MAX(id) FROM docs WHERE year = {}".format(year))
#             max_id = cur.fetchall()[0]['MAX(id)']
#
#             row = year - YEAR_START
#
#             if docs_or_sections == 'sections':
#                 min_id = doc_id_to_section_id_dict[min_id][0]
#                 max_id = doc_id_to_section_id_dict[max_id][1]
#
#
#             year_doc_matrix[row, min_id: max_id + 1] = 1
#
#         year_doc_matrix = year_doc_matrix.tocsc()
#         year_doc_matrix = csc_matrix(year_doc_matrix, dtype=np.uint8)
#         print("year_doc_matrix has {} elements. Shape: {}.".format(year_doc_matrix.getnnz(), year_doc_matrix.shape))
#         store_csr_matrix_to_file(year_doc_matrix, PATH_TOKENIZED + 'year_doc_matrix_{}.npz'.format(docs_or_sections))
#
#         return year_doc_matrix

def get_year_doc_id_list(docs_or_sections: str) -> list:

    """ Returns a list, wherein every value marks the first doc_id belonging to that year.
    e.g. year_doc_id_list[1910] -> first id belonging to year 1910
    year_doc_id_list[2015] -> highest doc_id + 1

    >>> year_doc_id_list = get_year_doc_id_list('docs')
    >>> ids_1901 = year_doc_id_list[0]
    >>> print(f'First 1901 doc id: {ids_1901[0]}. Last 1901 doc id: {ids_1901[1]}.')
    First 1901 doc id: 0. Last 1901 doc id: 183.

    :param docs_or_sections: 'docs' or 'sections'
    :return: list
    """

    doc_id_to_section_id_dict = get_doc_id_to_section_id_dict()

    try:
        year_doc_id_list = pickle.load(open(PATH_TOKENIZED + 'year_{}_id_list.pickle'.format(docs_or_sections), 'rb'))

    except IOError:
        from tobacco.utilities.databases import Database

        print("Creating new year_{}_id_list".format(docs_or_sections))

        db = Database("TOB_FULL")
        con, cur = db.connect()

        year_doc_id_list = []

        for year in range(1901, 2017):
            cur.execute("SELECT MIN(id), MAX(id) FROM docs WHERE year = {}".format(year))
            row = cur.fetchall()[0]
            min_doc_id = row['MIN(id)']
            max_doc_id = row['MAX(id)']

            if docs_or_sections == 'docs':
                year_doc_id_list.append((min_doc_id, max_doc_id))
                print(year, min_doc_id)
            elif docs_or_sections == 'sections':
                min_section_id = doc_id_to_section_id_dict[min_doc_id][0]
                max_section_id = doc_id_to_section_id_dict[max_doc_id][1]
                year_doc_id_list.append((min_section_id, max_section_id))
                print(year, min_section_id, max_section_id)


        pickle.dump(year_doc_id_list, open(PATH_TOKENIZED + 'year_{}_id_list.pickle'.format(docs_or_sections), 'wb'))

    return year_doc_id_list


if __name__ == "__main__":
    #    get_year_doc_transformation_matrix()
    pass