
from tobacco.utilities.databases import Database
from tobacco.configuration import PATH_TOKENIZED, DOC_COUNT, SECTION_COUNT
from tobacco.frequencies_preprocessing.preprocessing_sections import get_doc_id_to_section_id_dict
from tobacco.utilities.sparse_matrices import store_csr_matrix_to_file, load_csc_matrix_from_file, load_csr_matrix_from_file

from scipy.sparse import lil_matrix, csc_matrix
import numpy as np

import pickle
import gzip


def get_doc_types_to_idx_dict():
    '''
    Creates a dict of doc_type to id as well as id to doc_type
    :return:
    '''


    try:
        doc_types_and_idx_dict = pickle.load(gzip.open(PATH_TOKENIZED + 'doc_types_and_idx_dict.pickle', 'rb'))

    except IOError:

        print("No doc_types_and_idx_dict available. Creating now...")

        db = Database("TOB_FULL")
        con, cur = db.connect()

        cur.execute("select doc_type, count(*) as hits from doc_types group by doc_type order by count(*) desc;")

        rows = cur.fetchall()

        doc_types_and_idx_dict = {}

        ignored_doc_types = get_dtype_dict()['ignored']


        idx = 0
        for row in rows:

            doc_type = row['doc_type']
            total = row['hits']

            if doc_type in ignored_doc_types or total < 100:
                continue
            else:
                doc_types_and_idx_dict[doc_type] = idx
                idx += 1

        doc_types_and_idx_dict.update({d[1]:d[0] for d in doc_types_and_idx_dict.items()})

        pickle.dump(doc_types_and_idx_dict, gzip.open(PATH_TOKENIZED + 'doc_types_and_idx_dict.pickle', 'wb'))


    return doc_types_and_idx_dict


def get_doc_types_doc_matrix(docs_or_sections='docs'):
    '''
    Creates a transformation matrix (doc_types x doc) M such that
     M = transformation matrix (doc_types x doc)
     t = term vector
     x = vector of doc_type counts

     M * t = x

     This function creates the matrix M for later use


     7/28/17: storing this as a csc matrix leads to a longer indptr but faster multplication times. Don't know why.

    :param doc_types_and_idx_dict:
    :return:
    '''

    try:
        doc_types_doc_matrix = load_csc_matrix_from_file(PATH_TOKENIZED + 'doc_types_doc_matrix_{}'.format(docs_or_sections))
        doc_types_doc_matrix = csc_matrix(doc_types_doc_matrix, dtype=np.uint8)

    except IOError:

        print("No doc_doc_types_matrix available for {}. Creating now...".format(docs_or_sections))

        db = Database("TOB_FULL")
        con, cur = db.connect()

        n = DOC_COUNT
        if docs_or_sections == 'sections':
            n = SECTION_COUNT

        doc_types_and_idx_dict = get_doc_types_to_idx_dict()
        m = int(len(doc_types_and_idx_dict)/2)

        doc_types_doc_matrix = lil_matrix((m, n), dtype=np.float)
        doc_id_to_section_id_dict = get_doc_id_to_section_id_dict()

        print(doc_types_doc_matrix.shape)

        cur.execute("SELECT doc_id, doc_type FROM doc_types;")

        while True:
            row = cur.fetchone()
            if not row:
                break

            doc_id = row['doc_id']
            doc_type = row['doc_type']

            try:
                doc_type_id = doc_types_and_idx_dict[doc_type]
            except KeyError:
                continue

            if doc_id % 100000 == 0: print(doc_id)

            if docs_or_sections == 'docs':
                doc_types_doc_matrix[doc_type_id, doc_id] = 1
            elif docs_or_sections == 'sections':
                first_id, final_id = doc_id_to_section_id_dict[doc_id]
                for section_id in range(first_id, final_id+1):
                    doc_types_doc_matrix[doc_type_id, section_id] = 1

        doc_types_doc_matrix = doc_types_doc_matrix.tocsc()
        doc_types_doc_matrix = csc_matrix(doc_types_doc_matrix, dtype=np.uint8)
        store_csr_matrix_to_file(doc_types_doc_matrix, PATH_TOKENIZED + 'doc_types_doc_matrix_{}.npz'.format(docs_or_sections))


    return doc_types_doc_matrix


# def get_doc_type_preprocessing():
#
#     try:
#         data = pickle.load(gzip.open(PATH_TOKENIZED + 'doc_types.pickle', 'rb'))
#         doc_types_and_idx_dict = data['doc_types_and_idx_dict']
#         doc_doc_types_matrix = data['doc_doc_types_matrix'].T
#     except IOError:
#
#         print("Preprocessed doc types not available. Creating now.")
#
#         doc_types_and_idx_dict = create_doc_types_to_idx_dict()
#         doc_doc_types_matrix = create_doc_types_doc_matrix(doc_types_and_idx_dict)
#
#         print(doc_doc_types_matrix.getnnz())
#
#         data= {'doc_types_and_idx_dict': doc_types_and_idx_dict,
#                'doc_doc_types_matrix': doc_doc_types_matrix}
#
#         pickle.dump(data, gzip.open(PATH_TOKENIZED + 'doc_types.pickle', 'wb'))
#
#
#     return doc_types_and_idx_dict, doc_doc_types_matrix







def create_dtype_strings():
    '''
    Prints out all document types that appear at least 100 times by number of appearances

    '''

    db = Database("TOB_FULL")
    con, cur = db.connect()
    cur.execute("select doc_type, count(*) as hits from doc_types group by doc_type order by count(*) desc;")
    rows = cur.fetchall()

    for row in rows:
       if row['hits'] >= 100:
           print(u'({:30s}, 0, {:8d}, []),'.format('"{}"'.format(row['doc_type']), row['hits']))




DOC_TYPES = [
    ("letter"                      , 0,  2947169, ['internal communication']),
    ("report"                      , 0,  2029698, ['others']),
    ("memo"                        , 0,  1574909, ['internal communication']),
    ("email"                       , 0,  1219047, ['internal communication']),
    ("chart"                       , 1,  1207004, []),
    ("graph"                       , 1,  1079091, []),
    ("map"                         , 1,   958536, []),
    ("table"                       , 1,   910716, ['others']),
    ("form"                        , 1,   838109, ['others']),
    ("note"                        , 0,   680030, ['internal communication']),
    ("graphics"                    , 1,   419459, []),
    ("publication"                 , 0,   304312, ['others']),
    ("report, scientific"          , 0,   297201, ['internal scientific reports']),
    ("advertisement"               , 0,   280972, ['marketing documents']),
    ("list"                        , 1,   265469, []),
    ("promotional material"        , 0,   243094, ['marketing documents']),
    ("budget"                      , 0,   242683, ['others']),
    ("specification"               , 0,   242256, []),
    ("budget review"               , 0,   217413, ['others']),
    ("telex"                       , 0,   184741, ['internal communication']),
    ("bibliography"                , 1,   183224, []),
    ("computer printout"           , 1,   171367, []),
    ("handwritten"                 , 1,   162198, []),
    ("news article"                , 0,   148712, ['news reports']),
    ("agenda"                      , 0,   133454, ['others']),
    ("report, market research"     , 0,   130564, ['marketing documents']),
    ("speech"                      , 0,   127378, ['others']),
    ("presentation"                , 0,   117438, ['others']),
    ("minutes"                     , 0,   115053, ['internal communication']),
    ("drawing"                     , 1,   114229, []),
    ("draft"                       , 0,   110258, ['others']),
    ("invoice"                     , 0,   108764, ['others']),
    ("notes"                       , 0,   103771, ['internal communication']),
    ("financial"                   , 0,    95131, ['others']),
    ("abstract"                    , 0,    95086, ['scientific publications']),
    ("questionnaire"               , 0,    94296, ['others']),
    ("science"                     , 0,    92939, ['internal scientific reports']),                          # unclear
    ("agreement resolution"        , 0,    91674, ['others']),
    ("letter, consumer"            , 0,    89854, ['others']),
    ("advertising copy"            , 0,    87063, ['marketing documents']),
    ("brand review"                , 0,    84565, ['others']),
    ("graphic"                     , 1,    84442, []),
    ("manual"                      , 0,    83434, ['others']),
    ("smoke/tobacco analysis"      , 0,    81784, ['internal scientific reports']),
    ("blank page"                  , 1,    81482, []),
    ("printout"                    , 1,    80290, []),
    ("file folder"                 , 1,    75013, []),
    ("catalog"                     , 0,    71388, ['others']),
    ("other"                       , 1,    70976, []),
    ("revision"                    , 0,    67377, ['others']),
    ("handbook"                    , 0,    65803, ['others']),
    ("photograph"                  , 1,    65629, []),
    ("pleading"                    , 0,    62199, ['others']),
    ("email attachment"            , 0,    61010, ['others']),
    ("study"                       , 0,    56966, ['others']),
    ("loose email attachment"      , 0,    56018, ['others']),                          # does not seem to be has-a
    ("scientific article"          , 0,    55774, ['scientific publications']),
    ("contract"                    , 0,    55067, ['others']),
    ("pay request"                 , 0,    54657, ['others']),
    ("pamphlet"                    , 0,    54541, ['others']),
    ("publication, scientific"     , 0,    53716, ['scientific publications']),
    ("transcript"                  , 0,    53519, ['others']),
    ("tab page"                    , 1,    52958, []),
    ("press release"               , 0,    52860, ['others']),                          # maybe news?
    ("brand plan"                  , 0,    47676, ['others']),
    ("meeting materials"           , 0,    46419, ['others']),
    ("newsletter"                  , 0,    46060, ['others']),                          # unclear
    ("outline"                     , 0,    45907, ['others']),
    ("patent"                      , 0,    45788, ['others']),
    ("article"                     , 0,    44433, ['others']),                          # unclear, seems BAT only
    ("periodical"                  , 0,    41223, ['news reports']),
    ("report, marketing"           , 0,    37300, ['marketing documents']),
    ("footnote"                    , 1,    37189, []),
    ("resume"                      , 0,    36519, ['others']),
    ("tab sheet"                   , 1,    34470, []),
    ("filled in form"              , 1,    34400, []),
    ("proposal"                    , 0,    32828, ['others']),
    ("diagram"                     , 1,    32626, []),
    ("file"                        , 0,    32235, ['others']),
    ("report, formal r&d"          , 0,    31588, ['internal scientific reports']),
    ("transaction document"        , 0,    30740, ['others']),
    ("envelope"                    , 1,    30211, []),
    ("patent application"          , 0,    29347, ['others']),
    ("contract agreement"          , 0,    29036, ['others']),
    ("regulation"                  , 0,    28663, ['others']),
    ("newspaper article"           , 0,    26807, ['news reports']),
    ("routing slip"                , 0,    26544, ['others']),
    ("agreement"                   , 0,    26100, ['others']),
    ("legal"                       , 0,    25819, ['court documents']),
    ("lab protocol"                , 0,    25718, ['internal scientific reports']),
    ("marketing document"          , 0,    24962, ['marketing documents']),
    ("media transcript"            , 0,    24896, ['news reports']),
    ("media article"               , 0,    24880, ['news reports']),
    ("loose email"                 , 0,    23312, ['internal communication']),
    ("magazine article"            , 0,    23301, ['news reports']),
    ("raw data"                    , 0,    23135, ['others']),
    ("personnel information"       , 0,    22751, ['others']),
    ("log"                         , 0,    22731, ['others']),
    ("lrd abstract"                , 0,    21598, ['others']),
    ("organizational chart"        , 1,    21289, []),
    ("deposition"                  , 0,    21187, ['court documents']),
    ("delivery slip"               , 0,    20371, ['others']),
    ("media plan"                  , 0,    19992, ['others']),
    ("research proposal, scientific", 0,    19640, ['internal scientific reports']),                         # unclear
    ("testimony"                   , 0,    19496, ['court documents']),
    ("website, internet"           , 0,    18223, ['others']),
    ("script"                      , 0,    18133, ['others']),
    ("marketing research"          , 0,    17936, ['marketing documents']),
    ("scientific research"         , 0,    17685, ['internal scientific reports']),
    ("consumer response"           , 0,    17327, ['others']),                          # unclear
    ("law"                         , 0,    17172, ['others']),
    ("legal document"              , 0,    16737, ['court documents']),                          # unclear
    ("scientific publication"      , 0,    15516, ['scientific publications']),
    ("trial transcript"            , 0,    14684, ['court documents']),
    ("lab notebook"                , 0,    13607, ['others']),
    ("financial document"          , 0,    13557, ['others']),
    ("application, grant"          , 0,    13399, ['internal scientific reports']),
    ("shipping document"           , 0,    12414, ['others']),
    ("telephone record"            , 0,    12238, ['others']),
    ("deposition exhibit"          , 0,    12220, ['court documents']),
    ("summary"                     , 0,    11614, ['others']),
    ("legislation"                 , 0,    11610, ['others']),
    ("interoffice memo"            , 0,    11051, ['internal communication']),
    ("cigarette packages"          , 0,    10473, ['others']),
    ("attachment"                  , 0,     9911, ['others']),
    ("market research proposal"    , 0,     9906, ['marketing documents']),
    ("fax"                         , 0,     9635, ['internal communication']),
    ("excerpt"                     , 0,     9593, ['others']),
    ("news clipping"               , 0,     9590, ['news reports']),
    ("report, trip"                , 0,     9439, ['others']),
    ("internet"                    , 0,     9285, ['others']),
    ("cartons"                     , 0,     9099, ['others']),
    ("protocol"                    , 0,     9057, ['others']),
    ("newspaper"                   , 0,     9028, ['news reports']),
    ("scientific research proposal", 0,     9022, ['internal scientific reports']),
    ("pack"                        , 0,     8955, ['others']),
    ("survey questionnaire"        , 0,     8909, ['others']),
    ("response, consumer"          , 0,     8083, ['others']),                          # unclear
    ("legal brief"                 , 0,     7941, ['court documents']),
    ("binder cover"                , 1,     7916, []),
    ("report, r&d"                 , 0,     7229, ['internal scientific reports']),
    ("video"                       , 1,     7048, []),
    ("report, financial"           , 0,     7013, ['others']),
    ("calendar"                    , 0,     6889, ['others']),
    ("consumer letter"             , 0,     6643, ['others']),
    ("electronic file"             , 0,     6585, ['others']),
    ("receipt"                     , 1,     6564, ['others']),
    ("statistics"                  , 0,     6541, ['others']),
    ("statement"                   , 0,     6106, ['others']),
    ("survey"                      , 0,     6054, ['others']),
    ("procedures manual"           , 0,     6052, ['others']),
    ("preprinted form"             , 0,     5885, ['others']),
    ("collage"                     , 1,     5852, []),
    ("exhibit"                     , 0,     5758, ['court documents']),
    ("trial list"                  , 1,     5622, []),                          # this seems to contain trial exhibits?
    ("marketing plan"              , 0,     5587, ['marketing documents']),
    ("diary"                       , 0,     5521, ['others']),
    ("meeting notes"               , 0,     5368, ['internal communication']),
    ("records transfer list"       , 0,     5178, ['others']),
    ("notice"                      , 0,     5156, ['others']),
    ("scientific review"           , 0,     4973, ['internal scientific reports']),                              # contains also internal materials
    ("record transfer"             , 0,     4930, ['others']),
    ("flow chart"                  , 1,     4498, []),
    ("packaging"                   , 0,     4400, ['others']),
    ("work request"                , 0,     4093, ['others']),
    ("resolution"                  , 0,     4048, ['others']),
    ("inventory"                   , 0,     3980, ['others']),
    ("cover sheet"                 , 1,     3953, []),
    ("file folder label"           , 1,     3948, []),
    ("review"                      , 0,     3816, ['others']),
    ("report, progress"            , 0,     3811, ['others']),
    ("employee record"             , 0,     3795, ['others']),
    ("guidelines"                  , 0,     3769, ['others']),
    ("report, accession"           , 0,     3700, ['others']),
    ("procedures"                  , 0,     3664, ['others']),
    ("instructions"                , 0,     3663, ['others']),
    ("trial testimony"             , 0,     3369, ['court documents']),
    ("technical & scientific journal", 0,     3322, ['scientific publications']),
    ("datebook"                    , 0,     3030, ['others']),
    ("label"                       , 1,     3011, []),
    ("document dividers"           , 1,     2921, []),
    ("report, expense"             , 0,     2813, ['others']),
    ("patent, full"                , 0,     2765, ['others']),
    ("title page"                  , 1,     2629, []),
    ("deposition use"              , 0,     2448, ['court documents']),                          # seems to be the same as deposition exhibit
    ("email, draft"                , 0,     2281, ['internal communication']),
    ("direct mail"                 , 0,     2281, ['others']),
    ("meeting"                     , 0,     2206, ['others']),
    ("transmittal"                 , 0,     2139, ['others']),
    ("cartoon"                     , 1,     2071, []),
    ("grant folder"                , 0,     2065, ['others']),
    ("testimony, congressional"    , 0,     2059, ['others']),
    ("affidavit"                   , 0,     2054, ['others']),
    ("radio & television commercials", 0,     2039, ['marketing documents']),
    ("report, quarterly research"  , 0,     2026, ['others']),
    ("motion"                      , 0,     2017, ['court documents']),
    ("memo, draft"                 , 0,     1997, ['internal communication']),
    ("corporate"                   , 0,     1961, ['others']),
    ("catalog card"                , 0,     1928, ['others']),
    ("news letter"                 , 0,     1892, ['others']),                          # unclear
    ("grantee list"                , 0,     1794, ['others']),
    ("redbook"                     , 0,     1771, ['others']),
    ("profile"                     , 0,     1759, ['others']),
    ("transaction"                 , 0,     1744, ['others']),
    ("calculations"                , 1,     1707, []),
    ("rdr"                         , 0,     1707, ['others']),
    ("file sheet"                  , 1,     1687, []),
    ("form, blank"                 , 1,     1618, []),
    ("letter, draft"               , 0,     1609, ['others']),
    ("quotation"                   , 1,     1607, []),
    ("include"                     , 1,     1582, []),
    ("electronic attachment"       , 0,     1555, ['others']),
    ("press"                       , 0,     1499, ['others']),                          # unclear
    ("fax cover sheet"             , 1,     1498, []),
    ("consumer response letter"    , 0,     1493, ['others']),
    ("patent abstract"             , 0,     1487, ['others']),
    ("report, draft"               , 0,     1487, ['others']),
    ("market research report"      , 0,     1484, ['others']),
    ("appendix"                    , 1,     1473, []),
    ("report, annual"              , 0,     1467, ['others']),
    ("meeting minutes"             , 0,     1409, ['internal communication']),
    ("shipping/receiving"          , 1,     1404, []),
    ("brochure"                    , 0,     1400, ['others']),
    ("warning label"               , 0,     1291, ['others']),
    ("chemical information"        , 1,     1264, []),
    ("nested attachment"           , 0,     1215, ['others']),
    ("memo, conference"            , 0,     1210, ['others']),
    ("website"                     , 0,     1198, ['others']),
    ("legal language"              , 0,     1166, ['others']),                          # seems to contain contracts and other docs with legal language ???
    ("report, detailed billing"    , 0,     1113, ['others']),
    ("article, journal"            , 0,     1108, ['scientific publications']),                 # contains a few newspaper articles
    ("trial exhibit"               , 0,     1069, ['court documents']),
    ("cross reference sheet"       , 1,     1055, []),
    ("government publication"      , 0,     1016, ['others']),
    ("purchase order"              , 0,      989, []),
    ("research proposal"           , 0,      942, []),
    ("trade publication"           , 0,      859, []),
    ("slides"                      , 0,      846, []),
    ("agreement, draft"            , 0,      815, []),
    ("telegram"                    , 0,      813, []),
    ("cim"                         , 0,      791, []),
    ("picture"                     , 1,      790, []),
    ("application"                 , 0,      788, []),
    ("news release"                , 0,      747, []),
    ("financial statement"         , 0,      743, []),
    ("confidential"                , 0,      707, []),
    ("audio"                       , 1,      699, []),
    ("book"                        , 0,      695, []),
    ("blank form"                  , 0,      693, []),
    ("check"                       , 0,      667, []),
    ("computer disk"               , 0,      660, []),
    ("proceedings, conference"     , 0,      657, []),
    ("court document, draft"       , 0,      634, []),
    ("contract agreement resolution", 0,      627, []),
    ("purchase"                    , 0,      618, []),
    ("report, call"                , 0,      593, []),
    ("report, contact"             , 0,      592, []),
    ("interview"                   , 0,      576, []),
    ("loose e-mail attachment"     , 0,      541, []),
    ("trial exhibit, defendant"    , 0,      533, []),
    ("presentation materials"      , 0,      499, []),
    ("compilation"                 , 1,      476, []),
    ("summaries of publications"   , 0,      475, []),
    ("order"                       , 0,      466, []),
    ("graphics, draft"             , 0,      450, []),
    ("comments"                    , 0,      432, []),
    ("technical bulletin"          , 0,      394, []),
    ("shipping record"             , 0,      394, []),
    ("contract, draft"             , 0,      391, []),
    ("minutes, meeting"            , 0,      387, []),
    ("r&d report"                  , 0,      385, []),
    ("material safety data sheet"  , 0,      377, []),
    ("pleading, draft"             , 0,      376, []),
    ("formal legal document"       , 0,      375, []),
    ("expense report"              , 0,      343, []),
    ("testimony, osha"             , 0,      338, []),
    ("certificate"                 , 0,      333, []),
    ("ftcd, ftc submitted"         , 0,      324, []),
    ("technical specifications"    , 0,      324, []),
    ("media"                       , 0,      323, []),
    ("executive overview"          , 0,      320, []),
    ("tobacco institute"           , 0,      292, []),
    ("editorial"                   , 0,      286, []),
    ("attachment, embedded"        , 0,      283, []),
    ("report, litigation"          , 0,      279, []),
    ("exclude"                     , 0,      278, []),
    ("personnel"                   , 0,      276, []),
    ("policy"                      , 0,      276, []),
    ("file folder begin"           , 1,      274, []),
    ("personnel file"              , 0,      269, []),
    ("promotional item"            , 0,      268, []),
    ("procedural document"         , 0,      266, []),
    ("protocol, draft"             , 0,      258, []),
    ("chronology"                  , 0,      252, []),
    ("notebook"                    , 0,      250, []),
    ("testimony, legislative"      , 0,      242, []),
    ("regulatory"                  , 0,      234, []),
    ("legal memo"                  , 0,      234, []),
    ("statement, draft"            , 0,      231, []),
    ("index"                       , 1,      229, []),
    ("poster"                      , 0,      210, []),
    ("scholar award"               , 0,      207, []),
    ("brief"                       , 0,      200, []),
    ("assignment"                  , 0,      192, []),
    ("project summary"             , 0,      192, []),
    ("submission"                  , 0,      189, []),
    ("audiovisual"                 , 0,      187, []),
    ("response"                    , 0,      181, []),
    ("program planning"            , 0,      180, []),
    ("legal, other"                , 1,      178, []),
    ("memo to file"                , 0,      178, []),
    ("redacted"                    , 0,      174, []),
    ("rep"                         , 0,      166, []),
    ("package"                     , 0,      166, []),
    ("trial use"                   , 0,      163, []),
    ("trial exhibit, plaintiff"    , 0,      159, []),
    ("report, expert"              , 0,      156, []),
    ("oral testimony"              , 0,      155, []),
    ("bulletin"                    , 0,      154, []),
    ("subpoena"                    , 0,      151, []),
    ("insurance"                   , 0,      149, []),
    ("art-s"                       , 0,      148, []),
    ("offer"                       , 0,      146, []),
    ("research description"        , 0,      145, []),
    ("bid"                         , 0,      143, []),
    ("telephone report"            , 0,      136, []),
    ("interrogatory"               , 0,      136, []),
    ("internal"                    , 0,      127, []),
    ("court order"                 , 0,      125, []),
    ("business card"               , 0,      123, []),
    ("legal document, draft"       , 0,      123, []),
    ("medical"                     , 0,      122, []),
    ("telephone message"           , 0,      121, []),
    ("position statement"          , 0,      119, []),
    ("letter to the editor"        , 0,      118, []),
    ("minutes of patent meeting"   , 0,      117, []),
    ("departmental stamp/received stamp", 0,      115, []),
    ("registration form"           , 0,      115, []),
    ("zip files"                   , 0,      115, []),
    ("closing statement"           , 0,      113, []),
    ("litigation"                  , 0,      112, []),
    ("accounting record"           , 0,      103, []),
    ("report, research"            , 0,      100, []),
]


def get_dtype_dict():

    dtype_dict = {'valid': set(),
                  'ignored': set(),
                  'groups':{
                      'court documents': [],
                      'scientific publications': [],
                      'internal scientific reports': [],
                      'news reports': [],
                      'marketing documents': [],
                      'internal communication': [],
                      'others': []
                  }}

    for i in DOC_TYPES:
        if i[1] == 0:
            dtype_dict['valid'].add(i[0])
            for group in i[3]:
                dtype_dict['groups'][group].append(i[0])
        else:
            dtype_dict['ignored'].add(i[0])


    return dtype_dict



if __name__ == "__main__":
    #d =  get_dtype_dict()
    #for group in d['groups']:
    #    print(group, d['groups'][group])

#    get_doc_type_preprocessing()
#    get_doc_types_doc_matrix()
    get_doc_types_doc_matrix(docs_or_sections='docs')



'''








+--------------------------------+---------+
| doc_type                       | hits    |
+--------------------------------+---------+
| report, marketing              |   37300 |
| footnote                       |   37189 |
| resume                         |   36519 |
| tab sheet                      |   34470 |
| filled in form                 |   34400 |
| proposal                       |   32828 |
| diagram                        |   32626 |
| file                           |   32235 |
| report, formal r&d             |   31588 |
| transaction document           |   30740 |
| envelope                       |   30211 |
| patent application             |   29347 |
| contract agreement             |   29036 |
| regulation                     |   28663 |
| newspaper article              |   26807 |
| routing slip                   |   26544 |
| agreement                      |   26100 |
| legal                          |   25819 |
| lab protocol                   |   25718 |
| marketing document             |   24962 |
| media transcript               |   24896 |
| media article                  |   24880 |
| loose email                    |   23312 |
| magazine article               |   23301 |
| raw data                       |   23135 |
| personnel information          |   22751 |
| log                            |   22731 |
| lrd abstract                   |   21598 |
| organizational chart           |   21289 |
| deposition                     |   21187 |
| delivery slip                  |   20371 |
| media plan                     |   19992 |
| research proposal, scientific  |   19640 |
| testimony                      |   19496 |
| website, internet              |   18223 |
| script                         |   18133 |
| marketing research             |   17936 |
| scientific research            |   17685 |
| consumer response              |   17327 |
| law                            |   17172 |
| legal document                 |   16737 |
| scientific publication         |   15516 |
| trial transcript               |   14684 |
| lab notebook                   |   13607 |
| financial document             |   13557 |
| application, grant             |   13399 |
| shipping document              |   12414 |
| telephone record               |   12238 |
| deposition exhibit             |   12220 |
| summary                        |   11614 |
| legislation                    |   11610 |
| interoffice memo               |   11051 |
| cigarette packages             |   10473 |
| attachment                     |    9911 |
| market research proposal       |    9906 |
| fax                            |    9635 |
| excerpt                        |    9593 |
| news clipping                  |    9590 |
| report, trip                   |    9439 |
| internet                       |    9285 |
| cartons                        |    9099 |
| protocol                       |    9057 |
| newspaper                      |    9028 |
| scientific research proposal   |    9022 |
| pack                           |    8955 |
| survey questionnaire           |    8909 |
| response, consumer             |    8083 |
| legal brief                    |    7941 |
| binder cover                   |    7916 |
| report, r&d                    |    7229 |
| video                          |    7048 |
| report, financial              |    7013 |
| calendar                       |    6889 |
| consumer letter                |    6643 |
| electronic file                |    6585 |
| receipt                        |    6564 |
| statistics                     |    6541 |
| statement                      |    6106 |
| survey                         |    6054 |
| procedures manual              |    6052 |
| preprinted form                |    5885 |
| collage                        |    5852 |
| exhibit                        |    5758 |
| trial list                     |    5622 |
| marketing plan                 |    5587 |
| diary                          |    5521 |
| meeting notes                  |    5368 |
| records transfer list          |    5178 |
| notice                         |    5156 |
| scientific review              |    4973 |
| record transfer                |    4930 |
| flow chart                     |    4498 |
| packaging                      |    4400 |
| work request                   |    4093 |
| resolution                     |    4048 |
| inventory                      |    3980 |
| cover sheet                    |    3953 |
| file folder label              |    3948 |
| review                         |    3816 |
| report, progress               |    3811 |
| employee record                |    3795 |
| guidelines                     |    3769 |
| report, accession              |    3700 |
| procedures                     |    3664 |
| instructions                   |    3663 |
| trial testimony                |    3369 |
| technical & scientific journal |    3322 |
| datebook                       |    3030 |
| label                          |    3011 |
| document dividers              |    2921 |
| report, expense                |    2813 |
| patent, full                   |    2765 |
| title page                     |    2629 |
| deposition use                 |    2448 |
| email, draft                   |    2281 |
| direct mail                    |    2281 |
| meeting                        |    2206 |
| transmittal                    |    2139 |
| cartoon                        |    2071 |
| grant folder                   |    2065 |
| testimony, congressional       |    2059 |
| affidavit                      |    2054 |
| radio & television commercials |    2039 |
| report, quarterly research     |    2026 |
| motion                         |    2017 |
| memo, draft                    |    1997 |
| corporate                      |    1961 |
| catalog card                   |    1928 |
| news letter                    |    1892 |
| grantee list                   |    1794 |
| redbook                        |    1771 |
| profile                        |    1759 |
| transaction                    |    1744 |
| calculations                   |    1707 |
| rdr                            |    1707 |
| file sheet                     |    1687 |
| form, blank                    |    1618 |
| letter, draft                  |    1609 |
| quotation                      |    1607 |
| include                        |    1582 |
| electronic attachment          |    1555 |
| press                          |    1499 |
| fax cover sheet                |    1498 |
| consumer response letter       |    1493 |
| patent abstract                |    1487 |
| report, draft                  |    1487 |
| market research report         |    1484 |
| appendix                       |    1473 |
| report, annual                 |    1467 |
| meeting minutes                |    1409 |
| shipping/receiving             |    1404 |
| brochure                       |    1400 |
| warning label                  |    1291 |
| chemical information           |    1264 |
| nested attachment              |    1215 |
| memo, conference               |    1210 |
| website                        |    1198 |
| legal language                 |    1166 |
| report, detailed billing       |    1113 |
| article, journal               |    1108 |
| trial exhibit                  |    1069 |
| cross reference sheet          |    1055 |
| government publication         |    1016 |

+--------------------------------+---------+
| doc_type                       | hits    |
+--------------------------------+---------+
| letter                         | 2947169 |
| report                         | 2029698 |
| memo                           | 1574909 |
| email                          | 1219047 |
| chart                          | 1207004 |
| graph                          | 1079091 |
| map                            |  958536 |
| table                          |  910716 |
| form                           |  838109 |
| note                           |  680030 |
| graphics                       |  419459 |
| publication                    |  304312 |
| report, scientific             |  297201 |
| advertisement                  |  280972 |
| list                           |  265469 |
| promotional material           |  243094 |
| budget                         |  242683 |
| specification                  |  242256 |
| budget review                  |  217413 |
| telex                          |  184741 |
| bibliography                   |  183224 |
| computer printout              |  171367 |
| handwritten                    |  162198 |
| news article                   |  148712 |
| agenda                         |  133454 |
| report, market research        |  130564 |
| speech                         |  127378 |
| presentation                   |  117438 |
| minutes                        |  115053 |
| drawing                        |  114229 |
| draft                          |  110258 |
| invoice                        |  108764 |
| notes                          |  103771 |
| financial                      |   95131 |
| abstract                       |   95086 |
| questionnaire                  |   94296 |
| science                        |   92939 |
| agreement resolution           |   91674 |
| letter, consumer               |   89854 |
| advertising copy               |   87063 |
| brand review                   |   84565 |
| graphic                        |   84442 |
| manual                         |   83434 |
| smoke/tobacco analysis         |   81784 |
| blank page                     |   81482 |
| printout                       |   80290 |
| file folder                    |   75013 |
| catalog                        |   71388 |
| other                          |   70976 |
| revision                       |   67377 |
| handbook                       |   65803 |
| photograph                     |   65629 |
| pleading                       |   62199 |
| email attachment               |   61010 |
| study                          |   56966 |
| loose email attachment         |   56018 |
| scientific article             |   55774 |
| contract                       |   55067 |
| pay request                    |   54657 |
| pamphlet                       |   54541 |
| publication, scientific        |   53716 |
| transcript                     |   53519 |
| tab page                       |   52958 |
| press release                  |   52860 |
| brand plan                     |   47676 |
| meeting materials              |   46419 |
| newsletter                     |   46060 |
| outline                        |   45907 |
| patent                         |   45788 |
| article                        |   44433 |
| periodical                     |   41223 |
| report, marketing              |   37300 |
| footnote                       |   37189 |
| resume                         |   36519 |
| tab sheet                      |   34470 |
| filled in form                 |   34400 |
| proposal                       |   32828 |
| diagram                        |   32626 |
| file                           |   32235 |
| report, formal r&d             |   31588 |
| transaction document           |   30740 |
| envelope                       |   30211 |
| patent application             |   29347 |
| contract agreement             |   29036 |
| regulation                     |   28663 |
| newspaper article              |   26807 |
| routing slip                   |   26544 |
| agreement                      |   26100 |
| legal                          |   25819 |
| lab protocol                   |   25718 |
| marketing document             |   24962 |
| media transcript               |   24896 |
| media article                  |   24880 |
| loose email                    |   23312 |
| magazine article               |   23301 |
| raw data                       |   23135 |
| personnel information          |   22751 |
| log                            |   22731 |
| lrd abstract                   |   21598 |
| organizational chart           |   21289 |
| deposition                     |   21187 |
| delivery slip                  |   20371 |
| media plan                     |   19992 |
| research proposal, scientific  |   19640 |
| testimony                      |   19496 |
| website, internet              |   18223 |
| script                         |   18133 |
| marketing research             |   17936 |
| scientific research            |   17685 |
| consumer response              |   17327 |
| law                            |   17172 |
| legal document                 |   16737 |
| scientific publication         |   15516 |
| trial transcript               |   14684 |
| lab notebook                   |   13607 |
| financial document             |   13557 |
| application, grant             |   13399 |
| shipping document              |   12414 |
| telephone record               |   12238 |
| deposition exhibit             |   12220 |
| summary                        |   11614 |
| legislation                    |   11610 |
| interoffice memo               |   11051 |
| cigarette packages             |   10473 |
| attachment                     |    9911 |
| market research proposal       |    9906 |
| fax                            |    9635 |
| excerpt                        |    9593 |
| news clipping                  |    9590 |
| report, trip                   |    9439 |
| internet                       |    9285 |
| cartons                        |    9099 |
| protocol                       |    9057 |
| newspaper                      |    9028 |
| scientific research proposal   |    9022 |
| pack                           |    8955 |
| survey questionnaire           |    8909 |
| response, consumer             |    8083 |
| legal brief                    |    7941 |
| binder cover                   |    7916 |
| report, r&d                    |    7229 |
| video                          |    7048 |
| report, financial              |    7013 |
| calendar                       |    6889 |
| consumer letter                |    6643 |
| electronic file                |    6585 |
| receipt                        |    6564 |
| statistics                     |    6541 |
| statement                      |    6106 |
| survey                         |    6054 |
| procedures manual              |    6052 |
| preprinted form                |    5885 |
| collage                        |    5852 |
| exhibit                        |    5758 |
| trial list                     |    5622 |
| marketing plan                 |    5587 |
| diary                          |    5521 |
| meeting notes                  |    5368 |
| records transfer list          |    5178 |
| notice                         |    5156 |
| scientific review              |    4973 |
| record transfer                |    4930 |
| flow chart                     |    4498 |
| packaging                      |    4400 |
| work request                   |    4093 |
| resolution                     |    4048 |
| inventory                      |    3980 |
| cover sheet                    |    3953 |
| file folder label              |    3948 |
| review                         |    3816 |
| report, progress               |    3811 |
| employee record                |    3795 |
| guidelines                     |    3769 |
| report, accession              |    3700 |
| procedures                     |    3664 |
| instructions                   |    3663 |
| trial testimony                |    3369 |
| technical & scientific journal |    3322 |
| datebook                       |    3030 |
| label                          |    3011 |
| document dividers              |    2921 |
| report, expense                |    2813 |
| patent, full                   |    2765 |
| title page                     |    2629 |
| deposition use                 |    2448 |
| email, draft                   |    2281 |
| direct mail                    |    2281 |
| meeting                        |    2206 |
| transmittal                    |    2139 |
| cartoon                        |    2071 |
| grant folder                   |    2065 |
| testimony, congressional       |    2059 |
| affidavit                      |    2054 |
| radio & television commercials |    2039 |
| report, quarterly research     |    2026 |
| motion                         |    2017 |
| memo, draft                    |    1997 |
| corporate                      |    1961 |
| catalog card                   |    1928 |
| news letter                    |    1892 |
| grantee list                   |    1794 |
| redbook                        |    1771 |
| profile                        |    1759 |
| transaction                    |    1744 |
| calculations                   |    1707 |
| rdr                            |    1707 |
| file sheet                     |    1687 |
| form, blank                    |    1618 |
| letter, draft                  |    1609 |
| quotation                      |    1607 |
| include                        |    1582 |
| electronic attachment          |    1555 |
| press                          |    1499 |
| fax cover sheet                |    1498 |
| consumer response letter       |    1493 |
| patent abstract                |    1487 |
| report, draft                  |    1487 |
| market research report         |    1484 |
| appendix                       |    1473 |
| report, annual                 |    1467 |
| meeting minutes                |    1409 |
| shipping/receiving             |    1404 |
| brochure                       |    1400 |
| warning label                  |    1291 |
| chemical information           |    1264 |
| nested attachment              |    1215 |
| memo, conference               |    1210 |
| website                        |    1198 |
| legal language                 |    1166 |
| report, detailed billing       |    1113 |
| article, journal               |    1108 |
| trial exhibit                  |    1069 |
| cross reference sheet          |    1055 |
| government publication         |    1016 |

'''
