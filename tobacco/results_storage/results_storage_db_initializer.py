from tobacco.configuration import RESULTS_DB_NAME
from tobacco.utilities.databases import Database


def initialize_database():
    """Initializes all of the required tables for the results database.

    These are:
    - results_frequencies               (Stores the results for frequency queries)
    - results_passages                  (Stores mostly the metadata (topic model etc) found for one query)
    - results_passages_yearly           (Stores the text passages found for one year)
    - results_passages_yearly_default   (Stores yearly results for queries that use the standard settings)
                                        ??8/31/18 Why??

    :return:
    """

    db = Database(RESULTS_DB_NAME)
    con, cur = db.connect()

    cur.execute('''CREATE TABLE IF NOT EXISTS results_frequencies(
                      tokens                varchar(255)  NOT NULL,
                      doc_type_filters      varchar(255)  NOT NULL,
                      collection_filters    varchar(100)  NOT NULL,
                      availability_filters  varchar(100)  NOT NULL,
                      term_filters          varchar(100)  NOT NULL,

                      query_hash            varchar(32)    NOT NULL,

                      results               longblob      NOT NULL,
                      last_accessed         DATE          NOT NULL,
                      count_accessed        INT           NOT NULL,

                      UNIQUE INDEX results_frequencies_idx (query_hash)
                      )
                      ROW_FORMAT=COMPRESSED



    ''')

    cur.execute('''CREATE TABLE IF NOT EXISTS results_passages(
                      tokens                varchar(255)  NOT NULL,
                      doc_type_filters      varchar(255)  NOT NULL,
                      collection_filters    varchar(100)  NOT NULL,
                      availability_filters  varchar(100)  NOT NULL,

                      start_year            INT           NOT NULL,
                      end_year              INT           NOT NULL,
                      passage_length        INT           NOT NULL,
                      min_readability       FLOAT         NOT NULL,
                      passages_per_year     INT           NOT NULL,

                      query_hash            varchar(32)    NOT NULL,

                      results               longblob      NOT NULL,

                      last_accessed         DATE          NOT NULL,
                      count_accessed        INT           NOT NULL,

                      UNIQUE INDEX results_passages_idx (query_hash)
                      )
                      ROW_FORMAT=COMPRESSED



    ''')


    cur.execute('''CREATE TABLE IF NOT EXISTS results_passages_yearly(
                      tokens                varchar(255)  NOT NULL,
                      doc_type_filters      varchar(255)  NOT NULL,
                      collection_filters    varchar(100)  NOT NULL,
                      availability_filters  varchar(100)  NOT NULL,

                      year                  INT           NOT NULL,
                      passage_length        INT           NOT NULL,
                      complete              INT           NOT NULL,

                      query_hash            varchar(32)    NOT NULL,

                      results               longblob      NOT NULL,

                      UNIQUE INDEX results_passages_idx (query_hash, year)
                      )

                      ROW_FORMAT=COMPRESSED
    ''')


    cur.execute('''CREATE TABLE IF NOT EXISTS results_passages_yearly_default(
                      tokens                varchar(255)  NOT NULL,
                      doc_type_filters      varchar(255)  NOT NULL,
                      collection_filters    varchar(100)  NOT NULL,
                      availability_filters  varchar(100)  NOT NULL,

                      year                  INT           NOT NULL,
                      query_hash            varchar(32)    NOT NULL,

                      results               longblob      NOT NULL,

                      UNIQUE INDEX results_passages_idx (query_hash, year)
                      )

                      ROW_FORMAT=COMPRESSED
    ''')


    con.commit()


def delete_token(token):
    db = Database('RESULTS_RDS')
    con, cur = db.connect()

    cur.execute('delete from results_passages_yearly where tokens = "{}";'.format(token))
    cur.execute('delete from results_passages_yearly_default where tokens = "{}";'.format(token))
    cur.execute('delete from results_passages where tokens = "{}";'.format(token))
    con.commit()
    db = Database('RESULTS_LOCAL')
    con, cur = db.connect()

    cur.execute('delete from results_passages_yearly where tokens = "{}";'.format(token))
    cur.execute('delete from results_passages_yearly_default where tokens = "{}";'.format(token))
    cur.execute('delete from results_passages where tokens = "{}";'.format(token))
    con.commit()

if __name__ == "__main__":
    #initialize_database()
    delete_token("['ariel']")
    delete_token("['project ariel']")


