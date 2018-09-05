
import logging
import time
from subprocess import call

import MySQLdb as mdb
from MySQLdb import OperationalError
from tobacco.configuration import CURRENT_MACHINE
from tobacco.secrets import get_secret

logger = logging.getLogger('tobacco')


TOB_FULL = get_secret('db_tob_full_local_config')

TOB_RAW = get_secret('db_tob_raw_local_config')

TOB_NETWORKS = get_secret('db_tob_networks_config')

RESULTS_LOCAL = get_secret('db_results_local_config')

RESULTS_RDS= get_secret('db_results_rds_config')


class Database:
    # for future reference: this is fucking inefficient, and, yes, you should clean it up (use parameter directly)
    def __init__(self, name):
        self.config = name        #this dict contains all the necessary connection info
        if name == 'TOB_FULL':
            self.config = TOB_FULL
        elif name == 'RESULTS_RDS':
            self.config = RESULTS_RDS
        elif name == 'RESULTS_LOCAL':
            self.config = RESULTS_LOCAL
        elif name == 'TOB_NETWORKS':
            self.config = TOB_NETWORKS
        elif name == 'TOB_RAW':
            self.config = TOB_RAW
        else:
            raise NameError('No valid database selected. Selected: {}'.format(name))

    def connect(self):
        if 'database' in self.config.keys():
            con = mdb.connect(host=self.config['hostname'], port=self.config['port'], user=self.config['user'],
                              passwd=self.config['password'], db=self.config['database'], charset='utf8',
                              use_unicode=True)
            cur = con.cursor(mdb.cursors.SSDictCursor)
        else:
            con = mdb.connect(host=self.config['hostname'], port=self.config['port'], user=self.config['user'],
                              passwd=self.config['password'], charset='utf8', use_unicode=True)
            cur = con.cursor(mdb.cursors.SSDictCursor)
            cur.execute("USE tobacco;")

        # Jan 2016: Tried to go without these configs by adding them to my.cnf, led to timeouts -> back to using them.
        #read/write timeout set to max because sometimes processing the documents takes too much time for standard write timeout

        # if not (self.config == DJANGO_RDS or self.config == DJANGO_RDS2):
        #     cur.execute("SET GLOBAL net_read_timeout = 3600;")
        #     cur.execute("SET GLOBAL net_write_timeout = 3600;")

        if CURRENT_MACHINE == 'local':
            try:
                cur.execute("SET GLOBAL max_allowed_packet = 1073740800;")
            except OperationalError:
                pass

        return con, cur

    def get_connection_dict(self):
        return self.config

    def create_index(self, database, table, column):
        '''
        Creates an index on column if it does not yet exist
        :param table:
        :param column:
        :return: True if index is created, False if index already exists
        '''

        con, cur = self.connect()
        cur.execute("USE {};".format(database))
        try:
            # sample: "CREATE INDEX token on cancer_freq_raw(token)
            logger.info("Creating index {} on {}".format(column, table))
            cur.execute("CREATE INDEX {0} on {1}({0})".format(column, table))
            con.close()
            return True
        except mdb.Error as e:
            logger.info("Index on {} in table {} already exists. {}".format(column, table, e))
            con.close()
            return False

    def drop_index(self, database, table, column):
        con, cur = self.connect()
        cur.execute("USE {};".format(database))

        try:
            cur.execute("DROP INDEX {} on {}".format(column, table))
            logger.info("Dropping index {} from {}".format(column, table))
            con.close()
            return True
        except mdb.Error as e:
            logger.info("Index does not exist. \n{}".format(e))
            con.close()
            return False

    # def load_table(self, table_config):
    #     '''
    #     Loads specified table into database
    #     1) Creates database if not exists
    #     2) Downloads and extracts file
    #     3) Imports it into the database
    #
    #     :param table_config: The config dict of the table from db_config on S3
    #     '''
    #     from subprocess import Popen, PIPE
    #
    #     db_name = table_config['database']
    #     table_name = table_config['table']
    #     con, cur = self.connect()
    #
    #     # 1) check if database exists, create when necessary
    #     cur.execute("SHOW DATABASES LIKE '{}';".format(db_name))
    #     if len(cur.fetchall()) == 0:
    #         cur.execute("CREATE DATABASE {};".format(db_name))
    #         logger.info("Creating database {}".format(db_name))
    #     else: cur.execute("USE {}".format(db_name))
    #
    #     # 2) download and extract source file
    #     local_path = '/tobacco/{}.sql.gz'.format(table_name)
    #     load_file_from_s3(bucket='tobacco-database',
    #                       s3_path=table_config['s3_path'],
    #                       local_path=local_path)
    #     local_path = extract(local_path)
    #
    #     # 3) import the extracted *.sql file
    #     logger.info("Starting import of {}".format(local_path))
    #     with open(local_path, 'r') as f:
    #         command = ['mysql', '--user={}'.format(self.config['user']), '--password={}'.format(self.config['password']), db_name]
    #         proc = Popen(command, stdin = f)
    #         stdout, stderr = proc.communicate()
    #     logger.info("Finished importing {}\n".format(local_path))
    #
    #     # delete the local file
    #     os.remove(local_path)

    # def load_required_tables(self, table_names):
    #     '''
    #
    #     '''
    #     con, cur = self.connect()
    #
    #     db_config = load_db_config()
    #     for table_name in table_names:
    #         if not table_name in db_config: #return False if table does not yet exist
    #             return False
    #         cur.close()
    #         cur = con.cursor()
    #         table_config = db_config[table_name]
    #         try:
    #             cur.execute("SELECT * from {}.{} LIMIT 10;".format(table_config['database'], table_config['table']))
    #             logger.info("{} already present".format(table_name))
    #
    #         except mdb.Error as e:
    #             if e[0] == 1146: #1146 error -> table does not exist
    #                 logger.info("{} missing, loading table now.".format(table_name))
    #                 self.load_table(table_config)

    def dump_table(self,db_name, table_name):
        '''
        Dumps table from database to /tobacco, returns path
        '''


        # example: mysqldump -h localhost -P 3306 -u root --password=pass tobacco collection > collection.sql
        call("mysqldump -h {0} -P {1} -u {2} --password={3} --max_allowed_packet=1G {4} {5} > /tobacco/{5}.sql".format( #max_allowed_packet to allow dumping large documents
            self.config['hostname'], str(self.config['port']), self.config['user'], self.config['password'], db_name, table_name), shell=True)
        return "/tobacco/{}.sql".format(table_name)

    # def store_table_dump_on_s3(self, db_name, table_name, s3_path):
    #
    #     logger.info("config: ", db_name, table_name, s3_path)
    #
    #     # dump table to local
    #     logger.info("Dumping and compressing {}".format(table_name))
    #     file_path = self.dump_table(db_name, table_name)
    #     logger.info("File name ", file_path)
    #
    #     # compress dump
    #     compressed_path = compress(file_path)
    #
    #     # Store table on S3, update db_config
    #     store_db_table_in_s3(local_path=compressed_path,
    #                          s3_path=s3_path,
    #                          db_name = db_name,
    #                          table_name=table_name)

    # def store_yearly_sums_to_db(self, identifier, yearly_sums, start_year, end_year):
    #     '''
    #     Stores the yearly sums of an identifier to the database and uploads it to s3
    #
    #     :param identifier:
    #     :param yearly_sums:
    #     :param start_year:
    #     :param end_year:
    #     :return:
    #     '''
    #
    #     con, cur = self.connect()
    #
    #     cur.execute("DROP TABLE IF EXISTS {}_yearly_totals;".format(identifier))
    #     cur.execute('''CREATE TABLE IF NOT EXISTS {}_yearly_totals(
    #                           year int(4) NOT NULL UNIQUE,
    #                           total_tokens int NOT NULL);'''.format(identifier))
    #
    #     for year_idx, _ in enumerate(yearly_sums):
    #         logger.info(start_year + year_idx, yearly_sums[year_idx, 0])
    #         # yearly_sums is np defmatrix, so we always need the first column.
    #         year = start_year + year_idx
    #         tokens_of_year = yearly_sums[year_idx, 0]
    #         cur.execute('''INSERT INTO {}_yearly_totals (year, total_tokens)
    #                         VALUES ({}, {});'''.format(identifier, year, tokens_of_year))
    #
    #     # make sure that we cover the whole range between start year and end year before committing.
    #     assert start_year + len(yearly_sums) -1 == end_year
    #     con.commit()
    #
    #     # Dump table, compress, and store on s3, update db_config
    #     self.store_table_dump_on_s3(db_name='tobacco',
    #                                table_name='{}_yearly_totals'.format(identifier),
    #                                s3_path='/sub_database/{0}/{0}_yearly_totals.sql.gz'.format(identifier))

    # def load_yearly_sums(self, identifier):
    #     '''
    #     Loads yearly totals from database and returns start year, end year, and the totals.
    #
    #     :param identifier:
    #     :return:
    #     '''
    #
    #     logger.info("Loading yearly totals of {}".format(identifier))
    #
    #     self.load_required_tables(['{}_yearly_totals'.format(identifier)])
    #
    #     con, cur = self.connect()
    #
    #     cur.execute("SELECT min(year), max(year) from {}_yearly_totals;".format(identifier))
    #     rows = cur.fetchall()
    #     start_year = rows[0]['min(year)']
    #     end_year = rows[0]['max(year)']
    #     logger.infoc(start_year, end_year, end_year - start_year + 1)
    #
    #     totals_list = (end_year - start_year + 1) * [0]
    #
    #     cur.execute("SELECT * from {}_yearly_totals ORDER BY year ASC;".format(identifier))
    #     rows = cur.fetchall()
    #     for row in rows:
    #         totals_list[row['year']-start_year] = row['total_tokens']
    #     logger.info("Yearly totals: ", totals_list)
    #
    #     return start_year, end_year, totals_list


    def batch_insert(self, table_name, key_list, values, chunk_size=100):
        '''

        Automated batch inserts. Automatically produces chunks and inserts them

        :param table_name:
        :param key_list: list of keys, e.g. ['db_name', 'total', 'token_id']
        :param values: list of dicts containing the values from the key_list,
                        e.g. [{'db_name': 'pm', 'total': 2342, 'token_id': 23},...]
        :param chunk_size: number of entries to be inserted at once. Default: 100
        :return:
        '''

        con, cur = self.connect()

        # turns the list of dicts into a list of tuples,
        # e.g. [{'key': 1, 'value': 2}, {'key': 3, 'value': 4}] -> [(1,2), (3, 4)]
        insert_list = [tuple(entry[key] for key in key_list) for entry in values]
        # splits the insert list into chunks of size chunk_size (default: 100)
        insert_list_chunked = [insert_list[i:i+chunk_size] for i in range(0, len(insert_list), chunk_size)]
        del insert_list


        count = 0
        for chunk in insert_list_chunked:
            count += 1
            if count % 100 == 0: print("Inserted: {}".format(count*chunk_size))
            try:
                cur.executemany('''REPLACE INTO {} ({}) VALUES ({});'''.format(
                    table_name,
                    ", ".join(key_list), # e.g. 'key, value'
                    ", ".join(["%s" for key in key_list])), chunk) # e.g. '%s, %s
            # occasionally, there are lock wait timeout exceeded errors (-> insert takes too long) -> smaller chunks
            except mdb.OperationalError as e:
                logger.warning("batch insert lock wait timeout. reducing chunk size and waiting 1 minute.")
                logger.warning(e)
                time.sleep(6)
                self.batch_insert(table_name, key_list, values, chunk_size/4)
                return
            except mdb.ProgrammingError:
                logger.warning("Insert failed, printing last statement:")
                logger.warning(cur._last_executed)
                raise

        con.commit()
        con.close()



if __name__ == "__main__":
    db = Database("LOCAL")
    pass


