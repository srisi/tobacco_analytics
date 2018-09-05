import re
from collections import Counter


from tobacco.utilities.databases import Database


english_name_regex = re.compile('[A-Z][a-zA-Z]+[, -]+[A-Z]{1,3}')


def generate_node_db():

    initialize_green_db()

    # Step 1: Identify top 50 nodes
    node_counter = Counter()

    db = Database("TOB_FULL")
    con, cur = db.connect()
    con2, cur2 = db.connect()
    cur.execute('DELETE FROM green;')
    con.commit()

    author_and_recipient_commands = [
       '''SELECT recipients.recipient as node, docs.tid, "author" as main_person_is
                          FROM authors, recipients, docs
                          WHERE authors.author="Green, SJ" and authors.doc_id=recipients.doc_id AND docs.id=authors.doc_id;''',
       '''SELECT authors.author as node, authors.doc_id, docs.tid, "recipient" as main_person_is
                          FROM authors, recipients, docs
                          WHERE recipients.recipient="Green, SJ" and authors.doc_id=recipients.doc_id AND docs.id=authors.doc_id;''',
    ]

    for command in author_and_recipient_commands:
        cur.execute(command)
        while True:
            row = cur.fetchone()
            if not row: break
            else:
                for person in english_name_regex.findall(row['node']):
                    node_counter[person] += 1
#                    cur2.execute('INSERT INTO green(node, main_author_is, tid) VALUES("{}", "{}", "{}")'.format(
#                        person, row['main_person_is'], row['tid']
#                    ))

#    con2.commit()
    print(node_counter.most_common(50))
    top_50_nodes = [i[0] for i in node_counter.most_common(50)]

    # Step 2: insert
    for command in author_and_recipient_commands:
        cur.execute(command)
        while True:
            row = cur.fetchone()
            if not row: break
            else:
                for person in english_name_regex.findall(row['node']):
                    if person in top_50_nodes:
                        cur2.execute('INSERT INTO green(node, main_author_is, tid) VALUES("{}", "{}", "{}")'.format(
                            person, row['main_person_is'], row['tid']
                        ))

    con2.commit()


def initialize_green_db():

    db = Database('TOB_FULL')
    con, cur = db.connect()

    cur.execute('''CREATE TABLE IF NOT EXISTS green(
                    id              int NOT NULL AUTO_INCREMENT,
                    node            varchar(255)  NOT NULL,
                    main_author_is  varchar(10)   NOT NULL,
                    tid             varchar(10)   NOT NULL,
                    weight          float                 ,

                    PRIMARY KEY(id));''')
    con.commit()




if __name__ == "__main__":
    generate_node_db()

