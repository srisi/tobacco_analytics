import json
from IPython import embed
from nameparser import HumanName
import hashlib
from collections import defaultdict
import copy
import re

from nameparser.config import CONSTANTS
CONSTANTS.titles.remove(*CONSTANTS.titles)


SEARCH_NAME = 'john'


def parse_names_json():
    """
    Parses the names.json with all the tobacco industry employees and corporate entities pulled from
    https://solr.idl.ucsf.edu/solr/glossary/select?q=(*)&fq=collection:*&rows=46000&start=0&fl=&wt=json&sort=namesort+asc&facet=false



    :return:
    """

    names = {}

    with open('names.json') as jsonfile:
        data = json.load(jsonfile)['response']['docs']

    for doc in data:

        name = HumanName(doc['name'])



        names[doc['name']] = {
            'name_last': name.last,
            'name_first': name.first,
            'name_middle': name.middle,
            'name_suffix': name.suffix,
            'name_full': f'{name.first} {name.last} {name.suffix}',

            'name_orig': doc['name'],
            'position': doc['position'],
            'collection': doc['collection'],
            'id': doc['id_int']
        }
        print()
        print(name.last, ",", name.first, "|", doc['name'] )
        print(names[doc['name']])



    embed()

def parse_name_mysql():
    from tobacco.utilities.databases import Database

    db = Database('TOB_FULL')
    con, cur = db.connect()

    people_database = PeopleDatabase()

    cur.execute('''SELECT author, count(author) 
                          FROM authors WHERE lower(author) like "%{}%" 
                          group by author order by count(author) desc;'''.format(SEARCH_NAME))
    while True:
        row = cur.fetchone()
        if not row:
            break
        names_raw = row['author']
        names_raw = [name.strip() for name in names_raw.split("|")]
        for name_raw in names_raw:
            if len(name_raw) <= 1:
                continue
            people_database.add_person_raw(name_raw, count=row['count(author)'])



    people_database.merge_duplicates()


def parse_name(name_raw: str) -> HumanName:
    """
    Parses name and returns as human name
    >>> n = parse_name('TEAGUE CE JR')
    >>> n.last, n.first, n.middle, " ".join(n.affiliations).upper()
    ('Teague', 'C', 'E', 'JR')

    >>> n = parse_name('teague, ce jr')
    >>> n.last, n.first, n.middle, " ".join(n.affiliations).upper()
    ('Teague', 'C', 'E', 'JR')


    >>> n = parse_name('Teague, Claude Edward, Jr., Ph.D. ')
    >>> n.last, n.first, n.middle, " ".join(n.affiliations).upper()
    ('Teague', 'Claude', 'Edward', 'JR., PH.D.')

    >>> n = parse_name('Teague, J - BAT')
    >>> n.last, n.first, n.middle, " ".join(n.affiliations).upper()
    ('Teague', 'J', '', 'BAT')

    >>> n = parse_name('BAKER, T E - NATIONAL ASSOCIATION OF ATTORNEYS GENERAL')
    >>> n.last, n.first, n.middle, " ".join(n.affiliations).upper()
    ('Baker', 'T', 'E', 'NATIONAL ASSOCIATION OF ATTORNEYS GENERAL')

    >>> n = parse_name('BAKER-cj')
    >>> n.last, n.first, n.middle, " ".join(n.affiliations).upper()
    ('Baker', 'C', 'J', '')

    JR and SR are by default recognized as titles -> turn off through CONSTANTS.
    >>> n = parse_name('Baker, JR')
    >>> n.last, n.first, n.middle, " ".join(n.affiliations).upper()
    ('Baker', 'J', 'R', '')

    >>> n = parse_name('DUNN WL #')
    >>> n.last, n.first, n.middle, " ".join(n.affiliations).upper()
    ('Dunn', 'W', 'L', '')

    >>> n = parse_name('Dunn, W. L.')
    >>> n.last, n.first, n.middle, " ".join(n.affiliations).upper()
    ('Dunn', 'W', 'L', '')

    >>> n = parse_name('TEMKO SL, COVINGTON AND BURLING')
    >>> n.last, n.first, n.middle, " ".join(n.affiliations).upper()
    ('Temko', 'S', 'L', 'COVINGTON AND BURLING')

    >>> n = parse_name('Temko, Stanley L [Privlog:] TEMKO,SL')
    >>> n.last, n.first, n.middle, " ".join(n.affiliations).upper()
    ('Temko', 'Stanley', 'L', '')

    >>> n = parse_name('Temko-SL, Covington & Burling')
    >>> n.last, n.first, n.middle, " ".join(n.affiliations).upper()
    ('Temko', 'S', 'L', 'COVINGTON & BURLING')

    >>> n = parse_name('HENSON, A. (AMERICAN SENIOR VICE PRESIDENT AND GENERAL COUNSEL)')
    >>> n.last, n.first, n.middle, " ".join(n.affiliations).upper()
    ('Henson', 'A', '', 'AMERICAN SENIOR VICE PRESIDENT AND GENERAL COUNSEL')

    >>> n = parse_name('HENSON, A. (CHADBOURNE, PARKE, WHITESIDE & WOLFF, AMERICAN OUTSIDE COUNSEL) (HANDWRITTEN NOTES)')
    >>> n.last, n.first, n.middle, " ".join(n.affiliations).upper()
    ('Henson', 'A', '', 'CHADBOURNE, PARKE, WHITESIDE & WOLFF AMERICAN OUTSIDE COUNSEL) (HANDWRITTEN NOTES')

    >>> n = parse_name('Holtzman, A.,  Murray, J. ,  Henson, A. ,  Pepples, E. ,  Stevens, A. ,  Witt, S.')
    >>> n.last, n.first, n.middle, " ".join(n.affiliations).upper()
    ('Holtzman', 'A', '', '')

    >>> n = parse_name('Holtz, Jacob, Jacob & Medinger')
    >>> n.last, n.first, n.middle, " ".join(n.affiliations).upper()
    ('Holtz', 'Jacob', '', 'JACOB & MEDINGER')

    # This one breaks. But I don't think it can be avoided.
    >>> n = parse_name('Holtz, Jacob Alexander, Jacob & Medinger')
    >>> n.last, n.first, n.middle, " ".join(n.affiliations).upper()
    ('Holtz', '', '', 'JACOB ALEXANDER, JACOB & MEDINGER')





    :param name_raw:
    :return:
    """

    print(name_raw)

    # remove privlog info, e.g. 'Temko, Stanley L [Privlog:] TEMKO,SL'. It confuses the name parser
    privlog_id = name_raw.find('[Privlog:]')
    if privlog_id == 0:
        name_raw = name_raw[privlog_id:]
    elif privlog_id > 0:
        name_raw = name_raw[:name_raw.find('[Privlog:]')]
    else:
        pass

    # position is often attached with a dash, e.g. 'BAKER, T E - NATIONAL ASSOCIATION OF ATTORNEYS'
    if name_raw.find(" - ") > -1 and len(name_raw.split(' - ')) == 2:
        name_raw, extracted_position = name_raw.split(" - ")
        extracted_positions = {extracted_position.strip()}
    else:
        extracted_positions = set()

    print(name_raw)



    institution_regexes = [

        # TI/CTR
        '[,#] Tobacco Inst.+$',
        '[\(\,\#] ?SAB Exec.*$',

        # American Tobacco
        '[(,#] ?American .+$',
        '[\(\,\#] ?Amer Brands.*$',
        '[,#] American Tob',
        '[,#] Atco.*$',

        # PM
        '[\(\,\#] ?Philip Morris.*$',

        # RJR
        '[\(\,\#] ?RJR.*$',

        #### LAW FIRMS ####
        '[\(\,\#] ?Arnold &.*$',
        '[\(\,\#] ?Chadbourne.*$',
        '[,#] COVINGTON [AB&]*.+$',
        '[,#] Foster [&A]*.+$',
        '[,#] JACOB [A&]*.+$',

        '[\(\,\#] ?Philip Morris.*$',



        '[\(\,\#] Univ .*$',
    ]
    for institution in institution_regexes:
        extracted_institution = re.search(r'{}'.format(institution), name_raw, re.IGNORECASE)
        if extracted_institution:
            extracted_positions.add(extracted_institution.group().strip(',#() '))
            name_raw = name_raw[:name_raw.find(extracted_institution.group())]

    # remove #
    name_raw = name_raw.strip("#").strip()

    if name_raw[-2] == '-':
        name_raw = name_raw[:-2] + " " + name_raw[-1:]
    if len(name_raw) > 2 and name_raw[-3] == '-':
        name_raw = name_raw[:-3] + " " + name_raw[-2:]

    name = HumanName(name_raw)

    # e.g. Dunn W -> parsed as last name W. -> switch first/last
    if len(name.last) <= 2 and len(name.first) > 2:
        name.first, name.last = name.last, name.first

    # remove periods from initials
    if len(name.first) == 2 and name.first[1] == '.':
        name.first = name.first[0]
    if len(name.middle) == 2 and name.middle[1] == '.':
        name.middle = name.middle[0]

    # If first name is length 2 (Teague, CE), the two letters are most likely initials.
    if len(name.first) == 2:
        name.middle = name.first[1].upper()
        name.first = name.first[0].upper()

    # If first and middle initials have periods but not spaces -> separate, e.g. "R.K. Teague"
    if re.match('[a-zA-Z]\.[a-zA-Z]\.', name.first):
        name.middle = name.first[2]
        name.first = name.first[0]

    name.last = name.last.capitalize()
    name.first = name.first.capitalize()
    name.middle = name.middle.capitalize()

    # if multiple names are passed, they often end up in the middle name
    # e.g. 'Holtzman, A.,  Murray, J. ,  Henson, A.  -> only allow one comma or set to empty
    if name.middle.count(',') > 1:
        name.middle = ''

    if len(name.suffix) > 20 and name.suffix.count('.') > 2:
        name.suffix = ''

    name.affiliations = extracted_positions
    if name.suffix:
        name.affiliations.add(name.suffix)

#    name.affiliations = " ".join(extracted_positions).upper()

#    print(name_raw, extracted_positions)

#    name.position = (name.suffix + extracted_position).upper()

    return name



class Person:

    def __init__(self, last, first, middle='', affiliations={}, aliases=[], count=0):
        self.last = last
        self.first = first
        self.middle = middle
        self.affiliations = affiliations
        self.aliases = aliases
        self.count = count

    def __repr__(self):
        s = f'{self.first} {self.middle} {self.last}'
        if self.affiliations:
            s += "(" + ", ".join(self.affiliations) + ")"
            
        return s

    def copy(self):
        return Person(last=self.last, first=self.first, middle=self.middle,
                      affiliations=copy.deepcopy(self.affiliations),
                      aliases=copy.deepcopy(self.aliases), count=self.count)

    def __hash__(self):
        return hash(f'{self.last} {self.first} {self.middle} {self.affiliations}')

    def stemmed(self):
        return f'{self.last} {self.first} {self.middle}'

class PeopleDatabase:

    def __init__(self):
        self.people = set()
        self.aliases = {}

    def add_person_raw(self, name_raw: str, count=0):

        name_parsed = parse_name(name_raw)
        new_p = Person(last=name_parsed.last, first=name_parsed.first, middle=name_parsed.middle,
                       affiliations=name_parsed.affiliations, aliases=[name_raw], count=count)
        self.people.add(new_p)

    def merge_duplicates(self):

        last_names = set()
        for person in self.people:
            last_names.add(person.last)


        for last_name in last_names:
            while True:
                last_names_dict = defaultdict(list)
                for person in self.people:
                    last_names_dict[person.last].append(person)

                finished = self.merge_last_name(last_names_dict, last_name)
                if finished:
                    if last_name.find(SEARCH_NAME) > -1:
                        print("\nSUMMARY")
                        for n in last_names_dict[last_name]:
                            print(n.count, n, n.aliases)
                        print("\n")
                    break


    def merge_last_name(self, last_names_dict, last_name):

        # if only one name -> already finished
        if len(last_names_dict[last_name]) == 1:
            return True

        last_names_dict[last_name].sort(key=lambda x: x.count, reverse=True)

        for person1 in last_names_dict[last_name]:
#            print("person1", person1)
            for person2 in last_names_dict[last_name]:

                if person1 == person2:
                    continue

                # if no first and middle name -> continue
                if person1.first == '' and person1.middle == '':
                    continue
                if person2.first == '' and person2.middle == '':
                    continue

                if person1.first == person2.first and person1.middle == person2.middle:
                    self.merge_two_persons(person1, person2)
                    return False


                # if both have full first names and they don't match -> skip
                if len(person1.first) > 2 and len(person2.first) > 2 and person1.first != person2.first:
                    continue

                # if both have full middle names and they don't match -> skip
                if len(person1.middle) > 2 and len(person2.middle) > 2 and person1.middle != person2.middle:
                    continue

                # if initial of the first name is not the same -> skip
                if person1.first and person2.first and person1.first[0] != person2.first[0]:
                    continue



                # if both have at least first and middle initials
                if person1.first and person1.middle and person2.first and person2.middle:
                    if person1.first[0] != person2.first[0] or person1.middle[0] != person2.middle[0]:
                        continue



                    # if first and middle initials match -> merge
                    if person1.first[0] == person2.first[0] and person1.middle[0] == person2.middle[0]:
                        self.merge_two_persons(person1, person2)
                        return False    # we're not finished -> return False

                if len(person1.first) == 1 and not person1.middle:
                    pass
#                    print('single first name', person1)
                    continue
                if len(person2.first) == 1 and not person2.middle:
                    pass
#                    print('single first name', person2)
                    continue




                else:

                    continue


        return True

    def merge_two_persons(self, person1, person2):

        new_p = person1.copy()

        for attr in ['first', 'middle']:
            if len(getattr(person2, attr)) > len(getattr(person1, attr)):
                setattr(new_p, attr, getattr(person2, attr))
        try:
            new_p.affiliations = person1.affiliations.union(person2.affiliations)
        except:
            embed()
        new_p.aliases = person1.aliases + person2.aliases
        new_p.count = person1.count + person2.count

        if new_p.last.find(SEARCH_NAME) > -1:
            print("Merged.", new_p, person1.aliases, person2.aliases)

        try:
            self.people.remove(person1)
            self.people.remove(person2)
            self.people.add(new_p)
        except KeyError:
            print('k')
            embed()

    def find_related_person(self, person: Person):
        pass





if __name__ == "__main__":
#    parse_names_json()

    n = parse_name('TEAGUE CE JR')

    parse_name_mysql()
