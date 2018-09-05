'''
Possibly to be added in the future: location identifier for the distinctive terms
'''

#!/usr/bin/env python
# coding: utf-8
#
# The World Gazetteer provides a downloadable file that contains a list
# of all cities, towns, administrative divisions and agglomerations with
# their population, their English name parent country.
#
# Article:  http://answers.google.com/answers/threadview/id/774429.html
# Download: http://www.world-gazetteer.com/dataen.zip
from tobacco.configuration import PATH_TOKENIZED

import codecs


# can't just iterate over the fd as there are many lines with
# carriage returns in the middle of the line and things break.
def rows(fd):
    n = 1
    rest = ""
    while 1:
        chunk = fd.read(1024)
        if not chunk:
            break

        while 1:
            chunk = rest + chunk
            pos = chunk.find("\n")
            if pos > -1:
                pos += 1
                line, rest = chunk[:pos], chunk[pos:]
                yield n, line.replace("\r", "").replace("\n", "").split("\t")
                chunk = ""
                n += 1
            else:
                break


if __name__ == "__main__":

    location_terms = set()

    cities = set()
    fd = codecs.open(PATH_TOKENIZED + "locations.txt", "r", "utf-8")
    for n, row in rows(fd):
        columns = [row[8], row[9], row[1]]  # Country,Region,City

        # countries
        if len(row[8].split()) == 1:
            location_terms.add(row[8])
        # regions
        if len(row[9].split()) == 1:
            location_terms.add(row[9])

        # cities
        if len(row[1].split()) == 1:
            location_terms.add(row[1])

        if all(columns):
            cities.add(",".join(columns))

    #for line in sorted(cities):
    #    print (line)

    print(len(location_terms))

    a = '''Term	Count	Log-Likelihood (G2)	      	Term	Count	Log-Likelihood (G2)
device	182	1161.79		battelle	67	530.65
devices	133	941.18		sir	106	520.69
project	234	742.89		frangible	36	505.51
charles	126	722.63		aerosol	88	456.28
hughes	91	690.54		southampton	78	432.3
tube	116	678.66		tubes	67	418.27
nicotine	336	672.77		patent	88	386.71
ellis	93	658.11		agreed	93	378.15
orientated	49	630.42		probably	95	370.23
dr	242	610.27		work	143	299.06
Term	Count	Log-Likelihood (G2)	      	Term	Count	Log-Likelihood (G2)
lung	9416	37375.52		mortality	1098	3198.63
smoking	6465	6171.79		cause	1356	3155.96
cases	1813	4921.02		carcinoma	771	2983.55
patients	1487	4908.39		women	1327	2669.39
risk	2483	4714.51		lip	503	2586.05
disease	2003	4203.72		larynx	557	2541.27
deaths	1131	4148.99		breast	575	2517.04
death	1346	3898.57		causes	871	2371.21
cent	1142	3890.75		diseases	860	2246.85
incidence	1162	3785.25		years	1876	2214.42'''
    for t in a.split():
        t = t.capitalize()
        if t in location_terms:
            print(t, t in location_terms)


# Better turn this into a set. Use only single tokens, i.e. discard words with more than one word.