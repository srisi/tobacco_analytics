import json

import numpy as np

from IPython import embed


ALIASES = {
    'Bosnia and Herzegovina': ['bosnia', 'herzegovina', 'yugoslavia'],
    'Croatia': ['croatia', 'yugoslavia'],
    'Slovenia': ['slovenia', 'yugoslavia'],
    'Macedonia': ['macedonia', 'yugoslavia'],
    'Republic of Serbia': ['serbia', 'yugoslavia'],
    'Montenegro': ['montenegro', 'yugoslavia'],

    'Burkina Faso': ['burkina', 'upper volta'],
    'Myanmar': ['myanmar', 'burma'],
    'Russia': ['russia', 'soviet union'],
    'Somaliland': ['somalia'],
    'South Sudan': ['sudan'],
    'The Bahamas': ['bahamas'],
    'United Republic of Tanzania': ['tanzania'],
    'United States of America': ['united states'],
    'United Kingdom': ['united kingdom', 'england', 'wales', 'scotland', 'northern ireland'],

    'Democratic Republic of the Congo': ['congo'],
    'Republic of the Congo': ['congo']

}

POPULATION = {
    'Afghanistan':  (8.1, 34.66),
    'Angola': (4.1, 29.78),
    'Albania': (1.2, 41.32),
    'United Arab Emirates': 9.27,
    'Argentina': 44.27,
    'Armenia': 2.9,
    'Antarctica': 0.1,
    'French Southern and Antarctic Lands': 0.1,
    'Australia': 24.45,
    'Austria': 8.7,
    'Azerbaijan': 9.8,
    'Burundi': 10.86,
    'Belgium': 11.43,
    'Benin': 11.18,
    'Burkina Faso': 19.19,
    'Bangladesh': 164.67,
    'Bulgaria': 7.09,
    'The Bahamas': 0.4,
    'Bosnia and Herzegovina': 3.5,
    'Belarus': 9.47,
    'Belize': 0.38,
    'Bolivia': 11.05,
    'Brazil': 209.29,
    'Brunei': 0.43,
    'Bhutan': 0.81,
    'Botswana': 2.3,
    'Central African Republic': 4.66,
    'Canada': 36.62,
    'Switzerland': 8.48,
    'Chile': 18.05,
    'China': 1409.52,
    'Ivory Coast': 24.29,
    'Cameroon': 24.05,
    'Democratic Republic of the Congo': 84,
    'Republic of the Congo': 84,
    'Colombia': 49.06,
    'Costa Rica': 4.91,
    'Cuba': 11.48,
    'Northern Cyprus': 1.18,
    'Cyprus': 1.18,
    'Czech Republic': 81.34,
    'Germany': 82.11,
    'Djibouti': 0.96,
    'Denmark': 5.73,
    'Dominican Republic': 10.76,
    'Algeria': 41.32,
    'Ecuador': 16.62,
    'Egypt': 97.55,
    'Eritrea': 5.07,
    'Spain': 46.35,
    'Estonia': 1.31,
    'Ethiopia': 104.96,
    'Finland': 5.52,
    'Fiji': 0.91,
    'Falkland Islands': 0.01,
    'France': 64.98,
    'Gabon': 20.02,
    'United Kingdom': 66.18,
    'Georgia': 3.91,
    'Ghana': 28.83,
    'Guinea': 12.72,
    'Gambia': 2.1,
    'Guinea Bissau': 1.86,
    'Equatorial Guinea': 1.27,
    'Greece': 11.16,
    'Greenland': 0.06,
    'Guatemala': 16.91,
    'Guyana': 0.78,
    'Honduras': 9.26,
    'Croatia': 4.19,
    'Haiti': 10.98,
    'Hungary': 9.72,
    'Indonesia': 263.99,
    'India': 1339.19,
    'Ireland': 4.76,
    'Iran': 81.16,
    'Iraq': 38.27,
    'Iceland': 0.34,
    'Israel': 8.32,
    'Italy': 59.36,
    'Jamaica': 2.80,
    'Jordan': 9.70,
    'Japan': 127.48,
    'Kazakhstan': 18.20,
    'Kenya': 49.7,
    'Kyrgyzstan': 6.05,
    'Cambodia': 16.00,
    'South Korea': 50.98,
    'Kosovo': 1.92,
    'Kuwait': 4.14,
    'Laos': 6.86,
    'Lebanon': 6.08,
    'Liberia': 4.73,
    'Libya': 6.37,
    'Sri Lanka': 20.88,
    'Lesotho': 2.33,
    'Lithuania': 2.89,
    'Luxembourg': 0.58,
    'Latvia': 1.95,
    'Morocco': 35.74,
    'Moldova': 4.05,
    'Madagascar': 25.57,
    'Mexico': 129.16,
    'Macedonia': 2.08,
    'Mali': 18.54,
    'Myanmar': 53.37,
    'Montenegro': 0.63,
    'Mongolia': 3.08,
    'Mozambique': 29.67,
    'Mauritania': 4.42,
    'Malawi': 18.62,
    'Malaysia': 31.62,
    'Namibia': 2.53,
    'New Caledonia': 0.27,
    'Niger': 21.48,
    'Nigeria': 190.88,
    'Nicaragua': 6.22,
    'Netherlands': 17.04,
    'Norway': 5.31,
    'Nepal': 29.30,
    'New Zealand': 4.71,
    'Oman': 4.64,
    'Pakistan': 197.02,
    'Panama': 4.09,
    'Peru': 32.17,
    'Philippines': 104.92,
    'Papua New Guinea': 8.25,
    'Poland': 38.17,
    'Puerto Rico': 3.66,
    'North Korea': 25.49,
    'Portugal': 10.33,
    'Paraguay': 6.81,
    'Qatar': 2.64,
    'Romania': 19.68,
    'Russia': 143.99,
    'Rwanda': 12.21,
    'Western Sahara': 0.55,
    'Saudi Arabia': 32.94,
    'Sudan': 40.53,
    'South Sudan': 40.53,
    'Senegal': 15.85,
    'Solomon Islands': 0.61,
    'Sierra Leone': 7.56,
    'El Salvador': 6.38,
    'Somaliland': 14.74,
    'Somalia': 14.74,
    'Republic of Serbia': 8.80,
    'Suriname': 0.56,
    'Slovakia': 5.45,
    'Slovenia': 2.08,
    'Sweden': 9.91,
    'Swaziland': 1.37,
    'Syria': 18.27,
    'Chad': 14.9,
    'Togo': 7.79,
    'Thailand': 69.04,
    'Tajikistan': 8.92,
    'Turkmenistan': 5.76,
    'East Timor': 1.3,
    'Trinidad and Tobago': 1.37,
    'Tunisia': 11.53,
    'Turkey': 80.75,
    'Taiwan': 23.57,
    'United Republic of Tanzania': 57.31,
    'Uganda': 42.86,
    'Ukraine': 44.22,
    'Uruguay': 3.46,
    'United States of America': 324.46,
    'Uzbekistan': 31.91,
    'Venezuela': 31.98,
    'Vietnam': 95.54,
    'Vanuatu': 0.27,
    'West Bank': 4.92,
    'Yemen': 28.25,
    'South Africa': 56.72,
    'Zambia': 17.09,
    'Zimbabwe': 16.529
}

COUNTRY_CODES = {
    'Afghanistan':  4,
    'Angola': 24,
    'Albania': 8,
    'United Arab Emirates': 784,
    'Argentina': 32,
    'Armenia': 51,
    'Antarctica': 10,
    'French Southern and Antarctic Lands': 260,
    'Australia': 36,
    'Austria': 40,
    'Azerbaijan': 31,
    'Burundi': 108,
    'Belgium': 56,
    'Benin': 204,
    'Burkina Faso': 854,
    'Bangladesh': 50,
    'Bulgaria': 100,
    'The Bahamas': 44,
    'Bosnia and Herzegovina': 70,
    'Belarus': 112,
    'Belize': 84,
    'Bolivia': 68,
    'Brazil': 76,
    'Brunei': 96,
    'Bhutan': 64,
    'Botswana': 72,
    'Central African Republic': 140,
    'Canada': 124,
    'Switzerland': 756,
    'Chile': 152,
    'China': 156,
    'Ivory Coast': 384,
    'Cameroon': 120,
    'Democratic Republic of the Congo': 180,
    'Republic of the Congo': 178,
    'Colombia': 170,
    'Costa Rica': 188,
    'Cuba': 192,
    'Northern Cyprus': -99,
    'Cyprus': 196,
    'Czech Republic': 203,
    'Germany': 276,
    'Djibouti': 262,
    'Denmark': 208,
    'Dominican Republic': 214,
    'Algeria': 12,
    'Ecuador': 218,
    'Egypt': 818,
    'Eritrea': 232,
    'Spain': 724,
    'Estonia': 233,
    'Ethiopia': 231,
    'Finland': 246,
    'Fiji': 242,
    'Falkland Islands': 238,
    'France': 250,
    'Gabon': 266,
    'United Kingdom': 826,
    'Georgia': 268,
    'Ghana': 288,
    'Guinea': 324,
    'Gambia': 270,
    'Guinea Bissau': 624,
    'Equatorial Guinea': 226,
    'Greece': 300,
    'Greenland': 304,
    'Guatemala': 320,
    'Guyana': 328,
    'Honduras': 340,
    'Croatia': 191,
    'Haiti': 332,
    'Hungary': 348,
    'Indonesia': 360,
    'India': 356,
    'Ireland': 372,
    'Iran': 364,
    'Iraq': 368,
    'Iceland': 352,
    'Israel': 376,
    'Italy': 380,
    'Jamaica': 388,
    'Jordan': 400,
    'Japan': 392,
    'Kazakhstan': 398,
    'Kenya': 404,
    'Kyrgyzstan': 417,
    'Cambodia': 116,
    'South Korea': 410,
    'Kosovo': -99,
    'Kuwait': 414,
    'Laos': 418,
    'Lebanon': 422,
    'Liberia': 430,
    'Libya': 434,
    'Sri Lanka': 144,
    'Lesotho': 426,
    'Lithuania': 440,
    'Luxembourg': 442,
    'Latvia': 428,
    'Morocco': 504,
    'Moldova': 498,
    'Madagascar': 450,
    'Mexico': 484,
    'Macedonia': 807,
    'Mali': 466,
    'Myanmar': 104,
    'Montenegro': 499,
    'Mongolia': 496,
    'Mozambique': 508,
    'Mauritania': 478,
    'Malawi': 454,
    'Malaysia': 458,
    'Namibia': 516,
    'New Caledonia': 540,
    'Niger': 562,
    'Nigeria': 566,
    'Nicaragua': 558,
    'Netherlands': 528,
    'Norway': 578,
    'Nepal': 524,
    'New Zealand': 554,
    'Oman': 512,
    'Pakistan': 586,
    'Panama': 591,
    'Peru': 604,
    'Philippines': 608,
    'Papua New Guinea': 598,
    'Poland': 616,
    'Puerto Rico': 630,
    'North Korea': 408,
    'Portugal': 620,
    'Paraguay': 600,
    'Qatar': 634,
    'Romania': 642,
    'Russia': 643,
    'Rwanda': 646,
    'Western Sahara': 732,
    'Saudi Arabia': 682,
    'Sudan': 729,
    'South Sudan': 728,
    'Senegal': 686,
    'Solomon Islands': 90,
    'Sierra Leone': 694,
    'El Salvador': 222,
    'Somaliland': -99,
    'Somalia': 706,
    'Republic of Serbia': 688,
    'Suriname': 740,
    'Slovakia': 703,
    'Slovenia': 705,
    'Sweden': 752,
    'Swaziland': 748,
    'Syria': 760,
    'Chad': 148,
    'Togo': 768,
    'Thailand': 764,
    'Tajikistan': 762,
    'Turkmenistan': 795,
    'East Timor': 626,
    'Trinidad and Tobago': 780,
    'Tunisia': 788,
    'Turkey': 792,
    'Taiwan': 158,
    'United Republic of Tanzania': 834,
    'Uganda': 800,
    'Ukraine': 804,
    'Uruguay': 858,
    'United States of America': 840,
    'Uzbekistan': 860,
    'Venezuela': 862,
    'Vietnam': 704,
    'Vanuatu': 548,
    'West Bank': 275,
    'Yemen': 887,
    'South Africa': 710,
    'Zambia': 894,
    'Zimbabwe': 716
}


POPULATIONS = json.load(open('population.json'))

def get_population(country_name, year):

    fixed_populations = {
        'Antarctica': 0.1,
        'French Southern and Antarctic Lands': 0.1,
        'Falkland Islands': 0.1,
        'Western Sahara': 0.55,
        'East Timor': 1.3,
    }
    if country_name in fixed_populations:
        return fixed_populations[country_name]


    interpolated_populations = {
        'Taiwan': {1950: 7981454, 2016: 23464787}
    }
    if country_name in interpolated_populations:
        country_data = interpolated_populations[country_name]
        annual_growth = (country_data[2016] - country_data[1950]) / 66
        population = country_data[1950] + (year-1950) * annual_growth
        population_millions = population / 1000 / 1000
        return population_millions


    aliases = {
        'The Bahamas': 'Bahamas, The',
        'Brunei': "Brunei Darussalam",
        'Ivory Coast': "Cote d'Ivoire",
        'Democratic Republic of the Congo': "Congo, Dem. Rep.",
        'Republic of the Congo': "Congo, Rep.",
        'Northern Cyprus': 'Cyprus',
        'Egypt': "Egypt, Arab Rep.",
        'Gambia': "Gambia, The",
        'Guinea Bissau': "Guinea-Bissau",
        'Iran': "Iran, Islamic Rep.",
        'Kyrgyzstan': "Kyrgyz Republic",
        'South Korea': "Korea, Rep.",
        'North Korea': "Korea, Dem. People\u2019s Rep.",
        'Laos': "Lao PDR",
        'Macedonia': "Macedonia, FYR",
        'Russia': "Russian Federation",
        'Somaliland': 'Somalia',
        'Republic of Serbia': "Serbia",
        'Slovakia': "Slovak Republic",
        'Syria': "Syrian Arab Republic",
        'United Republic of Tanzania': "Tanzania",
        'United States of America': "United States",
        'Venezuela': "Venezuela, RB",
        'West Bank': "West Bank and Gaza",
        'Yemen': "Yemen, Rep.",
    }


    if country_name in aliases:
        country_name = aliases[country_name]

    if year < 1960:
        year = 1960
    if year > 2016:
        year = 2016

    population_millions = None
    for i in POPULATIONS:
        if i['Year'] == year and i['Country Name'] == country_name:
            population_millions = i['Value'] / 1000 / 1000
            return population_millions

    year_orig = year
    while year < 2016:
        year += 1
        for i in POPULATIONS:
            if i['Year'] == year and i['Country Name'] == country_name:
                population_millions = i['Value'] / 1000 / 1000

                print(f'Found population for {country_name} for {year} instead of {year_orig}.')
                return population_millions

    year = 2016
    while year > 1950:
        year -= 1
        for i in POPULATIONS:
            if i['Year'] == year and i['Country Name'] == country_name:
                population_millions = i['Value'] / 1000 / 1000

                print(
                    f'Found population for {country_name} for {year} instead of {year_orig}.')
                return population_millions

    raise ValueError(f'Country {country_name} not available. Year: {year_orig}')

def load_country_data():



    population_dict = {}
    for country_name in COUNTRY_CODES:
        print(country_name)
        population_dict[country_name] = 116 * [0.0]
        for year in range(1940, 2017):
            population_in_year = get_population(country_name, year)
            population_dict[country_name][year - 1901] = population_in_year

    countries_to_totals = {}
    country_data = {}

    from tobacco.frequencies_preprocessing.preprocessing_globals_loader import get_globals
    from tobacco.frequencies.calculate_ngrams import get_frequencies
    globals = get_globals()

    for name in COUNTRY_CODES:
        print(name)
        id = COUNTRY_CODES[name]

        if name in ALIASES:
            search_term = ALIASES[name]
        else:
            search_term = [name.lower()]


        d = {}
        collection_ids = {'at': 9, 'ba': 15, 'bw': 8, 'll': 7, 'pm': 5, 'rj': 6}

        for collection in ['at', 'ba', 'bw', 'll', 'pm', 'rj']:

            active_filters = {'doc_type': ['internal communication', 'marketing documents'],
                              'collection': [collection_ids[collection]], 'availability': [],
                              'term': []}

            res = get_frequencies(search_term, active_filters, globals, profiling_run=False)



            d[collection] = {'name': name}

            # add data from combined collections
            try:
                countries_to_totals[name] = res['data']['tokens'][0]['total']
                d[collection]['total'] = res['data']['tokens'][0]['total']
                d[collection]['mean_freq'] = np.mean(res['data']['tokens'][0]['frequencies'])
                d[collection]['counts'] = res['data']['tokens'][0]['counts']
                d[collection]['frequencies'] = res['data']['tokens'][0]['frequencies']

            # some countries don't exist in TA, e.g. French Southern and Antarctic Lands
            except KeyError:
                import traceback
                print(traceback.format_exc())
                d[collection]['total'] = 0
                d[collection]['mean_freq'] = 0.0
                d[collection]['counts'] = 116 * [0]
                d[collection]['frequencies'] = 116 * [0.0]

            print(d[collection]['counts'])

        country_data[id] = d



    output = {
        'country_data': country_data,
        'population_dict': population_dict
    }

    json.dump(output, open('/pcie/tobacco/tokenized/tobacco_flask_data/country_data.json', 'w'))

    embed()

def moving_avg(data, smoothing):


    if smoothing == 0: return data

    results = []
    print(data)
#    for datum in data:

    moving_avg = len(data) * [0]

    for idx in range(len(data)):
        section = data[max(0, idx - smoothing): idx+smoothing+1]
        moving_avg[idx] = np.mean(section)

    return moving_avg



if __name__ == '__main__':

    for year in range(1940, 1960):
        print(year, get_population('United States', year))
    load_country_data()