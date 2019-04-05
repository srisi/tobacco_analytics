import pandas as pd
from IPython import embed
import numpy as np


def calculate_cohort_data(year=1940, male_or_female='m', interval=10, pack_per_day_data=False):

    print(f"\n\n{year}: {male_or_female}")


    demographics = load_demographics_data(year, male_or_female)
    cohort_smoking = load_smoking_data(year, male_or_female, pack_per_day_data=False)
    cpd_smoking = load_smoking_data(year, male_or_female, pack_per_day_data=True)

    smokers = []
    cigs = []
    for cur_year in range(1880, year, interval):
        no_smokers = 1
        no_cpd = 0
        for i in range(interval):
            try:
                no_smokers += demographics[cur_year+i] * cohort_smoking[cur_year+i]
#                no_cpd += demographics[cur_year+i] * cpd_smoking[cur_year+i]
                no_cpd += 0
                for j in range(int(demographics[cur_year+i] * cohort_smoking[cur_year+i]/100)):
                    smokers.append(cur_year+i)
                for j in range(int(demographics[cur_year+i] * cpd_smoking[cur_year+i]/100)):
                    cigs.append(cur_year+i)
            except KeyError:
                pass

#        print(f'{year-cur_year} - {year - cur_year + interval-1}: {int(no_smokers)}. {int(no_cpd)}')
        print(f'{cur_year} - {cur_year + interval-1}: {int(no_smokers)}. {int(no_cpd)}. Cigs per day: {np.round(no_cpd/no_smokers, 1)}')

    print(f'Mean Smoker Age: {np.round(year - np.mean(smokers) , 1)}. Median Age: {np.round(year - np.median(smokers), 1)}')
    print(f'Mean Cig Age: {np.round(year - np.mean(cigs) , 1)}. Median Age: {np.round(year - np.median(cigs),1 )}')



def load_smoking_data(year=1940, male_or_female='m', pack_per_day_data=False):

    assert (male_or_female in ['m', 'f'])

    if male_or_female == 'm':
        male_or_female = 'male'
    else:
        male_or_female = 'female'

    df = pd.read_csv(f'{male_or_female}_smoking_prevalence.csv')

    df_cpd = pd.read_csv(f'{male_or_female}_cigs_per_day.csv')

    cohort_data = {}
    row = df.loc[df['Period (calendar year)'] == year]
    for cohort_year in row:
        if cohort_year == 'Period (calendar year)': continue

        if row[cohort_year].iloc[0] == '-':
            smoking_prevalence = 0.0
        else:
            smoking_prevalence = float(row[cohort_year].iloc[0]) / 100

        if pack_per_day_data and smoking_prevalence > 0:
            try:
                cpd = float(df_cpd.loc[df['Period (calendar year)'] == year][cohort_year].iloc[0])
            except:
                cpd = 0
            smoking_prevalence *= cpd

        for i in range(5):
            cohort_data[int(cohort_year)+i] = smoking_prevalence


    return cohort_data


def load_demographics_data(year=1940, male_or_female='m'):

    df = pd.read_csv('demographics.csv')

    out_dict = {}
    for row in df.iterrows():
        if row[1]['Age'] in ['85+', '100+']:
            continue

        age = int(row[1]['Age'])
        birth_year = int(year) - age

        try:
            number_of_people = row[1][f'{year}{male_or_female}']
        except:
            print("here")
            embed()

        out_dict[birth_year] = number_of_people

    return out_dict





if __name__ == "__main__":

    for sex in ['m']:
        for year in [1940, 1960, 1990, 2010]:
            calculate_cohort_data(year, sex, interval=1, pack_per_day_data=True)
