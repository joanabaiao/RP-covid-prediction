import pandas as pd
from datetime import datetime
import numpy as np

patient_info = pd.read_csv('dataset.csv')

patient_info['age'] = pd.to_numeric(patient_info['age'].str.rstrip('s'))

patient_info.sex.replace('female', 1, inplace=True)
patient_info.sex.replace('male', 2, inplace=True)

# State
patient_info.state.replace('released', 1, inplace=True)
patient_info.state.replace('isolated', 2, inplace=True)
patient_info.state.replace('deceased', 2, inplace=True)

# Countries
countries = {}
for i, country in enumerate(patient_info.country.unique(), start=1):
    countries.setdefault(country, i)

patient_info["country"] = patient_info["country"].apply(lambda x: countries[x])

# Provinces
provinces = {}
for i, province in enumerate(patient_info.province.unique(), start=1):
    provinces.setdefault(province, i)

patient_info["province"] = patient_info["province"].apply(lambda x: provinces[x])

# City
cities = {}
for i, city in enumerate(patient_info.city.unique(), start=1):
    cities.setdefault(city, i)

patient_info["city"] = patient_info["city"].apply(lambda x: cities[x])

# Cases
cases = {}
for i, case in enumerate(patient_info.infection_case.unique(), start=1):
    cases.setdefault(case, i)

patient_info["infection_case"] = patient_info["infection_case"].apply(lambda x: cases[x])

print(patient_info.info())

patient_info.to_csv('datasetA.csv', index=False)



