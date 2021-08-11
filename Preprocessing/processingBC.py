import pandas as pd
from datetime import datetime
import numpy as np

patient_info = pd.read_csv('dataset.csv')
patient_info.rename(columns = {'confirmed_date': 'date'}, inplace = True)

# seven_days = datetime.timestamp(pd.to_datetime("1970-01-08", format="%Y-%m-%d"))
# patient_info["contact_date"] = patient_info["confirmed_date"].apply(lambda x: x - seven_days)

policy = pd.read_csv('./Dataset/Policy.csv')
weather = pd.read_csv('./Dataset/Weather.csv').drop(columns=['code'])
region = pd.read_csv('./Dataset/Region.csv')
# case = pd.read_csv('./Dataset/Case.csv')
# search = pd.read_csv('./Dataset/SearchTrend.csv')
# seoul = pd.read_csv('./Dataset/SeoulFloating.csv')
# time = pd.read_csv('./Dataset/Time.csv')
# timeProvince = pd.read_csv('./Dataset/TimeProvince.csv')

date_format = "%Y-%m-%d"

# ----------------------------- Policy -----------------------------
policy_id_list = [2, 3, 4, 29, 30, 31, 32, 59]
policy = policy[['policy_id', 'start_date', 'end_date']]
policy = policy[policy['policy_id'].isin(policy_id_list)]

policy['start_date'] = policy['start_date'].apply(pd.to_datetime, errors='coerce', format=date_format)
policy['start_date'] = policy['start_date'].apply(lambda x: datetime.timestamp(x))

policy['end_date'].fillna('2020-06-30', inplace=True)
policy['end_date'] = policy['end_date'].apply(pd.to_datetime, errors='coerce', format=date_format)
policy['end_date'] = policy['end_date'].apply(lambda x: datetime.timestamp(x))

alerts = {1:[1579478400, 1580083200], # yellow alert
            2: [1580169600, 1582329600], # orange alert
            3: [1582416000, 1593471600]} # red alert

for k, (s, e) in alerts.items():
    patient_info.loc[patient_info['date'].between(s, e), 'policy_alert'] = k


distancing = {1: [1587337200, 1593471600], # weak distancing measures
            2: [1582934400, 1587250800]} # strong distancing measures

for k, (s, e) in distancing.items():
    patient_info.loc[patient_info['date'].between(s, e), 'policy_distancing'] = k

patient_info["policy_distancing"].fillna(0, inplace=True) # no distancing measures


mask = {1: [1590447600, 1593471600]} # mask obligation

for k, (s, e) in mask.items():
    patient_info.loc[patient_info['date'].between(s, e), 'policy_mask'] = k

patient_info["policy_mask"].fillna(0, inplace=True) # no mask obligation measures


# ----------------------------- Weather -----------------------------
weather_29june = weather[weather["date"] == "2020-06-29"]
weather_29june.loc[weather_29june["date"] == "2020-06-29", "date"] = '2020-06-30'
weather = pd.concat([weather, weather_29june])

weather.loc[weather["province"] == 'Chunghceongbuk-do', "province"] = 'Chungcheongbuk-do' # typo

weather["date"] = weather["date"].apply(pd.to_datetime, errors='coerce', format=date_format)
weather["date"] = (weather["date"].apply(lambda x: datetime.timestamp(x)))

init_date = datetime.timestamp(pd.to_datetime('2020-01-13', format=date_format))
missing_day = datetime.timestamp(pd.to_datetime('2020-06-30', format=date_format))

weather = weather[weather["date"] >= init_date].reset_index()

index_nan = weather[weather["most_wind_direction"].isna()].index[0]
weather.loc[weather["most_wind_direction"].isna(), "most_wind_direction"] = weather["most_wind_direction"].iloc[index_nan + 1] # substituir nan pelo valor do dia a seguir (pq tem dados mais parecidos)

weather_columns = weather.columns.drop(['index', 'date', 'province']).tolist()

new_patient_info = pd.merge(patient_info, weather, on=['date', 'province'], how='left').drop(columns='index')

idx_sejong = new_patient_info[new_patient_info['province']=='Sejong'].index.tolist()
idx_daejeon = weather[weather['province']=='Daejeon'].index.tolist()

for i in idx_sejong:
    for j in idx_daejeon:

        if (new_patient_info.at[i,'date'] == weather.at[j,'date']):

            for column in weather_columns:
                new_patient_info.at[i, column] = weather.at[j, column] # substituir os valores em falta de weather da provincia de Sejong pelos valores da provincia mais próxima (Daejeon)


# ----------------------------- Region -----------------------------

region = region[['province', 'city', 'latitude', 'longitude']]
new_patient_info = pd.merge(new_patient_info, region, on=['province', 'city'], how='left')


# ----------------------------- STR -> NUMBER -----------------------------
countries = {}
for i, country in enumerate(new_patient_info.country.unique(), start=1):
    countries.setdefault(country, i)

new_patient_info["country"] = new_patient_info["country"].apply(lambda x: countries[x])

provinces = {}
for i, province in enumerate(new_patient_info.province.unique(), start=1):
    provinces.setdefault(province, i)

new_patient_info["province"] = new_patient_info["province"].apply(
    lambda x: provinces[x])

cities = {}
for i, city in enumerate(new_patient_info.city.unique(), start=1):
    cities.setdefault(city, i)

new_patient_info["city"] = new_patient_info["city"].apply(lambda x: cities[x])

cases = {}
for i, case in enumerate(new_patient_info.infection_case.unique(), start=1):
    cases.setdefault(case, i)

new_patient_info["infection_case"] = new_patient_info["infection_case"].apply(
    lambda x: cases[x])

new_patient_info['age'] = pd.to_numeric(new_patient_info['age'].str.rstrip('s'))

new_patient_info.sex.replace('female', 1, inplace=True)
new_patient_info.sex.replace('male', 2, inplace=True)

new_patient_info.state.replace('released', 0, inplace=True)
new_patient_info.state.replace('isolated', 1, inplace=True)
new_patient_info.state.replace('deceased', 2, inplace=True)

state = new_patient_info["state"]
new_patient_info = new_patient_info.drop(columns = "state")
new_patient_info["state"] = state

# print(weather.info())
# print(patient_info.info())
# print(new_patient_info.info())
# print(new_patient_info.isna().sum())

# print(weather.isna().sum())
# print(new_patient_info.info())

new_patient_info.to_csv('datasetBC.csv', index=False)