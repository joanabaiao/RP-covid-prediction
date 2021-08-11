import pandas as pd
from datetime import datetime
import numpy as np

patient_info = pd.read_csv('./Dataset/PatientInfo.csv').drop(columns=['patient_id'])
timeAge = pd.read_csv('./Dataset/TimeAge.csv')
timeSex = pd.read_csv('./Dataset/TimeGender.csv')

# print(patient_info.info())
# print(patient_info.isna().sum())

# Corrigir dados
patient_info.loc[patient_info['province'] == 'Gwangju', 'city'] = 'Gwangju'
patient_info.loc[patient_info["city"] == 'Guri', "city"] = 'Guri-si' # typo
patient_info.loc[patient_info["city"] == 'Sejong', "province"] = 'Sejong' 
patient_info.loc[patient_info["city"] == 'Yangpyeong-si', "city"] = 'Yangpyeong-gun' # typo
patient_info.loc[patient_info["city"] == 'Suwon', "city"] = 'Suwon-si' # typo
patient_info.loc[patient_info["city"] == 'Dalsung-gun', "city"] = 'Dalseong-gun' # typo
patient_info.loc[patient_info["city"] == 'Kyeongsan-si', "city"] = 'Gyeongsan-si' # typo
patient_info.loc[patient_info["city"] == 'Icheon-dong', ["province", "city"]] = ['Seoul', 'Yongsan-gu'] # Icheon-dong é uma sub-parte de Yongsan-gu, que pertence a Seoul
patient_info.loc[patient_info["city"] == 'sankyeock-dong', "city"] = 'Daegu' # Sankyeock-dong não existe - metemos o nome da provincia
patient_info.loc[patient_info["city"] == 'Yeongcheon-si', "province"] = 'Gyeongsangbuk-do'
patient_info.loc[patient_info["city"] == 'Gyeongsan-si', "province"] = 'Gyeongsangbuk-do'

patient_info_grouped = patient_info.groupby(["sex", "age", "province"]).size().reset_index(name="size")

# TIME AGE
unique_ages = len(timeAge["age"].unique())
for i in reversed(range(unique_ages, len(timeAge))):
    timeAge["confirmed"][i] = timeAge["confirmed"][i] - timeAge["confirmed"][i-unique_ages]
    timeAge["deceased"][i] = timeAge["deceased"][i] - timeAge["deceased"][i-unique_ages]

# TIME GENDER
for i in reversed(range(2, len(timeSex))):
    timeSex["confirmed"][i] = timeSex["confirmed"][i] - timeSex["confirmed"][i-2]
    timeSex["deceased"][i] = timeSex["deceased"][i] - timeSex["deceased"][i-2]


def getAge(date, state):

    age_crop = timeAge[timeAge["date"] == date]
    if(state == 'deceased'):
        cases = age_crop.groupby(["age"]).deceased.sum()
    else:
        cases = age_crop.groupby(["age"]).confirmed.sum()

    age = cases.idxmax()

    return age


def getSex(date, state):

    sex_crop = timeSex[timeSex["date"] == date]
    if(state == 'deceased'):
        cases = sex_crop.groupby(["sex"]).deceased.sum()
    else:
        cases = sex_crop.groupby(["sex"]).confirmed.sum()

    sex = cases.idxmax()

    return sex


def getCity(age, sex, province, date):

    patient_info_city = patient_info.groupby(["sex", "age", "province", "city"]).size().reset_index(name="size")

    subset = patient_info_city.loc[(patient_info_city['age'] == age) & (patient_info_city['sex'] == sex) &
                                   (patient_info_city['province'] == province), ['city', 'size']]

    # Casos em que um paciente é o unico com esse genero e idade numa cidade com NaN
    if(subset.empty):
        subset = patient_info_city.loc[(patient_info_city['province'] == province), ['city', 'size']]

    idx = subset.loc[subset['size'] == subset['size'].max()].index[0]
    city = subset['city'][idx]

    # Substituir restantes etc pela cidade que aparece mais vezes numa determinada provincia
    if city == 'etc':
        patient_info_city = patient_info[patient_info['province'] == province]
        patient_info_city = patient_info_city[patient_info_city['city'] != 'etc']
        city = str(patient_info_city["city"].mode().values[0])

    return city


def getInfectionCase(province, city, date):

    patient_info_case = patient_info.groupby(["confirmed_date", "province", "city", "infection_case"]).size().reset_index(name="size")
    patient_info_case = patient_info_case[patient_info_case["infection_case"] != 'etc']

    subset = patient_info_case.loc[(patient_info_case['confirmed_date'] == date) & (patient_info_case['city'] == city) &
                                   (patient_info_case['province'] == province), ['infection_case', 'size']]

    # Casos em que um paciente é o unico nessa cidade, provincia e data com infection_case NaN
    if (subset.empty):
        infection_case = str(patient_info["infection_case"].mode().values[0])
    else:
        idx = subset.loc[subset['size'] == subset['size'].max()].index[0]
        infection_case = subset['infection_case'][idx]

    return infection_case


for i in range(len(patient_info)):

    date = patient_info["confirmed_date"][i]
    deceased_date = patient_info["deceased_date"][i]
    city = patient_info["city"][i]
    age = patient_info["age"][i]
    sex = patient_info["sex"][i]
    province = patient_info["province"][i]
    infection_case = patient_info["infection_case"][i]
    state = patient_info["state"][i]

    if((pd.isnull(age) or pd.isnull(sex)) and date >= '2020-03-02'):

        if pd.isnull(age):
            if (not pd.isnull(deceased_date)):
                patient_info.at[i, "age"] = getAge(deceased_date, state)
            else:
                patient_info.at[i, "age"] = getAge(date, state)

        if pd.isnull(sex):
            if (not pd.isnull(deceased_date)):
                patient_info.at[i, "sex"] = getSex(deceased_date, state)
            else:
                patient_info.at[i, "sex"] = getSex(date, state)

    else:  # patient_info com confirmed_date < '2020-03-02' & age ou sex nan

        if(pd.isnull(age) and not pd.isnull(sex)):

            subset = patient_info_grouped.loc[(patient_info_grouped['sex'] == sex)
                                              & (patient_info_grouped['province'] == province), ['age', 'size']]

            idx = subset.loc[subset['size'] == subset['size'].max()].index[0]
            patient_info.at[i, "age"] = subset['age'][idx]

            # print(patient_info.at[i, "age"])

        elif(not pd.isnull(age) and pd.isnull(sex)):

            subset = patient_info_grouped.loc[(patient_info_grouped['age'] == age)
                                              & (patient_info_grouped['province'] == province), ['sex', 'size']]

            idx = subset.loc[subset['size'] == subset['size'].max()].index[0]
            patient_info.at[i, "sex"] = subset['sex'][idx]

            # print(patient_info.at[i, "sex"])

        elif(pd.isnull(age) and pd.isnull(sex)):

            subset = patient_info_grouped.loc[(
                patient_info_grouped['province'] == province), ['sex', 'age', 'size']]

            idx = subset.loc[subset['size'] == subset['size'].max()].index[0]

            patient_info.at[i, "sex"] = subset['sex'][idx]
            patient_info.at[i, "age"] = subset['age'][idx]

            # print(patient_info.at[i, "sex"], patient_info.at[i, "age"])

    age = patient_info["age"][i]
    sex = patient_info["sex"][i]

    if (pd.isnull(city) or city == 'etc'):
        patient_info.at[i, 'city'] = getCity(age, sex, province, date)
        city = patient_info.at[i, 'city']

    if (pd.isnull(infection_case) or infection_case == 'etc'):
        patient_info.at[i, 'infection_case'] = getInfectionCase(province, city, date)


patient_info["contact_number"].fillna(patient_info["contact_number"].mode().values[0], inplace=True)
patient_info["confirmed_date"].fillna(str(patient_info["confirmed_date"].mode().values[0]), inplace=True)
patient_info["infected_by"].fillna(0, inplace=True)

date_format = "%Y-%m-%d"
patient_info["confirmed_date"] = patient_info["confirmed_date"].apply(pd.to_datetime, errors='coerce', format=date_format)
patient_info["confirmed_date"] = (patient_info["confirmed_date"].apply(lambda x: datetime.timestamp(x)))

patient_info["deceased_date"].fillna(str("1970-01-01"), inplace=True)
patient_info["deceased_date"] = patient_info["deceased_date"].apply(pd.to_datetime, errors='coerce', format=date_format)
patient_info["deceased_date"] = patient_info["deceased_date"].apply(lambda x: datetime.timestamp(x))
patient_info["deceased_date"] = patient_info["deceased_date"].replace(-3600, 0)

patient_info["released_date"].fillna(str("1970-01-01"), inplace=True)
patient_info["released_date"] = patient_info["released_date"].apply(pd.to_datetime, errors='coerce', format=date_format)
patient_info["released_date"] = patient_info["released_date"].apply(lambda x: datetime.timestamp(x))
patient_info["released_date"] = patient_info["released_date"].replace(-3600, 0)

patient_info.loc[patient_info['symptom_onset_date'] == ' ', 'symptom_onset_date'] = np.NaN
patient_info["symptom_onset_date"].fillna(str("1970-01-01"), inplace=True)
patient_info["symptom_onset_date"] = patient_info["symptom_onset_date"].apply(pd.to_datetime, errors='coerce', format=date_format)
patient_info["symptom_onset_date"] = patient_info["symptom_onset_date"].apply(lambda x: datetime.timestamp(x))
patient_info["symptom_onset_date"] = patient_info["symptom_onset_date"].replace(-3600, 0)

patient_info.contact_number.replace('-', patient_info["contact_number"].mode().values[0], inplace=True)
patient_info.loc[patient_info['contact_number'] == '6100000099', 'contact_number'] = patient_info["contact_number"].mode().values[0]
patient_info.loc[patient_info['contact_number']== '6100000098', 'infected_by'] = '6100000098'
patient_info.loc[patient_info['contact_number']== '6100000098', 'contact_number'] = patient_info["contact_number"].mode().values[0]
patient_info.loc[patient_info['contact_number']== '1000000796', 'infected_by'] = '1000000796'
patient_info.loc[patient_info['contact_number']== '1000000796', 'contact_number'] = patient_info["contact_number"].mode().values[0]
patient_info.loc[patient_info['infected_by'] == '1500000050, 1500000055', 'infected_by'] = '1500000050'

print(patient_info.info())
print(patient_info.isna().sum())

patient_info.to_csv('./dataset.csv', index=False)
