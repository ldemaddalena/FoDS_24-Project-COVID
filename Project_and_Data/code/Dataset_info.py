"""
Hallo zusammen, Ich habe hier versucht eine File aufzusetzen, woran wir alle arbeiten können. 
Man kann hier sehen was bereits gemacht wurde und vor allem auch wie es gemacht wurde um alles einheitlich zu gestalten.
Das Dataset ist im selben Ordner wie diese File.

- Ich habe einige print statements aukommentiert, alle auskommentierten code statements, welche man auch runnen könnte haben einen Abstand nach #
  Alle Textkommentare sind direkt nach dem #
- Alle Übertitel sind mit """ """ gekennzeichnet
Environment:
python == 3.11.8
matplotlib=3.8.3
numpy=1.26.0
pandas=2.1.1
scipy=1.12.0
seaborn=0.13.2
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts

data = pd.read_csv("../data/OWID-covid-data-28Feb2023.csv")



"""GETTING AN OVERVIEW:"""

# print(data.info())
# print(data.describe())
# print(data.head())

# Geographic and temporal data: ISO code, continent, location, and date.
# Case and death counts: Total and new cases/deaths, both raw and per million people.
# Hospitalization data: ICU and hospitalization counts, also provided per million.
# Testing and vaccination data: Including total tests conducted and vaccination rates.
# Miscellaneous metrics: Such as reproduction rates, population statistics, and excess mortality.

#Number of locations and what locations:
locations = data["location"].unique()  # -> not all are really countries (e.g. asia, africa etc.)
# print(countries)
n_locations = data["location"].nunique()
# print(n_locations)

#Doing the same for continents:
continents = data["continent"].unique() # -> one of the entries is nan? -> find out
# print(continents)
n_continents = data["continent"].nunique()
# print(n_continents)

#Locations per continent: (also to find all locations in nan)
loc_in_con = data.groupby("continent")["location"].apply(list)
# print(loc_in_con)  # -> now there is no more nan?


print(f'first entry: {data.date.min()}, last entry: {data.date.max()}')
# Observations from 01.01.2020 (before pandemic) to 27.02.2023
# -> span of over 3 years

print(data['iso_code'].nunique())
# observations from 248 regions coded in ISO 3166-1 alpha-3 format
# but also specially defined regions defined by OurWorldInData:
owid_codes= data[data['iso_code'].str.startswith('OWID')]['iso_code'].unique()
print(owid_codes)

missing_values_count = data.isnull().sum().sort_values(ascending=False)
print(missing_values_count.head(20))
# no missing values in date, location, and iso code
# most in mortality and icu metrics

"""DATA PREPROCESSING:"""

#Total number of data entries
print("Number of observations:", data.shape[0])
#Datatypes of each variable:
# print(data.dtypes)
#Taking a closer look at data types:
int_data = data.select_dtypes(include=("int64")).columns
# print("INTEGERS: ", int_data)
#We dont have any integers? -> We do but they are just not saved as such.
float_data = data.select_dtypes(include=("float64")).columns
# print("FLOATS: ", float_data)
object_data = data.select_dtypes(include=("object")).columns
# print("OBJECTS:", object_data)

# Date is saved as object
#->convert to daytime format
data['date']=pd.to_datetime(data['date'])



"""VISUALIZATIONS:""" 

#Trying to find out what continent had how many cases total:
cases_per_con = data.groupby("continent")["total_cases"].sum().reset_index()
plt.figure(figsize=(12,8))
fig1= sns.barplot(data=cases_per_con, x="continent", y="total_cases")
plt.title("total cases per continent")
plt.savefig("../output/tot_cases_per_cont.png")

cases_per_con = data.groupby("continent")["total_cases_per_million"].sum().reset_index()
plt.figure(figsize=(12,8))
fig2= sns.barplot(data=cases_per_con, x="continent", y="total_cases_per_million")
plt.title("total cases per continent per million")
plt.savefig("../output/tot_cases_per_cont_per_million.png")


# Problem: Most countries did not publish covid numbers on the weekends
# Effect: This leads to an oscillating graph for daily new cases
# This can be solved with weekly new cases
# either extract data from new_daily or use the smoothed cases count
daily_cases = data.groupby('date')['new_cases'].sum()

data_march= data[(data['date']>'2022-03-01')&(data['date']<'2022-04-01')]
daily_cases_march = data_march.groupby('date')['new_cases'].sum()
daily_cases_weekly= data.groupby(pd.Grouper(key='date', freq='W')).sum()['new_cases']
print(daily_cases_weekly)

missing_values_count = data.isnull().sum().sort_values(ascending=False)
print(missing_values_count.head(20))
# no missing values in date, location, and iso code
# most in mortality and icu metrics

print(daily_cases)

# Plotting the new cases per day
fig, (ax1,ax2,ax3, ax4)=plt.subplots(4,1,figsize=(10,20))

ax1.plot(daily_cases, label='New Cases Per Day')
ax1.set(title='COVID-19 New Cases Per Day',
        xlabel='Date',
        ylabel='Number of New Cases'
        )
ax1.legend()

ax2.plot(daily_cases_march, label='New Cases Per Day, March 2022')
ax2.set(title='COVID-19 New Cases Per Day',
        xlabel='Date',
        ylabel='Number of New Cases'
        )
ax2.legend()

ax3.plot(daily_cases_weekly, label='New Cases Per Day, Daily turned to weekly')
ax3.set(title='COVID-19 New Cases Per Day',
        xlabel='Date',
        ylabel='Number of New Cases'
        )

ax3.legend()

ax4.plot(data.groupby('date')['new_cases_smoothed'].sum(), label='New Cases Per Day, Using the smoothed count')
ax4.set(title='COVID-19 New Cases Per Day',
        xlabel='Date',
        ylabel='Number of New Cases'
        )

ax4.legend()

plt.subplots_adjust(hspace=0.5)

plt.savefig("../output/cases_trend.png")
