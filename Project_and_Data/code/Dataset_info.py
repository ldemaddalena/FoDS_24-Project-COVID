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

#Total number of data entries
# print("Number of observations:", data.shape[0])
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
# print(loc_in_con)  # -> now there is no more nan? there is no location saved under "the continent" nan?

#Missing values:
missing_values_per_variable = data.isnull().sum()
# print(missing_values_per_variable)
variables_with_no_missing_values = missing_values_per_variable[missing_values_per_variable == 0].index.tolist()
# print(variables_with_no_missing_values)
# print(missing_values_per_variable.sort_values(ascending=False))
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

#Datatypes of each variable:
# print(data.dtypes)
#Taking a closer look at data types:
int_data = data.select_dtypes(include=("int64")).columns
# print("INTEGERS: ", int_data)
#We dont have any integers? -> We do but they are just not saved as such.
float_data = data.select_dtypes(include=("float64")).columns
# print("FLOATS: ", float_data)
object_data = data.select_dtypes(include=("object")).columns
print("OBJECTS:", object_data)

#Converting Date to a datetime variable
data["date"] = pd.to_datetime(data["date"])
# print(data.dtypes)

# Date is saved as object
#->convert to daytime format
data['date']=pd.to_datetime(data['date'])


#OWID EXPLANATION
####################################################################################################################
#We have two different styles of saving data, where I think OWID contains the continents and non-OWID contains countries -> I dont think they should be mixed?
data_OWID = data[data['iso_code'].str.contains('OWID')] #Data frame containing OWID only
# print(data_OWID)
data_no_OWID = data.drop(data[data['iso_code'].str.contains('OWID')].index, axis=0) #Data frame containing all but OWID
# print(data_no_OWID)
grouped_OWID_iso = data_OWID.groupby("iso_code")
iso_codes_OWID = list(grouped_OWID_iso.groups.keys()) #trying to find out what OWID is about
# print(iso_codes_OWID)
grouped_OWID_loc = data_OWID.groupby("location")
loc_OWID = list(grouped_OWID_loc.groups.keys()) #OWID contains special summaries of data such as: Low income, European union as a whole etc.
# print(loc_OWID)
#I think all this OWID should be interpreted seperately from the rest of the data --> ASK TA
#####################################################################################################################
#USE DATA WITH RESPECT TO OWID AND NON OWID!!


#Creating a dataframe containing all entries grouped by isocode -> to have data over whole time period:
#getting a dataframe that contains sums and means according to column type
#Will allow us to see what country had the most deaths, deaths per million, cases etc. over the given time period
data_all_time_per_iso = data_no_OWID.groupby("iso_code").agg({col: ['min', 'max'] if col in ['date'] 
                                                              else ('sum' if col in ['new_cases','new_deaths','new_cases_per_million','new_deaths_per_million','new_tests','new_tests_per_thousand','new_vaccinations'] 
                                                                    else 'mean') for col in data_no_OWID.columns
                                                                    if col not in ['iso_code', 'continent', 'location', 'tests_units']
                                                                    and 'total' not in col.lower()
                                                                    and 'smoothed' not in col.lower()
                                                                    and 'excess' not in col.lower()})
# print(data_all_time_per_iso.columns)
# print(data_all_time_per_iso)
#data_all_time_per_iso is a helpful dataframe containing averaged and total data per country, go and print the two lines above for more information

"""VISUALIZATIONS:""" 

#Trying to find out what country had the most cases and cases per million:

tot_cases_per_country = data_all_time_per_iso["new_cases"]
sorted_tot_cases_per_country = tot_cases_per_country.sort_values(ascending=False, by="sum").head(15)
plt.figure(figsize=(12,8))
fig1= sns.barplot(data=sorted_tot_cases_per_country, x="iso_code", y="sum")
plt.title("total cases per country")
plt.savefig("../output/tot_cases_per_country.png")

tot_cases_per_country_per_million = data_all_time_per_iso["new_cases_per_million"]
sorted_tot_cases_per_country_per_million = tot_cases_per_country_per_million.sort_values(ascending=False, by="sum").head(15)
plt.figure(figsize=(12,8))
fig2= sns.barplot(data=sorted_tot_cases_per_country_per_million, x="iso_code", y="sum")
plt.title("total cases per country per million")
plt.savefig("../output/tot_cases_per_country_per_million.png")





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
