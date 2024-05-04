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
sklearn=1.4.2
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from scipy.stats import boxcox

data = pd.read_csv("../data/OWID-covid-data-28Feb2023.csv")



"""1. GETTING AN OVERVIEW:"""


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
num_data = data[float_data] 



"""2. DATA PREPROCESSING:"""

"""@FLORIN: hier schauen, was man auch noch rausnehmen könnte, teilweise wurden Dinge doppelt gemacht."""

"""2.1: MISSING VALUES:"""

missing_values_per_variable = data.isnull().sum()
# print(missing_values_per_variable)
variables_with_no_missing_values = missing_values_per_variable[missing_values_per_variable == 0].index.tolist()
# print(variables_with_no_missing_values)
# print(missing_values_per_variable.sort_values(ascending=False))
# print(f'first entry: {data.date.min()}, last entry: {data.date.max()}')
# Observations from 01.01.2020 (before pandemic) to 27.02.2023
# -> span of over 3 years


# print(data['iso_code'].nunique())
# observations from 248 regions coded in ISO 3166-1 alpha-3 format
# but also specially defined regions defined by OurWorldInData:
owid_codes= data[data['iso_code'].str.startswith('OWID')]['iso_code'].unique()
# print(owid_codes)


missing_values_count = data.isnull().sum().sort_values(ascending=False)
# print(missing_values_count.head(20))
# no missing values in date, location, and iso code
# most in mortality and icu metrics


"""2.2: Adjusting datatypes:"""

#Converting Date to a datetime variable
data["date"] = pd.to_datetime(data["date"])
# print(data.dtypes)
# Date is saved as object
#->convert to daytime format
data['date']=pd.to_datetime(data['date'])


"""2.3: Data cleaning"""

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
#percentage of OWID data:
# print((data_OWID.shape[0]/(data_no_OWID.shape[0]+ data_OWID.shape[0]))*100, "%")
#I think all this OWID should be interpreted seperately from the rest of the data --> ASK TA
#####################################################################################################################
#USE DATA WITH RESPECT TO OWID AND NON OWID!!


#Looking for duplicate rows:
duplicate_rows = data[data.duplicated()]
if not duplicate_rows.empty:
    print("Duplicate Rows:")
    print(duplicate_rows)
else:
    print("No duplicate rows found.")


#Looking for outliers, will be more useful after handling missing data:
"""
summary_stats = data.describe()
Q1 = summary_stats.loc['25%']
Q3 = summary_stats.loc['75%']
IQR = Q3 - Q1
# Define outliers using IQR rule
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Align DataFrame columns with summary statistics columns
data, lower_bound = data.align(lower_bound, axis=1, join='inner')
data, upper_bound = data.align(upper_bound, axis=1, join='inner')
outliers = data[(data < lower_bound) | (data > upper_bound)].dropna(axis=1, how='all')
if not outliers.empty:
    print("Outliers:")
    print(outliers)
else:
    print("No outliers found.")
"""
#Remove comments above later to find outliers


"""2.4: Feature scaling (Standardization or normalization)"""


#Testing whether the data is normally distributed or not:
#ONLY NUMERICAL DATA
#Using the anderson darling test, since the shapiro wilks is not good for data n > 5000 
distribution_results = {}

for column in num_data.columns:
    result = sts.anderson(num_data[column].dropna(), dist='norm')  ############################### I had to drop all na for it to work!!! -> maybe imputation needed
    test_stat = result.statistic 
    critical_val = result.critical_values
    #print(critical_val)                                
    #print(test_stat)                           
    if test_stat > critical_val[2]:
        result = "not normal"
    else:
        result = "normal"    
    distribution_results[column] = result


not_normal = []
for key, value in distribution_results.items():
    if value == "not normal":
        not_normal.append(key)
if len(not_normal) == 0:
    print("All numerical data seems to be normally distributed.")
elif len(not_normal) == len(num_data.columns):
    print("No variable is normally distributed.")
else:
    print("All data but", not_normal, "seems to be normally distributed")


# NOT A SINGLE VARIABLE IS NORMALLY DISTRIBUTED, SEE ALSO VISUALLY BELOW
"""
columns_per_row = 3
num_columns = len(num_data.columns)
num_rows = (num_columns - 1) // columns_per_row + 1
fig_width = 6 * columns_per_row
fig_height = 4 * num_rows
fig, axs = plt.subplots(num_rows, 4, figsize=(fig_width, fig_height))
axs = axs.flatten()
for i, column in enumerate(num_data.columns):
    ax = axs[i]
    ax.hist(num_data[column], bins=35, color='skyblue', edgecolor='black')
    ax.set_title(column)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig("../output/distributions_of_variables.png")
"""

#Numercial features:
#WE HAVE HUGE OUTLIERS THAT MEANS I WILL HAVE TO NORMALIZE IT WITH A METHOD ROBUST TO OUTLIERS
#I will be using some sort of transformation for the data (BOX COX, LOG tranform etc.)
#cannot work on data until nan is handled

    

"""3: SPECIAL DATA FRAME (USEFUL FORMAT) CREATION:"""


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
#
# print(data_all_time_per_iso)
#data_all_time_per_iso is a helpful dataframe containing averaged and total data per country, go and print the two lines above for more information


"""4. VISUALIZATIONS:""" 

"""

Unten auch schon?

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
"""


################# CASES VISUALIZATION ####################



# 1.1) Cases over time
# Problem: Most countries did not publish covid numbers on the weekends
# Effect: This leads to an oscillating graph for daily new cases
# This can be solved with weekly new cases
# either extract data from new_daily or use the smoothed cases count
daily_cases = data.groupby('date')['new_cases'].sum()
data_march= data[(data['date']>'2022-03-01')&(data['date']<'2022-04-01')]
daily_cases_march = data_march.groupby('date')['new_cases'].sum()
daily_cases_weekly= data.groupby(pd.Grouper(key='date', freq='W')).sum()['new_cases']

# Plotting
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
ax4.plot(data.groupby('date')['new_cases_smoothed'].sum(), label='New Cases Per Day, Daily turned to weekly')
ax4.set(title='COVID-19 New Cases Per Day',
        xlabel='Date',
        ylabel='Number of New Cases'
        )

ax4.legend()
plt.subplots_adjust(hspace=0.5)
plt.savefig('../output/newcasestotal.png')

# 1.2) Cases over time by continent compact
grouped_data = data[~(data['continent']==0)].groupby(['date', 'continent'])['new_cases_smoothed'].sum().reset_index()
plt.figure(figsize=(12, 8))
sns.lineplot(x='date', y='new_cases_smoothed', hue='continent', data=grouped_data)
plt.title('New COVID-19 Cases by Continent Over Time')
plt.xlabel('Date')
plt.ylabel('Number of New Cases')
plt.legend(title='Continent')
plt.grid(True)  
plt.savefig('../output/newcasesbycontinent')

#Cases over time by continent in subplots

grouped_data = data[~(data['continent']==0)].groupby(['continent','date'])['new_cases_smoothed'].sum().reset_index()
continents_of_interest=['North America','South America','Asia','Europe','Oceania','Africa']
fig, axs = plt.subplots(1,6,figsize=(24,8), sharey=True)
for i,continent in enumerate(continents_of_interest):
        sns.lineplot(x='date', y='new_cases_smoothed',  data=grouped_data[grouped_data['continent']==continent],ax=axs[i])
        axs[i].set_xlabel('Date')
        axs[i].set_ylabel('New Cases')
        axs[i].set_title(continent)
        axs[i].tick_params(axis='x', rotation=90)
        axs[i].grid(True)
plt.suptitle('New Cases by Continent')
plt.savefig('../output/casesbycontinentsp')

data.fillna(0, inplace=True)

#2.1) Total Cases vs Total Cases per Million by Continent
cases=['total_cases','total_cases_per_million']
fig, axs= plt.subplots(2,1,figsize=(10,10))
data_nozero=data[~(data['continent']==0)]
for i, variable in enumerate(cases):
        top_countries = data_nozero.groupby('continent')[variable].max().nlargest(10)
        print(top_countries)
        top_countries.plot(kind='bar', ax=axs[i],grid=True)
        axs[i].set_xlabel('Continent')
        axs[i].set_ylabel(variable)
plt.suptitle('Total Cases vs Total Cases per Million')
plt.subplots_adjust(hspace=0.5)
plt.savefig('../output/casesbycontinent.png')

#2.2) Total Cases vs Total Cases per Million by Country
cases=['total_cases','total_cases_per_million']
fig, axs= plt.subplots(2,1,figsize=(10,10))
data_noowid=data[~data['iso_code'].str.startswith("OWID_")]
for i, variable in enumerate(cases):
        top_countries = data_noowid.groupby('iso_code')[variable].max().nlargest(10)
        print(top_countries)
        top_countries.plot(kind='bar', ax=axs[i],grid=True)
        axs[i].set_xlabel('Country')
        axs[i].set_ylabel(variable)
plt.suptitle('Total Cases vs Total Cases per Million')
plt.subplots_adjust(hspace=0.5)
plt.savefig('../output/casesbycountry.png')

# 3) Scatter Plots with variables of interest
variables_of_interest = [
    'icu_patients_per_million', 'total_tests_per_thousand', 'total_vaccinations_per_hundred',
    'stringency_index', 'hospital_beds_per_thousand', 'aged_65_older'
]
fig, axs = plt.subplots(3, 2, figsize=(15, 15))
axs = axs.flatten()
for i, variable in enumerate(variables_of_interest):
    sns.scatterplot(x=variable, y='total_cases_per_million', data=data, ax=axs[i], markers='.')
    axs[i].set_title(f'Total Cases per Million vs. {variable}')
    axs[i].set_xlabel(variable)
    axs[i].set_ylabel('Total Cases per Million')
plt.tight_layout()
plt.savefig('../output/totcasescorr.png')


#Plotting new deaths 

daily_deaths = data.groupby('date')['new_deaths'].sum()
daily_deaths_smoothed = data.groupby('date')['new_deaths_smoothed'].sum()

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,20))

ax1.plot(daily_deaths, label='New Deaths Per Day')
ax1.set(title='COVID-19 New Deaths Per Day', xlabel='Date', ylabel='Number of New Deaths')
ax1.legend()

ax2.plot(daily_deaths_smoothed, label='New Deaths Per Day, Smoothed Data')
ax2.set(title='COVID-19 New Deaths Per Day', xlabel='Date', ylabel='Number of New Deaths')
ax2.legend()

plt.savefig("../output/new_deaths.png")

#Trying to find out if there is a difference in deaths per million among the continents.

new_deaths_per_million_continent = data.groupby(['date', 'continent'])['new_deaths_per_million'].sum().reset_index()
new_deaths_per_million_continent_smoothed = data.groupby(['date', 'continent'])['new_deaths_smoothed_per_million'].sum().reset_index()
total_cases_per_continent = data.groupby(['date', 'continent'])['total_cases_per_million'].sum().reset_index()
#print(new_deaths_per_million_continent)
#print(new_deaths_per_million_continent_smoothed)

continents_deaths = new_deaths_per_million_continent['continent'].unique()

#Creating suplots per Continent with the real counts
fig, axes = plt.subplots(len(continents_deaths), 1, figsize=(10, 6*len(continents_deaths)), sharex=True)

# Iterate over each continent and create a subplot
for i, continent in enumerate(continents_deaths):
    continent_data = new_deaths_per_million_continent[new_deaths_per_million_continent['continent'] == continent]
    sns.lineplot(data=continent_data, x='date', y='new_deaths_per_million', ax=axes[i])
    axes[i].set_title(f'New Deaths per Million in {continent} Over Time')
    axes[i].set_ylabel('New Deaths per Million')
    axes[i].set_xlabel('Date')
    axes[i].grid(True)
    
plt.tight_layout()

plt.savefig("../output/NewDeathsPerMillion_Subplots.png")



#Using the smoothed count
fig, axes = plt.subplots(len(continents_deaths), 1, figsize=(10, 6*len(continents_deaths)), sharex=True)

for i, continent in enumerate(continents_deaths):
    continent_data = new_deaths_per_million_continent_smoothed[new_deaths_per_million_continent_smoothed['continent'] == continent]
    sns.lineplot(data=continent_data, x='date', y='new_deaths_smoothed_per_million', ax=axes[i])
    axes[i].set_title(f'New Deaths per Million in {continent} Over Time')
    axes[i].set_ylabel('New Deaths per Million')
    axes[i].set_xlabel('Date')
    axes[i].grid(True)
    
plt.tight_layout()

plt.savefig("../output/NewDeathsPerMillionSmoothed_Subplots.png")


#Trying to make a statement about the severity by analyzing the new deaths per million divided by the total cases
# Merge the new deaths and total cases. Calculate new deaths per million divided by total cases per million
death_cases_merge = pd.merge(new_deaths_per_million_continent_smoothed, total_cases_per_continent, on=['date', 'continent'])
death_cases_merge['deaths_to_cases_ratio'] = death_cases_merge['new_deaths_smoothed_per_million'] / death_cases_merge['total_cases_per_million']


fig, axes = plt.subplots(len(continents_deaths), 1, figsize=(10, 6*len(continents_deaths)), sharex=True)

for i, continent in enumerate(continents_deaths):
    continent_data = death_cases_merge[death_cases_merge['continent'] == continent]
    sns.lineplot(data=continent_data, x='date', y='deaths_to_cases_ratio', ax=axes[i])
    axes[i].set_title(f'New Deaths per Million divided by Total Cases per Million in {continent} Over Time')
    axes[i].set_ylabel('New Deaths per Million / Total Cases per Million')
    axes[i].set_xlabel('Date')
    axes[i].grid(True)
    
plt.tight_layout()

plt.savefig("../output/NewDeathsToCasesRatio_Subplots.png")
