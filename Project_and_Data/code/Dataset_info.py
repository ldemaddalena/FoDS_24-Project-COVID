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

