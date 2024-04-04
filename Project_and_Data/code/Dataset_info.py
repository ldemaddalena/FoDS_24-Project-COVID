"""
Hallo zusammen, Ich habe hier versucht eine File aufzusetzen, woran wir alle arbeiten können. 
Man kann hier sehen was bereits gemacht wurde und vor allem auch wie es gemacht wurde um alles einheitlich zu gestalten.
Das Dataset ist im selben Ordner wie diese File.

Environment:
python == 3.11
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
#print(data.info())
#print(data.describe())
#print(data.head())
