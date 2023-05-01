#This file does an ANOVA test to see what features can be removed

import pandas as pd
from sklearn.feature_selection import f_classif

data = pd.read_csv("winequality-red.csv", sep=",")
data = data.to_numpy()

anova = f_classif(data[:, :-1], data[:, -1])

print("The f-statistics: ")
print(anova[0])
print("The p-values: ")
print(anova[1])
