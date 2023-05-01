'''
Copyright [2023] [Matteo Raso]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

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
