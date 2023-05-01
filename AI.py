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

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pickle

rng = np.random.default_rng()

data = pd.read_csv("winequality-red.csv", sep=",")
data = data.drop("residual sugar", axis=1)
data = data.to_numpy()

#Wines ranked at 3, 4, and 8 are outliers, so they'll be ignored.
data = np.delete(data, np.where(data[:, -1] == 3), axis=0)
data = np.delete(data, np.where(data[:, -1] == 4), axis=0)
data = np.delete(data, np.where(data[:, -1] == 8), axis=0)

rng.shuffle(data)
v_data = data[:120, :]
training_data = data[120:, :]

validation_inputs, validation_outputs = v_data[:, :-1], v_data[:, -1]
training_inputs, training_outputs = training_data[:, :-1], training_data[:, -1]
neigh = KNeighborsClassifier(n_neighbors=1, p=1, weights="distance", leaf_size=70)
neigh.fit(training_inputs, training_outputs)

accuracy = neigh.score(validation_inputs, validation_outputs)
print("The accuracy for this AI is " + str(accuracy))

pkl_filename = "wine_AI.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(neigh, file)
