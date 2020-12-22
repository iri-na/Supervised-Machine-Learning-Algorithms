import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "absences", "failures"]]

predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# keep this outside the loop so that we can still access the test data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

'''
acc = 0 # accuracy
while acc < 0.95:
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1) #10% of data split off into test samples

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

# save model so that it doesn't have to be trained again
with open("studentmodel.pickle", "wb") as f:
    pickle.dump(linear, f)
'''

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

'''
predictions = linear.predict(x_test)

# show the predicted G3, the attributes and the true value of G3
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
'''

p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("G3")
pyplot.show()
