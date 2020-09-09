import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
import warnings
import pickle 
warnings.filterwarnings("ignore")

#Importing dataset
dataset = pd.read_csv('forest_fire.html')
X = dataset.iloc[ ; , 1;-1]
y = dataset.iloc[: , :-1]

#Spliting dataset into Training set and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#Fitting the Logisitic Regression to training set 
from sklearn.linear_model import LogisticRegression
classifer = LogisticRegression(max_iter = 1000)
classifer.fit(X_train, y_train)

#Prediciting the test set result 
y_pred = classifer.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 

#packing the classifier using pickle file so that it can be used anytime without we have to train it again and again.


pickle.dumb(classifer , open('model.pkl', 'wb')
model = pickle.load(open('model.pkl','rb')

print(model.predict([[0,4,2]])) 