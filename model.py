import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

df = pd.read_csv(r"C:\Users\neman\Downloads\iris_new.csv")

x = df.drop(["species"], axis = 1)
y = df["species"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=10)

linear_model= LogisticRegression()
linear_model.fit(x_train,y_train)

file = open("iris_model.pkl",'wb')
pickle.dump(linear_model, file)
file.close()

columns_list = x.columns

file = open("iris_columns.obj", "wb")
pickle.dump(columns_list, file)
file.close()