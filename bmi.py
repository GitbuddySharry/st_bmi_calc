import streamlit as st
import pandas as pd
import numpy as np
st.title("hey Guys find out your BMI !!")
df=pd.read_csv("/content/500_Person_Gender_Height_Weight_Index.csv")
x=df.iloc[:,[1,2]].values
df=df.replace({'Index' : { 0 : "Extremely_Weak", 1 : "Weak", 2 : "Normal",3:"Overweight",4:"Obesity",5:"Extreme_Obesity" }})
y=df.iloc[:,-1].values
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=23,metric="euclidean")
model.fit(x,y)
xmin=np.min(x,axis=0)
xmax=np.max(x,axis=0)
height=st.slider("Height",float(xmin[0]),float(xmax[0]))
weight=st.slider("Weight",float(xmin[1]),float(xmax[1]))

y_pred=model.predict([[height,weight]])
print ("Your Body mass Index is :")

op=["Extremely_Weak","Weak","Normal","Overweight","Obesity","Extreme_Obesity"]
st.title(op[y_pred[0]])
