import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import altair as alt
import numpy as np

#Add Header to describe app 
st.markdown("### LinkedIn User Prediction App")

#Education 
educ = st.selectbox("Education level",
                   options = ["Less Than High School",
                             "High School Incomplete",
                             "High School Graduate",
                             "Some college or no degree",
                             "Two Year Degree",
                             "Four Year Degree",
                             "Some Postgraduate",
                             "Postgraduate Degree"])
#Education -> numerical 
if educ == "Less Than High School":
    educ_Num = 1 
elif educ == "High School Incomplete":
    educ_Num = 2 
elif educ == "High School Graduate":
    educ_Num = 3
elif educ == "Some college or no degree":
    educ_Num = 4
elif educ == "Two Year Degree":
    educ_Num = 5
elif educ == "Four Year Degree":
    educ_Num = 6
elif educ == "Some Postgraduate":
    educ_Num = 7
else: 
    educ_Num = 8

#Income
income = st.selectbox("Income Level",
                   options = ["<$10k",
                             "$10k-$20k",
                             "$20k-$30k",
                             "$30k-$40k",
                             "$40k-$50k",
                             "$50k-$75k",
                             "$75k-$100k",
                             "$100k-$150k",
                             "$150k+"])

#Income -> numerical 
if income == "<$10k":
    income_Num = 1 
elif income == "$10k-$20k":
    income_Num = 2 
elif income == "$20k-$30k":
    income_Num = 3
elif income == "$30k-$40k":
    income_Num = 4
elif income == "$40k-$50k":
    income_Num = 5
elif income == "$50k-$75k":
    income_Num = 6
elif income == "$75k-$100k":
    income_Num = 7
elif income == "$100k-$150k":
    income_Num = 8
else: 
    income_Num = 9

#Age
Age = st.number_input('Enter your age',
                min_value=1,
                max_value=98,
                value=1)

#Parent
parent = st.selectbox("Are you a Parent?",
                     options = ["Yes",
                               "No"])
#Parent -> numerical
if parent == "Yes":
    parent_Num = 1
else: 
    parent_Num = 0

#Married
Marital = st.selectbox("Are you Married?",
                     options = ["Yes",
                               "No"])
#Married -> numerical
if Marital == "Yes":
    Marital_Num = 1
else: 
    Marital_Num = 0

#Gender
Gender = st.selectbox("Do you Identify as Male or Female?",
                     options = ["Female",
                               "Male"])
#Gender -> numerical 
if Gender == "Female":
    Gender_Num = 1
else: 
    Gender_Num = 0



###TEST####


#Prediction 
s = pd.read_csv("social_media_usage.csv")
s_CD = pd.DataFrame({
    "education": np.where(s["educ2"]<=8,s["educ2"],np.nan),
    "income": np.where(s["income"]<=9, s["income"],np.nan),
    "age": np.where(s["age"]<=98,s["age"],np.nan),
    "parent":np.where(s["par"]==1,1,0),
    "marital":np.where(s["marital"]==1,1,0),
    "gender":np.where(s["gender"]==2,1,0),
    "sm_li":np.where(s["web1h"]==1,1,0)
})
ss=s_CD.dropna()
y= ss["sm_li"]
X = ss[["education","income","age","parent","marital","gender"]]
##Split Data to Train and Test Sets 
X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   stratify=y,
                                                   test_size=0.2,
                                                   random_state=987)
lr = LogisticRegression(class_weight='balanced')
#Fit the model with training data
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

person_Pred = [educ_Num, income_Num, Age,parent_Num,Marital_Num,Gender_Num]

## Probablity a user has LinkedIn
probs=lr.predict_proba([person_Pred])

probs_new = pd.DataFrame(probs,columns=["No","Yes"])
#probs_percentage = probs_new.iloc[:, [1]]

if round(probs_new.iloc[0]["Yes"],2) >= 0.50:
    LinkedIn_prob = str(round(probs_new.iloc[0]["Yes"],2))
else: 
    LinkedIn_prob= str(round(probs_new.iloc[0]["No"],2))

##Linkedin User Yes or No

predicted_class=lr.predict([person_Pred])

if predicted_class == 1:
    LinkedIn = "Is a LinkedIn User"
else: 
    LinkedIn = "Not a LinkedIn User"



#### Results 

if st.button('Predict if LinkedIn User'):
    LinkedIn +":"+ "     " + "Probability is  " + LinkedIn_prob
else: ''

