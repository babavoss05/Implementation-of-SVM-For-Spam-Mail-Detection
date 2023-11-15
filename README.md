# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Import the necessary python packages using import statements.
2.  Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().
3.  Split the dataset using train_test_split.
4.  Calculate Y_Pred and accuracy.
5.  Print all the outputs.
6. End the Program.

## Program:

Program to implement the SVM For Spam Mail Detection..
Developed by: gokul ,
RegisterNumber:  212221220013
```py
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```


## Output:
### data.head():
![image](https://github.com/AkilaMohan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/103019882/ce566a00-d51f-430e-ad0c-d77b856c85e9)

### data.info():
![image](https://github.com/AkilaMohan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/103019882/bd894c78-358f-4c81-953d-e4798d5ea3ce)

### DATA.ISNULL().SUM() :
![image](https://github.com/AkilaMohan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/103019882/52bc4193-abc5-4e9c-86de-46960277467d)


### Y_PRED :
![image](https://github.com/AkilaMohan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/103019882/631da2f7-7be8-4f1f-85de-9a0f3291a0ea)

### ACCURACY :
![image](https://github.com/AkilaMohan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/103019882/8c0368fd-05cd-4be8-9efa-2d5644099f9f)






## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
