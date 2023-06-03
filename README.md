# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: J.Archana priya
RegisterNumber:  212221230007
*/
```
```
import chardet
file = '/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd 
data = pd.read_csv("/content/spam.csv",encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()



x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)


from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)


y_pred = svc.predict(x_test)
y_pred



from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### Result
![image](https://github.com/Archana2003-Jkumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427594/f2f9dd50-ffd8-45e4-a6bd-768b35a6abff)
### data.head() 
![image](https://github.com/Archana2003-Jkumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427594/206adde3-b434-4db3-affa-68bfd97ae42e)
### data.info() 
![image](https://github.com/Archana2003-Jkumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427594/371eead9-fa5d-4637-a455-8815273949d3)
### data.isnull().sum()
![image](https://github.com/Archana2003-Jkumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427594/8afc9727-38b2-4094-87b9-c3d5d5efe1de)
### Y_prediction value
![image](https://github.com/Archana2003-Jkumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427594/2d4e3cd7-56e2-403e-b5f7-a1a580011236)
### Accuracy value
![image](https://github.com/Archana2003-Jkumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427594/8e79c495-0570-431b-b55d-90d8631efc86)
## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
