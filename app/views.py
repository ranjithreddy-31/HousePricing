from django.shortcuts import render
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Create your views here.
def index(request):
    housing =pd.read_csv('data.csv')
    k=housing['RM'].median()
    housing['RM'].fillna(k,inplace=True)
    X=housing.iloc[:,0:13].values
    y=housing.iloc[:,13:14].values
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
    from sklearn.linear_model import LinearRegression
    model=LinearRegression()
    model.fit(X_train,y_train)
    if request.method=='POST':
        sample=[0]*13
        crim=int(request.POST['crim'])
        zn=int(request.POST['zn'])
        indus=int(request.POST['indus'])
        chas=int(request.POST['chas'])
        nox=int(request.POST['nox'])
        rm=int(request.POST['rm'])
        age=int(request.POST['age'])
        dis=int(request.POST['dis'])
        rad=int(request.POST['rad'])
        tax=int(request.POST['tax'])
        ptratio=int(request.POST['ptratio'])
        b=int(request.POST['b'])
        lstat=10

        sample=[crim,zn,indus,chas,nox,rm,age,dis,rad,tax,ptratio,b,lstat]
        sample=np.array([sample])
        sample.reshape(-1,1)

        answer=model.predict(sample)
        ss='BEST PRICE FOR YOUR HOUSE IS '+str(int(round(answer[0][0])))+'$'
        return render(request,'index.html',{'ans':ss})
    else:

        return render(request,'index.html')
