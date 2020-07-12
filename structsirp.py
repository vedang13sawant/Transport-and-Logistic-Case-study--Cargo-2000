# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:47:26 2020

@author: vedan
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def preprocessingdata(data):
    data=data.drop(3942)
    #data=data.replace('?',0)
    #data=data.replace(np.nan,0)
    first_leg=segmentextractor(data,1,25)
    second_leg=segmentextractor(data,25,49)
    third_leg=segmentextractor(data,49,73)
    out_leg=segmentextractor(data,73,98)
    return first_leg , second_leg, third_leg,out_leg

    
def segmentextractor(df,start,end):
    col_header=list(df)
    segment=df[col_header[start:end]]
    return segment

def calculations_1(leg_1,s):
    leg_1=leg_1.replace('?',0)
    leg_1=leg_1.replace(np.nan,0)
    leg_1 = leg_1.astype(int)
    leg_1=leg_1.loc[(leg_1!=0).any(1)]
    leg_1["dlv_p"]=leg_1.iloc[:,0]+leg_1.iloc[:,2]+leg_1.iloc[:,5]
    leg_1["dlv_e"]=leg_1.iloc[:,1]+leg_1.iloc[:,3]+leg_1.iloc[:,6]
    leg_1["rcs_ontime"]=np.where((leg_1.iloc[:,1]<=leg_1.iloc[:,0]),"Time in Excess","Time Overdue")
    leg_1["dep_ontime"]=np.where((leg_1.iloc[:,3]<=leg_1.iloc[:,2]),"Time in Excess","Time Overdue")
    leg_1["rcf_ontime"]=np.where((leg_1.iloc[:,6]<=leg_1.iloc[:,5]),"Time in Excess","Time Overdue")
    leg_1["total_dlv"]=np.where((leg_1.iloc[:,9]<=leg_1.iloc[:,8]),"Time in Excess","Time Overdue")
    f, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True)
    f.suptitle(s,fontsize=20)
    sns.countplot(x="rcs_ontime",data=leg_1, palette="hls",ax=axes[0, 0])
    sns.countplot(x="dep_ontime",data=leg_1, palette="hls",ax=axes[0, 1])
    sns.countplot(x="rcf_ontime",data=leg_1, palette="hls",ax=axes[1, 0])
    sns.countplot(x="total_dlv",data=leg_1, palette="hls",ax=axes[1, 1])
    plt.show()
    return leg_1
    

def calculations_2(leg_2,s):
    leg_2=leg_2.replace('?',0)
    leg_2=leg_2.replace(np.nan,0)
    leg_2 = leg_2.astype(int)
    leg_2=leg_2.loc[(leg_2!=0).any(1)]
    leg_2["dlv_p"]=leg_2.iloc[:,0]+leg_2.iloc[:,3]
    leg_2["dlv_e"]=leg_2.iloc[:,1]+leg_2.iloc[:,4]
    #leg_2["rcs_ontime"]=np.where((leg_2.iloc[:,1]<=leg_2.iloc[:,0]),"Time in Excess","Time Overdue")
    leg_2["dep_ontime"]=np.where((leg_2.iloc[:,1]<=leg_2.iloc[:,0]),"Time in Excess","Time Overdue")
    leg_2["rcf_ontime"]=np.where((leg_2.iloc[:,4]<=leg_2.iloc[:,3]),"Time in Excess","Time Overdue")
    leg_2["total_dlv"]=np.where((leg_2.iloc[:,7]<=leg_2.iloc[:,6]),"Time in Excess","Time Overdue")
    f, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True)
    #sns.countplot(x="rcs_ontime",data=leg_2, palette="hls",ax=axes[0, 0])
    f.suptitle(s,fontsize=20)
    sns.countplot(x="dep_ontime",data=leg_2, palette="hls",ax=axes[0, 0])
    sns.countplot(x="rcf_ontime",data=leg_2, palette="hls",ax=axes[0, 1])
    sns.countplot(x="total_dlv",data=leg_2, palette="hls",ax=axes[1, 0])
    plt.show()
    return leg_2
    
def calculations_3(leg_3,s):
    leg_3=leg_3.replace('?',0)
    leg_3=leg_3.replace(np.nan,0)
    leg_3 = leg_3.astype(int)
    leg_3=leg_3.loc[(leg_3!=0).any(1)]
    print(leg_3)
    leg_3["dlv_p"]=leg_3.iloc[:,0]+leg_3.iloc[:,3]
    leg_3["dlv_e"]=leg_3.iloc[:,1]+leg_3.iloc[:,4]
    #leg_3["rcs_ontime"]=np.where((leg_3.iloc[:,1]<=leg_3.iloc[:,0]),"Time in Excess","Time Overdue")
    leg_3["dep_ontime"]=np.where((leg_3.iloc[:,1]<=leg_3.iloc[:,0]),"Time in Excess","Time Overdue")
    leg_3["rcf_ontime"]=np.where((leg_3.iloc[:,4]<=leg_3.iloc[:,3]),"Time in Excess","Time Overdue")
    leg_3["total_dlv"]=np.where((leg_3.iloc[:,7]<=leg_3.iloc[:,6]),"Time in Excess","Time Overdue")
    f, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True)
    #sns.countplot(x="rcs_ontime",data=leg_3, palette="hls",ax=axes[0, 0])
    f.suptitle(s,fontsize=20)
    sns.countplot(x="dep_ontime",data=leg_3, palette="hls",ax=axes[0, 0])
    sns.countplot(x="rcf_ontime",data=leg_3, palette="hls",ax=axes[0, 1])
    sns.countplot(x="total_dlv",data=leg_3, palette="hls",ax=axes[1, 0])
    plt.show()
    return leg_3


def overall_comparisions(leg):
    leg=leg.replace('?',0)
    leg = leg.astype(int)
    leg["total_p"]=leg.iloc[:,1]+leg.iloc[:,3]+leg.iloc[:,6]+leg.iloc[:,9]+leg.iloc[:,12]+leg.iloc[:,15]+leg.iloc[:,18]+leg.iloc[:,21]
    leg["total_e"]=leg.iloc[:,2]+leg.iloc[:,4]+leg.iloc[:,7]+leg.iloc[:,10]+leg.iloc[:,13]+leg.iloc[:,16]+leg.iloc[:,19]+leg.iloc[:,22]
    #print(leg)
    leg["total_ontime"]=np.where((leg['total_e']<=leg['total_p']),"Time in Excess","Time Overdue")
    return leg            

def correlation_matrix(leg1,leg2,leg3):
    column_names = ["dlv_p1", "dlv_e1", "dlv_p2","dlv_e2","dlv_p3","dlv_e3"]
    df = pd.DataFrame(columns = column_names)
    df["dlv_p1"]=leg1["dlv_p"]
    df["dlv_e1"]=leg1["dlv_e"]
    df["dlv_p2"]=leg2["dlv_p"]
    df["dlv_e2"]=leg2["dlv_e"]
    df["dlv_p3"]=leg3["dlv_p"]
    df["dlv_e3"]=leg3["dlv_e"]
    print(df)
    #df["rcf3"]=leg3["rcf_ontime"]
    corr=df.corr()
    corrlt=sns.heatmap(corr,cmap='coolwarm')
    plt.title("Correlation")
    plt.show()

def outreg(leg):
    corr=leg[['o_rcs_p','o_rcs_e','o_dep_1_p','o_dep_1_e','o_rcf_1_p','o_rcf_1_e']].corr()
    plt.figure(figsize=(12, 13))
    corrlt=sns.heatmap(corr,cmap='coolwarm')
    plt.title("Correlation")
    plt.show()
    
    X=leg[['o_rcs_p','o_dep_1_p','o_rcf_1_p']]
    Y=leg['dlv_e']
    reg = LinearRegression()
    reg.fit(X, Y)
    X2 = sm.add_constant(X)
    est = sm.OLS(Y, X2)
    est2 = est.fit()
    print(est2.summary())  
    y_pred=reg.predict(X)
    plt.scatter(Y,y_pred)
    #plt.plot(X,y_pred,color='red')
    plt.show()     


        
def reg(leg):
    corr=leg[['i1_rcs_p','i1_rcs_e','i1_dep_1_p','i1_dep_1_e','i1_rcf_1_p','i1_rcf_1_e']].corr()
    plt.figure(figsize=(12, 13))
    corrlt=sns.heatmap(corr,cmap='coolwarm')
    plt.title("Correlation")
    plt.show()
    
    X=leg[['i1_rcs_p','i1_dep_1_p','i1_rcf_1_p']]
    Y=leg['dlv_e']
    reg = LinearRegression()
    reg.fit(X, Y)
    X2 = sm.add_constant(X)
    est = sm.OLS(Y, X2)
    est2 = est.fit()
    print(est2.summary())  
    y_pred=reg.predict(X)
    plt.scatter(Y,y_pred)
    #plt.plot(X,y_pred,color='red')
    plt.show()      
    
def reg2(leg):
    leg=leg.replace('?',0)
    t= (leg != 0).any(axis=1)
    leg= leg.loc[t]
#    corr=leg[['i2_rcs_p','i2_rcs_e','i2_dep_1_p','i2_dep_1_e','i2_rcf_1_p','i2_rcf_1_e']].corr()
#    corrlt=sns.heatmap(corr,cmap='coolwarm')
#    plt.title("Correlation")
#    plt.show()
    
    X=leg[['i2_rcs_p','i2_dep_1_p','i2_rcf_1_p']]
    Y=leg['dlv_e']
    Y = Y.astype(float)
    reg = LinearRegression()
    reg.fit(X, Y)
    X2 = sm.add_constant(X)
    est = sm.OLS(Y, X2)
    est2 = est.fit()
    print(est2.summary())  
    y_pred=reg.predict(X)
    plt.scatter(Y,y_pred)
    #plt.plot(X,y_pred,color='red')
    plt.show()    


def correlation_all(leg1,leg2,leg3):
    columns=['i1_rcs_p', 'i1_rcs_e', 'i1_dep_1_p', 'i1_dep_1_e','i1_rcf_1_p', 'i1_rcf_1_e','i1_dep_2_p', 'i1_dep_2_e','i1_rcf_2_p',
       'i1_rcf_2_e','i1_dep_3_p', 'i1_dep_3_e','i1_rcf_3_p',
       'i1_rcf_3_e']
    df = pd.DataFrame(columns=columns)
    df['i1_rcs_p']=leg1['i1_rcs_p']
    df['i1_rcs_e']=leg1['i1_rcs_e']
    df['i1_dep_1_p']=leg1['i1_dep_1_p']
    df['i1_dep_1_e']=leg1['i1_dep_1_e']
    df['i1_rcf_1_p']=leg1['i1_rcf_1_p']
    df['i1_rcf_1_e']=leg1['i1_rcf_1_e']
    df['i1_dep_2_p']=leg2['i1_dep_2_p']
    df['i1_dep_2_e']=leg2['i1_dep_2_e']
    df['i1_rcf_2_p']=leg2['i1_rcf_2_p']
    df['i1_rcf_2_e']=leg2['i1_rcf_2_e']
    df['i1_dep_3_p']=leg3['i1_dep_3_p']
    df['i1_dep_3_e']=leg3['i1_dep_3_e']
    df['i1_rcf_3_p']=leg3['i1_rcf_3_p']
    df['i1_rcf_3_e']=leg3['i1_rcf_3_e']
    print(df)
    print(df.columns)
    corr=df.corr()
    print(corr)
    corrlt=sns.heatmap(corr,cmap='coolwarm')
    plt.title("Correlation")
    plt.show()
    
    
def main():
    data=pd.read_csv("C:\\Users\\vedan\\Desktop\\SIRP\\c2k_data_comma.csv")
    first_leg, second_leg, third_leg,out_leg=preprocessingdata(data)
    
    
    first_leg_1=segmentextractor(first_leg,1,9)
    first_leg_2=segmentextractor(first_leg,9,15)
    first_leg_3=segmentextractor(first_leg,15,21)
#
    l1=calculations_1(first_leg_1,"1 LEG-1 SEGMENT")
    l2=calculations_2(first_leg_2,"1 LEG-2 SEGMENT")
    l3=calculations_3(first_leg_3,"1 LEG-3 SEGMENT")
    print(l1.columns)
    
    columns=["rcs_ontime","dep_ontime","rcf_ontime","total_dlv"]
    ddf1 = pd.DataFrame(columns=columns)
    #ddf1=pd.Dataframe(columns=columns)
    ddf1['rcs_ontime']=l1['rcs_ontime']
    ddf1["dep_ontime"]=l1["dep_ontime"]
    ddf1["rcf_ontime"]=l1["rcf_ontime"]
    ddf1["total_dlv"]=l1["total_dlv"]
    ddf1=ddf1.replace('Time in Excess',1)
    ddf1=ddf1.replace('Time Overdue',0)
    print(ddf1.head())

    X_train, X_test, y_train, y_test = train_test_split(ddf1.drop(['total_dlv'],axis=1),ddf1['total_dlv'], test_size=0.30, random_state=101)
    #print(X_train)
    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
    print(X_test)
    Predictions = logmodel.predict(X_test)
    print(classification_report(y_test,Predictions))

    print(confusion_matrix(y_test, Predictions))
    cm=confusion_matrix(y_test, Predictions)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
    plt.title("Confusion Matrix")
    plt.show()
#    second_leg_1=segmentextractor(second_leg,1,9)    
#    second_leg_2=segmentextractor(second_leg,9,15)
#    second_leg_3=segmentextractor(second_leg,15,21)
###    
###    
#    l1=calculations_1(second_leg_1,"2 LEG-1 SEGMENT")
#    l2=calculations_2(second_leg_2,"2 LEG-2 SEGMENT")
#    l3=calculations_3(second_leg_3,"2 LEG-3 SEGMENT")
#    #correlation_matrix(l1,l2,l3)
##    
#    third_leg_1=segmentextractor(third_leg,1,9)
#    third_leg_2=segmentextractor(third_leg,9,15)
#    third_leg_3=segmentextractor(third_leg,15,21)
###    
#    l1=calculations_1(third_leg_1,"3 LEG-1 SEGMENT")
#    l2=calculations_2(third_leg_2,"3 LEG-2 SEGMENT")
#    l3=calculations_3(third_leg_3,"3 LEG-3 SEGMENT")
#    #correlation_matrix(l1,l2,l3)
#    
#    first_leg_dlv=segmentextractor(first_leg,21,23)
#    second_leg_dlv=segmentextractor(second_leg,21,23)
#    third_leg_dlv=segmentextractor(third_leg,21,23)
#    first_leg_dlv["dlv_ontime"]=np.where((first_leg_dlv.iloc[:,1]<=first_leg_dlv.iloc[:,0]),"Time in Excess","Time Overdue")
#    second_leg_dlv["dlv_ontime"]=np.where((second_leg_dlv.iloc[:,1]<=second_leg_dlv.iloc[:,0]),"Time in Excess","Time Overdue")
#    third_leg_dlv["dlv_ontime"]=np.where((third_leg_dlv.iloc[:,1]<=third_leg_dlv.iloc[:,0]),"Time in Excess","Time Overdue")
#    
#    f, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True)
#    f.suptitle("dlv-of all 3 segment",fontsize=20)
#    sns.countplot(x="dlv_ontime",data=first_leg_dlv, palette="hls",ax=axes[0, 0])
#    sns.countplot(x="dlv_ontime",data=second_leg_dlv, palette="hls",ax=axes[0, 1])
#    sns.countplot(x="dlv_ontime",data=third_leg_dlv, palette="hls",ax=axes[1, 0])
#    plt.show()
#    
#    first_leg=overall_comparisions(first_leg)
#    second_leg=overall_comparisions(second_leg)
#    third_leg=overall_comparisions(third_leg)
#    
#    f, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True)
#    f.suptitle("TOTAL TIME",fontsize=20)
#    sns.countplot(x="total_ontime",data=first_leg, palette="hls",ax=axes[0, 0])
#    sns.countplot(x="total_ontime",data=second_leg, palette="hls",ax=axes[0, 1])
#    sns.countplot(x="total_ontime",data=third_leg, palette="hls",ax=axes[1, 0])
#    plt.show()
#    print(third_leg.columns)
#    print("OUTPUT----------------------------------------")
#    out_leg_1=segmentextractor(out_leg,1,9)
#    out_leg_2=segmentextractor(out_leg,9,15)
#    out_leg_3=segmentextractor(out_leg,15,21)
##    print(out_leg_1.columns)
#    l1=calculations_1(out_leg_1,"O LEG-1 SEGMENT")
##    outreg(l1)
#    l2=calculations_2(out_leg_2,"O LEG-2 SEGMENT")
#    l3=calculations_3(out_leg_3,"O LEG-3 SEGMENT")
#    
    
    
if __name__=="__main__":
    main()

