# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 13:05:02 2017

@author: Rafiya
"""
import pandas as pd

df=pd.read_csv('data_no_null2.csv', encoding="ISO-8859-1")

df_2=pd.read_csv('data2.csv', encoding="ISO-8859-1")


dfnew=pd.DataFrame()

newval="Above"
i=0
for index, row in df.iterrows():
    i=i+1
    countyname=row['county']
    match=df_2.loc[df_2['County Name']==countyname]
    if(len(match)>0):
        newval=1
    else:
        newval=0
    row['Is HPSA']=newval
    dfnew=dfnew.append(row)
     
dfnew.to_csv('second_data.csv')