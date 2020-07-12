# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 23:39:30 2020

@author: vedan
"""

import pandas as pd
import numpy as np
from pyvis.network import Network
import networkx as nx


data=pd.read_csv("C:\\Users\\vedan\\Desktop\\SIRP\\c2k_data_comma.csv")
column_name=data.columns
#print(len(column_name))
#for i in range(len(column_name)):
data_leg1=data.iloc[:, 1:25]
data_leg2=data.iloc[:,25:49]
data_leg3=data.iloc[:,49:73]
data_leg4=data.iloc[:,73:97]
legs=data.iloc[:,97]
legs_list=[data_leg1,data_leg2,data_leg3,data_leg4]
print(legs_list[0].columns)
for i in range(len(legs_list)):
    legs_list[i]=legs_list[i].replace('?',0)
    legs_list[i]=legs_list[i].replace(np.nan,0)
    legs_list[i] = legs_list[i].astype(int)
columns=['ipath','n1','e1','n2','e2','n3','e3','n4','e4','n5','i1_hops']
df = pd.DataFrame(columns=columns)
df1= pd.DataFrame(columns=columns)
df2= pd.DataFrame(columns=columns)
# LEG 1
e1=legs_list[0]['i1_rcs_p']+legs_list[0]['i1_dep_1_p']+legs_list[0]['i1_rcf_1_p']
n2=legs_list[0]['i1_rcf_1_place']
e2=legs_list[0]['i1_dep_2_p']+legs_list[0]['i1_rcf_2_p']
n3=legs_list[0]['i1_rcf_2_place']   
e3=legs_list[0]['i1_dep_3_p']+legs_list[0]['i1_rcf_3_p']
n4=legs_list[0]['i1_rcf_3_place']   
e4=legs_list[0]['i1_dlv_p']
df['ipath']=legs_list[0]['i1_legid']
df['e1']=e1
df['n2']=n2
df['e2']=e2
df['n3']=n3
df['e3']=e3
df['n4']=n4
df['e4']=e4
df['n1']=1
df['n5']=4
df['i1_hops']=legs_list[0]['i1_hops']
df=df.drop(3942)

print()
e1=legs_list[1]['i2_rcs_p']+legs_list[1]['i2_dep_1_p']+legs_list[1]['i2_rcf_1_p']
n2=legs_list[1]['i2_rcf_1_place']
e2=legs_list[1]['i2_dep_2_p']+legs_list[1]['i2_rcf_2_p']
n3=legs_list[1]['i2_rcf_2_place']   
e3=legs_list[1]['i2_dep_3_p']+legs_list[1]['i2_rcf_3_p']
n4=legs_list[1]['i2_rcf_3_place']   
e4=legs_list[1]['i2_dlv_p']
df1['ipath']=legs_list[1]['i2_legid']
df1['e1']=e1
df1['n2']=n2
df1['e2']=e2
df1['n3']=n3
df1['e3']=e3
df1['n4']=n4
df1['e4']=e4
df1['n1']=5
df1['n5']=4
df1['i2_hops']=legs_list[1]['i2_hops']
df1=df1.drop(3942)



e1=legs_list[2]['i3_rcs_p']+legs_list[2]['i3_dep_1_p']+legs_list[2]['i3_rcf_1_p']
n2=legs_list[2]['i3_rcf_1_place']
e2=legs_list[2]['i3_dep_2_p']+legs_list[2]['i3_rcf_2_p']
n3=legs_list[2]['i3_rcf_2_place']   
e3=legs_list[2]['i3_dep_3_p']+legs_list[2]['i3_rcf_3_p']
n4=legs_list[2]['i3_rcf_3_place']   
e4=legs_list[2]['i3_dlv_p']
df2['ipath']=legs_list[2]['i3_legid']
df2['e1']=e1
df2['n2']=n2
df2['e2']=e2
df2['n3']=n3
df2['e3']=e3
df2['n4']=n4
df2['e4']=e4
df2['n1']=6
df2['n5']=4
df2['i3_hops']=legs_list[2]['i3_hops']
df2=df2.drop(3942)



for i in range(len(df)):
    if df['i1_hops'].values[i]==1:
        df['e2'].values[i]=df['e4'].values[i]
        df['n3'].values[i]=df['n5'].values[i]
        df['e4'].values[i]=0
        df['n5'].values[i]=0
    if df['i1_hops'].values[i]==2:
        df['e3'].values[i]=df['e4'].values[i]
        df['n4'].values[i]=df['n5'].values[i]
        df['e4'].values[i]=0
        df['n5'].values[i]=0
    if df1['i2_hops'].values[i]==1:
        df1['e2'].values[i]=df1['e4'].values[i]
        df1['n3'].values[i]=df1['n5'].values[i]
        df1['e4'].values[i]=0
        df1['n5'].values[i]=0
    if df1['i1_hops'].values[i]==2:
        df1['e3'].values[i]=df1['e4'].values[i]
        df1['n4'].values[i]=df1['n5'].values[i]
        df1['e4'].values[i]=0
        df1['n5'].values[i]=0
    if df2['i3_hops'].values[i]==1:
        df2['e2'].values[i]=df2['e4'].values[i]
        df2['n3'].values[i]=df2['n5'].values[i]
        df2['e4'].values[i]=0
        df2['n5'].values[i]=0
    if df2['i3_hops'].values[i]==2:
        df2['e3'].values[i]=df2['e4'].values[i]
        df2['n4'].values[i]=df2['n5'].values[i]
        df2['e4'].values[i]=0
        df2['n5'].values[i]=0


    

G = nx.Graph()
G.add_edge(df['n1'].values[0],df['n2'].values[0], weight=df['e1'].values[i])
for i in range(len(df)):
#    if(nx.has_path(G,df['n1'].values[i],df['n2'].values[i])):
#        continue
    if(df['n1'].values[i]==0 or df['n2'].values[i]==0):
        continue
    G.add_edge(df['n1'].values[i],df['n2'].values[i], weight=df['e1'].values[i])
    if(df['n3'].values[i]==0):
        continue
    G.add_edge(df['n2'].values[i],df['n3'].values[i], weight=df['e2'].values[i])
    if(df['n4'].values[i]==0):
        continue
    G.add_edge(df['n3'].values[i],df['n4'].values[i], weight=df['e3'].values[i])
    if(df['n4'].values[i]==0):
        continue
    G.add_edge(df['n4'].values[i],df['n5'].values[i], weight=df['e4'].values[i])
    
    
for i in range(len(df1)):
#    if(nx.has_path(G,df['n1'].values[i],df['n2'].values[i])):
#        continue
    if(df1['n1'].values[i]==0 or df1['n2'].values[i]==0):
        continue
    G.add_edge(df1['n1'].values[i],df1['n2'].values[i], weight=df1['e1'].values[i])
    if(df1['n3'].values[i]==0):
        continue
    G.add_edge(df1['n2'].values[i],df1['n3'].values[i], weight=df1['e2'].values[i])
    if(df1['n4'].values[i]==0):
        continue
    G.add_edge(df1['n3'].values[i],df1['n4'].values[i], weight=df1['e3'].values[i])
    if(df1['n4'].values[i]==0):
        continue
    G.add_edge(df1['n4'].values[i],df1['n5'].values[i], weight=df1['e4'].values[i])


for i in range(len(df2)):
#    if(nx.has_path(G,df['n1'].values[i],df['n2'].values[i])):
#        continue
    if(df2['n1'].values[i]==0 or df2['n2'].values[i]==0):
        continue
    G.add_edge(df2['n1'].values[i],df2['n2'].values[i], weight=df2['e1'].values[i])
    if(df2['n3'].values[i]==0):
        continue
    G.add_edge(df2['n2'].values[i],df2['n3'].values[i], weight=df2['e2'].values[i])
    if(df2['n4'].values[i]==0):
        continue
    G.add_edge(df2['n3'].values[i],df2['n4'].values[i], weight=df2['e3'].values[i])
    if(df2['n4'].values[i]==0):
        continue
    G.add_edge(df2['n4'].values[i],df2['n5'].values[i], weight=df2['e4'].values[i])    
    
#sp = dict(nx.all_pairs_shortest_path(G))

val1 = input("Enter your source: ") 
val2 = input("Enter your destination: ") 
p=nx.shortest_path(G,source=int(val1),target=int(val2),method='dijkstra')  
print(p)
val1 = input("Enter your source: ") 
val2 = input("Enter your destination: ")
p=nx.shortest_path(G,source=int(val1),target=int(val2),method='dijkstra')  
print(p)
val1 = input("Enter your value: ") 
val2 = input("Enter your value: ")
p=nx.shortest_path(G,source=int(val1),target=int(val2),method='dijkstra')    
print(p)
#print(nx.shortest_path(G, source=1, target=700, method='dijkstra')
#print(sp[1])
#G.add_node_from(df['n1'])


#G.from_pandas_edgelist(df, 'n1', 'n2','e1')
#G.add_weighted_edges_from(df['n1'], df['n2'],df['e1'])
