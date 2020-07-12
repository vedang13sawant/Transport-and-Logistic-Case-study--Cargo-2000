# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 23:48:49 2020

@author: vedan
"""
import pandas as pd
import numpy as np
from pyvis.network import Network
import networkx as nx
got_net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
got_net.barnes_hut()

#import networkx as nx
#G = nx.Graph()
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
#if legs_list[0]['i1_hops']==1: 
   
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
print(df)
sources = df['n1'].head(100)
targets = df['n2'].head(100)
weights = df['e1'].head(100)
edge_data = zip(sources, targets, weights)
for e in edge_data:
    src = e[0]
    dst = e[1]
    w = e[2]
    if(src==0 or dst==0):
        continue
    got_net.add_node(src, src, title=str(src))
    got_net.add_node(dst, dst, title=str(dst))
    got_net.add_edge(src, dst, value=w)

neighbor_map = got_net.get_adj_list()

sources = df['n2'].head(100)
targets = df['n3'].head(100)
weights = df['e2'].head(100)
edge_data = zip(sources, targets, weights)
for e in edge_data:
    src = e[0]
    dst = e[1]
    w = e[2]
    if(src==0 or dst==0):
        continue
    got_net.add_node(src, src, title=str(src))
    got_net.add_node(dst, dst, title=str(dst))
    got_net.add_edge(src, dst, value=w)

neighbor_map = got_net.get_adj_list()
sources = df['n3'].head(100)
targets = df['n4'].head(100)
weights = df['e3'].head(100)
edge_data = zip(sources, targets, weights)
for e in edge_data:
    src = e[0]
    dst = e[1]
    w = e[2]
    if(src==0 or dst==0):
        continue
    got_net.add_node(src, src, title=str(src))
    got_net.add_node(dst, dst, title=str(dst))
    got_net.add_edge(src, dst, value=w)

neighbor_map = got_net.get_adj_list()
sources = df['n4'].head(100)
targets = df['n5'].head(100)
weights = df['e4'].head(100)
edge_data = zip(sources, targets, weights)
for e in edge_data:
    src = e[0]
    dst = e[1]
    w = e[2]
    if(src==0 or dst==0):
        continue
    got_net.add_node(src, src, title=str(src))
    got_net.add_node(dst, dst, title=str(dst))
    got_net.add_edge(src, dst, value=w)

neighbor_map = got_net.get_adj_list()
#for node in got_net.nodes:
#    node["title"] += " Neighbors:<br>" + "<br>".join(neighbor_map[node["id"]])
#    node["value"] = len(neighbor_map[node["id"]])


got_net.show("gameofthrones.html")
#print(nx.shortest_path(got_net, source=1, target=256, method='dijkstra'))

##G = nx.add(df, 'n1', 'n2',['ipath','e1'])
#t=np.unique(df[['n1', 'n2','n3','n4','n5']].values)
#G.add_nodes(t,title=t)
#for i in range(len(df)):
#    G.add_edge(df['n1'].values[i], df['n2'].values[i], weight=df['e1'].values[i])
#nt = Network("500px", "500px")
## populates the nodes and edges data structures
#nt.from_nx(G)
#nt.show("nx.html")
##print(G[1][256]['ipath'])