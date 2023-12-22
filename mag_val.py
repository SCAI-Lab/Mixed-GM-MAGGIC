# -*- coding: utf-8 -*-
"""
Created on Thu Jan 06 09:44:13 2023

@author: liyan
"""
import time
import pandas as pd
# import matplotlib.pyplot as plt
import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

## using library causal-learn
from causallearn.search.ConstraintBased.PC import pc
# from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph

dfs = {}
N = [250,500,1000,2000,4000] # 

for n in N:
    dfs[n] = pd.read_csv("./maggic/simulate{}.csv".format(n))

features = list(dfs[4000].columns)

# true DAG of the simulation data

true_dag_path = './maggic/maggic_dag.txt'
true_dag = txt2generalgraph(true_dag_path) 
true_cpdag = dag2cpdag(true_dag)

        
# learn the DAG graph using constraint-based method
## learning over datasets with increasing sizes

from causallearn.graph.SHD import SHD
from causallearn.graph.ArrowConfusion import ArrowConfusion
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from joblib import Parallel, delayed


def const_run(cit, sig, n):
    start = time.time()
    dag = pc(dfs[n].to_numpy(), indep_test=cit, alpha=sig, verbose=True, 
             background_knowledge=None)
    end = time.time()
    
    # compute the SHD between learned graph and true graph
    shd = SHD(true_cpdag, dag.G).get_shd()
    runtime = (end - start)/60 # in mins
    
    # compute the arrow and adj F1
    arrow = ArrowConfusion(true_cpdag, dag.G)
    arrTP = arrow.get_arrows_tp()
    arrTN = arrow.get_arrows_tn()
    arrFP = arrow.get_arrows_fp()
    arrFN = arrow.get_arrows_fn()
    arrF1 = (arrTP*2)/(arrTP*2+arrFP+arrFN)
    
    adj = AdjacencyConfusion(true_cpdag, dag.G)
    adjTP = adj.get_adj_tp()
    adjTN = adj.get_adj_tn()
    adjFP = adj.get_adj_fp()
    adjFN = adj.get_adj_fn()
    adjF1 = (adjTP*2)/(adjTP*2+adjFP+adjFN)
    
    result = {"cit": cit,
              "sig": sig,
              "n": n,
              "shd": shd,
              "arrF1": arrF1,
              "adjF1": adjF1,
              "arrTP": arrTP,
              "arrTN": arrTN,
              "arrFP": arrFP,
              "arrFN": arrFN,
              "adjTP": adjTP,
              "adjTN": adjTN,
              "adjFP": adjFP,
              "adjFN": adjFN,
              "runtime": runtime}
    
    # write out all the edges
    with open('./maggic/DAG-mag{}-{}-{}.txt'.format(n, cit, sig), 'w') as f:
        print(dag.G, file=f)
        
    with open('./maggic/DAG-mag{}-{}-{}-res.txt'.format(n, cit, sig), 'w') as f1:
        print(result, file=f1)
        
    return result

cits = ["ppi", "ppi2w"] # "gsq", "fisherz", "chisq",  "kci",
sigs = [0.001, 0.01, 0.05]


parallel = Parallel(n_jobs=-1, verbose=True)
results = parallel(delayed(const_run)(cit,sig,n) for cit in cits for sig in sigs for n in N)        
'''

# using score-based method: GES
from causallearn.search.ScoreBased.GES import ges

def score_run(score, n):
    start = time.time()
    dag = ges(dfs[n].to_numpy(), score_func=score, maxP=None, parameters=None)
    end = time.time()
    
    # compute the SHD between learned graph and true graph
    shd = SHD(true_cpdag, dag["G"]).get_shd() 
    runtime = (end - start)/60 
    
    # compute the arrow and adj F1
    arrow = ArrowConfusion(true_cpdag, dag["G"])
    arrTP = arrow.get_arrows_tp()
    arrTN = arrow.get_arrows_tn()
    arrFP = arrow.get_arrows_fp()
    arrFN = arrow.get_arrows_fn()
    arrF1 = (arrTP*2)/(arrTP*2+arrFP+arrFN)
    
    adj = AdjacencyConfusion(true_cpdag, dag["G"])
    adjTP = adj.get_adj_tp()
    adjTN = adj.get_adj_tn()
    adjFP = adj.get_adj_fp()
    adjFN = adj.get_adj_fn()
    adjF1 = (adjTP*2)/(adjTP*2+adjFP+adjFN)
    
    result = {"score": score,
              "n": n,
              "shd": shd,
              "arrF1": arrF1,
              "adjF1": adjF1,
              "arrTP": arrTP,
              "arrTN": arrTN,
              "arrFP": arrFP,
              "arrFN": arrFN,
              "adjTP": adjTP,
              "adjTN": adjTN,
              "adjFP": adjFP,
              "adjFN": adjFN,
              "runtime": runtime}
    
    # write out all the edges
    with open('./maggic/DAG-mag{}-{}.txt'.format(n, score), 'w') as f2:
        print(dag["G"], file=f2)
        
    with open('./maggic/DAG-mag{}-{}-res.txt'.format(n, score), 'w') as f3:
        print(result, file=f3)
        
    return result

scores = ["local_score_BIC", 'local_score_BDeu',"local_score_CV_general"] # , "local_score_marginal_general"

parallel = Parallel(n_jobs=-1, verbose=True)
results2 = parallel(delayed(score_run)(score, n) for score in scores for n in N)

# consolidate the results
shds = {}
runtimes = {}

for result in results:
    cit = result["cit"]
    sig = result["sig"]
    n = result["n"]
    shds['{}-{}-{}'.format(cit,sig,n)] = result["shd"]
    runtimes['{}-{}-{}'.format(cit,sig,n)] = result["runtime"]
    
for result in results2:
    score = result["score"]
    n = result["n"]
    shds['{}-{}'.format(score,n)] = result["shd"]
    runtimes['{}-{}'.format(score,n)] = result["runtime"]

with open("maggic-shd.txt", 'w') as f4: 
    for key, value in shds.items(): 
        f4.write('%s:%s\n' % (key, value))
        
with open("maggic-runtime.txt", 'w') as f5: 
    for key, value in runtimes.items(): 
        f5.write('%s:%s\n' % (key, value))
'''