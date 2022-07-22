#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 18:12:41 2022

@author: karimkhalil
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class Cluster():
    
    def __init__(self, N, K, D):
        self.N = N
        self.K = K
        self.D = D
        self.means = np.random.randn(self.K,self.D) + np.random.randint(0,5, size=(self.K,self.D))
        
    def euc (self, x,y):
        x = np.array(x)
        y = np.array(y)
        return sum((x-y)**2)
    
    def cost(self, X, R, M):
        """
        X: data matrix hoding the data
        R: Responsibility matrix holding the weights of the K clusters in the columns
        M: cluster centers matrix holding. centers converges to the mean in all iterations
        """
        cost = 0
        for k in range(len(M)):
            # method 1
            # for n in range(len(X)):
            #     cost += R[n,k]*d(M[k], X[n])
    
            # method 2
            diff = X - M[k]
            sq_distances = (diff * diff).sum(axis=1)
            cost += (R[:,k] * sq_distances).sum()
            
            return cost
        
    def create_sample(self):
        # ranges = np.zeros((int(self.N / self.K), self.D))
        X = np.random.randn(self.N, self.D)
        size_k = self.N / self.K
        l= []
        
        ###### prepare ranges list
        beg = 0
        end = size_k
        for k in range(self.K):
            beg = int(k * size_k)
            # print(f'from: {beg}')
            
            for d in range(self.D):
                end = int(beg + size_k)
            l.append((beg, end))
            # print(f'to: {end}')
            
        ## create array with means  
        X = np.random.randn(self.N , self.D)
        
        ## scale arrays with the means array
        for i,j in enumerate(l):
            X[j[0]:j[1]] = X[j[0]:j[1]] + self.means[i]

        return X
    

        
    def clusters_hard(self, max_iter = 20):
        
        cluster_centers = np.zeros((self.K,self.D))
        X = self.create_sample()
        
        X = np.column_stack([X, np.zeros((self.N,3))]) # extend sample array by columns for distance , identities & number of iterations

        # random selection of cluster centers
        for k in range(self.K):
            z = np.random.choice(self.N)
            cluster_centers[k] = X[z,:2]
            
        cluster_centers_original = cluster_centers.copy()
        cluster_id = X[:,2]
        euc_dist = X[:,3]
        saved_cluster_id = []
        x_alliter = []

        
        for it in range(max_iter):
            old_cluster_id = cluster_id.copy()
            saved_cluster_id.append(old_cluster_id)
            X[:, 4] = it
            
            for n in range(self.N): # for every observation
                l = []
                for k in range(self.K): # for every observation calculate the euc distance of the 3 points k
                    dist = self.euc(X[n,:2], cluster_centers[k])
                    l.append(dist)
    
                lowest = min(l)
                id = l.index(lowest)
        
                X[n,2] = id
                cluster_id[n] = id
        
                X[n,3] = dist
                euc_dist[n] = dist
                
            ## 2: recalculate means & update cluster centers
        
            for k in range(self.K):
                cluster_centers[k,:] = X[X[:,2] == k, :2].mean(axis=0) ## new cluster center
        
            x_alliter.append(X.copy())
        
            ## check for convergence
            if np.all(old_cluster_id == cluster_id):
                print(f'converged on step: {it}')
                break
            
        XX = np.concatenate(x_alliter)
        df = pd.DataFrame(XX , columns = ['x' , 'y' , 'cluster' , 'euc' , 'iteration'])
        df_cost = df.groupby('iteration')['euc'].sum().reset_index()
        

        return X, cluster_centers , df , df_cost , x_alliter
    
    def clusters_soft(self, max_iter=20, beta = 3.0):
        R = np.zeros((self.N, self.K)) ## responsibility matrix holding the weights
        cluster_centers = np.zeros((self.K,self.D))
        X = self.create_sample()
        
        X = np.column_stack([X, np.zeros((self.N,3))]) # extend sample array by columns for distance , identities & number of iterations
        
        
        #random selection of cluster centers
        for k in range(self.K):
            z = np.random.choice(self.N)
            cluster_centers[k] = X[z,:2]
            
        cluster_centers_original = cluster_centers.copy()
        cluster_id = X[:,2]
        euc_dist = X[:,3]
        saved_cluster_id = []
        x_alliter = []
        costs = []
        
        for it in range(max_iter):
            old_cluster_id = cluster_id.copy()
            saved_cluster_id.append(old_cluster_id)
            X[:, 4] = it
            
            for n in range(self.N): # for every observation
                l = []
                for k in range(self.K): # for every observation calculate the euc distance of the 3 points k
                    dist = self.euc(X[n,:2], cluster_centers[k])
                    exp = np.exp(-0.3 * dist)
                    l.append(exp)
                    
                l2 = [i/ sum(l) for i in l]
                    
                R[n] = l2
                
            cluster_centers = R.T@X[:,:2]/R.sum(axis=0, keepdims=True).T


            c = self.cost(X[:,:2], R, cluster_centers)
            costs.append(c)
            # print(c)
            
            
            if it > 0:
                if np.abs(costs[-1] - costs[-2]) < 1e-5:
                    break
        
            random_colors = np.random.random((self.K, 3))
            colors = R.dot(random_colors)
            
            # x_alliter.append(X.copy())
        
            ## check for convergence
            # if np.all(old_cluster_id == cluster_id):
            #     print(f'converged on step: {it}')
            #     break
            
        # XX = np.concatenate(x_alliter)
        # df = pd.DataFrame(XX , columns = ['x' , 'y' , 'cluster' , 'euc' , 'iteration'])
        # df_cost = df.groupby('iteration')['euc'].sum().reset_index()
        

        return X, R , cluster_centers   , colors           
            

cluster = Cluster(400, 4 , 2)
cluster.clusters_soft(100)
# data , cluster_centers , df, df_cost , all_iter = cluster.clusters_hard()
data , R,  cluster_centers , colors = cluster.clusters_soft(100)
# data , cluster_centers , df, df_cost , all_iter = cluster.clusters_hard(1)

# print(exp)
# print(exp/sum(exp))
# print(sum(exp/sum(exp)))
# plt.scatter(data[:,0] , data[:,1] , s=40 , marker="o")
# plt.show()

plt.scatter(data[:,0] , data[:,1] , s=40, c=colors, marker="o")
plt.scatter(cluster_centers[:,0] , cluster_centers[:,1] , s=200, c='black' , marker="*")



