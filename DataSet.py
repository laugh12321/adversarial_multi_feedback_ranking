'''
Created on July 18, 2019
Processing datasets. 
@author: zhangpeng bo (zhang26162@gmail.com)
'''
import numpy as np
import pandas as pd
from time import time
import scipy.sparse as sp

def load_ratings(filename):
    """
    loads the dataset, ignoring temporal information

    Args:
        path (str): path pointing to folder with interaction data `ratings.dat`

    Returns:
        ratings (:obj:`pd.DataFrame`): overall interaction instances (rows)
            with three columns `[user, item, rating]`
        m (int): no. of unique users in the dataset
        n (int): no. of unique items in the dataset
    """
    # ratings = pd.read_csv(filename, sep='	', names=['user', 'item', 'rating', 'timestamp'])
    ratings = pd.read_csv(filename, sep='::', names=['user', 'item', 'rating', 'timestamp'])
    ratings.drop('timestamp', axis=1, inplace=True)
    
    return ratings


class DataSet(object):
    def __init__(self, path):
        self.trainList,  self.num_users, self.num_items = self.load_ratings(path + ".train.rating")
        self.testRatings, _, _ = self.load_ratings(path + ".test.rating")
        
        self.trainMatrix = self.get_trainMatrix()
        
    def load_ratings(self, filename):
        """
        loads the dataset, ignoring temporal information

        Args:
            path (str): path pointing to folder with interaction data `ratings.dat`

        Returns:
            ratings (:obj:`pd.DataFrame`): overall interaction instances (rows)
                with three columns `[user, item, rating]`
            m (int): no. of unique users in the dataset
            n (int): no. of unique items in the dataset
        """
        ratings = pd.read_csv(filename, sep='	', names=['user', 'item', 'rating', 'timestamp'])
        #ratings = pd.read_csv(filename, sep='::', names=['user', 'item', 'rating', 'timestamp'])
        ratings.drop('timestamp', axis=1, inplace=True)
        
        m = max(ratings['user']); m += 1
        n = max(ratings['item']); n += 1
        
        return ratings, m, n
        
        
    def get_trainMatrix(self):
        mat = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        
        for i in range(len(self.trainList)):
            user, item, rating = int(self.trainList['user'][i]), int(self.trainList['item'][i]),\
                                 float(self.trainList['rating'][i]) 
            if (rating > 0):
                mat[user, item] = 1.0
        return mat