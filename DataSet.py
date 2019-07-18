'''
Created on Aug 8, 2016
Processing datasets. 
@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import pandas as pd
import numpy as np
from time import time

class Dataset(object):
    '''
    Loading the data file
        trainMatrix: load rating records as sparse matrix for class Data
        trianList: load rating records as list to speed up user's feature retrieval
        testRatings: load leave-one-out rating test for class Evaluate
        testNegatives: sample the items not rated by user
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_training_file_as_matrix(path + ".train.rating")
        self.trainList = self.load_training_file_as_list(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)
        self.num_users, self.num_items = self.trainMatrix.shape

        self.train_ratings, self.m, self.n = self.load_ratings(path + ".train.rating")
        self.test_ratings, _, _ = self.load_ratings(path + ".test.rating")

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_training_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()
        print("already load the trainMatrix...")
        return mat

    def load_training_file_as_list(self, filename):
        # Get number of users and items
        u_ = 0
        lists, items = [], []
        with open(filename, "r") as f:
            line = f.readline()
            index = 0
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                if u_ < u:
                    index = 0
                    lists.append(items)
                    items = []
                    u_ += 1
                index += 1
                #if index<300:
                items.append(i)
                line = f.readline()
        lists.append(items)
        print("already load the trainList...")
        return lists

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
        ratings.drop('timestamp', axis=1, inplace=True)

        m = ratings['user'].unique().shape[0]
        n = ratings['item'].unique().shape[0]

        # Contiguation of user and item IDs
        user_rehasher = dict(zip(ratings['user'].unique(), np.arange(m)))
        item_rehasher = dict(zip(ratings['item'].unique(), np.arange(n)))
        ratings['user'] = ratings['user'].map(user_rehasher).astype(int)
        ratings['item'] = ratings['item'].map(item_rehasher)

        return ratings, m, n