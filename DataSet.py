import pandas as pd
from collections import OrderedDict  # Ex - 01: 使字典对象有序
from sampling import (get_pos_level_dist, get_neg_level_dist)

class DataSet(object):

    def __init__(self, path, beta):
        self.beta = beta
        self.train_inter_df = self.load_rating_file_as_df(path + '.train.rating')
        self.test_inter_df = self.load_rating_file_as_df(path + '.test.rating')
        self.train_inter_pos, self. train_inter_neg = self.get_pos_neg_splits(self.train_inter_df)
        self.pos_level_dist, self.neg_level_dist = self.get_overall_level_distributions(self.train_inter_pos,
                                                                             self.train_inter_neg, self.beta)
        self.train_inter_pos_dict = self.get_pos_channel_item_dict(self.train_inter_pos)                                                        

    def load_rating_file_as_df(self, filename):
        names = ['user', 'item', 'rating', 'timestamp']  
        rating_inter_df = pd.read_csv('filename', sep='	', header=None, names=names)  
        return rating_inter_df

    def get_pos_neg_splits(self, train_inter_df):
        """
        Calculates the rating mean for each user and splits the train
        ratings into positive (greater or equal as every user's
        mean rating) and negative ratings (smaller as mean ratings)

        Args:
            train_inter_df (:obj:`pd.DataFrame`): `M` training instances (rows)
                with three columns `[user, item, rating]`

        Returns:
            train_inter_pos (:obj:`pd.DataFrame`): training instances (rows)
                where `rating_{user}` >= `mean_rating_{user}
            train_inter_neg (:obj:`pd.DataFrame`): training instances (rows)
                where `rating_{user}` < `mean_rating_{user}
        """
        user_mean_ratings = \
            train_inter_df[['user', 'rating']].groupby('user').mean().reset_index()
        user_mean_ratings.rename(columns={'rating': 'mean_rating'},
                                inplace=True)

        train_inter_df = train_inter_df.merge(user_mean_ratings, on='user')
        train_inter_pos = train_inter_df[
            train_inter_df['rating'] >= train_inter_df['mean_rating']]
        train_inter_neg = train_inter_df[
            train_inter_df['rating'] < train_inter_df['mean_rating']]

        return train_inter_pos, train_inter_neg

    def get_overall_level_distributions(self, train_inter_pos, train_inter_neg, beta):
        """
        Computes the frequency distributions for discrete ratings

        Args:
            train_inter_pos (:obj:`pd.DataFrame`): training instances (rows)
                where `rating_{user}` >= `mean_rating_{user}
            train_inter_neg (:obj:`pd.DataFrame`): training instances (rows)
                where `rating_{user}` < `mean_rating_{user}
            beta (float): share of unobserved feedback within the overall
                negative feedback

        Returns:
            pos_level_dist (dict): positive level sampling distribution
            neg_level_dist (dict): negative level sampling distribution
        """

        pos_counts = train_inter_pos['rating'].value_counts().sort_index(
                ascending=False)
        neg_counts = train_inter_neg['rating'].value_counts().sort_index(
                ascending=False)

        pos_level_dist = get_pos_level_dist(pos_counts.index.values,
                                            pos_counts.values)
        neg_level_dist = get_neg_level_dist(neg_counts.index.values,
                                            neg_counts.values, beta)

        return pos_level_dist, neg_level_dist

    def get_pos_channel_item_dict(self, train_inter_pos):
        """
        Creates buckets for each possible rating in `train_inter_pos`
        and subsumes all observed (user, item) interactions with
        the respective rating within

        Args:
            train_inter_pos (:obj:`pd.DataFrame`): training instances (rows)
                where `rating_{user}` >= `mean_rating_{user}

        Returns:
            train_inter_pos_dict (dict): collection of all (user, item) interaction
                tuples for each positive feedback channel
        """

        pos_counts = train_inter_pos['rating'].value_counts().sort_index(
            ascending=False)
        train_inter_pos_dict = OrderedDict() # 见 Ex - 01

        for key in pos_counts.index.values:
            u_i_tuples = [tuple(x) for x in  # tuple() 将列表转换为元组
                        train_inter_pos[train_inter_pos['rating'] == key][['user', 'item']].values]
            train_inter_pos_dict[key] = u_i_tuples

        return train_inter_pos_dict