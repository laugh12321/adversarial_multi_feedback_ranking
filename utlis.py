'''
Created on July 18, 2019
Utils. 
@author: zhangpeng bo (zhang26162@gmail.com)
'''
import numpy as np
from collections import OrderedDict 

def get_channels(inter_df):
    channels = list(inter_df['rating'].unique())
    channels.sort()

    return channels[::-1]

def get_pos_level_dist(weights, level_counts, mode='non-uniform'):
    """
    Returns the sampling distribution for positive
    feedback channels L using either a `non-uniform` or `uniform` approach

    Args:
        weights (:obj:`np.array`): (w, ) `w` rating values representing distinct
            positive feedback channels
        level_counts (:obj:`np.array`): (s, ) count `s` of ratings for each
            positive feedback channel
        mode (str): either `uniform` meaning all positive levels are
            equally relevant or `non-uniform` which imposes
            a (rating*count)-weighted distribution of positive levels

    Returns:
        dist (dict): positive channel sampling distribution
    """
    if mode == 'non-uniform':
        nominators = (1.0 / weights) * level_counts
        denominator = sum(nominators)
        dist = nominators / denominator
    else:
        n_levels = len(weights)
        dist = np.ones(n_levels) / n_levels

    dist = dict(zip(list(weights), dist))

    return dist

def get_neg_level_dist(weights, level_counts, mode):
    """
    Compute negative feedback channel distribution

    Args:
        weights (:obj:`np.array`): (w, ) `w` rating values representing distinct
            negative feedback channels
        level_counts (:obj:`np.array`): (s, ) count `s` of ratings for each
            negative feedback channel
        mode: either `uniform` meaning all negative levels are
            equally relevant or `non-uniform` which imposes
            a (rating*count)-weighted distribution of negative levels

    Returns:
        dist (dict): negative channel sampling distribution
    """
    if mode == 'non-uniform':
        nominators = (1.0 / weights) * level_counts
        # nominators = [(1.0/weight) * count for weight, count in zip(weights, level_counts)]
        denominator = sum(nominators)
        if denominator != 0:
            dist = list(nom / denominator for nom in nominators)
        else:
            dist = [0.0] * len(nominators)
    else:
        n_levels = len(weights)
        dist = [1 / n_levels] * n_levels

    if np.abs(np.sum(dist)-1) > 0.00001:
        logging.warning('Dist sum unequal 1.') 
        print("Dist sum unequal 1.")

    dist = dict(zip(list(weights), dist))

    return dist 

def get_pos_neg_splits(train_inter_df):
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

def get_overall_level_distributions(train_inter_pos, train_inter_neg, mode):
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
                                        pos_counts.values, mode)
    neg_level_dist = get_neg_level_dist(neg_counts.index.values,
                                        neg_counts.values, mode)

    return pos_level_dist, neg_level_dist

def get_pos_channel_item_dict(train_inter_pos):
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
    train_inter_pos_dict = OrderedDict() 

    for key in pos_counts.index.values:
        u_i_tuples = [tuple(x) for x in 
                      train_inter_pos[train_inter_pos['rating'] == key][['user', 'item']].values]
        train_inter_pos_dict[key] = u_i_tuples

    return train_inter_pos_dict

def get_user_reps(m, d, train_inter, test_inter, channels, beta):
    """
    Creates user representations that encompass user latent features
    and additional user-specific information
    User latent features are drawn from a standard normal distribution

    Args:
        m (int): no. of unique users in the dataset
        d (int): no. of latent features for user and item representations
        train_inter (:obj:`pd.DataFrame`): `M` training instances (rows)
            with three columns `[user, item, rating]`
        test_ratings (:obj:`pd.DataFrame`): `M` testing instances (rows)
                with three columns `[user, item, rating]`
        channels ([int]): rating values representing distinct feedback channels
        beta (float): share of unobserved feedback within the overall
            negative feedback

    Returns:
        user_reps (dict): representations for all `m` unique users
    """
    user_reps = {}
    train_inter = train_inter.sort_values('user')

    for user_id in range(m):
        user_reps[user_id] = {}
        user_reps[user_id]['embed'] = np.random.normal(size=(d,))
        user_item_ratings = train_inter[train_inter['user'] == user_id][['item', 'rating']]
        user_reps[user_id]['mean_rating'] = user_item_ratings['rating'].mean()
        user_reps[user_id]['items'] = list(user_item_ratings['item'])
        user_reps[user_id]['all_items'] = list(set(user_reps[user_id]['items']).union(
                                               set(list(test_inter[test_inter['user'] == user_id]['item']))))
        user_reps[user_id]['pos_channel_items'] = OrderedDict()
        user_reps[user_id]['neg_channel_items'] = OrderedDict()
        for channel in channels:
            if channel >= user_reps[user_id]['mean_rating']:
                user_reps[user_id]['pos_channel_items'][channel] = \
                    list(user_item_ratings[user_item_ratings['rating'] == channel]['item'])
            else:
                user_reps[user_id]['neg_channel_items'][channel] = \
                    list(user_item_ratings[user_item_ratings['rating'] == channel]['item'])

        pos_channels = np.array(list(user_reps[user_id]['pos_channel_items'].keys()))
        neg_channels = np.array(list(user_reps[user_id]['neg_channel_items'].keys()))
        pos_channel_counts = [len(user_reps[user_id]['pos_channel_items'][key]) for key in pos_channels]
        neg_channel_counts = [len(user_reps[user_id]['neg_channel_items'][key]) for key in neg_channels]

        user_reps[user_id]['pos_channel_dist'] = \
            get_pos_level_dist(pos_channels, pos_channel_counts, 'non-uniform')

        if sum(neg_channel_counts) != 0:
            user_reps[user_id]['neg_channel_dist'] = \
                get_neg_level_dist(neg_channels, neg_channel_counts, 'non-uniform')

            # correct for beta
            for key in user_reps[user_id]['neg_channel_dist'].keys():
                user_reps[user_id]['neg_channel_dist'][key] = \
                    user_reps[user_id]['neg_channel_dist'][key] * (1 - beta)
            user_reps[user_id]['neg_channel_dist'][-1] = beta

        else:
            # if there is no negative feedback, only unobserved remains
            user_reps[user_id]['neg_channel_dist'] = {-1: 1.0}

    return user_reps