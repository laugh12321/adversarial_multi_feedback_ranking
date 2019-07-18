import numpy as np

def get_pos_channel(pos_level_dist):
    """
    Samples a positive feedback channel

    Args:
        pos_level_dist (dict): positive channel sampling distribution

    Returns:
        L (int): positive feedback channel
    """
    levels = list(pos_level_dist.keys())
    probabilities = list(pos_level_dist.values())
    L = np.random.choice(levels, p=probabilities)
    '''
    numpy.random.choice(a, size=None, replace=True, p=None)
    Ex: 从 a 中产生一个大小为 size 的有重复替换的，概率为 p 的随机采样

    - a 如果为 ndarray 数组，随机样本在该数组获取数据元素，
        如果是整形数据，则随机样本生成类似 np.arange(n)    
    - size 随机样本大小
    - replace 采样样本是否含有重复值
    - p 为 a 中元素的出现概率，默认 a 中元素等概率出现
    '''
    return L


def get_pos_user_item(L, train_inter_pos_dict):
    """
    Sample user u, positive feedback channel and item

    Args:
        L (int): positive feedback channel
        train_inter_pos_dict (dict): collection of all (user, item) interaction
                    tuples for each positive feedback channel

    Returns:
        (int, int, int): (u, i, L) user ID, positive item ID and feedback channel
    """
    pick_idx = np.random.randint(0, len(train_inter_pos_dict[L]))
    u, i = train_inter_pos_dict[L][pick_idx]

    return u, i


def get_neg_channel(user_rep):
    """
    Conditional negative level sampler
    Samples negative level based on the user-specific negative level
    sampling distribution (also influenced by beta)

    Args:
        user_rep (dict): user representation

    Returns:
        N (int): negative feedback channel
    """
    levels = list(user_rep['neg_channel_dist'].keys())
    probabilities = list(user_rep['neg_channel_dist'].values())
    N = np.random.choice(levels, p=probabilities)

    return N


def get_neg_item(user_rep, N, n, u, i, pos_level_dist, train_inter_pos_dict,
                 mode='uniform'):
    """
    Samples the negative item `j` to complete the update triplet `(u, i, j)

    If the sampled negative level `N` is actually an explicit negative channel,
    we sample uniformly from the items in the user's negative channel

    If the samples negative level `N` is actually the unobserved channel,
    we sample uniformly from all items the user did not interact with
    for mode == `uniform` and non-uniformly if the mode == `non-uniform`

    Args:
        user_rep (dict): user representation
        N (int): N (int): negative feedback channel
        n (int): no. of unique items in the dataset
        u (int): user ID
        i (int): positive item ID
        pos_level_dist (dict): positive channel sampling distribution
        train_inter_pos_dict (dict): collection of all (user, item) interaction
            tuples for each positive feedback channel
        mode (str): `uniform` or `non-uniform` mode to sample negative items

    Returns:
        j (int): sampled negative item ID
    """
    if N != -1:
        # sample uniformly from negative channel
        neg_items = list(user_rep['neg_channel_items'][N])
        j = np.random.choice(neg_items)

    else:
        if mode == 'uniform':
            # sample item uniformly from unobserved channel
            j = np.random.choice(np.setdiff1d(np.arange(n), user_rep['items']))

        elif mode == 'non-uniform':
            # sample item non-uniformly from unobserved channel
            L = get_pos_channel(pos_level_dist)
            pos_channel_interactions = train_inter_pos_dict[L]
            n_pos_interactions = len(pos_channel_interactions)
            pick_trials = 0  # ensure sampling despite
            u_other, i_other = u, i
            while u == u_other or i == i_other:
                pos_channel_interactions = train_inter_pos_dict[L]
                pick_idx = np.random.randint(n_pos_interactions)
                u_other, i_other = pos_channel_interactions[pick_idx]
                pick_trials += 1
                if pick_trials == 10:
                    # Ensures that while-loop terminates if sampled L does
                    # not provide properly different feedback
                    L = get_pos_channel(pos_level_dist)
                    pos_channel_interactions = train_inter_pos_dict[L]
                    n_pos_interactions = len(pos_channel_interactions)

            j = i_other

    return j
    
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
        nominators = weights * level_counts
        denominator = sum(nominators)
        dist = nominators / denominator
    else:
        n_levels = len(weights)
        dist = np.ones(n_levels) / n_levels

    dist = dict(zip(list(weights), dist))

    return dist

def get_neg_level_dist(weights, level_counts, mode='non-uniform'):
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
        nominators = [weight * count for weight, count in zip(weights, level_counts)]
        denominator = sum(nominators)
        if denominator != 0:
            dist = list(nom / denominator for nom in nominators)
        else:
            dist = [0] * len(nominators)
    else:
        n_levels = len(weights)
        dist = [1 / n_levels] * n_levels

    if np.abs(np.sum(dist)-1) > 0.00001:
        print("Dist sum unequal 1.")

    dist = dict(zip(list(weights), dist))

    return dist