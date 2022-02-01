import numpy as np

def dist_generators(dist_params):
    
    dist_name = dist_params['name']
    dist_params = dist_params[dist_name]
    if dist_name == 'uniform':
        u_pos_params = dist_params['positive_dist']
        u_neg_params = dist_params['negative_dist']
        u_pos = lambda : np.random.uniform(u_pos_params['low'], u_pos_params['high'], 1)[0]
        u_neg = lambda : np.random.uniform(u_neg_params['low'], u_neg_params['high'], 1)[0]
        return u_pos, u_neg
    elif dist_name == 'normal':
        normal_pos_params = dist_params['positive_dist']
        normal_neg_params = dist_params['negative_dist']
        normal_pos = lambda : np.random.normal(normal_pos_params['mu'], normal_pos_params['std'], 1)[0]
        normal_neg = lambda : np.random.normal(normal_neg_params['mu'], normal_neg_params['std'], 1)[0]
        return normal_pos, normal_neg
    elif dist_name == 'bernoulli':
        bernoulli_pos_params = dist_params['positive_dist']
        bernoulli_neg_params = dist_params['negative_dist']
        bernoulli_pos = lambda : np.random.binomial(bernoulli_pos_params['n'], bernoulli_pos_params['p'], 1)[0]
        bernoulli_neg = lambda : np.random.binomial(bernoulli_neg_params['n'], bernoulli_neg_params['p'], 1)[0]
        return bernoulli_pos, bernoulli_neg
    


def extra_dist_generator(dist_params, scale):
    
    dist_name = dist_params['name']
    dist_params = dist_params[dist_name]
    if dist_name == 'uniform':
        u_pos_params = dist_params['positive_dist']
        extra_dist = lambda : np.random.uniform(u_pos_params['low']+scale, u_pos_params['high']+scale, 1)[0]
        return extra_dist
    
    elif dist_name == 'normal':
        normal_pos_params = dist_params['positive_dist']
        extra_dist = lambda : np.random.normal(normal_pos_params['mu']+scale, normal_pos_params['std'], 1)[0]
        return extra_dist
    
    elif dist_name == 'bernoulli':
        bernoulli_pos_params = dist_params['positive_dist']
        extra_dist = lambda : np.random.binomial(bernoulli_pos_params['n'], bernoulli_pos_params['p']/scale, 1)[0]
        return extra_dist
    