import numpy as np

def largest_tile(state):
    return np.max(state)

def summary(stats) -> str: # TODO better summary
    '''Takes dictionary in form of largest_tile:n_achieved'''
    s = ''
    n_episodes = sum(stats.values())
    maxtiles_sorted_by_key = [(k, v) for k,v in sorted(stats.items(), key=lambda x: x[0])]
    accumulated_achieved = {}
    for highest in sorted(stats.keys(), reverse=True):
        s += '{}:\t{}\n'.format(highest, stats[highest])
    return s
