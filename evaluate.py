import numpy as np

def largest_tile(state):
    return np.max(state)

def summary(stats) -> str: # TODO better summary
    '''Takes dictionary in form of largest_tile:n_achieved'''
    s = ''
    n_episodes = sum(stats.values())
    maxtiles_sorted_by_key = [(k, v) for k,v in sorted(stats.items(), key=lambda x: x[0])]
    accumulated_achieved = {}
    for highscore in stats:
        accumulated_score = 0
        for highscore_ in stats:
            if highscore_ < highscore:
                accumulated_score += stats[highscore_]
        accumulated_achieved[highscore] = accumulated_score
    s += 'Score\tN_achieved\t%acc_achieved'
    for highest in sorted(stats.keys(), reverse=True):
        s += f'{highest}:\t{stats[highest]}\t{round(accumulated_achieved[highest]/n_episodes,2)*100}%\n'
    return s
