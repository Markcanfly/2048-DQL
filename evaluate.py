import numpy as np

def largest_tile(state):
    return np.max(state)

def summary(stats) -> str: # TODO better summary
    '''Takes dictionary in form of largest_tile:n_achieved'''
    s = ''

    n_episodes = sum(stats.values())
    # Calculate average score
    avg_score = 0
    for score, n in stats.items():
        avg_score += score * n / n_episodes
    s += f'Average score: {round(avg_score)}\n'
    # Draw table
    maxtiles_sorted_by_key = [(k, v) for k,v in sorted(stats.items(), key=lambda x: x[0])]
    accumulated_achieved = {}
    for highscore in stats:
        accumulated_score = 0
        for highscore_ in stats:
            if highscore_ >= highscore:
                accumulated_score += stats[highscore_]
        accumulated_achieved[highscore] = accumulated_score
    s += 'Score N_achieved %Achieved\n'
    for highest in sorted(stats.keys(), reverse=True):
        s += "{:<5} {:<10} {:<6}%\n".format(highest, stats[highest], round(accumulated_achieved[highest]/n_episodes*100,2))
    return s
