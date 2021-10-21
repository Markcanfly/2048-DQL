'''
Basic implementation of the 2048 game
'''
from enum import Enum
from random import random
from inspect import cleandoc as cleanindent

class Game:
    class Direction(Enum):
        UP = 'up'
        DOWN = 'down'
        LEFT = 'left'
        RIGHT = 'right'
    def __init__(self):
        self.grid = [[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]]
    def shift(self, direction: Direction):
        pass # TODO
    def add_random_tile(self):
        pass # TODO
    @property
    def game_lost(self) -> bool:
        pass # TODO
    def step(self, direction: Direction) -> bool:
        '''Takes a step in the game
        Args:
            direction: The direction of the step
        Returns:
            game_running: whether after the step, the game is still on
        '''
        self.shift(direction)
        self.add_random_tile()
        return self.game_lost
    def __str__(self):
        return(cleanindent(f'''-----------------
                               | {0} | {1} | {2} | {3} |
                               ----+---+---+----
                               | {4} | {5} | {6} | {7} |
                               ----+---+---+----
                               | {8} | {9} | {10} | {11} |
                               ----+---+---+----
                               | {12} | {13} | {14} | {15} |
                               -----------------
                               '''.format(*[y for x in self.grid for y in x])))
