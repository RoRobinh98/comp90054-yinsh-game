import re
from collections import deque
from copy import deepcopy

import numpy as np

from template import Agent
from Yinsh.yinsh_model import YinshGameRule
import random, time

THINKTIME = 0.95
SIMULATE_MAX_DEPTH = 20
LAMBDA = 0.95
GAMMA = 0.95


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.game_rule = YinshGameRule(2)
        self.count = 0

    def GetActions(self, state):
        return self.game_rule.getLegalActions(state, self.id)

    def DoAction(self, state, action):
        score = state.agents[self.id].score
        state = self.game_rule.generateSuccessor(state, action, self.id)
        return state.agents[self.id].score - score

    def Q_function(self, reward, power, para=LAMBDA):
        return reward * (para ** power)

    def GetBoard(self, state):
        board = state.board
        str_board = str(board)
        str_board = re.sub('\\[', '', str_board)
        str_board = re.sub('\\]', '', str_board)
        return str_board

    def SelectAction(self, actions, game_state):

        start_time = time.time()

        # final action chosen by MCT. random it first incase out of time
        random_action = random.choice(actions)

        # used for storing value, count, and action for backpropagation
        state_value = dict()
        state_count = dict()
        state_action = dict()

        while time.time() - start_time < THINKTIME:

            mct_action = None
            reward = 0

            # expand node
            expand_state = deepcopy(game_state)

            current_actions = actions
            current_board = self.GetBoard(game_state)
            queue_board = deque()
            queue_action = deque()
            queue_board.append(current_board)

            while current_board in state_value and len(current_actions) > 0 and not reward:

                if time.time() - start_time >= THINKTIME:
                    return random_action

                # multi arm bandit greedy method
                if random.uniform(0, 1) > GAMMA and current_board in state_action:
                    select_action = state_action[current_board]
                else:
                    select_action = random.choice(current_actions)

                # selection part
                if not mct_action:
                    mct_action = select_action
                queue_action.append(select_action)
                next_state = deepcopy(expand_state)
                reward = self.DoAction(next_state, select_action)
                current_board = self.GetBoard(next_state)
                queue_board.append(current_board)
                current_actions = self.GetActions(next_state)
                expand_state = next_state

            # simulation part
            depth = 0
            if len(current_actions) > 0 and reward == 0:
                while not reward and len(current_actions) > 0 and depth < SIMULATE_MAX_DEPTH:

                    if time.time() - start_time >= THINKTIME:
                        return random_action

                    simulate_action = random.choice(current_actions)
                    next_state = deepcopy(expand_state)
                    reward = self.DoAction(next_state, simulate_action)
                    current_actions = self.GetActions(next_state)
                    expand_state = next_state
                    depth += 1

            # get Q-value
            value = self.Q_function(reward, depth)
            queue_action.append('token')

            # backpropagation
            while len(queue_board):

                if time.time() - start_time >= THINKTIME:
                    return random_action

                board = queue_board.pop()
                action = queue_action.pop()

                if board in state_value:
                    if len(queue_board) == 0 and value > state_value[board]:
                        random_action = mct_action
                    if value > state_value[board]:
                        state_value[board] = value
                        if action != 'token':
                            state_action[board] = action
                    state_count[board] += 1
                else:
                    state_value[board] = value
                    state_count[board] = 1
                    if action != 'token':
                        state_action[board] = action

                value *= LAMBDA

        return random_action
