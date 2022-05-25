import sys

sys.path.append('agents/t_045/')
import json

from copy import deepcopy
from template import Agent
from Yinsh.yinsh_model import YinshGameRule
from Yinsh.yinsh_utils import ILLEGAL_POS
import random, time, heapq
import numpy as np

TIMELIMIT = 0.95


class myAgent():
    def __init__(self, _id):
        self.id = _id
        self.game_rule = YinshGameRule(2)

        self.weight = [1, 0.1, 0.1, 0.2, 0.1, 0.2]
        self.round = 0
        self.RING_BOARD = [[-1, -1, -1, -1, -1, -1, 4, 4, 4, 4, -1],
                           [-1, -1, -1, -1, 4, 5, 5, 5, 5, 5, 4],
                           [-1, -1, -1, 4, 5, 6, 6, 6, 6, 5, 4],
                           [-1, -1, 4, 5, 6, 7, 7, 7, 6, 5, 4],
                           [-1, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4],
                           [-1, 5, 6, 7, 8, 9, 8, 7, 6, 5, -1],
                           [4, 5, 6, 7, 8, 8, 7, 6, 5, 4, -1],
                           [4, 5, 6, 7, 7, 7, 6, 5, 4, -1, -1],
                           [4, 5, 6, 6, 6, 6, 5, 4, -1, -1, -1],
                           [4, 5, 5, 5, 5, 5, 4, -1, -1, -1, -1],
                           [-1, 4, 4, 4, 4, -1, -1, -1, -1, -1, -1]]

    def GetActions(self, state):
        return self.game_rule.getLegalActions(state, self.id)

    def DoAction(self, state, action):
        score = state.agents[self.id].score
        state = self.game_rule.generateSuccessor(state, action, self.id)
        return state.agents[self.id].score - score

    def SelectRing(self, state):
        current_max = -1
        possible_list = []
        ring_1, ring_2 = state.ring_pos
        for pos in ring_1:
            self.RING_BOARD[pos[0]][pos[1]] = -1
        for pos in ring_2:
            self.RING_BOARD[pos[0]][pos[1]] = -1
        for i in range(10):
            for j in range(10):
                current_max = max(current_max, self.RING_BOARD[i][j])
        for i in range(10):
            for j in range(10):
                if self.RING_BOARD[i][j] == current_max:
                    possible_list += [(i, j)]
        return random.choice(possible_list)

    def SelectAction(self, actions, currentState):
        self.round += 1
        start_time = time.time()

        best_action = random.choice(actions)
        if self.round <= 5:
            best_action['place pos'] = self.SelectRing(currentState)
            return best_action

        best_Q = -999
        for action in actions:
            if time.time() - start_time > TIMELIMIT:
                # print(time.time() - start_time)
                # print("time out!!!")
                break
            Q_value = self.getQValue(deepcopy(currentState), action)
            # print("get qValue:" + str(Q_value))
            if Q_value > best_Q:
                best_Q = Q_value
                best_action = action
        return best_action

    def getQValue(self, state, action):
        features = self.getFeatures(state, action)
        # print("get features: " + str(features))
        if len(features) != len(self.weight):
            print("features and weight not matched!")
            return -999
        Q_value = 0
        for i in range(len(features)):
            Q_value += features[i] * self.weight[i]
        return Q_value

    def getFeatures(self, state, action):
        self_counter = 2 * (self.id + 1)
        opponent_counter = 2 * (2 - self.id)
        features = []

        current_state = deepcopy(state)
        opponent_ring = len(current_state.ring_pos[1 - self.id])
        current_danger = 0
        current_danger = self.getDangercombine(current_state, self.id, opponent_ring, current_danger)
        next_state = deepcopy(state)

        # feature1
        score = self.DoAction(next_state, action)
        if score > 1:
            print("score is greater than 1: " + str(score))
        features.append(score)

        # feature2
        current_board = "".join(map(str, next_state.board))
        self_counter_score = current_board.count(str(self_counter)) / 51
        features.append(self_counter_score)

        # feature3
        oppo_counter_score = current_board.count(str(opponent_counter)) / 51
        features.append(-oppo_counter_score)

        # feature4
        features.append(self.getStepScore(next_state.board) / 21)

        # feature5 棋盘中我方环周围的对方环数量
        features.append(self.getComponentsAround(next_state, 2 * (1 - self.id) + 1) / 6)

        # feature6 danger combines
        next_danger = self.getDangercombine(next_state, self.id, opponent_ring, current_danger)
        if current_danger != 0:
            danger_feature = (current_danger - next_danger) / current_danger
        else:
            danger_feature = 0
        features.append(danger_feature)

        # feature how many positions are colinear with self rings
        # colinearPos = set()
        # for r in self.getSelfRingsPos(next_state.board):
        #     for i in range(11):
        #         if (r[0],i) not in ILLEGAL_POS and i != r[1]:
        #             colinearPos.add((r[0],i))
        #         if (i,r[1]) not in ILLEGAL_POS and i != r[0]:
        #             colinearPos.add((i,r[1]))
        #         if i != 0 and r[0] - i >= 0 and r[0] + i <= 10 and (r[0]-i,r[1]+i) not in ILLEGAL_POS:
        #             colinearPos.add((r[0]-i,r[1]+i))
        #         if i != 0 and r[0] + i <= 10 and r[0] - i >= 0 and (r[0]+i,r[1]-i) not in ILLEGAL_POS:
        #             colinearPos.add((r[0]+i,r[1]-i))
        # features.append(len(colinearPos)/51)

        return features

    def getSelfRingsPos(self, board):
        rings = []
        for i in range(11):
            for j in range(11):
                if board[i][j] == 2 * self.id + 1:
                    rings.append((i, j))
        return rings

    def getOppoRingsPos(self, board):
        rings = []
        for i in range(11):
            for j in range(11):
                if board[i][j] == 3 - 2 * self.id:
                    rings.append((i, j))
        return rings

    def getStepScore(self, board):
        self_ring = 2 * (self.id + 1) - 1
        max_value = 0
        hValue = 51
        for i in range(1, 10):
            for j in range(11):
                if j + 4 <= 10:
                    value1 = 0
                    horizon = [board[i][j], board[i][j + 1], board[i][j + 2], board[i][j + 3], board[i][j + 4]]
                    if horizon.count(self_ring) > 0:
                        value1 += 10
                    if 5 in horizon:
                        continue
                    else:
                        horizonValue = self.HeuristicValue(horizon, self.id)

                        horizonValue = min(hValue, horizonValue)
                        value1 += (5 - horizonValue) * 2
                    max_value = max(max_value, value1)

                if i + 4 <= 10:
                    value2 = 0
                    vertical = [board[i][j], board[i + 1][j], board[i + 2][j], board[i + 3][j], board[i + 4][j]]
                    if vertical.count(self_ring) > 0:
                        value2 += 10
                    if 5 in vertical:
                        continue
                    else:
                        verticalValue = self.HeuristicValue(vertical, self.id)
                        verticalValue = min(hValue, verticalValue)
                        value2 += (5 - verticalValue) * 2
                    max_value = max(max_value, value2)

                if i + 4 <= 10 and j - 4 >= 0:
                    value3 = 0
                    slant = [board[i][j], board[i + 1][j - 1], board[i + 2][j - 2], board[i + 3][j - 3],
                             board[i + 4][j - 4]]
                    if slant.count(self_ring) > 0:
                        value3 += 10
                    if 5 in slant:
                        continue
                    else:
                        slantValue = self.HeuristicValue(slant, self.id)
                        slantValue = min(hValue, slantValue)
                        value3 += (5 - slantValue) * 2
                    max_value = max(max_value, value3)

        # print(max_value)
        return max_value

    def HeuristicValue(self, list, id):
        if 5 in list:
            return 999
        if (id == 0 and 3 in list) or (id == 1 and 1 in list):
            return 999

        simplify_list = [list[0]]
        hValue = 0
        for i in range(1, 5):
            if list[i] == list[i - 1] and (list[i - 1] == 2 or list[i - 1] == 4):
                continue
            simplify_list.append(list[i])

        for i in simplify_list:
            if i == (id * 2 + 1) or i == 0:
                hValue += 1

        if 0 in list:
            hValue += 1

        if id == 0:
            hValue += max(simplify_list.count(4) - simplify_list.count(1), 0)

        if id == 1:
            hValue += max(simplify_list.count(2) - simplify_list.count(3), 0)

        # print("get a h value: "+ str(hValue))
        return hValue

    def getComponentsAround(self, state, component):
        components = 0
        self_rings = state.ring_pos[self.id]
        board = state.board
        larger_board = []
        larger_board.insert(0, [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        for i in range(11):
            bList = board[i].tolist()
            bList.insert(0, 5)
            bList.append(5)
            larger_board.append(bList)
        larger_board.append([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        new_board = np.array(larger_board)
        for self_ring in self_rings:
            (i, j) = self_ring
            # print("r: " + str((i, j)))
            if new_board[i + 1][j] == component:
                components += 1
            if new_board[i][j + 1] == component:
                components += 1
            if new_board[i - 1][j] == component:
                components += 1
            if new_board[i][j - 1] == component:
                components += 1
            if new_board[i + 1][j + 1] == component:
                components += 1
            if new_board[i - 1][j - 1] == component:
                components += 1
        # print(components / len(self_rings))
        return components / len(self_rings)

    def getDangercombine(self, state, id, current_ring, current_danger):
        next_ring = len(state.ring_pos[1 - self.id])
        if current_ring > next_ring:
            # print("yes")
            return 2 * current_danger

        board = state.board
        dangers = 0
        if id == 0:
            opponent = 4
        else:
            opponent = 2
        for i in range(1, 10):
            for j in range(11):
                if j + 4 <= 10:
                    horizon = [board[i][j], board[i][j + 1], board[i][j + 2], board[i][j + 3], board[i][j + 4]]
                    if horizon.count(opponent) > 3:
                        dangers += 1
                if i + 4 <= 10:
                    vertical = [board[i][j], board[i + 1][j], board[i + 2][j], board[i + 3][j], board[i + 4][j]]
                    if vertical.count(opponent) > 3:
                        dangers += 1

                if i + 4 <= 10 and j - 4 >= 0:
                    slant = [board[i][j], board[i + 1][j - 1], board[i + 2][j - 2], board[i + 3][j - 3],
                             board[i + 4][j - 4]]
                    if slant.count(opponent) > 3:
                        dangers += 1

        return dangers
