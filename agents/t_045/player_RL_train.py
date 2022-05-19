
import sys
sys.path.append('agents/t_045/')
import json

from copy import deepcopy
from template import Agent
from Yinsh.yinsh_model import YinshGameRule
import random, time, heapq

TIMELIMIT = 0.95
EPS = 0.001
ALPHA = 0.1
GAMMA = 0.9



class myAgent():
    def __init__(self, _id):
        self.id = _id
        self.game_rule = YinshGameRule(2)

        self.weight = [0, 0, 0, 0]
        self.round = 0
        with open("agents/t_045/weight.json", 'r', encoding='utf-8') as fw:
            self.weight = json.load(fw)['weight']
        print(self.weight)
        with open("agents/t_045/heuristic_chart.json", 'r', encoding='utf-8') as fw:
            self.hValue = json.load(fw)

    def GetActions(self, state):
        return self.game_rule.getLegalActions(state, self.id)

    def DoAction(self, state, action):
        score = state.agents[self.id].score
        state = self.game_rule.generateSuccessor(state, action, self.id)
        return state.agents[self.id].score - score

    def SelectAction(self, actions, currentState):
        self.round += 1
        start_time = time.time()
        
        best_action = random.choice(actions)
        if self.round <= 5:
            return best_action

        best_Q = -999

        if random.uniform(0,1) < 1 - EPS:
            for action in actions:
                if time.time() - start_time > TIMELIMIT:
                    print("time out!!!")
                    break
                Q_value = self.getQValue(deepcopy(currentState), action)
                if Q_value > best_Q:
                    best_Q = Q_value
                    best_action = action
        else:
            Q_value = self.getQValue(deepcopy(currentState), best_action)
            best_Q = Q_value
        features = self.getFeatures(deepcopy(currentState), best_action)

        next_state = deepcopy(currentState)
        reward = self.DoAction(next_state, best_action)

        next_actions = self.GetActions(next_state)
        best_next_Q = -999
        for next_action in next_actions:
            if time.time() - start_time > TIMELIMIT:
                print("time out!!!")
                break
            next_Q_value = self.getQValue(deepcopy(next_state), next_action)
            best_next_Q = max(best_next_Q, next_Q_value)

        for i in range(len(features)):
            self.weight[i] = self.weight[i] + ALPHA * (reward + GAMMA * best_next_Q - best_Q) * features[i]
        with open("agents/t_045/weight.json", 'w', encoding='utf-8') as f:
            json.dump({'weight': self.weight}, f)
        print(self.weight)

        # print(time.time() - start_time)
        return best_action

    def getQValue(self, state, action):
        features = self.getFeatures(state, action)
        #print("get features: " + str(features))
        if len(features) != len(self.weight):
            print("features and weight not matched!")
            return -999
        Q_value = 0
        for i in range(len(features)):
            Q_value += features[i]*self.weight[i]
        return Q_value

    def getFeatures(self, state, action):
        self_counter = 2*(self.id + 1)
        opponent_counter = 2*(2-self.id)
        features = []

        #feature1
        next_state = deepcopy(state)
        score = self.DoAction(next_state, action)
        if score > 1:
            print("score is greater than 1: " + str(score))
        features.append(score)

        #feature2
        current_board = "".join(map(str, next_state.board))
        self_counter_score = current_board.count(str(self_counter))/51
        features.append(self_counter_score)

        #feature3
        oppo_counter_score = current_board.count(str(opponent_counter))/51
        features.append(oppo_counter_score)

        #feature4
        features.append(self.getStepScore(next_state.board)/21)
        return features

    def getStepScore(self, board):
        self_ring = 2*(self.id+1)-1
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

        #print(max_value)
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

        #print("get a h value: "+ str(hValue))
        return hValue