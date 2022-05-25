import sys

sys.path.append('agents/t_045/')
import json

from copy import deepcopy
from Yinsh.yinsh_model import YinshGameRule
import random, time
import numpy as np

TIMELIMIT = 0.95

def listFunc(l1, l2):
        o = -1
        for i in range(len(l1)):
            if l1[i] in l2:
                o = i
                break
        return o

class myAgent():
    def __init__(self, _id):
        self.id = _id
        self.game_rule = YinshGameRule(2)

        self.weight = [1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.1]
        self.round = 0
        with open("agents/t_045/weight_train.json", 'r', encoding='utf-8') as fw:
            self.weight = json.load(fw)['weight']
                # print(self.weight)
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
        current_self_score = current_state.agents[self.id].score
        current_oppo_score = current_state.agents[1 - self.id].score
        opponent_ring = len(current_state.ring_pos[1 - self.id])
        current_danger = 0
        current_danger = self.getDangercombine(current_state, self.id, opponent_ring, current_danger)
        current_board = "".join(map(str, current_state.board))
        current_self_counter = current_board.count(str(self_counter))
        current_oppo_counter = current_board.count(str(opponent_counter))

        next_state = deepcopy(state)

        # feature1 If the agent will get a score in the next step
        score = self.DoAction(next_state, action)
        if score > 1:
            print("score is greater than 1: " + str(score))
        features.append(score)

        # feature2 Self-counters left on board if such action taken compared with counters left before action taken
        next_board = "".join(map(str, next_state.board))
        next_self_counter = next_board.count(str(self_counter))
        next_self_score = next_state.agents[self.id].score
        if next_self_score - current_self_score:
            features.append(1)
        else:
            features.append((next_self_counter - current_self_counter)/51)

        # feature3 Opponent’s counters left on board if such action taken compared with counters left before action taken
        next_oppo_counter = next_board.count(str(opponent_counter))
        next_oppo_score = next_state.agents[1-self.id].score
        if next_oppo_score - current_oppo_score:
            features.append(-1)
        else:
            features.append(-(next_oppo_counter - current_oppo_counter)/51)

        # feature4 The inverse of heuristic value of next action
        features.append(self.getStepScore(next_state.board))

        # feature5 The number of opponent's rings around our rings on the board
        features.append(self.getComponentsAround(next_state, 2 * (1 - self.id) + 1) / 6)

        # feature6 danger combines
        next_danger = self.getDangercombine(next_state, self.id, opponent_ring, current_danger)
        if current_danger != 0:
            danger_feature = (current_danger-next_danger)/current_danger
        else:
            danger_feature = 0
        features.append(danger_feature)

        # feature7 how many legal ring moves for oppo
        legalMoveNum = 0
        # feature8 how many counters controlled by oppo rings
        cntrControlNum = 0
        for r in self.getOppoRingsPos(next_state.board):
            for v in self.getRingLines(next_state.board, r).values():
                if v:
                    # index of first non-blank
                    idx1 = listFunc(v,[1,2,3,4,5])
                    if idx1 == -1:
                        legalMoveNum = legalMoveNum + len(v)
                    else:
                        vv = v[idx1:]
                        flag = listFunc(vv,[0,1,3,5])
                        if flag == -1:
                            legalMoveNum = legalMoveNum + idx1
                        else:
                            # index of first interruption 
                            idx2 = idx1 + flag
                            if v[idx2] == 0:
                                legalMoveNum = legalMoveNum + idx1 + 1
                                cntrControlNum = cntrControlNum + flag
                            else:
                                legalMoveNum = legalMoveNum + idx1
        # print(legalMoveNum)
        # print(cntrControlNum)
        features.append(-legalMoveNum/51)
        features.append(-cntrControlNum/51)
        
        return features

    def getSelfRingsPos(self, board):
        rings = []
        for i in range(11):
            for j in range(11): 
                if board[i][j] == 2 * self.id + 1:
                    rings.append((i,j))
        return rings

    def getOppoRingsPos(self, board):
        rings = []
        for i in range(11):
            for j in range(11): 
                if board[i][j] == 3 - 2 * self.id:
                    rings.append((i,j))
        return rings

    def getRingLines(self, board, ring):
        l = dict()
        l['w'], l['e'], l['n'], l['s'], l['ws'], l['en'] = ([] for i in range(6))
        for i in reversed(range(ring[1])):
            l['w'].append(board[ring[0]][i])
        for i in range(ring[1]+1,11):
            l['e'].append(board[ring[0]][i])
        for i in reversed(range(ring[1])):
            l['n'].append(board[i][ring[1]])
        for i in range(ring[0]+1,11):
            l['w'].append(board[i][ring[1]])
        for i in range(1,min(11-ring[0],ring[1]+1)):
            l['ws'].append(board[ring[0]+i][ring[1]-i])
        for i in range(1,min(11-ring[1],ring[0]+1)):
            l['en'].append(board[ring[0]-i][ring[1]+i])
        # print(l)
        return l

    def getStepScore(self, board):
        self_ring = 2 * (self.id + 1) - 1
        hValue = 51
        for i in range(1, 10):
            for j in range(11):
                if j + 4 <= 10:
                    horizon = [board[i][j], board[i][j + 1], board[i][j + 2], board[i][j + 3], board[i][j + 4]]
                    if 5 in horizon:
                        continue
                    else:
                        horizonValue = self.HeuristicValue(horizon, self.id)
                        hValue = min(hValue, horizonValue)
                if i + 4 <= 10:
                    vertical = [board[i][j], board[i + 1][j], board[i + 2][j], board[i + 3][j], board[i + 4][j]]
                    if 5 in vertical:
                        continue
                    else:
                        verticalValue = self.HeuristicValue(vertical, self.id)
                        hValue = min(hValue, verticalValue)

                if i + 4 <= 10 and j - 4 >= 0:
                    slant = [board[i][j], board[i + 1][j - 1], board[i + 2][j - 2], board[i + 3][j - 3],
                             board[i + 4][j - 4]]
                    if 5 in slant:
                        continue
                    else:
                        slantValue = self.HeuristicValue(slant, self.id)
                        hValue = min(hValue, slantValue)

        # print(1/hValue)
        if hValue != 0:
            return 1/hValue
        else:
            return 1.0

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
            return 2*current_danger

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
