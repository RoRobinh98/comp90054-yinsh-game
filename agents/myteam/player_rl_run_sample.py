import time, random
import json
from template import Agent
from Yinsh.yinsh_model import YinshGameRule
from copy import deepcopy

THINKTIME = 0.85

class myAgent():
    def __init__(self, _id):
        self.id = _id
        self.game_rule = YinshGameRule(2)
        self.weight = [0, 0, 0, 0]
        self.round = 0
        # self.hValue = 
        with open("agents/myteam/rl_weights_sample.json", 'r', encoding='utf-8') as f:
            self.weight = json.load(f)['weight']

    def GetActions(self, state):
        return self.game_rule.getLegalActions(state, self.id)

    def DoAction(self, state, action):
        score = state.agents[self.id].score
        state = self.game_rule.generateSuccessor(state, action, self.id)
        return state.agents[self.id].score - score

    def CalRemainStepsFeature(self, state):
        return None

    def CalFeatures(self, game_state, action):
        state = deepcopy(game_state)
        features = []
        self_counter = 2 * (self.id + 1)
        oppo_counter = 2 * (1 - self.id + 1)

        score = self.DoAction(state, action)
        features.append(score/3)
        t_board = "".join(map(str,state.board))
        features.append(t_board.count(str(self_counter))/51)
        features.append(t_board.count(str(oppo_counter))/51)

        return features

    def CalQValue(self, game_state, action):
        features = self.CalFeatures(game_state, action)
        if len(self.weight) != len(features):
            print("Weights and features don't match.")
            return -99999
        else:
            q = 0
            for i in range(len(features)):
                q = q + features[i] * self.weight[i]
            return q

    def SelectAction(self, actions, game_state):
        self.round = self.round + 1
        start_time = time.time()
        best_q = -99999
        best_action = random.choice(actions)
        if self.round <= 5:
            return best_action
        else:
            pass
        for action in actions:
            if time.time() - start_time > THINKTIME:
                print("Time out.")
                break
            q = self.CalQValue(deepcopy(game_state), action)
            if q > best_q:
                best_q = q
                best_action = action
        print(time.time() - start_time)
        return best_action
