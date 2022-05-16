from copy import deepcopy

from template import Agent
from Yinsh.yinsh_model import YinshGameRule
import random, time, heapq

THINKTIME = 0.55


class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)


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
        return state.agents[self.id].score > score

    def GetHeuristic(self, state):
        board = state.board
        hValue = 999
        for i in range(1, 10):
            for j in range(11):
                if j + 4 <= 10:
                    horizon = [board[i][j], board[i][j + 1], board[i][j + 2], board[i][j + 3], board[i][j + 4]]
                    horizonValue = self.HeuristicValue(horizon, self.id)
                    hValue = min(hValue, horizonValue)
                if i + 4 <= 10:
                    vertical = [board[i][j], board[i + 1][j], board[i + 2][j], board[i + 3][j], board[i + 4][j]]
                    verticalValue = self.HeuristicValue(vertical, self.id)
                    hValue = min(hValue, verticalValue)
                if i + 4 <= 10 and j - 4 >= 0:
                    slant = [board[i][j], board[i + 1][j - 1], board[i + 2][j - 2], board[i + 3][j - 3],
                             board[i + 4][j - 4]]
                    slantValue = self.HeuristicValue(slant, self.id)
                    hValue = min(hValue, slantValue)

        return hValue

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

        return hValue

    def SelectAction(self, actions, game_state):

        start_time = time.time()
        myPQ = PriorityQueue()
        visited = set()
        best_g = dict()
        startState = deepcopy(game_state)
        startNode = (startState, 0, [])
        myPQ.push(startNode, 0)

        while not myPQ.isEmpty() and time.time() - start_time < THINKTIME:

            state, cost, path = myPQ.pop()

            if (not state in visited) or cost < best_g.get(state):
                visited.add(state)
                best_g[state] = cost
                new_actions = self.GetActions(state)
                for a in new_actions:
                    next_state = deepcopy(state)
                    next_path = path + [a]
                    next_cost = cost + 1
                    reward = self.DoAction(next_state, a)
                    if reward:
                        print(f'Move {len(next_path)}, path found:', next_path)
                        return next_path[0]
                    else:
                        priority = cost + 1 + self.GetHeuristic(next_state)
                        myPQ.push((next_state, next_cost, next_path), priority)

        return random.choice(actions)
