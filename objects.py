from rl_agent import RLAgent

class Result:
    def __init__(self, agent: RLAgent, score: int, dist_to_next_cp: float):
        self.agent = agent
        self.score = score
        self.dist_to_next_cp = dist_to_next_cp

    def __lt__(self, other):
        if self.score < other.score:
            return True
        elif self.score == other.score:
            return self.dist_to_next_cp > other.dist_to_next_cp
        else:
            return False

    def __le__(self, other):
        return self < other or self == other

    def __eq__(self, other):
        return self.score == other.score and self.dist_to_next_cp == other.dist_to_next_cp

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        if self.score > other.score:
            return True
        elif self.score == other.score:
            return self.dist_to_next_cp < other.dist_to_next_cp
        else:
            return False

    def __ge__(self, other):
        return self > other or self == other