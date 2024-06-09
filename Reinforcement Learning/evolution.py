from core import index_loop
from objects import Result
from rl_agent import RLAgent
from os import listdir
import json


class Entity:
    """
    Reinforcement Learning agent entity with name, settings, agent, ...
    Savable and loadable.
    """

    def __init__(self):
        # parameters
        self.name = ""
        self.acceleration = 0
        self.max_speed = 0
        self.rotation_speed = 0
        self.friction = 0

        self.shape = []

        # info
        self.gen_count = 0
        self.max_score = 0
        self.lap_time = 0

        # result with agent to save
        self.agent = None

    # Add 1 to gen count
    def increment_gen_count(self):
        self.gen_count += 1

    # Get agent
    def get_agent(self):
        return self.agent

    # Get random agent with this shape
    def get_random_agent(self):
        state_size = self.shape[0]
        action_size = self.shape[1]
        return RLAgent(state_size, action_size)

    # set new result and max score
    def set_agent_from_result(self, result: Result):
        self.agent = result.agent
        self.max_score = result.score

    def get_car_parameters(self):
        return {
            "acceleration": self.acceleration,
            "max_speed": self.max_speed,
            "rotation_speed": self.rotation_speed,
            "friction": self.friction
        }

    def get_save_parameters(self):
        return {
            "name": self.name,
            "acceleration": self.acceleration,
            "max_speed": self.max_speed,
            "rotation_speed": self.rotation_speed,
            "shape": self.shape,
            "max_score": self.max_score,
            "lap_time": self.lap_time,
            "gen_count": self.gen_count,
            "friction": self.friction
        }

    def set_parameters_from_dict(self, par: dict):
        # get from dict, if not in set default
        self.name = par.get("name", self.name)

        self.acceleration = par.get("acceleration", self.acceleration)
        self.max_speed = par.get("max_speed", self.max_speed)
        self.rotation_speed = par.get("rotation_speed", self.rotation_speed)
        self.shape = par.get("shape", self.shape)
        self.friction = par.get("friction", self.friction)

        self.gen_count = par.get("gen_count", self.gen_count)
        self.max_score = par.get("max_score", self.max_score)
        self.lap_time = par.get("lap_time", self.lap_time)

    def save_file(self, save_name="", folder="saves"):
        # get name
        savefiles = listdir(folder)
        savename = save_name
        name_count = 0
        while savename + ".json" in savefiles:
            name_count += 1
            savename = "%s(%s)" % (name_count, name_count)
        savefile = {
            "settings": self.get_save_parameters(),
            "agent_state_dict": self.agent.policy.state_dict()
        }
        with open(folder + "/" + savename + ".json", "w") as json_file:
            json.dump(savefile, json_file)
        print("Saved ", savename)

    def load_file(self, path):
        try:
            with open(path) as json_file:
                file_data = json.load(json_file)

            file_parameters = file_data["settings"]
            agent_state_dict = file_data["agent_state_dict"]

            self.set_parameters_from_dict(file_parameters)
            state_size = self.shape[0]
            action_size = self.shape[1]
            self.agent = RLAgent(state_size, action_size)
            self.agent.policy.load_state_dict(agent_state_dict)

            print(f"Loaded {path}")
        except:
            print(f"Failed to load: {path}")
            return False

class Evolution:
    def __init__(self):
        self.best_result = Result(None, -1, 0, 0)
        self.mutation_rate = 0.1  # Adjust this value as needed

    def load_generation(self, agent: RLAgent, population: int):
        return self.get_new_generation([agent], population)

    def get_new_generation(self, agents: [RLAgent], population: int):
        return [agents[index_loop(i, len(agents))].reproduce(self.mutation_rate) for i in range(population)]

    def get_new_generation_from_results(self, results: [Result], population: int, to_add_count=3):
        best_agents = []
        sorted_results = sorted(results, reverse=True)

        to_add = to_add_count if len(
            sorted_results) >= to_add_count else len(sorted_results)
        for i in range(to_add):
            best_agents.append(sorted_results[i].agent)

        return self.get_new_generation(best_agents, population)

    def find_best_result(self, results: [Result]):
        current_best_result = max(results)
        self.best_result = current_best_result if current_best_result > self.best_result else self.best_result
        return self.best_result


class CustomEvolution(Evolution):
    pass
