import numpy as np
import json
from neural_network import NeuralNetwork
from core import index_loop
from objects import Result
from os import listdir

class Entity:
    def __init__(self):
        self.name = ""
        self.acceleration = 0
        self.max_speed = 0
        self.rotation_speed = 0
        self.friction = 0
        self.shape = []
        self.gen_count = 0
        self.max_score = 0
        self.lap_time = 0
        self.nn = None

    def increment_gen_count(self):
        self.gen_count += 1

    def get_nn(self):
        return self.nn

    def get_random_nn(self):
        nn = NeuralNetwork(self.shape)
        nn.set_random_weights()
        return nn

    def set_nn_from_result(self, result: Result):
        self.nn = result.nn
        self.lap_time = result.lap_time
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
        if not save_name.endswith(".json"):
            save_name += ".json"
        save_file = {
            "settings": self.get_save_parameters(),
            "weights": [np_arr.tolist() for np_arr in self.nn.weights],
            "biases": [np_arr.tolist() for np_arr in self.nn.biases]
        }
        with open(folder + "/" + save_name, "w") as json_file:
            json.dump(save_file, json_file)
        print("Saved ", save_name)

    def load_file(self, path):
        with open(path) as json_file:
            file_raw = json.load(json_file)

        file_parameters = file_raw["settings"]
        file_weights = file_raw["weights"]
        #file_biases = file_raw["biases"]

        self.nn = NeuralNetwork(file_parameters["shape"])
        self.nn.set_weights(file_weights)
        self.set_parameters_from_dict(file_parameters)

        print(f"Loaded {path}")

class Evolution:
    def __init__(self, mutation_rate=0.2, elitism_rate=0.1, crossover_rate=0.8):
        self.best_result = Result(None, -1, 0, 0)
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.crossover_rate = crossover_rate

    def load_generation(self, nn: NeuralNetwork, nn_stg: dict, population: int):
        return self.get_new_generation([nn], population)

    def get_new_generation(self, nns, population: int):
        new_generation = []

        # Randomly select elite individuals
        elite_count = int(self.elitism_rate * population)
        if elite_count > len(nns):
            elite_count = len(nns)  # Limit elite count to the population size
        elites = np.random.choice(nns, size=elite_count, replace=False)
        new_generation.extend(elites)

        while len(new_generation) < population:
            parent1 = self.select_parent(nns)
            parent2 = self.select_parent(nns)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_generation.append(child)

        return new_generation

    def get_new_generation_from_results(self, results, population: int, to_add_count=3):
        sorted_results = sorted(results, reverse=True)
        best_nns = [sorted_results[i].nn for i in range(min(to_add_count, len(sorted_results)))]
        return self.get_new_generation(best_nns, population)

    def find_best_result(self, results):
        current_best_result = max(results)
        self.best_result = current_best_result if current_best_result > self.best_result else self.best_result
        return self.best_result

    def crossover(self, parent1: NeuralNetwork, parent2: NeuralNetwork):
        child = NeuralNetwork(parent1.shape)
        for i in range(len(parent1.weights)):
            mask = np.random.rand(*parent1.weights[i].shape) < 0.5  # Randomly select from parents
            child.weights[i] = np.where(mask, parent1.weights[i], parent2.weights[i])
        return child

    def mutate(self, network: NeuralNetwork):
        mutated_network = network.get_deep_copy()

        # Mutate weights
        for i in range(len(mutated_network.weights)):
            mutation_mask = np.random.rand(*mutated_network.weights[i].shape) < self.mutation_rate
            mutation = np.random.normal(0, 0.1, mutated_network.weights[i].shape)  # Gaussian mutation
            mutated_network.weights[i] += np.where(mutation_mask, mutation, 0)

        # Mutate biases
        for i in range(len(mutated_network.biases)):
            mutation_mask = np.random.rand(*mutated_network.biases[i].shape) < self.mutation_rate
            mutation = np.random.normal(0, 0.1, mutated_network.biases[i].shape)  # Gaussian mutation
            mutated_network.biases[i] += np.where(mutation_mask, mutation, 0)

        return mutated_network

    def select_parent(self, networks):
    # Select a random contender as the parent
        return np.random.choice(networks)