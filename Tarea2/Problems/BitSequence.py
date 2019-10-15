from ..src.Problem import Problem
from ..src.GeneticAlgorithm import GeneticAlgorithm
import random


class BitSequence(Problem):
    def __init__(
        self, target_number, pop_size, mutation_rate, number_of_genes, max_iter
    ):
        self.target = target_number
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.number_of_genes = number_of_genes
        self.max_iter = max_iter

    def fitness_function(self, individual):
        result = 0
        exp = 1
        for gen in individual[::-1]:
            result += int(gen) * exp
            exp *= 2

        return -abs(self.target - result)

    def gen_factory(self):
        if random.random() > 0.5:
            return "1"
        return "0"

    def individual_factory(self):
        return [self.gen_factory() for _ in range(self.number_of_genes)]

    def termination_condition(self, individual_fitness):
        return individual_fitness == 0

    def run(self):
        simulation = GeneticAlgorithm(
            self.pop_size,
            self.mutation_rate,
            self.fitness_function,
            self.individual_factory,
            self.gen_factory,
            self.termination_condition,
            self.max_iter,
        )
        data = simulation.simulate()
        final_population = simulation.getPopulation()
        return (final_population, data)
