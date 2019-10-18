import random
import numpy as np
from src.GeneticAlgorithm import GeneticAlgorithm
from src.Problem import Problem
import sys
sys.path.append('../')


class Knapsack01(Problem):
    def __init__(
        self,
        pop_size,
        mutation_rate,
        max_iter
    ):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.max_iter = max_iter
        self.number_of_genes = 5
        self.max_weight = 15
        self.choices = ((12, 4), (2, 2), (1, 2), (1, 1), (4, 10))

    def fitness_function(self, individual):
        value = 0
        for index, gene in enumerate(individual):
            if gene == "1":
                value += self.choices[index][1]

        return value

    def gen_factory(self):
        if random.random() > 0.5:
            return "1"
        return "0"

    def individual_factory(self):
        return [self.gen_factory() for _ in range(self.number_of_genes)]

    def termination_condition(self, individual_fitness):
        return individual_fitness == 15

    def mutate(self, gene):
        if gene == "0":
            return "1"
        return "0"

    def individual_viability_condition(self, individual):
        weight = 0
        for index, gene in enumerate(individual):
            if gene == "1":
                weight += self.choices[index][0]

        return weight <= 15

    def viable_individual_factory(self):
        individual = [self.gen_factory() for _ in range(self.number_of_genes)]
        while not self.individual_viability_condition(individual):
            individual = [self.gen_factory()
                          for _ in range(self.number_of_genes)]
        return individual

    def viable_mutation(self, individual, index):
        new_individual = individual.copy()
        # this mutation creates a viable individual
        if individual[index] == "1":
            return "0"

        # check if mutation creates a viable individual
        new_individual[index] = "1"
        if self.individual_viability_condition(new_individual):
            return "1"

        # not viable individual, no mutation
        else:
            return "0"

    def viable_crossover(self, parent_1, parent_2):
        # 1 to len -2 (ex [1,2,3,4,5]=> index from 1 to 3) so it chooses a part of both parents
        gen_breakpoint_index = np.random.randint(1, len(parent_1)-1)

        if np.random.random() < 0.5:
            child = parent_1[:gen_breakpoint_index] + \
                parent_2[gen_breakpoint_index:]
        else:
            child = parent_2[:gen_breakpoint_index] + \
                parent_1[gen_breakpoint_index:]

        if self.individual_viability_condition(child):
            return child

        for i in range(len(parent_1)):
            new_child = child.copy()
            new_child[i] = parent_1[i]
            if self.individual_viability_condition(new_child):
                return new_child
            new_child[i] = parent_2[i]
            if self.individual_viability_condition(new_child):
                return new_child

    def run(self):
        simulation = GeneticAlgorithm(
            pop_size=self.pop_size,
            mutation_rate=self.mutation_rate,
            fitness_function=self.fitness_function,
            individual_factory=self.individual_factory,
            gene_factory=self.gen_factory,
            mutate=self.mutate,
            termination_condition=self.termination_condition,
            max_iter=self.max_iter,
            individual_viability_check=True,
            individual_viability_condition=self.individual_viability_condition,
            viable_mutation=self.viable_mutation,
            viable_crossover=self.viable_crossover,
            viable_individual_factory=self.viable_individual_factory,
        )
        data = simulation.simulate()
        final_population = simulation.getPopulation()
        self.print_results(final_population, data)
        return (final_population, data)
