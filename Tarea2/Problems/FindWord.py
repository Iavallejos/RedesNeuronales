import random
from src.GeneticAlgorithm import GeneticAlgorithm
from src.Problem import Problem
import sys
sys.path.append('../')


class FindWord(Problem):
    def __init__(
        self,
        target_word,
        pop_size,
        mutation_rate,
        max_iter
    ):
        self.target_word = target_word.lower()
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.max_iter = max_iter
        self.number_of_genes = len(target_word)

    def fitness_function(self, individual):
        result = 0
        for i in range(len(self.target_word)):
            if individual[i] == self.target_word[i]:
                result += 1

        return result

    def gen_factory(self):
        genes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                 'n', 'ñ', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        index = random.randint(0, len(genes)-1)
        return genes[index]

    def individual_factory(self):
        return [self.gen_factory() for _ in range(self.number_of_genes)]

    def termination_condition(self, individual_fitness):
        return individual_fitness == len(self.target_word)

    def mutate(self, gene):
        new_gen = self.gen_factory()
        while new_gen == gene:
            new_gen = self.gen_factory()

        return new_gen

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
        )
        data = simulation.simulate()
        final_population = simulation.getPopulation()
        self.print_results(final_population, data)
        return (final_population, data)
