import random
from src.GeneticAlgorithm import GeneticAlgorithm
from src.Problem import Problem
import sys
sys.path.append('../')


class FindWord(Problem):
    """Finds a given word via GeneticAlgorithm"""

    def __init__(
        self,
        target_word,
        pop_size,
        mutation_rate,
        max_iter
    ):
        """Creates a new FindWord Problem
        Parameters
        ----------
        target_word : string
            The word to find
        pop_size : int
            The number of existing individuals at any given time for the GeneticAlgorithm
        mutation_rate : float - from 0.0 to 1.0
            The probability of any gene in an individual to spontaneously change into another for the GeneticAlgorithm
        max_iter : positive int
            The maximum number of generations to occur in the simulation for the GeneticAlgorithm
        """
        self.target_word = target_word.lower()
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.max_iter = max_iter
        self.number_of_genes = len(target_word)

    def fitness_function(self, individual):
        """Returns the fitness of the given individual

        It calculates the number of equal characters in the same place between the individual and the target word
        """
        result = 0
        for i in range(len(self.target_word)):
            if individual[i] == self.target_word[i]:
                result += 1

        return result

    def gen_factory(self):
        """Returns a character between "a" and "z" """
        genes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                 'n', 'ñ', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        index = random.randint(0, len(genes)-1)
        return genes[index]

    def individual_factory(self):
        """Returns a new individual"""
        return [self.gen_factory() for _ in range(self.number_of_genes)]

    def termination_condition(self, individual_fitness):
        """The condition to terminate the simulation True if it's fulfulled

        The condition is individual_fitness equals to len of the target word, that means the individual is the same as the target
        """
        return individual_fitness == len(self.target_word)

    def mutate(self, gene):
        """Returns a new diferent gene"""
        new_gen = self.gen_factory()
        while new_gen == gene:
            new_gen = self.gen_factory()

        return new_gen

    def run(self, silent=False):
        """Runs the simulation and prints the results"""
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
        data = simulation.simulate(silent)
        final_population = simulation.getPopulation()
        if not silent:
            self.print_results(final_population, data)
        return (final_population, data)
