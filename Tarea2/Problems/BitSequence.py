import random
from src.GeneticAlgorithm import GeneticAlgorithm
from src.Problem import Problem
import sys
sys.path.append('../')


class BitSequence(Problem):
    """Finds a representation using a sequence of bits of an int via GeneticAlgorithm"""

    def __init__(
        self, target_number, pop_size, mutation_rate, number_of_genes, max_iter
    ):
        """Creates a new BitSequence Problem
        Parameters
        ----------
        target : positive int
            The number to represent
        pop_size : int
            The number of existing individuals at any given time for the GeneticAlgorithm
        mutation_rate : float - from 0.0 to 1.0
            The probability of any gene in an individual to spontaneously change into another for the GeneticAlgorithm
        number_of_genes : int
            The length of the sequence, if the length is too short it will return the closest number
        max_iter : positive int
            The maximum number of generations to occur in the simulation for the GeneticAlgorithm
        """
        self.target = target_number
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.number_of_genes = number_of_genes
        self.max_iter = max_iter

    def fitness_function(self, individual):
        """Returns the fitness of the given individual

        It calculates the decimal representation of the individual and returns the negative of the absolute value of the difference between the target and the individual
        """
        result = 0
        exp = 1
        for gene in individual[::-1]:
            result += int(gene) * exp
            exp *= 2

        return -abs(self.target - result)

    def gen_factory(self):
        """Flips a coin and returns "1" or "0" """
        if random.random() > 0.5:
            return "1"
        return "0"

    def individual_factory(self):
        """Returns a new individual"""
        return [self.gen_factory() for _ in range(self.number_of_genes)]

    def termination_condition(self, individual_fitness):
        """The condition to terminate the simulation True if it's fulfulled

        The condition is individual_fitness equals to 0, that means the individual is the same as the target
        """
        return individual_fitness == 0

    def mutate(self, gene):
        """Returns a new diferent gene"""
        if gene == '0':
            return '1'
        return '0'

    def set_pop_size(self, pop_size):
        """Sets new pop_size"""
        self.pop_size = pop_size

    def set_mutation_rate(self, mutation_rate):
        """Sets new mutation_rate"""
        self.mutation_rate = mutation_rate

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
