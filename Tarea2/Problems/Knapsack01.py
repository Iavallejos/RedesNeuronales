import random
import numpy as np
from src.GeneticAlgorithm import GeneticAlgorithm
from src.Problem import Problem
import sys
sys.path.append('../')


class Knapsack01(Problem):
    """Finds the best combination for the 0-1-Knapsack problem with 5 given boxes via GeneticAlgorithm"""

    def __init__(
        self,
        pop_size,
        mutation_rate,
        max_iter
    ):
        """Creates a new Knapsack01 Problem
        Parameters
        ----------
        pop_size : int
            The number of existing individuals at any given time for the GeneticAlgorithm
        mutation_rate : float - from 0.0 to 1.0
            The probability of any gene in an individual to spontaneously change into another for the GeneticAlgorithm
        max_iter : positive int
            The maximum number of generations to occur in the simulation for the GeneticAlgorithm
        """
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.max_iter = max_iter
        self.number_of_genes = 5
        self.max_weight = 15
        self.choices = ((12, 4), (2, 2), (1, 2), (1, 1), (4, 10))

    def fitness_function(self, individual):
        """Returns the fitness of the given individual

        It calculates value of the individual
        """
        value = 0
        for index, gene in enumerate(individual):
            if gene == "1":
                value += self.choices[index][1]

        return value

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

        The condition is individual_fitness equals to 15 which is the best combination value, that means the individual is the best
        """
        return individual_fitness == 15

    def mutate(self, gene):
        """Returns a new diferent gene"""
        if gene == "0":
            return "1"
        return "0"

    def individual_viability_condition(self, individual):
        """The condition that makes a individual viable, True if it's met

        The condition is that the weight is less than 15
        """
        weight = 0
        for index, gene in enumerate(individual):
            if gene == "1":
                weight += self.choices[index][0]

        return weight <= 15

    def viable_individual_factory(self):
        """Returns a new viable individual

        it creates new individuals until it creates a viable one
        """
        individual = [self.gen_factory() for _ in range(self.number_of_genes)]
        while not self.individual_viability_condition(individual):
            individual = [self.gen_factory()
                          for _ in range(self.number_of_genes)]
        return individual

    def viable_mutation(self, individual, index):
        """Returns a valid mutation for the gene in the given index for the given individual
        
        if the given gene is "1" returns "0"
        if the given gene is "0" it calculates de viability, if it is viable returns "1", else returns "0"
        """
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
        """Returns a valid offpring between the 2 parents
        
        First it uses the basic method:
            Get a random index between 1 and the len of the parents minus 1,
            then brakes the parents in the index,
            then it flips a coin and uses the first part of one parent and the second of the other
        
        If this doesn't creates a viable child:
            It tries changing one gene for the parents one for every gene
            and if it creates a viable child it returns it

        Note
        ----
        This method creates a viable child, it was tested using all the combinations
        """
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
            individual_viability_check=True,
            individual_viability_condition=self.individual_viability_condition,
            viable_mutation=self.viable_mutation,
            viable_crossover=self.viable_crossover,
            viable_individual_factory=self.viable_individual_factory,
        )
        data = simulation.simulate(silent)
        final_population = simulation.getPopulation()
        if not silent:
            self.print_results(final_population, data)
        return (final_population, data)
