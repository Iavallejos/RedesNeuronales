import numpy as np


class GeneticAlgotirhm:
    def __init__(
        self,
        pop_size,
        mutation_rate,
        fitness_function,
        individual_factory,
        gene_factory,
        termination_condition,
        max_iter,
    ):
        self.__pop_size = pop_size
        self.__mutation_rate = mutation_rate
        self.__fitness_function = fitness_function
        self.__individual_factory = individual_factory
        self.__gene_factory = gene_factory
        self.__termination_condition = termination_condition
        self.__max_iter = max_iter

        self.__population = [self.individual_factory() for _ in range(self.pop_size)]
        self.__population_fitness = []

    def getPopulation(self):
        return self.__population

    def __evaluate(self):
        self.__population_fitness = [
            self.__fitness_function(individual) for individual in self.__population
        ]

    def __reproduce(self):
        new_population = [__produce_offspring() for _ in range(self.__pop_size)]
        self.__population = new_population

    def __produce_offspring(self):
        pass  # TODO

    def simulate(self):
        best_individual = []
        worst_individual = []
        generation_average = []
        for i in range(self.__max_iter + 1):
            self.__evaluate()

            best_individual.append(np.max(self.__population_fitness))
            worst_individual.append(np.max(self.__population_fitness))
            generation_average.append(np.average(self.__population_fitness))

            results = map(self.__termination_condition, self.__population_fitness)
            if True in results:
                break

            self.__reproduce()

        return {
            "best_individual": best_individual,
            "worst_individual": worst_individual,
            "generation_average": generation_average,
        }
