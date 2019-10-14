import numpy as np


class GeneticAlgorithm:
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

        self.__population = [self.__individual_factory()
                             for _ in range(self.__pop_size)]
        self.__population_fitness = []

    def getPopulation(self):
        return self.__population

    def __evaluate(self):
        self.__population_fitness = [
            self.__fitness_function(individual) for individual in self.__population
        ]

    def __produce_offspring(self, total_fitness):
        parent_1_fitness, parent_2_fitness = np.random.randint(
            1, total_fitness, 2)
        total = 0
        parent_1 = None
        parent_2 = None
        for index, individual in enumerate(self.__population):
            total += self.__population_fitness[index]
            if total >= parent_1_fitness:
                parent_1 = individual
            if total >= parent_2_fitness:
                parent_2 = individual
            if not (parent_1 is None or parent_2 is None):
                break

        # TODO create the new individual

    def __reproduce(self):
        total_fitness = np.sum(self.__population_fitness)
        new_population = [self.__produce_offspring(total_fitness)
                          for _ in range(self.__pop_size)]
        self.__population = np.reshape(
            new_population, (1, 2*len(new_population)))[0].tolist()

    def simulate(self):
        best_individual = []
        worst_individual = []
        generation_average = []
        for _ in range(self.__max_iter + 1):
            self.__evaluate()

            best_individual.append(np.max(self.__population_fitness))
            worst_individual.append(np.min(self.__population_fitness))
            generation_average.append(np.average(self.__population_fitness))

            results = map(self.__termination_condition,
                          self.__population_fitness)
            if True in results:
                break

            self.__reproduce()

        return {
            "best_individual": best_individual,
            "worst_individual": worst_individual,
            "generation_average": generation_average,
        }
