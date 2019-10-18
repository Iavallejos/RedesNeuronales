import numpy as np


class GeneticAlgorithm:
    def __init__(
        self,
        pop_size,
        mutation_rate,
        fitness_function,
        individual_factory,
        gene_factory,
        mutate,
        termination_condition,
        max_iter,
        individual_viability_check=False,
        individual_viability_condition=None,
        viable_mutation=None,
        viable_crossover=None,
        viable_individual_factory=None,
    ):
        self.__pop_size = pop_size
        self.__mutation_rate = mutation_rate
        self.__fitness_function = fitness_function
        self.__individual_factory = individual_factory
        self.__gene_factory = gene_factory
        self.__termination_condition = termination_condition
        self.__max_iter = max_iter
        self.__mutate = mutate
        self.__individual_viability_check = individual_viability_check
        self.__viable_individual_factory = viable_individual_factory

        if individual_viability_check:
            if individual_viability_condition is None:
                raise NotImplementedError("individual_viability_condition")
            if viable_mutation is None:
                raise NotImplementedError("viable_mutation")
            if viable_crossover is None:
                raise NotImplementedError("viable_crossover")
            if viable_individual_factory is None:
                raise NotImplementedError("viable_individual_factory")

        self.__individual_viability_condition = individual_viability_condition
        self.__viable_mutation = viable_mutation
        self.__viable_crossover = viable_crossover

        if individual_viability_check:
            self.__population = [self.__viable_individual_factory()
                                 for _ in range(self.__pop_size)]
        else:
            self.__population = [self.__individual_factory()
                                 for _ in range(self.__pop_size)]

        self.__population_fitness = []

    def getPopulation(self):
        return self.__population

    def __evaluate(self):
        self.__population_fitness = [
            self.__fitness_function(individual) for individual in self.__population
        ]

    def __produce_offspring(self, total_fitness, mode=0):
        if mode == 0 and self.__population_fitness[0] >= 0:
            # Roulette
            parent_1_fitness, parent_2_fitness = np.random.randint(
                0, total_fitness, 2)
            total = 0
            parent_1 = None
            parent_2 = None
            for index, individual in enumerate(self.__population):
                total += abs(self.__population_fitness[index])
                if total >= parent_1_fitness:
                    parent_1 = individual
                if total >= parent_2_fitness:
                    parent_2 = individual
                if not (parent_1 is None or parent_2 is None):
                    break
        else:
            # Tournament where best 2 win
            participants = np.random.randint(0, self.__pop_size, 8)
            max_1 = None
            max_2 = None
            for participant in participants:
                if max_1 is None:
                    max_1 = participant

                elif self.__population_fitness[participant] > self.__population_fitness[max_1]:
                    max_2 = max_1
                    max_1 = participant

                elif max_2 is None or self.__population_fitness[participant] > self.__population_fitness[max_2]:
                    max_2 = participant

            parent_1 = self.__population[max_1]
            parent_2 = self.__population[max_2]

        if self.__individual_viability_check:
            child = self.__viable_crossover(parent_1, parent_2)
        else:
            gen_breakpoint_index = np.random.randint(1, len(parent_1)-1)

            if np.random.random() < 0.5:
                child = parent_1[:gen_breakpoint_index] + \
                    parent_2[gen_breakpoint_index:]
            else:
                child = parent_2[:gen_breakpoint_index] + \
                    parent_1[gen_breakpoint_index:]

        # mutation
        for i in range(len(child)):
            if np.random.random() < self.__mutation_rate:
                if self.__individual_viability_check:
                    child[i] = self.__viable_mutation(child, i)
                else:
                    child[i] = self.__mutate(child[i])

        return child

    def __reproduce(self):
        total_fitness = np.sum(self.__population_fitness)
        new_population = [self.__produce_offspring(
            total_fitness) for _ in range(self.__pop_size)]
        self.__population = new_population

    def simulate(self):
        best_individual_data = []
        worst_individual_data = []
        generation_average_data = []
        historic_best_individual = []
        print("{:-^50}".format("START"))
        for i in range(self.__max_iter + 1):
            self.__evaluate()

            actual_best_individual_fitness = np.max(self.__population_fitness)
            actual_worst_individual_fitness = np.min(self.__population_fitness)
            actual_average_individual_fitness = np.average(
                self.__population_fitness)
            actual_best_individual = self.__population[np.argmax(
                self.__population_fitness)]

            print("iteration {}:".format(i))
            print("\tBest Individual fitness:    {}".format(
                actual_best_individual_fitness))
            print("\tWorst Individual fitness:   {}".format(
                actual_worst_individual_fitness))
            print("\tAverage Individual fitness: {}".format(
                actual_average_individual_fitness))
            print("\tActual Best Individual: ", end="")
            for gene in actual_best_individual:
                print(gene, end="")
            print("\n", end="")

            best_individual_data.append(actual_best_individual_fitness)
            worst_individual_data.append(actual_worst_individual_fitness)
            generation_average_data.append(actual_average_individual_fitness)
            historic_best_individual.append(actual_best_individual)

            results = map(self.__termination_condition,
                          self.__population_fitness)
            if True in results:
                break

            self.__reproduce()

        print("{:-^50}".format("END"))
        return {
            "best_individual_data": best_individual_data,
            "worst_individual_data": worst_individual_data,
            "generation_average_data": generation_average_data,
            "historic_best_individual": historic_best_individual
        }
