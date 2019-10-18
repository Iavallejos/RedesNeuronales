class Problem:
    def fitness_function(self, individual):
        raise NotImplementedError

    def gen_factory(self):
        raise NotImplementedError

    def individual_factory(self):
        raise NotImplementedError

    def termination_condition(self, individual_fitness):
        raise NotImplementedError

    def mutate(self, gene):
        raise NotImplementedError

    def individual_viability_condition(self, individual):
        raise NotImplementedError

    def viable_mutation(self, individual, index):
        raise NotImplementedError

    def viable_crossover(self, parent_1, parent_2):
        raise NotImplementedError

    def viable_individual_factory(self):
        raise NotImplementedError

    def print_results(self, population, data):
        print("Best Individuals fitness:   ", end="")
        print(data["best_individual_data"])
        print("Worst Individuals fitness:  ", end="")
        print(data["worst_individual_data"])
        print("Generation Average fitness: ", end="")
        print(data["generation_average_data"])
        print("\nFinal best individual: ", end="")

        final_best_individual = data["historic_best_individual"][-1]
        for gene in final_best_individual:
            print(gene, end="")
        print("\n", end="")

    def run(self):
        raise NotImplementedError
