class Problem:
    def fitness_function(self, individual):
        raise NotImplementedError

    def gen_factory(self):
        raise NotImplementedError

    def individual_factory(self):
        raise NotImplementedError

    def termination_condition(self, individual_fitness):
        raise NotImplementedError

    def mutate(self, gen):
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
        for gen in final_best_individual:
            print(gen, end="")
        print("\n", end="")

    def run(self):
        raise NotImplementedError
