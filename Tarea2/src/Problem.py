class Problem:
    def fitness_function(self, individual):
        raise NotImplementedError

    def gen_factory(self):
        raise NotImplementedError

    def individual_factory(self):
        raise NotImplementedError

    def termination_condition(self, individual_fitness):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
