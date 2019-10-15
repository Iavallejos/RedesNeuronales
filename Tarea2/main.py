from Problems.BitSequence import BitSequence


def main():
    problem = BitSequence(
        target_number=121,
        pop_size=100,
        mutation_rate=0.1,
        number_of_genes=10,
        max_iter=10
    )

    population, data = problem.run()


if __name__ == "__main__":
    main()
