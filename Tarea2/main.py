from Problems.BitSequence import BitSequence
from Problems.FindWord import FindWord


def main():

    '''
    problem = BitSequence(
        target_number=121,
        pop_size=100,
        mutation_rate=0.1,
        number_of_genes=10,
        max_iter=10
    )
    '''
    problem = FindWord(
        target_word='papel',
        pop_size=500,
        mutation_rate=0.1,
        max_iter=50
    )
    
    population, data = problem.run()


if __name__ == "__main__":
    main()
