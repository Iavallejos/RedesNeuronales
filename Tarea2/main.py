from Problems.BitSequence import BitSequence
from Problems.FindWord import FindWord
from Problems.Knapsack01 import Knapsack01


def main():
    """It creates and runs one of the created problems"""
    problem = BitSequence(
        target_number=121,
        pop_size=100,
        mutation_rate=0.1,
        number_of_genes=10,
        max_iter=10
    )
    
    '''
    problem = FindWord(
        target_word='helloworld',
        pop_size=1000,
        mutation_rate=0.1,
        max_iter=500
    )
    '''
    '''
    problem = Knapsack01(
        pop_size = 5,
        mutation_rate = 0.1,
        max_iter = 20
    )
    '''
    population, data = problem.run()
    

if __name__ == "__main__":
    main()
