from Problems.BitSequence import BitSequence
from Problems.FindWord import FindWord
from Problems.Knapsack01 import Knapsack01

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def main(graph=False):
    """It creates and runs one of the created problems

    If graph is true, it also creates 2 figures

    """

    max_iter = 20
    problem = BitSequence(
        target_number=615781,
        pop_size=100,
        mutation_rate=0.1,
        number_of_genes=20,
        max_iter=max_iter
    )

    '''
    problem = FindWord(
        target_word='helloworld',
        pop_size=1000,
        mutation_rate=0.1,
        max_iter=max_iter
    )
    '''
    '''
    problem = Knapsack01(
        pop_size = 5,
        mutation_rate = 0.1,
        max_iter = max_iter
    )
    '''
    _, data = problem.run()


    # From here is the code to create the graphs for the BitSequence Problem
    historic_best_individual = data["historic_best_individual"]

    del data["historic_best_individual"]

    sns.set(style="darkgrid")
    sns.set_context("paper")
    plt.figure(figsize=(10, 7.5))
    df = pd.DataFrame(data=data)
    ax = sns.lineplot(data=df)
    ax.set(xlabel="Generation", ylabel="Fitness",
           title="Fitness evolution for BitSequence problem")
    plt.xticks(np.arange(0, len(historic_best_individual), 1.0))
    plt.tight_layout()

    plt.figure(figsize=(10, 7.5))
    heat_data = []
    pops = [i for i in range(50, 1001, 50)]
    muts = [i/10 for i in range(0, 11, 1)]
    for pop in range(50, 1001, 50):
        problem.set_pop_size(pop)
        ndata = []
        for mutation in range(0, 11, 1):
            problem.set_mutation_rate(mutation/10)
            _, problem_data = problem.run(silent=True)
            metric = (
                max_iter - len(problem_data["historic_best_individual"]))/max_iter
            ndata.append(metric)

        heat_data.append(ndata)

    ax2 = sns.heatmap(heat_data, annot=True, linewidths=.5,
                      xticklabels=muts, yticklabels=pops)
    ax2.set(xlabel="Mutation Rate", ylabel="Population Size",
            title="Heat map BitSequence problem population size v/s mutation rate")
    plt.tight_layout()
    
    plt.show()


if __name__ == "__main__":
    main()
