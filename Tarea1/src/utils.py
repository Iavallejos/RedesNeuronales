import numpy as np

def calculate_cost(results, expected):
   cost = np.sum(0.5 * (results - expected)**2) / len(expected)

   return cost

