import numpy as np
from scipy.optimize import minimize


stocks = ['Stock A', 'Stock B', 'Stock C', 'Stock D', 'Stock E']
beta = np.array([1.2, 0.8, 1.1, 1.3, 0.9])
momentum = np.array([0.75, 0.85, 0.9, 0.65, 0.8])


n_stocks = len(stocks)
target_weighted_beta = 1
max_weight = 0.50


def objective(weights):
    return -np.dot(momentum, weights)


# Constraints
def constraint_sum_weights(weights):
    return np.sum(weights) - 1


def constraint_weighted_beta(weights):
    return np.dot(weights, beta) - target_weighted_beta


constraints = [
    {'type': 'eq', 'fun': constraint_sum_weights},
    {'type': 'eq', 'fun': constraint_weighted_beta}
]


bounds = [(0, max_weight) for _ in range(n_stocks)]


initial_weights = np.random.uniform(0, max_weight, n_stocks)
initial_weights /= np.sum(initial_weights)


result = minimize(objective, initial_weights, constraints=constraints, bounds=bounds, method='SLSQP')

if __name__ == '__main__':
    if result.success:
        optimal_weights = result.x
        print("Optimal Stock Weights:", optimal_weights)
        print("Portfolio Beta:", np.dot(optimal_weights, beta))
        print("Portfolio Momentum:", np.dot(optimal_weights, momentum))
    else:
        print("Optimization failed:", result.message)