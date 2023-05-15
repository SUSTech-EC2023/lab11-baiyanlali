import numpy as np
import random
import matplotlib.pyplot as plt


# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# A function to perform a forward pass through the neural network. You can design your neural network however you want.
def predict(weights, inputs):
    layer1_weights = weights[:6].reshape(2, 3)
    layer2_weights = weights[6:].reshape(3, 1)

    layer1 = sigmoid(np.dot(inputs, layer1_weights))
    output = sigmoid(np.dot(layer1, layer2_weights))

    return output


# TODO: Implement a simple evolutionayr learning algorithm to to optimize the weights of a feedforward neural network for the XOR problem.
xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_outputs = np.array([[0], [1], [1], [0]])


def fitness(ind):
    output = predict(ind, xor_inputs)
    return np.sum(np.abs(xor_outputs - output))


def init_population(pop_size=100):
    population = []
    for i in range(pop_size):
        population.append(np.zeros(9))
    return population


def crossover(p1, p2):
    c1 = np.zeros((9))
    c2 = np.zeros((9))
    for i in range(9):
        c1[i] = p1[i] if random.random() < 0.5 else p2[i]
        c2[i] = p2[i] if random.random() < 0.5 else p1[i]
    return c1, c2


def mutation(ind: np.ndarray, mutation_prob):
    for i in range(9):
        if random.random() < mutation_prob:
            ind[i] += random.gauss(0, 1)
    return ind


def run_genetic_programming(population_size, crossover_probability, mutation_probability, number_of_generations,
                            max_depth, elitism_size):
    best_fitness_values = []

    # Initialize the population
    population = init_population(population_size)

    for generation in range(number_of_generations):
        print(f"Generation {generation + 1}/{number_of_generations}")

        # Evaluate the population
        fitness_values = lambda populaiton: [fitness(ind) for ind in population]
        # Select the best individuals for elitism
        population = sorted(population, key=fitness_values)
        elite_individuals = population[:elitism_size]

        # Select parents
        parents = []
        for _ in range((population_size - elitism_size) // 2):
            p1 = min(random.sample(population, k=10), key=fitness_values)  # Increase selection pressure
            p2 = min(random.sample(population, k=10), key=fitness_values)  # Increase selection pressure
            parents.append((p1, p2))

        # Create offspring
        offspring = []
        for parent1, parent2 in parents:
            if random.random() < crossover_probability:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            offspring.append(mutation(child1, mutation_probability))
            offspring.append(mutation(child2, mutation_probability))

        population = elite_individuals + offspring

        best_fitness = min(fitness_values(population))
        best_fitness_values.append(best_fitness)

    plt.plot(best_fitness_values)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Fitness Curve")
    plt.show()

    best_individual = min(population, key=fitness_values)
    print("Best individual:", best_individual)
    print("Predict:", predict(best_individual, xor_inputs))
    print("Fitness:", fitness(best_individual))


# if __name__ == '__main__':
#     print(fitness(np.array([[0], [1], [0], [0]])))


if __name__ == '__main__':
    population_size = 100
    crossover_probability = 0.8
    mutation_probability = 0.3
    number_of_generations = 50
    max_depth = 6
    elitism_size = 5

    # Run the genetic programming algorithm with the given hyperparameters
    run_genetic_programming(population_size, crossover_probability, mutation_probability, number_of_generations,
                            max_depth, elitism_size)
