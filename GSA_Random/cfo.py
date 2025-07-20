import numpy as np
import random
import math
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom  # Added to prettify XML output
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

# Population Initialization with random values within bounds
def init_population(size, dim, lb, ub):
    population = np.random.uniform(low=lb, high=ub, size=(size, dim))
    return population

# Objective function definitions
def sphere_function(solution):
    return np.sum(solution ** 2)

def gear_train(solution):
    x1, x2, x3, x4 = solution
    result = ((1 / 6.931) - (x3 * x2) / (x1 * x4)) ** 2
    return result

def pressure_vessel(solution):
    x1, x2, x3, x4 = solution
    g1 = -x1 + 0.0193 * x3
    g2 = -x2 + 0.00954 * x3
    g3 = 1296000 - (4 / 3) * math.pi * (x3 * 3) - math.pi * (x3 * 2) * (x4)
    g4 = x4 - 240
    if g1 <= 0 and g2 <= 0 and g3 <= 0 and g4 <= 0:
        return 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3 * x3 + 3.1661 * x1 * x1 * x4 + 19.84 * x1 * x1 * x3
    else:
        return 1e10

# Objective function selector
def objective_function_selector(problem):
    if problem == 1:
        return sphere_function
    elif problem == 2:
        return gear_train
    elif problem == 3:
        return pressure_vessel
    else:
        raise ValueError("Invalid problem number")

def trim(solution, lb, ub):
    return np.clip(solution, lb, ub)

# Chameleon Flower Optimization (CFO) Function
def cfo(population, dim, itr, objective_function, lb, ub):
    iteration_results = []  # To store the best solution and fitness of each iteration
    best_fitness_values = []  # List to keep track of best fitness values for plotting

    # Evaluate initial fitness
    fitness_values = np.array([objective_function(individual) for individual in population])
    best_idx = np.argmin(fitness_values)
    best_solution = population[best_idx]
    best_fitness = fitness_values[best_idx]

    for t in range(itr):
        for i in range(len(population)):
            # Randomly select a neighbor
            neighbor_idx = random.randint(0, len(population) - 1)
            while neighbor_idx == i:
                neighbor_idx = random.randint(0, len(population) - 1)

            # Calculate the color change (position update) based on the neighbor's fitness
            if fitness_values[neighbor_idx] < fitness_values[i]:
                change_factor = random.uniform(0.5, 1.5)
                population[i] += change_factor * (population[neighbor_idx] - population[i])
            else:
                change_factor = random.uniform(-1, 0)
                population[i] += change_factor * (population[i] - population[neighbor_idx])

            # Clip the new position to be within the bounds
            population[i] = trim(population[i], lb, ub)

            # Evaluate new fitness
            new_fitness = objective_function(population[i])
            fitness_values[i] = new_fitness

            # Update the best solution found
            if new_fitness < best_fitness:
                best_solution = population[i]
                best_fitness = new_fitness

        print("\nIteration ", t, "\nBest Solution:", best_solution)
        print("Best Fitness:", best_fitness)

        # Save the best solution and fitness for the current iteration
        iteration_results.append((best_solution.copy(), best_fitness))
        best_fitness_values.append(best_fitness)  # Append current best fitness for plotting

    # Return the best solution found along with all iteration results
    return best_solution, best_fitness, iteration_results, best_fitness_values

# Function to write results to an XML file and prettify it
def write_results_to_xml(best_solution, best_fitness, iteration_results, filename="results.xml"):
    root = ET.Element("Results")
    
    # Adding overall best solution and fitness
    overall_best_elem = ET.SubElement(root, "OverallBest")
    solution_elem = ET.SubElement(overall_best_elem, "BestSolution")
    fitness_elem = ET.SubElement(overall_best_elem, "BestFitness")
    
    # Only take x1 and x2 from the best solution
    solution_elem.text = f"x1: {best_solution[0]}, x2: {best_solution[1]}"
    fitness_elem.text = str(best_fitness)
    
    # Adding details of each iteration in a table-like structure
    iterations_elem = ET.SubElement(root, "IterationsTable")
    
    # Adding column headers
    header_elem = ET.SubElement(iterations_elem, "Header")
    x1_header = ET.SubElement(header_elem, "Column")
    x1_header.text = "x1"
    x2_header = ET.SubElement(header_elem, "Column")
    x2_header.text = "x2"
    fitness_header = ET.SubElement(header_elem, "Column")
    fitness_header.text = "Fitness"
    
    # Adding data for each iteration
    for i, (solution, fitness) in enumerate(iteration_results):
        row_elem = ET.SubElement(iterations_elem, "Row", iteration=str(i + 1))
        
        # Adding x1, x2, and fitness in the row
        x1_elem = ET.SubElement(row_elem, "x1")
        x1_elem.text = str(solution[0])
        
        x2_elem = ET.SubElement(row_elem, "x2")
        x2_elem.text = str(solution[1])
        
        fitness_elem = ET.SubElement(row_elem, "Fitness")
        fitness_elem.text = str(fitness)

    # Convert the tree to a string and prettify
    xml_string = ET.tostring(root)
    pretty_xml = minidom.parseString(xml_string).toprettyxml(indent="  ")

    # Write the prettified XML to a file
    with open(filename, "w") as fh:
        fh.write(pretty_xml)

# Function to plot the best fitness values over iterations
def plot_fitness(iterations, best_fitness_values):
    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), best_fitness_values, marker='o', linestyle='-', color='b')
    plt.title('Best Fitness Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness Value')
    plt.grid()
    plt.show()

# Main function to run the algorithm with user input for other parameters
def main():
    print("Select problem to solve:")
    print("1. Sphere Function")
    print("2. Gear Train")
    print("3. Pressure Vessel")
    problem = int(input("Enter problem number (1-3): "))

    # Define problem-specific bounds and dimensions
    if problem == 1:
        lb = [-100, -100]
        ub = [100, 100]
        dim = len(lb)
    elif problem == 2:
        lb = [12, 12, 12, 12]
        ub = [60, 60, 60, 60]
        dim = len(lb)
    elif problem == 3:
        lb = [0, 0, 10, 10]
        ub = [100, 100, 200, 200]
        dim = len(lb)
    else:
        raise ValueError("Invalid problem number")

    # Get user input for parameters
    pop_size = int(input("Enter population size: "))
    itr = int(input("Enter number of iterations: "))

    # Initialize population randomly within the bounds
    population = init_population(pop_size, dim, lb, ub)
    print("Initial Population:")
    print(population)

    # Select the objective function based on user input
    objective_function = objective_function_selector(problem)
    
    # Run the CFO algorithm
    best_solution, best_fitness, iteration_results, best_fitness_values = cfo(population, dim, itr, objective_function, lb, ub)

    print("\nBest Solution:", best_solution)
    print("Best Fitness:", best_fitness)

    # Write the results to an XML file, including all iterations
    write_results_to_xml(best_solution, best_fitness, iteration_results)

    # Plot the best fitness values over iterations
    plot_fitness(itr, best_fitness_values)

if __name__ == "__main__":
    main()
