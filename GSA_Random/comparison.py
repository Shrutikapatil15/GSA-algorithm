import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# Function to parse fitness values from the XML file
def parse_fitness_from_xml(filename):
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        raise ValueError(f"File '{filename}' does not exist or is empty.")
    
    tree = ET.parse(filename)
    root = tree.getroot()
    
    fitness_values = []
    
    # Get the IterationsTable element
    iterations_table = root.find("IterationsTable")
    if iterations_table is None:
        raise ValueError(f"No 'IterationsTable' element found in {filename}. Please check the structure.")
    
    for row in iterations_table:
        fitness_elem = row.find("Fitness")
        
        # Check if the Fitness element exists and has a valid text
        if fitness_elem is not None and fitness_elem.text is not None:
            try:
                fitness = float(fitness_elem.text)
                fitness_values.append(fitness)
            except ValueError:
                print(f"Invalid fitness value in file '{filename}' at iteration {row.attrib.get('iteration')}")
        else:
            print(f"Missing 'Fitness' element or empty value in file '{filename}' at iteration {row.attrib.get('iteration')}")
    
    return fitness_values

# Plotting the combined results
def plot_combined_results(gsa_results, pso_results, cfo_results):
    plt.figure(figsize=(10, 6))
    
    # Plot GSA results
    plt.plot(gsa_results, label='GSA', marker='o', linestyle='-')
    
    # Plot PSO results
    plt.plot(pso_results, label='PSO', marker='o', linestyle='-')
    
    # Plot CFO results
    plt.plot(cfo_results, label='CFO', marker='o', linestyle='-')

    plt.title('Comparison of Best Fitness Values Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.legend()
    plt.grid()
    plt.xticks(range(max(len(gsa_results), len(pso_results), len(cfo_results))))  # Show all iterations on the x-axis
    plt.show()  # Display the combined plot

# Main function to parse the XML files and plot the combined results
def main_combined():
    try:
        gsa_results = parse_fitness_from_xml('gsa_results.xml')
        pso_results = parse_fitness_from_xml('pso_results.xml')
        cfo_results = parse_fitness_from_xml('cfo_results.xml')
    
        plot_combined_results(gsa_results, pso_results, cfo_results)
    
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main_combined()
