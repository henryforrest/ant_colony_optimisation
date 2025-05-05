import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time

# Parameters
NUM_ANTS = 10
EVAPORATION_RATE = 0.4


def construction_graph(bins, items):
    random.seed(1) # random seed which was changed for testing
    graph = [[random.random() for x in range(bins)] for y in range(items)] #create construction graph with random values between 0 and 1
    return graph


def fitness_calculation(bin_allocation):
    # Calculate the total weight for each bin
    bin_totals = [sum(bin) for bin in bin_allocation]

    # Find the max and min weighted bins
    biggest = max(bin_totals)
    smallest = min(bin_totals)

    # Return the difference
    return biggest - smallest


def initialize_bins(bins):
    """Initialize empty bins for each ant."""
    return [[] for _ in range(bins)] # return an array of empty arrays for number of bins


def place_item_in_bin(graph, item, bins):
    """Place an item in a bin based on pheromone levels and heuristic."""
    pheromones = [graph[item][bin_index] for bin_index in range(bins)]  # gather all the pheromone values or each bin for this single item
    bin_choice = random.choices(range(bins), weights=pheromones, k=1)[0] #randomly choose bin with pheromones as weightings
    return bin_choice # return number corresponding to bin item is placed in


def update_pheromones(graph, all_bin_allocations, all_fitnesses):
    """Update the pheromone values on the graph """
    for item in range(len(graph)):
        for bin_index in range(len(graph[item])):
            graph[item][bin_index] *= EVAPORATION_RATE # multiply every value in the pheromone graph by the evaporation rate

    # Deposit new pheromone based on fitness
    for allocation, fitness in zip(all_bin_allocations, all_fitnesses):
        pheromone_deposit = 100/fitness # pheromone deposit for each bin allocation
        for item, bin_index in enumerate(allocation):
            graph[item][bin_index] += pheromone_deposit # add pheromone deposit to each path taken by an ant


def ant_colony_bin_packing(num_bins, num_items, item_weights):
    graph = construction_graph(num_bins, num_items) # declare construction graph
    best_allocation = None  # declare best allocation with keyword none
    best_fitness = float('inf') # declare best fitness with float equal to infinity so that any new fitness is better
    fitness_per_iteration = []

    for iteration in range(int(10000/NUM_ANTS)):
        all_bin_allocations = [] # declare all allocations as empty
        all_fitnesses = [] # declare all fitnesses as empty

        # Each ant creates a solution
        for ant in range(NUM_ANTS):
            bins = initialize_bins(num_bins) # get empty array of arrays for each bin
            allocation = [] # empty array to put bin allocation of each weighted item in

            # Place each item in a bin
            for item in range(num_items):
                chosen_bin = place_item_in_bin(graph, item, num_bins) # get bin choice
                bins[chosen_bin].append(item_weights[item])  #place item weight in the array for the specified bin
                allocation.append(chosen_bin)  # append the bin choice to the array

            # Calculate fitness using the provided fitness function
            fitness = fitness_calculation(bins) # get fitness for this ant
            all_bin_allocations.append(allocation) # append bin allocation
            all_fitnesses.append(fitness) #append fitness

            # Update best solution
            if fitness < best_fitness:
                best_fitness = fitness
                best_allocation = allocation

        # Update pheromones after all ants have made their choices
        update_pheromones(graph, all_bin_allocations, all_fitnesses)

        print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}") # print iteration number and current best fitness
        fitness_per_iteration.append(best_fitness)

    return best_allocation, best_fitness, fitness_per_iteration # return values to help with making graphs


""" ========== BPP1 ========= """
def bpp1():
    # BPP1
    num_bins = 10
    num_items = 500
    item_weights = [i for i in range(1, num_items + 1)]  # BPP1 weights
    best_allocation, best_fitness, fitness = ant_colony_bin_packing(num_bins, num_items, item_weights)
    print("Best Allocation:", best_allocation)
    print("Best Fitness:", best_fitness)

    return fitness, best_allocation


""" ========== BPP2 ========= """
def bpp2():
    num_bins = 50
    num_items = 500
    item_weights = [i ** 2 / 2 for i in range(1, num_items + 1)]  # BPP2 weights
    best_allocation, best_fitness, fitness = ant_colony_bin_packing(num_bins, num_items, item_weights)
    print("Best Allocation:", best_allocation)
    print("Best Fitness:", best_fitness)

    return fitness, best_allocation


"""============ GRAPHS ============="""
def heatmap():
    num_ants_values = [5, 10, 15, 20]
    evaporation_rates = [0.5, 0.6, 0.7, 0.8, 0.9]

    heatmap_data = np.zeros((len(num_ants_values), len(evaporation_rates)))

    # Loop through each combination of parameters
    for i, ants in enumerate(num_ants_values):
        for j, evaporation in enumerate(evaporation_rates):
            # Set parameters for this run
            NUM_ANTS = ants
            EVAPORATION_RATE = evaporation

            # Run the bin-packing algorithm and get the final fitness score ie the best score
            fitness_progress = bpp2()
            final_fitness = fitness_progress[-1]

            # Store the result in the heatmap data array
            heatmap_data[i, j] = final_fitness

    # Plotting the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, xticklabels=evaporation_rates, yticklabels=num_ants_values, cmap="YlGnBu")
    plt.xlabel("Evaporation Rate")
    plt.ylabel("Number of Ants")
    plt.title("Parameter Sensitivity Heatmap (Final Fitness Score)")
    plt.show()


def graph1():
    x, y = bpp2()
    plt.plot(x)
    EVAPORATION_RATE = 0.9
    x2, y2 = bpp2()
    plt.plot(x2, linestyle="dashed")
    plt.legend(['e = 0.6', 'e = 0.9'])
    plt.xlabel("Number of iterations")
    plt.ylabel("Best fitness")
    plt.show()


def bin_distribution():
    fitness, allocations = bpp2()

    totals = [0] * 50
    count = 1

    for x in allocations:
        totals[x] += ((count**2)/2)
        count += 1

    print(totals)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(totals)), totals, color='skyblue')
    plt.xlabel('Bin Index')
    plt.ylabel('Total Weight in Bin')
    plt.title('Bin Weight Distribution after Final Iteration')
    plt.xticks(range(len(totals)))
    plt.show()





bpp1()
bpp2()
