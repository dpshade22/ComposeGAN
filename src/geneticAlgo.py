from keras.models import load_model
import random
import pandas as pd
from model import predictCSV

model = load_model("models/gru/79onBigData/model.h5")
"./"

# Define the size of the population and the number of generations to evolve
populationSize = 100
numEpochs = 1000

# Define the probability of crossover and mutation
crossoverProbability = 0.8
mutationProbability = 0.2
randomAdd = 0.05
eliteismRate = 0.3

# Define the range of values for each column in the CSV file
trackRange = range(1, 3)
noteRange = range(0, 127)
velocityRange = range(0, 127)
timeRange = range(0, 480)

GENE_LENGTH = 1000


# Define the fitness function, which computes the difference between the target melody and the notes in the CSV file
def fitness(csvDF):
    return predictCSV(csvDF, model)


# Define the function to generate a random CSV file
def generateMidiDFs(numGenomes):
    csv_data = []
    for i in range(numGenomes):
        genome = []
        for j in range(GENE_LENGTH):
            track = random.choice(trackRange)
            note = random.choice(noteRange)
            velocity = random.choice(velocityRange)
            time = random.choice(timeRange)
            row = [track, note, velocity, time]
            genome.append(row)
        df = pd.DataFrame(genome, columns=["track", "note", "velocity", "time"])
        csv_data.append(df)
    return csv_data


# Define the function to perform crossover between two dataframes
def crossover(df1, df2):
    split_index = random.randint(1, len(df1) - 1)
    newDF = pd.concat([df1[:split_index], df2[split_index:]], ignore_index=True)
    return newDF


# Define the function to perform mutation on a dataframe
def mutate(individual, numMutations=1):
    row_to_mutate = random.randint(0, len(individual) - 1)
    col_to_mutate = random.choice(["track", "note", "velocity", "time"])
    individual.at[row_to_mutate, col_to_mutate] = random.choice(f"{col_to_mutate}Range")

    return individual


def evolvePopulation(pop, epoch, eliteismRate=0.2, random_select=0.05, mutateRate=0.1):
    """
    Evolve the population by retaining the top `retain` percent of the population, adding random new genes,
    and mutating some of the existing genes.

    :param pop: The population to evolve
    :param retain: The fraction of the population to retain (default: 0.5)
    :param random_select: The fraction of the population to randomly add (default: 0.05)
    :param mutate: The fraction of genes to mutate (default: 0.01)
    :return: The evolved population
    """

    # Sort the population by fitness
    graded = [(fitness(individual), individual) for individual in pop]
    graded = [x[1] for x in sorted(graded, reverse=True)]

    # Calculate the number of individuals to retain
    retain_length = int(len(graded) * eliteismRate)

    # Retain the best individual from the previous generation
    best_individual = max([(fitness(individual), individual) for individual in pop])[1]
    parents = [best_individual]
    parents += graded[:retain_length]

    # Add random new individuals to the population
    while len(parents) < len(pop):
        if random_select > random.random():
            parents.append(generateMidiDFs(1)[0])

    # Mutate some of the individuals
    for individual in parents:
        if mutateRate > random.random():
            mutate(individual, random.randint(1, 150))

    # # Add the children to the parents to create the new population
    # parents.extend(children)
    pop = parents

    # Check if the maximum fitness has been reached
    max_fitness = max([fitness(individual) for individual in pop])

    if max_fitness >= 0.80:
        return pop, max_fitness, True
    else:
        return pop, max_fitness, False


def main():
    # Initialize the population
    pop = generateMidiDFs(populationSize)
    indEx = pop[0]
    epoch = 0

    # Evolve the population until the target is found
    found = False
    while not found:
        pop, max_fitness, found = evolvePopulation(
            pop, epoch, eliteismRate, randomAdd, mutationProbability
        )
        print("Epoch:", epoch, "\nMax Fitness:", max_fitness, "\n")
        epoch += 1

    # Print the result
    graded = [(fitness(individual), individual) for individual in pop]
    graded = [x[1] for x in sorted(graded, reverse=True)]

    print("Generations: " + str(epoch))
    print("Best individual: " + str(graded[0]))
    graded[0].to_csv("bestIndividual.csv")


main()
