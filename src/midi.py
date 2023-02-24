import random, pandas as pd
from keras.models import load_model
from model import predictCSV, saveMidiFromCSV

model = load_model("models/gru/79onBigData/model.h5")

# Define the range of values for each column in the CSV file
trackRange = range(1, 2)
noteRange = range(21, 127)
velocityRange = range(0, 127)
timeRange = range(0, 240)

GENE_LENGTH = 1000


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


def gradedPopulation(population):
    population = [
        (predictCSV(individual, model), individual) for individual in population
    ]
    return [x[1] for x in sorted(population, reverse=True)]


def mutateBest(bestIndividual, populationSize=100, numMutations=3, copyProb=0.15, transProb=0.15):
    mutatedPop = [bestIndividual]
    for x in range(populationSize):
        individualCopy = bestIndividual.copy()

        for _ in range(numMutations):
            if random.random() < transProb:
                # Transpose a random group of MIDI messages
                group_size = random.randint(1, 4)
                start_index = random.randint(0, len(bestIndividual) - group_size)
                delta = random.randint(-6, 6)

                for i in range(start_index, start_index + group_size):
                    for col in ["note", "velocity"]:
                        current_val = individualCopy.at[i, col]
                        new_val = min(
                            max(current_val + delta, eval(f"{col}Range").start),
                            eval(f"{col}Range").stop - 1,
                        )
                        individualCopy.at[i, col] = new_val
            if random.random() < copyProb:
                # Copy a random MIDI message from the best individual
                row_to_copy = random.randint(0, len(bestIndividual) - 1)
                individualCopy.iloc[row_to_copy] = bestIndividual.iloc[row_to_copy]

            else:
                # Mutate a random MIDI message in the individual copy
                row_to_mutate = random.randint(0, len(bestIndividual) - 1)
                col_to_mutate = random.choice(["track", "note", "velocity", "time"])

                current_val = individualCopy.at[row_to_mutate, col_to_mutate]
                delta_val = random.randint(-12, 12)
                new_val = min(
                    max(current_val + delta_val, eval(f"{col_to_mutate}Range").start),
                    eval(f"{col_to_mutate}Range").stop - 1,
                )

                individualCopy.at[row_to_mutate, col_to_mutate] = new_val

        mutatedPop.append(individualCopy)

    return mutatedPop



def evolve():
    population = generateMidiDFs(5)
    currBest = 0
    currCount = 0
    elitestRate = 0.3
    while currBest < 0.8:
        population = gradedPopulation(population)
        population = mutateBest(population[0], 5, 50)
        currBest = predictCSV(population[0], model)
        print(currBest)

        currCount += 1

        if (currCount % 20) == 0:
            population[0].to_csv(f"./genTesting/{currBest[0][0]}.csv", index=False)
            saveMidiFromCSV(
                f"./genTesting/{currBest[0][0]}.csv",
                f"./genTesting/midis/{currBest[0][0]}.mid",
            )

    population[0].to_csv(f"Testing{currBest}.csv", index=False)


evolve()
