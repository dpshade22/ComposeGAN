import random, pandas as pd
from keras.models import load_model
from model import predictCSV, saveMidiFromCSV

model = load_model("models/gru/79onBigData/model.h5")
"./"

# Define the range of values for each column in the CSV file
trackRange = range(1, 3)
noteRange = range(0, 127)
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


def mutateBest(bestIndividual, populationSize=100, numMutations=3):
    mutatedPop = [bestIndividual]
    for x in range(populationSize):
        individualCopy = bestIndividual.copy()

        for _ in range(numMutations):
            row_to_mutate = random.randint(0, len(bestIndividual) - 1)
            col_to_mutate = random.choice(["track", "note", "velocity", "time"])

            individualCopy.at[row_to_mutate, col_to_mutate] = random.choice(
                eval(f"{col_to_mutate}Range")
            )

        mutatedPop.append(individualCopy)

    return mutatedPop


def mutationLoop():
    population = generateMidiDFs(5)
    currBest = 0
    currCount = 0
    while currBest < 0.8:
        population = gradedPopulation(population)
        population = mutateBest(population[0], 5, 10)
        currBest = predictCSV(population[0], model)
        print(currBest)

        currCount += 1

        if (currCount % 20) == 0:
            population[0].to_csv(f"./genTesting/{currBest}.csv", index=False)
            saveMidiFromCSV(
                f"./genTesting/{currBest}.csv", f"./genTesting/midis/{currBest}.mid"
            )

    population[0].to_csv(f"Testing{currBest}.csv", index=False)


mutationLoop()
