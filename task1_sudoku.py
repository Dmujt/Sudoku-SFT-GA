#####################################
# Task 1: GA for Sudoku
# Dena Mujtaba
#####################################
import time
import random
import math
import numpy as np

#puzzle to solve, 0s indicate empty
puzzleProblem = [[5,1,7,6,0,0,0,3,4],[2,8,9,0,0,4,0,0,0],[3,4,6,2,0,5,0,9,0],[6,0,2,0,0,0,0,1,0],[0,3,8,0,0,6,0,4,7],[0,0,0,0,0,0,0,0,0],[0,9,0,0,0,0,0,7,8],[7,0,3,4,0,0,5,6,0],[0,0,0,0,0,0,0,0,0]]
puzzleProblem = np.array(puzzleProblem)
#get the size of the board
N = int(math.sqrt(len(puzzleProblem)))

# SETUP PARAMS
POPULATION_SIZE = 100 
NUM_GENERATIONS = 10 
SELECTION_RATE = 0.5
MUTATION_RATE = 0.5

# initialize the population
def populationGeneration(puzzle):
    pop = []
    for individual in range(0, POPULATION_SIZE):
        newPuzzle = np.array(puzzle, copy=True)
        for row in range(0, N**2):
            for col in range(0, N**2):
                if newPuzzle[row][col] == 0:
                    newPuzzle[row][col] = random.randint(1, (N**2) + 1)
        pop.append(newPuzzle)
    return np.array(pop)

#main class to solve the 
class SudokuGA():

    #constructor for GA
    #@param p initial population
    def __init__(self, p):
        # check arguments for GA setup parameters
        self.initialPopulation = p

    def run(self):
        population = self.initialPopulation
        fitnessValue = self.fitness(population)
        generation = 0
        while(self.terminate(fitnessValue, generation)):
            #select
            
            #crossover
            
            #mutate
            
            #new fitness
            fitnessValue = self.fitness(population)
            generation += 1
            print("---- Generation: %s - Fitness: %s ----" % (generation, fitnessValue))    
            
        return self.initialPopulation[0]
        
    # if the algo should terminate
    def terminate(self, fitnessValue, generation):
        return (fitnessValue == 100 or generation < NUM_GENERATIONS)
    
    # will return child
    def crossover(self, p1, p2):
        return p1

    # mutate x
    def mutate(self, x):
        return x

    def fitness(self):
        return 0
    
    def select(self):
        return 0

if __name__ == "__main__":
    startTime = time.time()
    population = populationGeneration(puzzleProblem)
    ga = SudokuGA(population)
    sol = ga.run()
    
    print("Solution: %s" % sol)
    print("---- Population Length: %s ----" % len(population))    
    print("---- %s seconds ----" % (time.time() - startTime))