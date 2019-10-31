#####################################
# Task 1: GA for Sudoku
# Dena Mujtaba
#####################################
import time
import random
import math
import numpy as np
from numpy.random import choice
random.seed(14)

#puzzle to solve, 0s indicate empty
puzzleProblem = [[5,1,7,6,0,0,0,3,4],[2,8,9,0,0,4,0,0,0],[3,4,6,2,0,5,0,9,0],[6,0,2,0,0,0,0,1,0],[0,3,8,0,0,6,0,4,7],[0,0,0,0,0,0,0,0,0],[0,9,0,0,0,0,0,7,8],[7,0,3,4,0,0,5,6,0],[0,0,0,0,0,0,0,0,0]]
puzzleProblem = np.array(puzzleProblem)

fixedValues = []
#determine what the index of the 0 values of the puzzle are, for crossover
for r in puzzleProblem:
    rArr = []
    for idx, c in enumerate(r):
        if c == 0:
            rArr.append(idx)
    fixedValues.append(rArr)
    
print(fixedValues)
#get the size of the board
N = int(math.sqrt(len(puzzleProblem)))

# SETUP PARAMS
POPULATION_SIZE = 500
NUM_GENERATIONS = 500
MUTATION_RATE = 0.75

# initialize the population
def populationGeneration(puzzle):
    pop = []
    for individual in range(0, POPULATION_SIZE):
        newPuzzle = np.array(puzzle, copy=True)
        for row in range(0, (N**2)):
            for col in range(0,(N**2)):
                if newPuzzle[row][col] == 0:
                    # set a random float here
                    # we use floats to keep track of which values were added/can be modified
                    newPuzzle[row][col] =float(random.randint(1, (N**2)))
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
        fitnessEval = self.fitness(population)
        generation = 0
        solution = population[fitnessEval[1]]
        print("---- Generation: %s - Fitness: %s ----" % (generation, round(fitnessEval[0],3) ))    
        while not (self.terminate(fitnessEval[0], generation)):
            new_population = []
            
            for puz in population:
                #select
                parents = self.select(population, fitnessEval[2])
                
                #crossover
                newpuz = self.crossover(parents[0], parents[1])
                
                #mutate
                if ( random.uniform(0, 1.0) <= MUTATION_RATE):
                    newpuz = self.mutate(newpuz)
                    
                new_population.append(newpuz)
                
            #new fitness
            population = new_population
            fitnessEval = self.fitness(population)
            generation += 1
            solution = population[fitnessEval[1]]
            #print(solution)
            print("---- Generation: %s - Fitness: %s ----" % (generation, round(fitnessEval[0], 3)))    
            
        return solution
        
    # if the algo should terminate
    def terminate(self, fitnessValue, generation):
        return (fitnessValue >= 1.0 or generation >= NUM_GENERATIONS)
    
    # will return child
    def crossover(self, p1, p2):
        rRange = range(0, random.randint(0, ((N**2) - 1)))
        cRange = range(0, random.randint(0, ((N**2) - 1)))
        
        #p1[0:r]
        #p2[r:((N**2))]
        child = list(p1)
        
        #modify rows and columns according to ranges
        for r in range(0, N**2):
            for c in range(0, N**2):
                if c in fixedValues[r] and (r in rRange or c in cRange):
                    #can modify, select from p2
                    child[r][c] = p2[r][c]

        return np.array(child)

    # mutate x
    def mutate(self, x):
        rRange = random.randint(0, (len(fixedValues) - 1))
        cRange = random.randint(0, (len(fixedValues[rRange]) - 1) )
        newPuzzle = x
        newPuzzle[rRange][cRange] = random.randint(1, (N**2))
        return newPuzzle

    #select 2 individuals to crossover
    def select(self, population, probabilities):
        K = 5
        #print(probabilities)
        draw = random.choices(population=population, 
                       k=K,
             weights=probabilities)
        
        r1 = random.randint(0, K-1)
        r2 = random.randint(0, K-1)

        return [draw[r1], draw[r2]]

    #find fitness of the indidividual puzzle
    def individualFitness(self, puz):
        puzError = (N**2)*3
        totalError = float(puzError) #want to get the lowest
        #check columns
        for c in puz.T:
            if (N**2) == len(set(c)):
                puzError = puzError - 1
                
        #check rows
        for r in puz:
            if (N**2) == len(set(r)):
                puzError = puzError - 1
                
        #check boxes
        for br in range(0, (N**2),3):
            for bc in range(0, (N**2), 3):
                box = [puz[br][bc], puz[br][bc + 1],puz[br][bc + 2],
                        puz[br + 1][bc],puz[br + 1][bc + 1], puz[br + 1][bc + 2], 
                        puz[br + 2][bc], puz[br + 2][bc + 1], puz[br + 2][bc + 2] ]
                
                if (N**2) == len(set(box)):
                    puzError = puzError - 1
 
        return (1-(puzError/totalError))
    
    #find fitness individual
    def fitness(self, population):
        fittestIndex = 0
        fitnessValue = 0 #want to get the lowest
        probs = []
        for idx, puz in enumerate(population):
            puzFitness = self.individualFitness(puz)
            if puzFitness >= fitnessValue:
                fittestIndex = idx
                fitnessValue = puzFitness
            probs.append(puzFitness)
                
        return (fitnessValue, fittestIndex, probs) 

if __name__ == "__main__":
    startTime = time.time()
    population = populationGeneration(puzzleProblem)
    ga = SudokuGA(population)
    sol = ga.run()
    
    print("Solution:\n %s" % sol)
    print(puzzleProblem)
    print("---- Population Length: %s ----" % len(population))    
    print("---- %s seconds ----" % (time.time() - startTime))