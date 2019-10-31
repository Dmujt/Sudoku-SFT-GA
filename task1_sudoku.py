#####################################
# Task 1: GA for Sudoku
# Dena Mujtaba
#####################################
import time
import random
import math
import numpy as np

random.seed(54)

#sample puzzles. Select the one to solve by entering below in
# the puzzleProblem numpy array
puzzleProblem2 = [
        [0,0,0,3],
        [0,0,0,2],
        [1,0,0,0],
        [4,0,0,0]
        ]

puzzleProblem3 = [[0,0,0,2,6,0,7,0,1],
                 [6,8,0,0,7,0,0,9,0],
                 [8,9,0,0,0,4,5,0,0],
                 [0,2,0,1,0,0,0,4,0],
                 [0,0,4,6,0,2,9,0,0],
                 [0,5,0,0,0,3,0,2,8],
                 [0,0,9,3,0,0,0,7,4],
                 [0,4,0,0,5,0,0,3,6],
                 [7,0,3,0,1,8,0,0,0]]

puzzleProblem4 = [
        [0,1,0,6,0,0,5,0,0,13,0,4,0,10,0,0],
        [0,16,0,5,13,0,0,0,2,1,14,3,0,9,6,0],
        [14,8,10,0,6,0,0,0,15,7,0,5,0,0,11,16],
        [4,3,2,0,11,0,0,0,8,0,0,16,0,0,5,0],
        [0,10,16,0,0,12,11,0,7,4,3,0,0,0,0,6],
        [0,0,12,7,5,0,0,9,14,0,8,6,10,11,2,0],
        [3,0,0,2,7,0,1,0,0,11,0,0,0,15,14,8],
        [6,0,4,11,0,15,0,13,5,2,12,0,7,0,16,1],
        [0,11,8,0,0,10,15,0,6,16,13,14,3,0,12,2],
        [13,7,6,0,0,5,14,0,0,0,4,2,9,0,0,15],
        [0,0,14,15,0,0,6,8,9,12,7,0,1,16,13,5],
        [9,0,0,0,0,13,4,16,0,0,15,8,6,14,0,0],
        [8,5,0,0,16,0,7,0,0,0,0,12,11,13,15,3],
        [12,0,0,0,15,0,0,0,0,9,0,0,0,2,4,0],
        [10,0,0,3,0,0,13,12,0,0,0,0,0,0,0,7],
        [16,0,0,0,1,0,0,0,3,6,10,7,0,0,0,0]
        ]
#puzzle to solve, 0s indicate empty

puzzleProblem = np.array(puzzleProblem3)

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
POPULATION_SIZE = 1000
NUM_GENERATIONS = 100
MUTATION_RATE = 0.25
CROSS_RATE = .75

# initialize the population
def populationGeneration(puzzle):
    pop = []
    for individual in range(0, POPULATION_SIZE):
        newPuzzle = []
        for row in range(0, (N**2)):
            puzRow = []
            opts = list(range(1, N**2 + 1))
            
            #first only use available numbers
            for col in puzzle[row]:
                if col != 0:
                    opts.pop(opts.index(col))
                    
            for col in range(0,(N**2)):
                if puzzle[row][col] == 0:
                    # set a random float here
                    # we use floats to keep track of which values were added/can be modified
                    puzRow.append(opts.pop(random.randint(0, len(opts) - 1)))
                else: 
                    puzRow.append(puzzle[row][col])
            newPuzzle.append(puzRow)
        pop.append(np.array(newPuzzle))

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
            
            for i in range(0, POPULATION_SIZE):
                #select
                parent1,parent2 = self.select(population, fitnessEval[2], solution, fitnessEval[1])
                
                #crossover
                newpuz, newpuz2 = self.crossover(parent1, parent2)
                
                #mutate
                if ( random.uniform(0, 1.0) <= MUTATION_RATE):
                    newpuz = self.mutate(newpuz)
                    
                new_population.append(newpuz2)
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
        return (generation >= NUM_GENERATIONS or fitnessValue >= 1.0)
    
    # will return child
    def crossover(self, p1, p2):
        rRange = random.randint(0, len(np.concatenate(np.array(fixedValues)).ravel().tolist()))
        #print(rRange)
        #p1[0:r]
        #p2[r:((N**2))]
        child = list(p1)
        child2 = list(p2)
        
        rc = 0
        #modify rows and columns according to ranges
        for ridx, row in enumerate(fixedValues):
            for cidx, c in enumerate(row):
                if rc >= rRange:
                    #set the p1 to p2
                    child[ridx][c] = p2[ridx][c]
                    child2[ridx][c] = p1[ridx][c]

                rc +=1
                
        return (np.array(child), np.array(child2))

    # mutate x
    def mutate(self, x):
        newPuzzle = x
        rowToChange = random.randint(0, N**2 - 1)
        opts = fixedValues[rowToChange]
        
        if len(opts) >= 2:
            drawCol = random.choices(population=opts, 
                k=2)
            
            newPuzzle[rowToChange][drawCol[0]] = newPuzzle[rowToChange][drawCol[1]]
        
        return newPuzzle

    #select 2 individuals to crossover
    def select(self, population, probabilities, fittest,fittestIndex):
        K = 10
        #print(probabilities)
        matingpool = list(population)
        matingpool.pop(fittestIndex)
        probs = list(probabilities)
        probs.pop(fittestIndex)
        draw = random.choices(population=matingpool, 
                       k=K,
             weights=probs)
        
        r1 = fittest
        r2 = draw[random.randint(0, K-1)]
        return (r1, r2)

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
        for br in range(0, (N**2),N):
            for bc in range(0, (N**2), N):
                box = []
                
                for i in range(0, N):
                    for z in range(0, N):
                        box.append(puz[br + i][bc + z])
                
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
            probs.append(puzFitness*100)
                
        return (fitnessValue, fittestIndex, probs, population[fittestIndex]) 

if __name__ == "__main__":
    startTime = time.time()
    population = populationGeneration(puzzleProblem)
    ga = SudokuGA(population)
    sol = ga.run()
    
    print("Solution:\n %s" % sol)
    print(puzzleProblem)
    print("---- Population Length: %s ----" % len(population))    
    print("---- %s seconds ----" % (time.time() - startTime))