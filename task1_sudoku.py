#####################################
# Task 1: GA for Sudoku
# Dena Mujtaba
#####################################

# SETUP PARAMS
populationSize = 100
numGenerations = 10

# initialize the population
def populationGeneration():
    return []

#main class to solve the 
class SudokuGA():

    #constructor for GA
    def __init__(self):
        # check arguments for GA setup parameters
        print("Starting...")

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
    population = []