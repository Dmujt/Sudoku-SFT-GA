#########################################
# Task 2 Trail Ant Problem
# GP Code Implementation From DEAP tutorial: 
#https://github.com/DEAP/deap/blob/master/examples/gp/ant.py
##########################################
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import random
import logging
import sys
from functools import partial
import copy
import numpy as np
import matplotlib.pyplot as plt
import operator
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)


def progn(*args):
    for arg in args:
        arg()

def prog2(out1, out2): 
    return partial(progn,out1,out2)

def prog3(out1, out2, out3):     
    return partial(progn,out1,out2,out3)  

def if_then_else(condition, out1, out2):
    out1() if condition() else out2()

#ANT simulator from DEAP documentation
class AntSimulator(object):
    direction = ["north","east","south","west"]
    dir_row = [1, 0, -1, 0]
    dir_col = [0, 1, 0, -1]
    
    def __init__(self, max_moves):
        self.max_moves = max_moves
        self.moves = 0
        self.eaten = 0
        self.routine = None
        
    def _reset(self):
        self.row = self.row_start 
        self.col = self.col_start 
        self.dir = 1
        self.moves = 0  
        self.eaten = 0
        self.matrix_exc = copy.deepcopy(self.matrix)

    @property
    def position(self):
        return (self.row, self.col, self.direction[self.dir])
            
    def turn_left(self): 
        if self.moves < self.max_moves:
            self.moves += 1
            self.dir = (self.dir - 1) % 4

    def turn_right(self):
        if self.moves < self.max_moves:
            self.moves += 1    
            self.dir = (self.dir + 1) % 4
        
    def move_forward(self):
        if self.moves < self.max_moves:
            self.moves += 1
            self.row = (self.row + self.dir_row[self.dir]) % self.matrix_row
            self.col = (self.col + self.dir_col[self.dir]) % self.matrix_col
            if self.matrix_exc[self.row][self.col] == "food":
                self.eaten += 1
            self.matrix_exc[self.row][self.col] = "passed"

    def sense_food(self):
        ahead_row = (self.row + self.dir_row[self.dir]) % self.matrix_row
        ahead_col = (self.col + self.dir_col[self.dir]) % self.matrix_col        
        return self.matrix_exc[ahead_row][ahead_col] == "food"
   
    def if_food_ahead(self, out1, out2):
        return partial(if_then_else, self.sense_food, out1, out2)
   
    def run(self,routine):
        self._reset()
        while self.moves < self.max_moves:
            routine()
    
    def parse_matrix(self, matrix):
        self.matrix = list()
        for i, line in enumerate(matrix):
            self.matrix.append(list())
            for j, col in enumerate(line):
                if col == "#":
                    self.matrix[-1].append("food")
                elif col == ".":
                    self.matrix[-1].append("empty")
                elif col == "S":
                    self.matrix[-1].append("empty")
                    self.row_start = self.row = i
                    self.col_start = self.col = j
                    self.dir = 1
        self.matrix_row = len(self.matrix)
        self.matrix_col = len(self.matrix[0])
        self.matrix_exc = copy.deepcopy(self.matrix)

#define number of moves
ant = AntSimulator(600)
pset = gp.PrimitiveSet("MAIN", 0)
pset.addPrimitive(ant.if_food_ahead, 2)
pset.addPrimitive(prog2, 2)
pset.addPrimitive(prog3, 3)
pset.addTerminal(ant.move_forward)
pset.addTerminal(ant.turn_left)
pset.addTerminal(ant.turn_right)

#max fitness
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#individual trees
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalArtificialAnt(individual):
    # Transform the tree expression to functionnal Python code
    routine = gp.compile(individual, pset)
    # Run the generated routine
    ant.run(routine)
    return ant.eaten,

toolbox.register("evaluate", evalArtificialAnt)
#Tournament selection with K=7
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    random.seed(69)

    trail_file = open("santafe_trail.txt")
    ant.parse_matrix(trail_file)
    
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Avg", np.mean)
    stats.register("Std", np.std)
    stats.register("Min", np.min)
    stats.register("Max", np.max)
    
    pop, logbook = algorithms.eaSimple(pop, toolbox, 
                                       0.7, 0.1, 100,
                                       stats,
                                       halloffame=hof,
                                       verbose=True)
    
    logging.info("Best individual is %s, %s", gp.compile(hof[0], pset=pset), hof[0].fitness)
    print(logbook)
    gen = logbook.select("gen")
    fit_mins = logbook.select("Min")
    fit_maxs = logbook.select("Max")
    fit_avgs= logbook.select("Avg")

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_avgs, label="Avg Fitness")
    line2 = ax1.plot(gen, fit_mins, label="Min Fitness")
    line3 = ax1.plot(gen, fit_maxs, label="Max Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    
    lns = [line1, line2, line3]
    ax1.legend(lns,labels= ["Avg Fitness", "Min Fitness", "Max Fitness"], loc="bottom right")
    
    plt.show()

    return pop, hof, stats

if __name__ == "__main__":
    main()