import pygad
import numpy
import random
random.seed(42)
numpy.random.seed(42)
from statistics import mean,pstdev
from scipy.stats import sem

function_inputs = [4,-2,3.5,5,-11,-4.7,4,-2,3.5,5,-11,-4.74,-2,3.5,5,-11,-4.74,-2,3.5,5,-11,-4.74,-2,3.5,5,-11,-4.74,-2,3.5,5,-11,-4.7]
desired_output = 320

fitness_history=[]

def fitness_func(solution, solution_idx):
    output = numpy.sum(solution*function_inputs)
    #print(output)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    fitness_history.append(fitness)
    #print(fitness)
    
    return fitness


initial_population = numpy.zeros((8, 32))

ga_instance = pygad.GA(num_generations=100,
                       sol_per_pop=8,
                       num_genes=len(function_inputs),
                      # parent_selection_type='tournament',
                       #K_tournament=1,
                       num_parents_mating=4,
                       fitness_func=fitness_func,
                       #crossover_probability=0.001,
                       initial_population = initial_population,
                       mutation_type='random',
                       #mutation_by_replacement=True)
                       mutation_probability=0.9)

#print(ga_instance.initial_population)
ga_instance.run()



solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
#print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
print("mean = {}".format(mean(fitness_history)))
print("std dev = {}".format(pstdev(fitness_history)))
print("std error = {}".format(sem(fitness_history)))

ga_instance.plot_result(title='K_tournament = 5 Best Fitness Value = {}'.format(numpy.round(solution_fitness, 6)))

#filename = 'gene'
#ga_instance.save(filename=filename)
