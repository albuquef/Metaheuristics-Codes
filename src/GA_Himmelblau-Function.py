import numpy as np
# import matplotlib as plt
import matplotlib.pyplot as plt
import random as rd

import GA_Functions_Continuous as gafc

# chromosome_test = np.array([1,1,0,1,1,0,0,1,1,0,0,1,  # y variable
#                             0,1,1,0,1,0,1,0,1,1,0,0]) # x variable

chromosome_test = np.array([0,0,1,1,1,0,  # y variable
                            1,1,1,0,1,1]) # x variable

print('-'*30)
print('\nTesting new chromossome for decoding,')
print(f'chromosome is: \n x = {chromosome_test[(len(chromosome_test)//2):]}  \
      \n y = {chromosome_test[0:(len(chromosome_test)//2)]}')
print('-'*30)

decoded_x,decoded_y,obj_value = gafc.Objetive_value(chromosome_test)
print(f'decoded x = {decoded_x:.3f}\ndecoded y = {decoded_y:.3f}')
print(f'objetive function value: {obj_value:.3f}')
print('-'*30)


chromosome_size = len(chromosome_test)
population_size = 20
all_solutions = np.empty((0,chromosome_size))

for i in range(population_size):  # 20 random solutions
    rd.shuffle(chromosome_test)
    all_solutions = np.vstack((all_solutions, chromosome_test))

# for i in range(population_size):
#     print_XY(all_solutions[i],i)

# parents
parents = gafc.Find_Parents(all_solutions)
# for i in range(2):
#     print_XY(parents[i],i)

parent_1 = np.array([0,0,1,1,1,0,
                     0,1,0,1,0,1,])

parent_2 = np.array([1,0,0,1,0,1,
                     1,1,0,0,1,1,])

# 2-point crossover
childrens = gafc.Crossover(parent_1,parent_2)

print(childrens)

child_1 = np.array([0,0,0,0,0,0,
                     0,0,0,0,0,0,])
child_2 = np.array([1,1,1,1,1,1,
                     1,1,1,1,1,1,])
#mutation
# mutated_childrens = Mutation(childrens[0],childrens[1])
mutated_childrens = gafc.Mutation(child_1,child_2,1)         
print(mutated_childrens[0])
print(mutated_childrens[1])



