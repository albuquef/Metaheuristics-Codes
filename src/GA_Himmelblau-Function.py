import numpy as np
# import matplotlib as plt
import matplotlib.pyplot as plt
import random as rd
import time

import GA_Functions_Continuous as gafc

# chromosome_test = np.array([1,1,0,1,1,0,0,1,1,0,0,1,  # y variable
#                             0,1,1,0,1,0,1,0,1,1,0,0]) # x variable

# parameters: 
prob_crossover = 1  
prob_mutation = 0.6
population = 200
generations = 180

inicial_solution = np.array([0,1,0,0,0,1,0,0,1,0,1,1,1,   # y variable
                            0,1,1,1,0,0,1,0,1,1,0,1,1])  # x variable
chromosome_size = len(inicial_solution)

pool_solutions = np.empty((0,chromosome_size))
vbest_generation = np.empty((0,chromosome_size+1))


for i in range(population): 
    rd.shuffle(inicial_solution)
    pool_solutions = np.vstack((pool_solutions,inicial_solution)) # create the inicial population

start_time = time.time()

for i in range(generations):
    
    # print(f'-> generation: {i+1}')
    new_population =  np.empty((0,chromosome_size))
    obj_new_population = np.empty((0,chromosome_size+1))
    
    best_for_plotting = np.empty((0,chromosome_size+1))
    
    for j in range(int(population/2)):
        # print(f'- family: {j+1}')
        parent_1, parent_2 = gafc.Find_Parents(pool_solutions) # parents
        child_1,child_2 = gafc.Crossover(parent_1,parent_2, prob_crossover) # crossover -> childs
        mutated_child_1, mutated_child_2 = gafc.Mutation(child_1,child_2,prob_mutation) # childs mutated or not
        # child_1, child_2 = gafc.Mutation(child_1,child_2,prob_mutation) # childs mutated or not
        
        obj_child_1 = np.hstack((gafc.Objetive_value(mutated_child_1)[2],mutated_child_1)) # obj value, child_solution 
        obj_child_2 = np.hstack((gafc.Objetive_value(mutated_child_2)[2],mutated_child_2))
        
        
        new_population = np.vstack((new_population,mutated_child_1,mutated_child_2))
        obj_new_population = np.vstack((obj_new_population,obj_child_1,obj_child_2))

    
    pool_solutions = np.copy(new_population) # same size of solutions
    
    # to find the best obj. value of each generation
    best_for_plotting = np.array(sorted(obj_new_population,key=lambda x:x[0])) 
    # sorted to pick the first -> best value 
    vbest_generation = np.vstack((vbest_generation,best_for_plotting[0]))


end_time = time.time()


sorted_last_population = np.array(sorted(obj_new_population,key=lambda x:x[0]))
sorted_vbest_generation = np.array(sorted(vbest_generation,key=lambda x:x[0]))

best_convergence = sorted_last_population[0]
# print("Final Solution (Convergence):",best_convergence[1:]) # final solution entire chromosome [0:obj_value, 1:chromosome]
# gafc.print_XY(best_convergence[1:],0)
best_solution = sorted_vbest_generation[0]
# print("Final Solution (Best):",best_solution[1:]) # final solution entire chromosome
# gafc.print_XY(best_solution[1:],0) 

# get decoded x, decoded y and obj_value
print('-'*30)
best_convergence_final = gafc.Objetive_value(best_convergence[1:])
print(f'Convergence solution: \ndecoded x = {best_convergence_final[0]:.3f}\ndecoded y = {best_convergence_final[1]:.3f}')
print(f'objetive function value: {best_convergence_final[2]:.3f}')
print('-'*30)
best_solution_final = gafc.Objetive_value(best_solution[1:])
print(f'Best Solution: \ndecoded x = {best_solution_final[0]:.3f}\ndecoded y = {best_solution_final[1]:.3f}')
print(f'objetive function value: {best_solution_final[2]:.3f}')
print('-'*30)
# print(vbest_generation)
# print('-'*30)


# ------ ploting -------
best_obj_val_convergence = best_convergence[0]
best_obj_val_overall = best_solution[0] # only obj value


plt.plot(vbest_generation[:,0]) 

plt.axhline(y=best_obj_val_convergence,color='r',linestyle='--') # create a red line for convergence and best
plt.axhline(y=best_obj_val_overall,color='r',linestyle='--')


plt.title("Z Reached Through Generations",fontsize=20,fontweight='bold') # title and labels
plt.xlabel("Generation",fontsize=18,fontweight='bold')
plt.ylabel("Z",fontsize=18,fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')


if sorted_vbest_generation[-1][0] > 2:  # k is the high of text use after, need to be porporcional to bounds
    k = 0.8
elif sorted_vbest_generation[-1][0] > 1:
    k = 0.5
elif sorted_vbest_generation[-1][0] > 0.5:
    k = 0.3
elif sorted_vbest_generation[-1][0] > 0.3:
    k = 0.2
else:
    k = 0.1

xyz1 = (generations/2.4,best_obj_val_convergence) # position of arrow and text (location of x axis, location of y axis)
xyzz1 = (generations/2.2,best_obj_val_convergence+k) # without k the high of text and arrow is the same

plt.annotate("At Convergence: %0.5f" % best_obj_val_convergence,xy=xyz1,xytext=xyzz1,
             arrowprops=dict(facecolor='black',shrink=1,width=1,headwidth=5),
             fontsize=12,fontweight='bold')


xyz2 = (generations/6,best_obj_val_overall) # position of arrow and text
xyzz2 = (generations/5.4,best_obj_val_overall+(k/2))

plt.annotate("Minimum Overall: %0.5f" % best_obj_val_overall,xy=xyz2,xytext=xyzz2,
             arrowprops=dict(facecolor='black',shrink=1,width=1,headwidth=5),
             fontsize=12,fontweight='bold')


plt.show()