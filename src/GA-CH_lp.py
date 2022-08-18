# Genetic Algorithm (GA) with penalty (Constraint Handling)

'''
LP Problem:
min Z = 0.5*((x**4)+(y**4)-(16*x**2)-(16*y**2)+(5*x)+(y*5))
st.
    -5 <= x <=5
    -5 <= y <=5
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random as rd

def print_XY(chromosome):
    print('-'*30)
    print(f'Binary solution is: \n x = {chromosome[(len(chromosome)//2):].astype(int)}  \
      \n y = {chromosome[0:(len(chromosome)//2)].astype(int)}')

def SumBits_XY(chromosome):
    
    z,t = 0,1
    xbit_sum = 0
    for i in range((len(chromosome)//2)):
        xbit = chromosome[-t]*(2**z)
        xbit_sum += xbit
        t+=1
        z+=1
        
    z,t = 0,1+(len(chromosome)//2)
    ybit_sum = 0
    for i in range((len(chromosome)//2)):
        ybit = chromosome[-t]*(2**z)
        ybit_sum += ybit
        t+=1
        z+=1
    
    return xbit_sum,ybit_sum

def Objetive_value(chromosome):
    lb_x,ub_x = -5,5
    lb_y,ub_y = -5,5
    len_x =  len_y = (len(chromosome)//2)

    precision_x = (ub_x - lb_x)/((2**len_x)-1)
    precision_y = (ub_y - lb_y)/((2**len_y)-1)

    x_bit_sum, y_bit_sum = SumBits_XY(chromosome) # sum(bit*2^z)

    decoded_x = (x_bit_sum*precision_x) + lb_x
    decoded_y = (y_bit_sum*precision_y) + lb_y
    # print(f'decoded x = {decoded_x:.3f}\ndecoded y = {decoded_y:.3f}')

    obj_value = 0.5*((decoded_x**4)+(decoded_y**4)-(16*decoded_x**2)-
                 (16*decoded_y**2)+(5*decoded_x)+(decoded_y*5))

    # print(f'objetive function value: {obj_value:.3f}')
    
    return decoded_x,decoded_y,obj_value

def Penalty(array, penalty_value):
    
    decoded_x,decoded_y,obj = Objetive_value(array) 
    
    Sum_Penalties = 0
    if decoded_x > 5 or decoded_x < -5:
        Sum_Penalties += penalty_value
    if decoded_y > 5 or decoded_y < -5:
        Sum_Penalties += penalty_value

    return Sum_Penalties

def Find_Parents(solutions):
    
    parents = np.empty((0,np.size(solutions,1)))
    sample_size_parents = 3
    
    for i in range(2):

        indice_list = np.random.choice(len(solutions),sample_size_parents,replace=False) # select 3 random parents 
        # replace => get unique numbers (don't duplicated)

        pos_parents = [-1]*sample_size_parents
        obj_pos_parents = [-1]*sample_size_parents
        for i in range(sample_size_parents):
            pos_parents[i] = solutions[indice_list[i]]
            obj_pos_parents[i] = Objetive_value(pos_parents[i])[2]
            
        # print(indice_list)
        # for i in range(len(pos_parents)):
        #     print_XY(pos_parents[i],i)
        #     print(f'obj value: {obj_pos_parents[i]:.3f}')
        #     print('-'*30)    
             
        index_min_parent = np.argmin(obj_pos_parents)
        parents = np.vstack((parents,pos_parents[index_min_parent]))
    
    # two rows, for each parent
    parent1,parent2  = parents[0,:], parents[1,:]
    return parent1,parent2

def Crossover(parent_1,parent_2,prob=1): # 2-point crossover
    
    child_1, child_2 = np.empty((0,len(parent_1))), np.empty((0,len(parent_2)))

    random_num = np.random.rand()
    
    if random_num < prob:
        
        child_1, child_2 = np.copy(parent_1), np.copy(parent_2) 
        
        index_1, index_2 = np.random.randint(0,len(parent_1)), np.random.randint(0,len(parent_2))
        while index_1 == index_2:
            index_2 = np.random.randint(0,len(parent_2))

        index_initial,index_final = min(index_1,index_2), max(index_1,index_2)
        # print(index_initial,index_final)
        
        child_1[index_initial:index_final+1] = np.copy(parent_2[index_initial:index_final+1])
        child_2[index_initial:index_final+1] = np.copy(parent_1[index_initial:index_final+1])
        
        # vet_parent_1 = np.copy(parent_1[index_initial:index_final+1])
        # vet_parent_2 = np.copy(parent_2[index_initial:index_final+1])        
        # for i in range(index_initial, index_final+1):
        #     child_1[i] = vet_parent_2[i-index_initial]
        #     child_2[i] = vet_parent_1[i-index_initial]
            
    else:
        child_1, child_2 = np.copy(parent_1), np.copy(parent_2) 

    return child_1, child_2

def Muatation_child(child, prob_mutation):
    
    mutated_child = np.copy(child) # mutated_indices = []
    t = 0
    for i in child:
        # print(f'current index: {t}')
        random_num = np.random.rand()
        if random_num < prob_mutation:
            # print(f'mutated index: {t}') # mutated_indices.append(t)
            child[t] = 1 if child[t] == 0 else 0 # change value of index
            mutated_child = child
        t+=1
    return mutated_child

def Mutation(child_1,child_2,prob_mutation=0.2): # mutating the 2 childrens bit-flipping/flip-bit mutatation
    
    mutated_child_1, mutated_child_2 = Muatation_child(child_1, prob_mutation), Muatation_child(child_2, prob_mutation)
    return mutated_child_1, mutated_child_2

### VARIABLES ###
Seed = 0 # Seed for the random number generator
prob_crossover = 1 # Probability of crossover
prob_mutation = 0.2 # Probability of mutation
K = 3 # For Tournament selection
pop = 30 # Population per generation
gen = 30 # Number of generations
print('-'*30)
print('(GA) with penalty (Constraint Handling)')
print('-'*30)
print('Parameters:')
print(f'-Population size: {pop} \n-Number of Generations: {gen}')
print(f'-Prob. of crossover: {prob_crossover} \n-Prob. of mutation: {prob_mutation}')


# Where the first 13 represent Y and the second 13 represent X
Init_Sol = np.array([0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, # y
                     1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]) # x

# create the random inicial sol
rand_solutions = np.empty((0,len(Init_Sol)))
for i in range(pop): # Shuffles the elements in the vector n times and stores them
    rd.shuffle(Init_Sol)
    rand_solutions = np.vstack((rand_solutions,Init_Sol))
    
print_XY(Init_Sol)
x,y,obj = Objetive_value(Init_Sol) 
print(f'\n(x,y) Initial solution: ({x:.3f},{y:.3f})')
print(f'Initial solution value: {obj:.3f}\n')

Final_Best_in_Generation_X = []
Worst_Best_in_Generation_X = []

For_Plotting_the_Best = np.empty((0,len(Init_Sol)+1))

One_Final_Guy = np.empty((0,len(Init_Sol)+2))
Final_solution = []

Min_for_all_Generations_for_Mut_1 = np.empty((0,len(Init_Sol)+1))
Min_for_all_Generations_for_Mut_2 = np.empty((0,len(Init_Sol)+1))

Min_for_all_Generations_for_Mut_1_1 = np.empty((0,len(Init_Sol)+2))
Min_for_all_Generations_for_Mut_2_2 = np.empty((0,len(Init_Sol)+2))

Min_for_all_Generations_for_Mut_1_1_1 = np.empty((0,len(Init_Sol)+2))
Min_for_all_Generations_for_Mut_2_2_2 = np.empty((0,len(Init_Sol)+2))



Final_solutions = []
Generation = 1 
for i in range(gen):
    
    New_Population = np.empty((0,len(Init_Sol))) # Saving the new generation
    
    All_in_Generation_X_1 = np.empty((0,len(Init_Sol)+1))
    All_in_Generation_X_2 = np.empty((0,len(Init_Sol)+1))
    
    Min_in_Generation_X_1 = []
    Min_in_Generation_X_2 = []
    
    Save_Best_in_Generation_X = np.empty((0,len(Init_Sol)+1))
    Final_Best_in_Generation_X = []
    Worst_Best_in_Generation_X = []
    
    # print('\n--> GENERATION: #',Generation)
    
    Family = 1
    for j in range(int(pop/2)): # range(int(pop/2))
            
        # print('\n--> FAMILY: #',Family)
    
        # Tournament Selection
        Parent_1, Parent_2 = Find_Parents(rand_solutions)
        
        # Crossover
        Child_1, Child_2 = Crossover(Parent_1,Parent_2,prob_crossover)
        
        # Mutation
        Mutated_Child_1, Mutated_Child_2 = Mutation(Child_1,Child_2,prob_mutation)
        
        # Calculate fitness values of mutated children
            
        fit_val_muted_children = np.empty((0,2))
        
        
        obj_MC_1 = Objetive_value(Mutated_Child_1)[2] # For mutated child #1
        obj_MC_2 = Objetive_value(Mutated_Child_2)[2] # For mutated child #2
        
        # print('\nBefore Penalty FV at Mutated Child #1 at Gen #',Generation,':', obj_MC_1)
        # print('Before Penalty FV at Mutated Child #2 at Gen #',Generation,':', obj_MC_2)
        
        P1 = Penalty(Mutated_Child_1,20) # penalty_value = 20
        obj_MC_1 = obj_MC_1 + P1
        
        P2 = Penalty(Mutated_Child_2,20)
        obj_MC_2 = obj_MC_2 + P2
        
        # print('After Penalty FV at Mutated Child #1 at Gen #',Generation,':', obj_MC_1)
        # print('After Penalty FV at Mutated Child #2 at Gen #',Generation,':', obj_MC_2)
        
        
        All_in_Generation_X_1_1_temp = Mutated_Child_1[np.newaxis]
        All_in_Generation_X_1_1 = np.column_stack((obj_MC_1, All_in_Generation_X_1_1_temp))
        
        All_in_Generation_X_2_1_temp = Mutated_Child_2[np.newaxis]
        All_in_Generation_X_2_1 = np.column_stack((obj_MC_2, All_in_Generation_X_2_1_temp))
        
        All_in_Generation_X_1 = np.vstack((All_in_Generation_X_1,All_in_Generation_X_1_1))
        All_in_Generation_X_2 = np.vstack((All_in_Generation_X_2,All_in_Generation_X_2_1))
        
        Save_Best_in_Generation_X = np.vstack((All_in_Generation_X_1,All_in_Generation_X_2))
        
        New_Population = np.vstack((New_Population,Mutated_Child_1,Mutated_Child_2))
        
        index_min_1 = np.argmin(All_in_Generation_X_1[:,:1]) # save the best in each generation
        Min_in_Generation_X_1 = All_in_Generation_X_1[index_min_1]
        Min_in_Generation_X_1 = Min_in_Generation_X_1[np.newaxis]
        
        index_min_2 = np.argmin(All_in_Generation_X_2[:,:1]) # save the best in each generation
        Min_in_Generation_X_2 = All_in_Generation_X_2[index_min_2]
        Min_in_Generation_X_2 = Min_in_Generation_X_2[np.newaxis]
        
        Family = Family+1
        
    
    index_min = np.argmin(Save_Best_in_Generation_X[:,:1]) # save the best in each generation
    Final_Best_in_Generation_X = Save_Best_in_Generation_X[index_min]
    Final_Best_in_Generation_X = Final_Best_in_Generation_X[np.newaxis]
    
    For_Plotting_the_Best = np.vstack((For_Plotting_the_Best,Final_Best_in_Generation_X))
    
    index_max = np.argmax(Save_Best_in_Generation_X[:,:1]) # save the best in each generation
    Worst_Best_in_Generation_X = Save_Best_in_Generation_X[index_max]
    Worst_Best_in_Generation_X = Worst_Best_in_Generation_X[np.newaxis]
    
    # Elitism, the best in the generation lives
    
    
    Darwin_Guy = Final_Best_in_Generation_X[:]
    Not_So_Darwin_Guy = Worst_Best_in_Generation_X[:]
    
    Darwin_Guy = Darwin_Guy[0:,1:].tolist()
    Not_So_Darwin_Guy = Not_So_Darwin_Guy[0:,1:].tolist()
    
    Best_1 = np.where((New_Population == Darwin_Guy).all(axis=1))
    Worst_1 = np.where((New_Population == Not_So_Darwin_Guy).all(axis=1))
    
    New_Population[Worst_1] = Darwin_Guy
    
    n_list = New_Population
    
    # save the minimum for each gen
    Min_for_all_Generations_for_Mut_1 = np.vstack((Min_for_all_Generations_for_Mut_1,Min_in_Generation_X_1))
    Min_for_all_Generations_for_Mut_2 = np.vstack((Min_for_all_Generations_for_Mut_2,Min_in_Generation_X_2))
    
    Min_for_all_Generations_for_Mut_1_1 = np.insert(Min_in_Generation_X_1, 0, Generation)
    Min_for_all_Generations_for_Mut_2_2 = np.insert(Min_in_Generation_X_2, 0, Generation)
    
    Min_for_all_Generations_for_Mut_1_1_1 = np.vstack((Min_for_all_Generations_for_Mut_1_1_1,Min_for_all_Generations_for_Mut_1_1))
    Min_for_all_Generations_for_Mut_2_2_2 = np.vstack((Min_for_all_Generations_for_Mut_2_2_2,Min_for_all_Generations_for_Mut_2_2))
    
    Generation = Generation+1
    
    Best_Guy_So_Far = np.vstack((Min_for_all_Generations_for_Mut_1_1_1,Min_for_all_Generations_for_Mut_2_2_2))


One_Final_Guy = np.vstack((Min_for_all_Generations_for_Mut_1_1_1,Min_for_all_Generations_for_Mut_2_2_2))
    


index_min = np.argmin(One_Final_Guy[:,1])
Final_solution = One_Final_Guy[index_min]
Final_solution = Final_solution[np.newaxis]

# print('Min in all Generations:',Final_solution.tolist())
# print('The Lowest Cost is:',Final_solution[:,1])

# print(f'\nFinal Solution: {np.squeeze(Final_solution[:,2:])}')
bin_final_solution = np.squeeze(Final_solution[:,2:])
decoded_x , decoded_y, obj = Objetive_value(bin_final_solution)

print_XY(bin_final_solution)
print(f'\n(x,y) Final solution: ({decoded_x:.4f}, {decoded_y:.4f})')
print(f'Final solution value: {float(np.squeeze(Final_solution[:,1])):.4f}')
print(f'At Generation: {int(np.squeeze(Final_solution[:,0]))}')


print('\n\nploting...')
# plotting
Look = (Final_solution[:,1])

plt.plot(For_Plotting_the_Best[:,0])
plt.axhline(y=Look,color='r',linestyle='--')
plt.title('Z Reached Through Generations',fontsize=20,fontweight='bold')
plt.xlabel('Generations',fontsize=18,fontweight='bold')
plt.ylabel('Z',fontsize=18,fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
xyz=(Generation/4, Look)
xyzz = (Generation/3.7, Look+0.5)
plt.annotate('Minimum Reached at: %0.3f' % Look, xy=xyz, xytext=xyzz,
             arrowprops=dict(facecolor='black', shrink=1,width=1,headwidth=5),
             fontsize=12,fontweight='bold')
plt.show()



