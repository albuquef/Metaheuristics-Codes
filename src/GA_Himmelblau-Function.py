import numpy as np
# import matplotlib as plt
import matplotlib.pyplot as plt
import random as rd

def print_XY(chromossome, i):
    # print('-'*30)
    print(f'chromosome {i+1} is: \n x = {chromossome[(len(chromossome)//2):]}  \
      \n y = {chromossome[0:(len(chromossome)//2)]}')
    print('-'*30)

def Himmelblau_Function(x,y): # min ((x**2)+y-11)**2 + (x+(y**2)-7)**2 : HF
    z = ((x**2)+y-11)**2+(x+(y**2)-7)**2
    return z

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
    lb_x,ub_x = -6,6
    lb_y,ub_y = -6,6
    len_x =  len_y = (len(chromosome)//2)

    precision_x = (ub_x - lb_x)/((2**len_x)-1)
    precision_y = (ub_y - lb_y)/((2**len_y)-1)

    x_bit_sum, y_bit_sum = SumBits_XY(chromosome) # sum(bit*2^z)

    decoded_x = (x_bit_sum*precision_x) + lb_x
    decoded_y = (y_bit_sum*precision_y) + lb_y
    # print(f'decoded x = {decoded_x:.3f}\ndecoded y = {decoded_y:.3f}')

    obj_value = Himmelblau_Function(decoded_x,decoded_y)
    # print(f'objetive function value: {obj_value:.3f}')
    
    return decoded_x,decoded_y,obj_value

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




# chromosome_test = np.array([1,1,0,1,1,0,0,1,1,0,0,1,  # y variable
#                             0,1,1,0,1,0,1,0,1,1,0,0]) # x variable

chromosome_test = np.array([0,0,1,1,1,0,  # y variable
                            1,1,1,0,1,1]) # x variable

print('-'*30)
print('\nTesting new chromossome for decoding,')
print(f'chromosome is: \n x = {chromosome_test[(len(chromosome_test)//2):]}  \
      \n y = {chromosome_test[0:(len(chromosome_test)//2)]}')
print('-'*30)

decoded_x,decoded_y,obj_value = Objetive_value(chromosome_test)
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
parents = Find_Parents(all_solutions)
# for i in range(2):
#     print_XY(parents[i],i)

parent_1 = np.array([0,0,1,1,1,0,
                     0,1,0,1,0,1,])

parent_2 = np.array([1,0,0,1,0,1,
                     1,1,0,0,1,1,])

# 2-point crossover
childrens = Crossover(parent_1,parent_2)

print(childrens)

child_1 = np.array([0,0,0,0,0,0,
                     0,0,0,0,0,0,])
child_2 = np.array([1,1,1,1,1,1,
                     1,1,1,1,1,1,])
#mutation
# mutated_childrens = Mutation(childrens[0],childrens[1])
mutated_childrens = Mutation(child_1,child_2,1)         
print(mutated_childrens[0])
print(mutated_childrens[1])


exit()

