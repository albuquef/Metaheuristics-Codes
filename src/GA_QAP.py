import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random as rd

   
'''
https://www.localsolver.com/docs/last/exampletour/qap.html
A distance is specified for each pair of locations,
and a flow (or weight) is specified for each pair of facilities 
(e.g. the amount of supplies transported between the pair).
'''

def find_Parents(solutions_list):
    
    Parents = np.empty((0,len(solutions_list[0])))
            
    for i in range(2):
        
        # 3 differents and unique numbers                    
        index_1, index_2, index_3 =  rd.sample(range(0,len(solutions_list)), 3)
        indexes = [index_1,index_2,index_3] 
        
        # 3 possibles parents, differents
        pos_Parent = [-1]*3
        obj_Parents = [-1]*3
        cont = 0
        for i in indexes:
            pos_Parent[cont] =  solutions_list[i]
            cont+=1
        
        for i in range(3):
            obj_Parents[i] = Solution_Cost(pos_Parent[i])
        
        #get the best(min) parent 
        index_min_parent = np.argmin(obj_Parents)
    
        Parents = np.vstack((Parents,pos_Parent[index_min_parent]))
            
    return Parents[0], Parents[1]

def fit_Crossover(Parent_1,Parent_2):
    
    rand1, rand2 =  rd.sample(range(0,len(Parent_1)), 2)

    pos_initial,pos_final = min(rand1,rand2), max(rand1,rand2)
    # print(pos_initial, pos_final)

    seg_parent_1 = Parent_1[pos_initial:pos_final+1]
    seg_parent_2 = Parent_2[pos_initial:pos_final+1]

    child_1, child_2 = [-1]*solution_size, [-1]*solution_size
    # child_1[pos_initial:pos_final+1] = Parent_2[pos_initial:pos_final+1]
    # child_2[pos_initial:pos_final+1] = Parent_1[pos_initial:pos_final+1]
    child_1[pos_initial:pos_final+1] = seg_parent_2[0:(pos_final-pos_initial)+1]
    child_2[pos_initial:pos_final+1] = seg_parent_1[0:(pos_final-pos_initial)+1]

    # child_1 already have segment_2, similar to child_2
    for i in range(0,pos_initial):
        for elem in seg_parent_1:
            if (elem not in child_1 and child_1[i] == -1):
                child_1[i] = elem
        for elem in seg_parent_2:
            if (elem not in child_2 and child_2[i] == -1):
                child_2[i] = elem


    for i in range(pos_final+1, solution_size):
        for elem in seg_parent_1:
            if (elem not in child_1 and child_1[i] == -1):
                child_1[i] = elem
        for elem in seg_parent_2:
            if (elem not in child_2 and child_2[i] == -1):
                child_2[i] = elem

    for i in range(solution_size):
        for elem in Parent_1:
            if (elem not in child_1 and child_1[i] == -1):
                child_1[i] = elem
        for elem in Parent_2:
            if (elem not in child_2 and child_2[i] == -1):
                child_2[i] = elem
                
    return child_1, child_2      

def child_Mutation(Child,prob_mutation = 0.3):
    
    mutated_child = np.copy(Child) # mutated_indices = []

    random_num = np.random.rand()
    if random_num < prob_mutation:
        
        rand1, rand2 =  rd.sample(range(0,len(Parent_1)), 2)
        pos_initial,pos_final = min(rand1,rand2), max(rand1,rand2)
        
        # print(pos_initial,pos_final)
        
        child_segment = Child[pos_initial:pos_final+1]
        # print(child_segment)
        
        Child[pos_initial:pos_final+1] = list(reversed(child_segment[0:(pos_final-pos_initial)+1]))

        mutated_child = Child


    return mutated_child



# calculate de cost of solution
def Solution_Cost(array):
    New_Dist_DF = Dist.reindex(columns=array, index=array)
    New_Dist_Arr = np.array(New_Dist_DF)
    # Make a dataframe of the cost of the initial solution
    Objfun1_start = pd.DataFrame(New_Dist_Arr*Flow)
    Objfun1_start_Arr = np.array(Objfun1_start)
    sum_start_int = sum(sum(Objfun1_start_Arr))
    return(sum_start_int)






Dist = pd.DataFrame([[0,1,2,3,1,2,3,4],[1,0,1,2,2,1,2,3],[2,1,0,1,3,2,1,2],
                      [3,2,1,0,4,3,2,1],[1,2,3,4,0,1,2,3],[2,1,2,3,1,0,1,2],
                      [3,2,1,2,2,1,0,1],[4,3,2,1,3,2,1,0]],
                    columns=["A","B","C","D","E","F","G","H"],
                    index=["A","B","C","D","E","F","G","H"])

Flow = pd.DataFrame([[0,5,2,4,1,0,0,6],[5,0,3,0,2,2,2,0],[2,3,0,0,0,0,0,5],
                      [4,0,0,0,5,2,2,10],[1,2,0,5,0,10,0,0],[0,2,0,2,10,0,5,1],
                      [0,2,0,2,0,5,0,10],[6,0,5,10,0,1,10,0]],
                    columns=["A","B","C","D","E","F","G","H"],
                    index=["A","B","C","D","E","F","G","H"])

inicial_solution = ["D","A","C","B","G","E","F","H"] # inicial solution


print(f'inicial solution: {inicial_solution}')
print(f'inicial solution cost: {Solution_Cost(inicial_solution)}')


### VARIABLES ###
### VARIABLES ###
prob_crossover = 1 # Probability of crossover
prob_mutation = 0.3 # Probability of mutation
K = 3 # For Tournament selection - ps: not used
pop = 100 # Population per generation
gen = 30 # Number of generations
### VARIABLES ###
### VARIABLES ###


X0 = inicial_solution[:] #copy inicial solution


# 1- Randomly generate n solutions, for generation #1
solution_size = len(X0)
solutions_list = np.empty((0,solution_size))

for i in range(int(pop)): # Shuffles the elements in the vector n times and stores them
    rnd_sol = rd.sample(inicial_solution,solution_size)
    solutions_list = np.vstack((solutions_list,rnd_sol))
    
    
# -------------- testing GA FUNCTIONS  ------------------
# Parent_1, Parent_2 = find_Parents(solutions_list)
# print(Parent_1)
# print(Parent_2)
# Child_1, Child_2 = fit_Crossover(Parent_1,Parent_2)
# print(Child_1)
# print(Child_2)

# prob_mutation = 1
# mutation_child_1 = child_Mutation(Child_1,prob_mutation)
# mutation_child_2 = child_Mutation(Child_2,prob_mutation)
# print(mutation_child_1)
# print(mutation_child_2)
#-----------------------------------------------


Final_Best_in_Generation_X = [] # tracking proposes
Worst_Best_in_Generation_X = []

plotting_best = np.empty((0,solution_size+1))

One_Final_Guy = np.empty((0,solution_size+2))
One_Final_Guy_Final = []

Min_for_all_Generations_for_Mut_1 = np.empty((0,len(X0)+1)) # plus the fitness value
Min_for_all_Generations_for_Mut_2 = np.empty((0,len(X0)+1))

Min_for_all_Generations_for_Mut_1_1 = np.empty((0,len(X0)+2)) # plus fitness value and generation
Min_for_all_Generations_for_Mut_2_2 = np.empty((0,len(X0)+2))

Min_for_all_Generations_for_Mut_1_1_1 = np.empty((0,len(X0)+2))
Min_for_all_Generations_for_Mut_2_2_2 = np.empty((0,len(X0)+2))


Generation = 1 

for i in range(gen):
    
    
    new_population = np.empty((0,solution_size)) # Saving the new generation
    
    All_in_Generation_X_1 = np.empty((0,len(X0)+1))
    All_in_Generation_X_2 = np.empty((0,len(X0)+1))
    
    Min_in_Generation_X_1 = []
    Min_in_Generation_X_2 = []
    
    Save_Best_in_Generation_X = np.empty((0,len(X0)+1))
    Final_Best_in_Generation_X = []
    Worst_Best_in_Generation_X = []
    
    
    # print()
    # print("--> GENERATION: #",Generation)
    
    Family = 1
    
    for j in range(int(pop/2)): # range(int(pop/2))
        
        # print()
        # print("--> FAMILY: #",Family)
          
        
        # Tournament Selection to find Parents        
        Parent_1, Parent_2 = find_Parents(solutions_list)
    
        # Where to crossover
        Child_1, Child_2 = fit_Crossover(Parent_1,Parent_2)
            
    
        # Mutation Child 
        Mutated_Child_1 = child_Mutation(Child_1,prob_mutation)
        Mutated_Child_2 = child_Mutation(Child_2,prob_mutation)
        # Mutated_Child_1 = child_Mutation(Child_1,np.random.rand())
        # Mutated_Child_2 = child_Mutation(Child_2,np.random.rand())
        
        Total_Cost_Mut_1 = Solution_Cost(Mutated_Child_1) 
        Total_Cost_Mut_2 = Solution_Cost(Mutated_Child_2) 
        
        
        # print()
        # print("FV at Mutated Child #1 at Gen #",Generation,":", Total_Cost_Mut_1)
        # print("FV at Mutated Child #2 at Gen #",Generation,":", Total_Cost_Mut_2)
        
        
        Mutated_Child_1 = np.copy(Mutated_Child_1)
        Mutated_Child_2 = np.copy(Mutated_Child_2)
        
        
        All_in_Generation_X_1_1_temp = Mutated_Child_1[np.newaxis]
        All_in_Generation_X_1_1 = np.column_stack((Total_Cost_Mut_1, All_in_Generation_X_1_1_temp)) # cost and solution appended
        
        All_in_Generation_X_2_1_temp = Mutated_Child_2[np.newaxis]
        All_in_Generation_X_2_1 = np.column_stack((Total_Cost_Mut_2, All_in_Generation_X_2_1_temp))
        
        All_in_Generation_X_1 = np.vstack((All_in_Generation_X_1,All_in_Generation_X_1_1))
        All_in_Generation_X_2 = np.vstack((All_in_Generation_X_2,All_in_Generation_X_2_1))
        
        
        Save_Best_in_Generation_X = np.vstack((All_in_Generation_X_1,All_in_Generation_X_2))
        
        
        new_population = np.vstack((new_population,Mutated_Child_1,Mutated_Child_2)) # append in the new population
        
        t = 0
        
        R_1 = []
        for i in All_in_Generation_X_1: # each generation get the best solution
            
            if (All_in_Generation_X_1[t,:1]) <= min(All_in_Generation_X_1[:,:1]):
                R_1 = All_in_Generation_X_1[t,:]
            t = t+1
              
        Min_in_Generation_X_1 = R_1[np.newaxis]
        
        
        t = 0
        R_2 = []
        for i in All_in_Generation_X_2:
            
            if (All_in_Generation_X_2[t,:1]) <= min(All_in_Generation_X_2[:,:1]):
                R_2 = All_in_Generation_X_2[t,:]
            t = t+1
                
        Min_in_Generation_X_2 = R_2[np.newaxis]
        
        
        Family = Family+1
    
    t = 0
    R_Final = []
    
    for i in Save_Best_in_Generation_X:
        
        if (Save_Best_in_Generation_X[t,:1]) <= min(Save_Best_in_Generation_X[:,:1]):
            R_Final = Save_Best_in_Generation_X[t,:]
        t = t+1
    
    Final_Best_in_Generation_X = R_Final[np.newaxis]
    
    
    
    plotting_best = np.vstack((plotting_best,Final_Best_in_Generation_X))
    
    t = 0
    R_22_Final = []
    
    for i in Save_Best_in_Generation_X:
        
        if (Save_Best_in_Generation_X[t,:1]) >= max(Save_Best_in_Generation_X[:,:1]):
            R_22_Final = Save_Best_in_Generation_X[t,:]
        t = t+1
    
    Worst_Best_in_Generation_X = R_22_Final[np.newaxis]
    
    
    
    
    # Elitism, the best in the generation lives
    
    Darwin_Guy = Final_Best_in_Generation_X[:]
    Not_So_Darwin_Guy = Worst_Best_in_Generation_X[:]
    
    Darwin_Guy = Darwin_Guy[0:,1:].tolist()
    Not_So_Darwin_Guy = Not_So_Darwin_Guy[0:,1:].tolist()
    
    
    Best_1 = np.where((new_population == Darwin_Guy).all(axis=1))
    Worst_1 = np.where((new_population == Not_So_Darwin_Guy).all(axis=1))
    
    
    new_population[Worst_1] = Darwin_Guy
    
    
    solutions_list = new_population
    

    Min_for_all_Generations_for_Mut_1 = np.vstack((Min_for_all_Generations_for_Mut_1,Min_in_Generation_X_1))
    Min_for_all_Generations_for_Mut_2 = np.vstack((Min_for_all_Generations_for_Mut_2,Min_in_Generation_X_2))
    
    Min_for_all_Generations_for_Mut_1_1 = np.insert(Min_in_Generation_X_1, 0, Generation)
    Min_for_all_Generations_for_Mut_2_2 = np.insert(Min_in_Generation_X_2, 0, Generation)
    
    Min_for_all_Generations_for_Mut_1_1_1 = np.vstack((Min_for_all_Generations_for_Mut_1_1_1,Min_for_all_Generations_for_Mut_1_1))
    Min_for_all_Generations_for_Mut_2_2_2 = np.vstack((Min_for_all_Generations_for_Mut_2_2_2,Min_for_all_Generations_for_Mut_2_2))
    
    Generation = Generation+1
    



One_Final_Guy = np.vstack((Min_for_all_Generations_for_Mut_1_1_1,Min_for_all_Generations_for_Mut_2_2_2))
    
t = 0
Final_Here = []
for i in One_Final_Guy:
    
    if (One_Final_Guy[t,1]) <= min(One_Final_Guy[:,1]):
        Final_2 = []
        Final_2 = [One_Final_Guy[t,1]]
        Final_Here = One_Final_Guy[t,:]
    t = t+1
        
A_2_Final = min(One_Final_Guy[:,1])

One_Final_Guy_Final = Final_Here[np.newaxis]

print()
print("Min in all Generations:",One_Final_Guy_Final)

print("The Lowest Cost is:",One_Final_Guy_Final[:,1])

Look = (One_Final_Guy_Final[:,1]).tolist()
Look = float(Look[0])
Look = int(Look)

plt.plot(plotting_best[:,0])
plt.axhline(y=Look,color="r",linestyle='--')
plt.title("Cost Reached Through Generations",fontsize=20,fontweight='bold')
plt.xlabel("Generations",fontsize=18,fontweight='bold')
plt.ylabel("Cost (Flow * Solution_Cost)",fontsize=18,fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
xyz=(Generation/2, Look)
xyzz = (Generation/1.95, 1)
plt.annotate("Minimum Reached at: %s" % Look, xy=xyz, xytext=xyzz,
             arrowprops=dict(facecolor='black', shrink=0.001,width=1,headwidth=5),
             fontsize=12,fontweight='bold')
plt.show()

print()
# print("Initial Solution:",X0)
print("Final Solution:",One_Final_Guy_Final[:,2:])
print("The Lowest Cost is:",One_Final_Guy_Final[:,1])
print("At Generation:",One_Final_Guy_Final[:,0])
print()
# print("### METHODS ###")
# print("# Selection Method = Tournament Selection")
# print("# Crossover = C1 (order) but 2-point selection")
# print("# Mutation = #1- Inverse")
# print("# Other = Elitism")
# print("### METHODS ###")
# print()
# print("### VARIABLES ###")
# print("prob_crossover = %s" % prob_crossover)
# print("prob_mutation = %s" % prob_mutation)
# print("K = %s" % K)
# print("pop = %s" % pop)
# print("gen = %s" % gen)
# print("### VARIABLES ###")
            
            
            
            
            
            
            