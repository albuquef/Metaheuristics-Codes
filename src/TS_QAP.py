# Tabu Search
# Local search, deterministic, tabu list
# Set M,L, sigma # N is based in sigma 
from tokenize import Token
import numpy as np
import random as rd
import pandas as pd
import itertools as itr

#  --> DYNAMIC TABU LIST <--  Legth is dynamic
### --> Short-term and long-term memories <-- ###

# Quadratic Assignment Problem (QAP):
# Given a set of facilities and locations along with the 
# flows between facilities and the distances between locations, 
# the objective of the Quadratic Assignment Problem is to assign 
# each facility to a location in such a way as to minimize the total cost.
'''
https://www.localsolver.com/docs/last/exampletour/qap.html
A distance is specified for each pair of locations,
and a flow (or weight) is specified for each pair of facilities 
(e.g. the amount of supplies transported between the pair).
'''

# Instance:
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

X0 = ["D","A","C","B","G","E","F","H"] # inicial solution

# Make a dataframe of the initial solution
New_Dist_DF = Dist.reindex(columns=X0, index=X0)
New_Dist_Arr = np.array(New_Dist_DF)

# Make a dataframe of the cost of the initial solution
Objfun1_start = pd.DataFrame(New_Dist_Arr*Flow)
Objfun1_start_Arr = np.array(Objfun1_start)
sum_start_int = sum(sum(Objfun1_start_Arr))
# print(f'inicial solution: {X0}')
# print(f'incial solution cost: {sum_start_int}') # sum of the sum of the rows

copy_inicial_solution = np.copy(X0)


###

Runs = 60 # iterations
Legth_tabu_list = 10
Tabu_List = np.empty((0,len(X0)+1))

final_solution = []
Save_Solutions_Here = np.empty((0,len(X0)+1))


for i in range(Runs):
    
    # print(f'--> iteration: {i+1}')
    
    # all combinations with swap two departaments
    list_comb = list(itr.combinations(X0,2)) 
    # print(list_comb, len(list_comb))
    
    all_solutions_comb =  np.empty((0,len(X0)))
    
    cont_comb = 0
    for comb in list_comb:
        x_swap = []
        # dept_1, dept_2 = comb #list_comb[cont_comb]
        dept_1, dept_2 = list_comb[cont_comb]
        # print(X0)
        # pos1 = X0.index(dept_1)
        # pos2 = X0.index(dept_2)
        pos1 = np.where(X0 == dept_1)
        pos2 = np.where(X0 == dept_2)
        # print(pos1,pos2)
        
        x_swap = np.copy(X0)
        x_swap[pos1], x_swap[pos2] = x_swap[pos2], x_swap[pos1]
        # print(x_swap)

        all_solutions_comb = np.vstack((all_solutions_comb,x_swap))
        cont_comb+=1

    obj_solution_current = np.empty((0,len(X0)+1))
    obj_all_solutions = np.empty((0,len(X0)+1))
    
    # cont_n = 0
    for solution in all_solutions_comb:
        
        #calculate the cost:
        New_Dist_DF = Dist.reindex(columns=solution, index=solution)
        New_Dist_Arr = np.array(New_Dist_DF)
        Objfun1_start = pd.DataFrame(New_Dist_Arr*Flow) # objective function
        Objfun1_start_Arr = np.array(Objfun1_start)
        
        Total_Cost_solution = sum(sum(Objfun1_start_Arr)) 
        solution = solution[np.newaxis] #  increase the dimension (obj, [solution])
        
        obj_solution_current = np.column_stack((Total_Cost_solution, solution))
        obj_all_solutions = np.vstack((obj_all_solutions,obj_solution_current))
        #cont_n+=1
    
    obj_all_solutions_ordered = np.array(sorted(obj_all_solutions,key=lambda x: x[0]))
    
    #check if solution already in tabu list
    
    t=0
    current_solution = obj_all_solutions_ordered[t]
    while current_solution[0] in Tabu_List[:,0]:
        current_solution = obj_all_solutions_ordered[t]
        t+=1
    
    if len(Tabu_List) >= Legth_tabu_list:
        Tabu_List = np.delete(Tabu_List, (Legth_tabu_list-1), axis=0)
     
    Tabu_List = np.vstack((current_solution,Tabu_List))
    
    Save_Solutions_Here = np.vstack((current_solution,Save_Solutions_Here)) 
    
    
    if (i+1)%10 == 0: # each 10 iterations 
        
        r1, r2, r3 =  rd.sample(range(1,len(X0)+1), 3) # np.random.randint(1,len(X0)+1)
        
        # 3-swap
        x_swap = np.copy(current_solution)
        # swap r1 to r2
        x_swap[r1], x_swap[r2] = x_swap[r2], x_swap[r1]
        # swap new r1 with r3 
        x_swap[r1], x_swap[r3] = x_swap[r3], x_swap[r1]
    
    # uptaded x0
    X0 = current_solution[1:]
    
    # Change length of Tabu List every 5 runs, between 5 and 20, dynamic Tabu list
    if (i+1)%5 == 0 or (i+1)%10 == 0:
        Legth_tabu_list = np.random.randint(5,20)
        


index_min = np.argmin(Save_Solutions_Here[:,0])
final_solution = Save_Solutions_Here[index_min,:]
final_solution = final_solution[np.newaxis]

print()
print()
print("DYNAMIC TABU LIST")
print()
print("Initial Solution:",copy_inicial_solution)
print("Initial Cost:", sum_start_int)
print()
print("Min in all Iterations:",final_solution)
print("The Lowest Cost is:",final_solution[:,0])