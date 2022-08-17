# Notations: mew (n of parents), lambda (n of children), M (n of iterations)
# a (1/5 rule)

import numpy as np
from matplotlib import pyplot as plt
import random as rd
import math as mt


import warnings
warnings.filterwarnings("ignore")

def Branin_rcos_Function(x1,x2):
    return ((x2 - ((5.1/(4*mt.pi**2))*x1**2) + ((5/mt.pi)*x1) - 6)**2
            + 10*(1-(1/(8*mt.pi)))*mt.cos(x1) + 10)
    


num_parents = 120
num_childrens = num_parents*6 # Where the ratio of Mew to Lambda is 1:6
a = 0.8
Stopping_Criteria = 30

Initial_X1 = 0
Initial_X2 = 8

Initial_Sig_1, Initial_Sig_2 = 1.25, 1.00
Value_Sig_1, Value_Sig_2 = Initial_Sig_1, Initial_Sig_2

Initial_Variables = (Initial_X1,Initial_Sig_1,
                     Initial_X2,Initial_Sig_2)



Obj_Fun = Branin_rcos_Function(Initial_X1,Initial_X2)

print(f'\nInicial solution: ({Initial_X1},{Initial_X2})')
print(f'Inicial solution value: {Obj_Fun:.4f}')


New_Pop = np.empty((0,5))

vrand_solutions = np.empty((0,5))

Best_Child_from_X = np.empty((0,5))

Final_Best_in_Generation_X = []
Final_Best = []

# create rand population (mew)
for i in range(num_parents):
    X1_Rand = np.random.uniform(-5, 10)
    X2_Rand = np.random.uniform(0, 15)
    obj_fun_current = Branin_rcos_Function(X1_Rand,X2_Rand)
    rand_sol = (obj_fun_current,X1_Rand,Initial_Sig_1,X2_Rand,Initial_Sig_2)
    vrand_solutions = np.vstack((vrand_solutions,rand_sol))

# method
See_Parents = np.empty((0,5))
Save_the_Best_Childs = np.empty((0,5))

Gen = 1  
for i in range(Stopping_Criteria):
    
    # print()
    # print('Generation at:',Gen)
    
    New_Pop = np.empty((0,5))
    
    n_list_Parents_2 = np.empty((0,5))
    n_list_Children_2 = np.empty((0,5))
    
    All_Parents = np.empty((0,5))
    All_Children = np.empty((0,5))
    
    One_Fifth_Final = 0
    
    for j in range(num_parents):
        
        One_Fifth = 0
        
        
        #------- cobination (parents) --------
        r1, r2, r3, r4 , r5 =  rd.sample(range(0,num_parents), 5)
        
        rand_p1,rand_p2,rand_p3 = vrand_solutions[r1,:],vrand_solutions[r2,:], vrand_solutions[r3,:]
        rand_p4,rand_p5 =vrand_solutions[r4,:],vrand_solutions[r5,:]
        
        temp_sol = (rand_p2[1],rand_p3[2],rand_p4[3],rand_p5[4]) # diagonal 
        temp_sol = np.array(temp_sol)
        
        rand_sol = (((rand_p1[[1]]+temp_sol[[0]])/2),((rand_p1[[2]]+temp_sol[[1]])/2),
                        ((rand_p1[[3]]+temp_sol[[2]])/2),((rand_p1[[4]]+temp_sol[[3]])/2))
        
        obj_parent = Branin_rcos_Function(rand_sol[0],rand_sol[2]) # x1,x2
        
        rand_parent = np.append(obj_parent,rand_sol)
        See_Parents = np.vstack((See_Parents,rand_parent))
        
        # tracking parents
        n_list_Parents_1 = (obj_parent,rand_parent[1],Value_Sig_1,rand_parent[3],Value_Sig_2)
        n_list_Parents_2 = np.vstack((n_list_Parents_2,n_list_Parents_1))
        
        All_in_Generation_P = np.column_stack((obj_parent,rand_parent[1],
                                               Value_Sig_1,rand_parent[3],
                                               Value_Sig_2))
        
        All_Parents = np.vstack((All_Parents,All_in_Generation_P))
        
        #------- create childrens --------
        
        Child_Num = 1
        Child_from_X = np.empty((0,5))
        # each parent create n childrens, num_parents*n / num_parents = n (n=6)
        
        for k in range(int(num_childrens/num_parents)): #int(Lambda/Mew)
            '''
            print()
            print('Child #',Child_Num)
            '''
            Sig_Rand_1 = Value_Sig_1*np.random.normal(0,1) #gauss. dist.
            Sig_Rand_2 = Value_Sig_2*np.random.normal(0,1)
            '''
            print('Dana:',Sig_Rand)
            '''
            X1_new = rand_parent[1] + Sig_Rand_1
            X2_new = rand_parent[3] + Sig_Rand_2
            
            obj_child = Branin_rcos_Function(X1_new,X2_new)
            
            if obj_child < obj_parent: # minimization problem
                One_Fifth += 1
            else:
                One_Fifth += 0
                
            All_in_Generation_C = np.column_stack((obj_child,X1_new,Sig_Rand_1,
                                                   X2_new,Sig_Rand_2))
            
            All_Children = np.vstack((All_Children,All_in_Generation_C))
            
            Child_from_X = np.vstack((Child_from_X,All_in_Generation_C))
            
            Child_Num = Child_Num+1
            
        One_Fifth_Final += One_Fifth
        
        
    index_min = np.argmin(All_Children[:,:1])
    Final_Best_in_Generation_X = All_Children[index_min]
    # print('\nFinal_Best_in_Generation_X:',Final_Best_in_Generation_X)

    Save_the_Best_Childs = np.vstack((Save_the_Best_Childs,Final_Best_in_Generation_X))
    
    # The 1/5th rule
    Final_Ratio_of_Success = One_Fifth_Final/num_childrens

    Mod_G = Gen%10
    if Mod_G == 5 or Mod_G == 0: # every 5 generations
        if Final_Ratio_of_Success > 1/5:
            Value_Sig_1, Value_Sig_2 = Value_Sig_1/a, Value_Sig_2/a
        elif Final_Ratio_of_Success < 1/5:
            Value_Sig_1, Value_Sig_2 = Value_Sig_1*a, Value_Sig_2*a

    # All the parents and all the genrated children
    New_Pop_Parents = np.array(sorted(All_Parents,key=lambda x: x[0]))
    New_Pop_Children = np.array(sorted(All_Children,key=lambda x: x[0]))

    
    # get a part of the new parents and childrens
    percent_parents = 0.1
    percent_childs = 0.9
    New_Population_Parents = New_Pop_Parents[:int(num_parents*(percent_parents)),:]
    New_Population_Children = New_Pop_Children[:int(num_parents*(percent_childs)),:]
    
    # new population
    New_Pop = np.vstack((New_Population_Parents,New_Population_Children))
    # New_Pop_New = np.vstack((New_Population_Parents,New_Population_Children))
    # shuffle the parents and childrens in the list
    np.random.shuffle(New_Pop)
    
    vrand_solutions = New_Pop

    Gen = Gen+1
    

index_min = np.argmin(Save_the_Best_Childs[:,:1]) # save the best in each generation
Best_in_All = Save_the_Best_Childs[index_min]

# Final_Best = np.concatenate((Best_in_All[[0]],Best_in_All[[1]],Best_in_All[[3]])) # (obj,x1,x2)
# Final_Best = Final_Best[np.newaxis]
Final_Best = [Best_in_All[0],Best_in_All[1],Best_in_All[3]]


print(f'\nFinal solution: ({Final_Best[1]:.4f},{Final_Best[2]:.4f})')
print(f'The best value is: {Final_Best[0]:.4f}')

# plotting
# Here = (Final_Best[:,0]).tolist()
# Here = float(Here[0])
Here = float(Final_Best[0])

plt.plot(Save_the_Best_Childs[:,0])
plt.axhline(y=Here,color='r',linestyle='--')
plt.title('The Branin rcos Function',fontsize=20,fontweight='bold')
plt.xlabel('# of Iterations',fontsize=18,fontweight='bold')
plt.ylabel('Value of f(x1,x2)',fontsize=18,fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
xyz=(Gen/2.5, Here)
xyzz = (Gen/2.4, Here+0.0025)
plt.annotate('Minimum Reached at: %.4f' % Here, xy=xyz, xytext=xyzz,
             arrowprops=dict(facecolor='black', shrink=0.001,width=1,headwidth=5),
             fontsize=12,fontweight='bold')
plt.show()