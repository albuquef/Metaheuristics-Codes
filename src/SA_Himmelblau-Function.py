import numpy as np
# import matplotlib as plt
import matplotlib.pyplot as plt

np.seterr(divide = "ignore", invalid = "ignore") # ignore the waring of zero division
import warnings
warnings.filterwarnings('ignore')

def Himmelblau_Function(x,y):
    z = ((x**2)+y-11)**2+(x+(y**2)-7)**2
    return z

def SA_probability_function(obj_possible,obj_current,T):
    return 1/(np.exp((obj_possible-obj_current)/T))


#inicial solution
x,y = 2, 1
 
# print('\n')
print('-'*30)
print(f'\nincial solution: ({x:.3f},{y:.3f})')
print(f'inicial objective value: {Himmelblau_Function(x,y):.3f}')
x_best, y_best = x,y
obj_best = Himmelblau_Function(x,y)

# hyperparameters
T0 = 1000 #inicial temperature
temp_for_plot = T0 #plotting
M = 300 #how many times will decreaese temp. 
N = 15 #how many times you want to search your neighborhood
alpha = 0.85 # decrease the temp.

k = 0.1 #helps reduce the step-size, small step side for the current solution

temp_values, obj_values = [], [] #plot temperatures and obj values

for i in range(M): # how many time decrese the temp: iterations
    for j in range(N): # neightboorhood searches 
        
        rand_1, rand_x = np.random.rand(), np.random.rand()
        step_size_x = k*rand_x if rand_1 >= 0.5 else -k*rand_x
            
        rand_2, rand_y = np.random.rand(), np.random.rand()
        step_size_y = k*rand_y if rand_2 >= 0.5 else -k*rand_y
            
        # x,y temporary: can be accept or not
        x_temp = x + step_size_x
        y_temp = y + step_size_y
        
        obj_val_temp = Himmelblau_Function(x_temp,y_temp)
        obj_val_current = Himmelblau_Function(x,y)
        
        
        rand_num = np.random.rand()
        if obj_val_temp <= obj_val_current:
            x,y = x_temp,y_temp
            obj_val_current = obj_val_temp
        # if is a worse solution accept using probability_function
        elif rand_num <= SA_probability_function(obj_val_temp,obj_val_current,T0):
            x,y = x_temp,y_temp
            obj_val_current = obj_val_temp
        
        #save best solution
        if obj_val_temp <= obj_best:
            x_best,y_best = x_temp,y_temp
            obj_best = obj_val_temp
    
    temp_values.append(T0)
    obj_values.append(obj_val_current)          
    T0 = alpha*T0 #decrese temperature


print('-'*30)
print(f'final solution: ({x:.3f},{y:.3f})')
print(f'final objective value: {Himmelblau_Function(x,y):.3f}')
print('-'*30)
print('Ploting...')
plt.plot(temp_values,obj_values)
plt.title('Z at Temperature Values', fontsize=20,fontweight='bold')
plt.xlabel('Temperature', fontsize=18,fontweight='bold')
plt.ylabel('Z', fontsize=18,fontweight='bold')

plt.xlim(temp_for_plot,0)
plt.xticks(np.arange(min(temp_values),max(temp_values),100), fontweight='bold')
plt.yticks(fontweight='bold')

plt.show()