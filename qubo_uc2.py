import pyqubo
import neal
import numpy as np
import re
import greedy
#Double check all functions and variables
#Why isn't the program favoring unit 1?


# a = [int(item) for item in [item for item in input("Input binary cost coefficients for power units:  ").replace("["," ").replace("]"," ").replace(","," ").split()]]
# b = [int(item) for item in [item for item in input("Input linear cost coefficients for power units:  ").replace("["," ").replace("]"," ").replace(","," ").split()]]
# c = [int(item) for item in [item for item in input("Input quadratic cost coefficients for power units:  ").replace("["," ").replace("]"," ").replace(","," ").split()]]
# Pmax = [int(item) for item in [item for item in input("Input max production output for units:  ").replace("["," ").replace("]"," ").replace(","," ").split()]]
# Pmin = [int(item) for item in [item for item in input("Input min production output for units:  ").replace("["," ").replace("]"," ").replace(","," ").split()]]
# L = int(input("Input load demand (power consumption):  "))
# N = int(input("N bins for discretization... N=  "))
#
# #raise error if not all parameters have the same length
# params = [a,b,c,Pmax,Pmin]
# it = iter(params)
# the_len = len(next(it))
# if not all(len(l) == the_len for l in it):
#     print("\n")
#     raise ValueError('All parameters must have the same length')

#Load and Answer from table III
a= [1000, 970, 700, 680, 450, 370, 480, 660, 665, 670]
b = [16.19, 17.26, 16.60, 16.50, 19.70, 22.26, 27.74, 25.92, 27.27, 27.79]
c = [0.00048, 0.00031, 0.002, 0.00211, 0.00398, 0.00712, 0.0079, 0.00413, 0.00222, 0.00173]
Pmax = [455, 455, 130, 130, 162, 80, 85, 55, 55, 55]
Pmin = [150, 150, 20, 20, 25, 20, 25, 10, 10, 10]
L=700 #hour 1
# hour = int(input('Select Hour for Load: '))
# Load = [700,750,850,950,1000,1100,1150,1200,1300,1400,1450,1500,1400,1300,1200,1050,1000,1100,1200,1400,1300,1100,900,800]
# L = Load[hour-1]
# n= int(input('Select log_2 of number of bins for discretization: '))
n=3
N= 2**n
print(f"Number of bins, N = {N}")
# N=5
# Pdiff = [a - b for a,b in zip(Pmax,Pmin)]
# print(Pdiff)

from time import time
start = time()
h = [ (Pmax[i] - Pmin[i])/(N-1) for i in range(len(b))]
z = pyqubo.Array.create('z',shape=(len(b),n+1), vartype='BINARY')
v = [ z[i][0] for i in range(len(b))] #whether unit i is online or not
p = [[0 for j in range(n+1)] for i in range(len(b))]
psteps= [[0 for j in range(n+1)] for i in range(len(b))]
for i in range(len(b)):
    for j in range(n+1):
        if j ==0:
            p[i][j] = Pmin[i]*v[i]
            psteps[i][j] = Pmin[i]
            # print(i,j)
        else:
            p[i][j] = (2**(j-1))*h[i]*z[i][j]
            psteps[i][j] = (2**(j-1))*h[i]
            # print(i,j)
# p = [[ (Pmin[i] + (2**j)*h[i]*z[i][j]) for j in range(n)] for i in range(len(b))]
# psteps = [[ (Pmin[i] + (2**j)*h[i]) for j in range(n)] for i in range(len(b))]
# #Penalty Strengths
A = 50000
B = 20

#
# for i in Load:
#     L = i
objective = []
objective.append( sum([ b[i]*sum(p[i])  for i in range(len(b))]) )
objective.append( sum([a[i]* v[i] for i in range(len(b))]) )
objective.append( sum([c[i]*sum(p[i])**2 for i in range(len(b))]) )

penalties=[]


penalties.append( B*( sum([ sum(p[i]) for i in range(len(b))]) - L)**2 )
penalties.append( A*( sum([ (sum([z[i][j] for j in range(1,len(z[i]))]) - n*v[i]) for i in range(len(b)) ]))) #modify inequality constraint
# # penalties.append( A*sum([ (v[i] -1) for i in range(len(b))]) )
penalties.append( 0.5*A*( sum(v)**2 ))



model1 = sum(objective) + sum(penalties)

model = model1.compile()
bqm = model.to_bqm()

sa = neal.SimulatedAnnealingSampler()
solver_greedy = greedy.SteepestDescentSolver()

sampleset = sa.sample(bqm, num_reads=5000)

sampleset = solver_greedy.sample(bqm, intitial_states=sampleset,num_reads=100000)
# sa = greedy.SteepestDescentComposite(neal.SimulatedAnnealingSampler())
# sampleset = sa.sample(bqm,num_reads=10000)
# print("method 1")
# decoded_samples = model.decode_sampleset(sampleset)
# best_sample = min(decoded_samples, key = lambda x: x.energy)
# values= [k for k,v in best_sample.sample.items() if v==1]
# values.sort()

# print("method 2")
sampleset = sampleset.aggregate().lowest()
decoded_samples = model.decode_sampleset(sampleset)
values= [k for k,v in decoded_samples[0].sample.items() if v==1]
values.sort()
#print(best_sample.sample)
end = time()


print(values)
pval_from_z = [i.replace("z","").replace("["," ").replace("]","").split() for i in values]
kset = set()
[kset.add(int(k)) for k,v in pval_from_z]
for k in kset:
    ksum= sum([psteps[int(kval)][int(v)] for kval,v in pval_from_z if int(kval) ==k])
    print(f'p[{k}] = {ksum}')

# pval_from_z2 = [str(psteps[int(k)][int(v)]) for k,v in pval_from_z]
# regex = re.compile(r'(\d+)')
# pval_final = [float('.'.join([i for i in regex.findall(j)[2:]])) for j in pval_from_z2]

#[print(f"p[{int(k)}][{int(v)}] = {psteps[int(k)][int(v)]}") for k,v in pval_from_z]
TotalSum = sum([psteps[int(k)][int(v)] for k,v in pval_from_z])
print(f"\nTotal production: {TotalSum} / {L}")

print(f"Number of units: {len(kset)} / {len(p)}")
print(f"Min energy = {decoded_samples[0].energy}")
# # print(f'A = {A}, B = {B}, C = {C}')
[print(f'p[{int(k)}] =  {psteps[int(k)]}') for k in kset]
diff = end - start
print(f"{diff} seconds")
