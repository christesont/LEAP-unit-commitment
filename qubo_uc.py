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
# N= int(input('Select number of bins for discretization: '))
N=5
# Pdiff = [a - b for a,b in zip(Pmax,Pmin)]
# print(Pdiff)

from time import time
start = time()
h = [ (Pmax[i] - Pmin[i])/N for i in range(len(b))]
z = pyqubo.Array.create('z',shape=(len(b),N+1), vartype='BINARY')
v = [ (1- sum(z[i])) for i in range(len(b))] #whether unit i is online or not
p = [[ (Pmin[i] + j*h[i])*z[i][j] for j in range(N+1)] for i in range(len(b))]
psteps = [[ Pmin[i] + j*h[i] for j in range(N+1)] for i in range(len(b))]
# #Penalty Strengths
A = 50000
B = 25
C = 100000
#
# for i in Load:
#     L = i
objective = []
objective.append( sum([ b[i]*sum(p[i])  for i in range(len(b))]) )
objective.append( sum([a[i]*(1 - v[i]) for i in range(len(b))]) )
objective.append( sum([c[i]*sum(p[i])**2 for i in range(len(b))]) )

penalties=[]

penalties.append( A*sum([ (sum(z[i]) -1) for i in range(len(b))]) )
penalties.append( 0.8*A*sum([ (sum(z[i]) -1)**2 for i in range(len(b))]) )
penalties.append( B*( sum([ sum(p[i]) for i in range(len(b))]) - L)**2 )
###Does this constraint work?
# penalties.append( A*sum([ (v[i] + sum(z[i]) -1)**2 for i in range(len(b))]) )

#Would it make sense to add a conditional constraint? If sum(p[i]) exceeds Pmax[i], add a large penalty
# penalties.append(10*A*[sum(p[i])**2 if 1.00001*sum(p[i]) > Pmax[i]*sum(z[i])**2 else 0.001*sum(p[i]) for i in range(len(b))])
# penalties.append( B*sum( [(Pmax[i] - sum(p[i]))**2 for i in range(len(b))]) )

#get some more common penalties? selectively apply the common constraints to the appropriate units
#COMMON CONSTRAINTS
#
model1 = sum(objective) + sum(penalties)

model = model1.compile()
bqm = model.to_bqm()

# sa = neal.SimulatedAnnealingSampler()
# solver_greedy = greedy.SteepestDescentSolver()
#
# sampleset = sa.sample(bqm, num_reads=2000)
# sampleset = solver_greedy.sample(bqm, intitial_states=sampleset,num_reads=10000)
sa = greedy.SteepestDescentComposite(neal.SimulatedAnnealingSampler())
sampleset = sa.sample(bqm,num_reads=10000)
# print("method 1")
# decoded_samples = model.decode_sampleset(sampleset)
# best_sample = min(decoded_samples, key = lambda x: x.energy)
# values= [k for k,v in best_sample.sample.items() if v==1]
# values.sort()
# print(values)
# print("method 2")
sampleset = sampleset.aggregate().lowest()
decoded_samples = model.decode_sampleset(sampleset)
values= [k for k,v in decoded_samples[0].sample.items() if v==1]
values.sort()
#print(best_sample.sample)
end = time()


print(values)
# [print(p[i]) for i in range(len(p))]
pval_from_z = [i.replace("z","").replace("["," ").replace("]","").split() for i in values]
pval_from_z2 = [str(p[int(k)][int(v)]) for k,v in pval_from_z]
regex = re.compile(r'(\d+)')
pval_final = [float('.'.join([i for i in regex.findall(j)[2:]])) for j in pval_from_z2]

[print(f"p[{int(k)}][{int(v)}] = {psteps[int(k)][int(v)]}") for k,v in pval_from_z]
print(f"\nTotal production: {sum(pval_final)} / {L}")
print(f"Number of units: {len(pval_final)} / {len(p)}")
print(f"Min energy = {decoded_samples[0].energy}")
# print(f'A = {A}, B = {B}, C = {C}')
[print(f'p[{int(k)}] =  {psteps[int(k)]}') for k,v in pval_from_z]
diff = end - start
print(f"{diff} seconds")
