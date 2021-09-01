'''Module with the load and resistance factors from ASCE 07

D = dead load
Di = weight of ice
E = earthquake load
F = load due to fluids with well-defined pressures and maximum heights 
Fa = flood load
H = load due to lateral earth pressure, ground water pressure, or
pressure of bulk materials
L = live load
Lr = roof live load
R = rain load
S = snow load
T = self-straining load
W = wind load
Wi = wind-on-ice
'''
from cirsoc_402.load.loadfactor import LoadFactors
from cirsoc_402.load.loadfactor import LoadFactorDict

# ASCE 07 2.3.2 and 2.3.4
ultimate = LoadFactorDict()
# 1.4 D
idx = 1
# 1.4 D
ultimate['LC'+str(idx)] = LoadFactors(D=1.4)

idx += 1
# 1.2 D + 1.6 L + 0.5 Lr
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, L=1.6, Lr=0.5)

idx += 1
# 1.2 D + 1.6 L + 0.5 S
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, L=1.6, S=0.5)

idx += 1
# 1.2 D + 1.6 L + 0.5 R
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, L=1.6, R=0.5)

idx += 1
# 1.2 D + 1.6 Lr + L
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, Lr=1.6, L=1.0)

idx += 1
# 1.2 D + 1.6 Lr + 0.5 W
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, Lr=1.6, R=0.5)

idx += 1
# 1.2 D + 1.6 S + L
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, S=1.6, L=1.0)

idx += 1
# 1.2 D + 1.6 S + 0.5 W
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, S=1.6, R=0.5)

idx += 1
# 1.2 D + 1.6 R + L
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, R=1.6, L=1.0)

idx += 1
# 1.2 D + 1.6 R + 0.5 W
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, R=1.6, W=0.5)

idx += 1
# 1.2 D + W + L + 0.5 Lr
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, W=1, L=1, Lr=0.5)

idx += 1
# 1.2 D + W + L + 0.5 S
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, W=1, L=1, S=0.5)

idx += 1
# 1.2 D + W + L + 0.5 R
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, W=1, L=1, R=0.5)

idx += 1
# 1.2 D + E + L + 0.2 S
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, E=1, L=1, S=0.2)

idx += 1
# 0.9 D + W
ultimate['LC'+str(idx)] = LoadFactors(D=0.9, W=1.0)

idx += 1
# 0.9 D + E
ultimate['LC'+str(idx)] = LoadFactors(D=0.9, E=1.0)

idx += 1
# 1.2 D + 1.6 L + 0.2 Di + 0.5 S
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, L=1.6, Di=0.2, S=0.5)

idx += 1
# 1.2 D + L + Di + Wi + 0.5 S
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, L=1, Di=1, Wi=1.0, S=0.5)

idx += 1
# 0.9 D + Wi
ultimate['LC'+str(idx)] = LoadFactors(D=0.9, Wi=1.0)


# ASCE 07 2.4.1 and 2.4.3
service = LoadFactorDict()

idx = 1
# D
service['LC'+str(idx)] = LoadFactors(D=1)

idx += 1
# D + L
service['LC'+str(idx)] = LoadFactors(D=1, L=1)

idx += 1
# D + Lr
service['LC'+str(idx)] = LoadFactors(D=1, Lr=1)

idx += 1
# D + S
service['LC'+str(idx)] = LoadFactors(D=1, S=1)

idx += 1
# D + R
service['LC'+str(idx)] = LoadFactors(D=1, R=1)

idx += 1
# D + 0.75 L + 0.75 Lr
service['LC'+str(idx)] = LoadFactors(D=1, L=0.75, Lr=0.75)

idx += 1
# D + 0.75 L + 0.75 S
service['LC'+str(idx)] = LoadFactors(D=1, L=0.75, S=0.75)

idx += 1
# D + 0.75 L + 0.75 R
service['LC'+str(idx)] = LoadFactors(D=1, L=0.75, R=0.75)

idx += 1
# D + 0.6 W
service['LC'+str(idx)] = LoadFactors(D=1, W=0.6)

idx += 1
# D + 0.7 E
service['LC'+str(idx)] = LoadFactors(D=1, E=0.7)

idx += 1
# D + 0.75 L + 0.75 0.6 W + 0.75 Lr
service['LC'+str(idx)] = LoadFactors(D=1, L=0.75, W=0.75 * 0.6, Lr=0.75)

idx += 1
# D + 0.75 L + 0.75 0.6 W + 0.75 S
service['LC'+str(idx)] = LoadFactors(D=1, L=0.75, W=0.75 * 0.6, S=0.75)

idx += 1
# D + 0.75 L + 0.75 0.6 W + 0.75 R
service['LC'+str(idx)] = LoadFactors(D=1, L=0.75, W=0.75 * 0.6, R=0.75)

idx += 1
# D + 0.75 L + 0.75 0.7 E + 0.75 S
service['LC'+str(idx)] = LoadFactors(D=1, L=0.75, E=0.75 * 0.7, S=0.75)

idx += 1
# 0.6 D + 0.6 W
service['LC'+str(idx)] = LoadFactors(D=0.6, W=0.6)

idx += 1
# 0.6 D + 0.7 E
service['LC'+str(idx)] = LoadFactors(D=0.6, E=0.7)

idx += 1
# D + L + 0.7 Di
service['LC'+str(idx)] = LoadFactors(D=1, L=1, Di=0.7)

idx += 1
# D +  0.7 Di + 0.7 Wi + S
service['LC'+str(idx)] = LoadFactors(D=1, Di=0.7, Wi=0.7, S=1)

idx += 1
# 0.6 D +  0.7 Di + 0.7 Wi
service['LC'+str(idx)] = LoadFactors(D=0.6, Di=0.7, Wi=0.7)
