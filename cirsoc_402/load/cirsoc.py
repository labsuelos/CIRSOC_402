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

# CIRSOC 201 9.2.1
# CIRSOC 301 A.4.2
ultimate = LoadFactorDict()

idx = 1
# 1.4 D + 1.4 F
ultimate['LC'+str(idx)] = LoadFactors(D=1.4, F=1.4)

idx += 1
# 1.2 D + 1.2 F + 1.2 T + 1.6 H + 1.6 L + 0.5 Lr
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, F=1.2, T=1.2, L=1.6, H=1.6, Lr=0.5)

idx += 1
# 1.2 D + 1.2 F + 1.2 T + 1.6 H + 1.6 L + 0.5 S
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, F=1.2, T=1.2, L=1.6, H=1.6, S=0.5)

idx += 1
# 1.2 D + 1.2 F + 1.2 T + 1.6 H + 1.6 L + 0.5 R
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, F=1.2, T=1.2, L=1.6, H=1.6, R=0.5)

idx += 1
# 1.2 D + 1.6 Lr + L
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, L=1, Lr=1.6)

idx += 1
# 1.2 D + 1.6 S + L
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, L=1, S=1.6)

idx += 1
# 1.2 D + 1.6 R + L
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, L=1, R=1.6)

idx += 1
# 1.2 D + 1.6 Lr + 0.8 W
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, W=0.8, Lr=1.6)

idx += 1
# 1.2 D + 1.6 S + 0.8 W
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, W=0.8, S=1.6)

idx += 1
# 1.2 D + 1.6 R + 0.8 W
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, W=0.8, R=1.6)

idx += 1
# 1.2 D + 1.6 W + L + Lr
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, W=1.6, L=1, Lr=1)

idx += 1
# 1.2 D + 1.6 W + L + 0.5 S
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, W=1.6, L=1, S=0.5)

idx += 1
# 1.2 D + 1.6 W + L + 0.5 R
ultimate['LC'+str(idx)] = LoadFactors(D=1.2, W=1.6, L=1, R=0.5)

idx += 1
# 1.2 D + E + L + Lr + S
ultimate['LC'+str(idx)] = LoadFactors(D=1.2 ,E=1, L=1, Lr=1, S=1)

idx += 1
# 0.9 D + 1.6 W + 1.6 H
ultimate['LC'+str(idx)] = LoadFactors(D=0.9, W=1.6, H=1.6)

idx += 1
# 0.9 D + E + 1.6 H
ultimate['LC'+str(idx)] = LoadFactors(D=0.9, E=1, H=1.6)


# CIRSOC 301 A-L1
service = LoadFactorDict()

# D 
idx = 1
service['LC'+str(idx)] = LoadFactors(D=1)

# D + F + L + Lr + S + R + H
idx += 1
service['LC'+str(idx)] = LoadFactors(D=1, F=1, L=1, Lr=1, S=1, R=1, H=1)

# D + F + W
idx += 1
service['LC'+str(idx)] = LoadFactors(D=1, F=1, W=1)

# D + F + T
idx += 1
service['LC'+str(idx)] = LoadFactors(D=1, F=1, T=1)

# D + F + 0.7 L + 0.7 Lr + 0.7 S + 0.7 R + 0.7 H + 0.7 W
idx += 1
service['LC'+str(idx)] = LoadFactors(D=1, F=1, L=0.7, Lr=0.7, S=0.7, R=0.7, H=0.7, W=0.7)

# D + F + 0.7 W + 0.7 T
idx += 1
service['LC'+str(idx)] = LoadFactors(D=1, F=1, W=0.7, T=0.7)

# D + F + 0.7 L + 0.7 Lr + 0.7 S + 0.7 R + 0.7 T
idx += 1
service['LC'+str(idx)] = LoadFactors(D=1, F=1, L=0.7, Lr=0.7, S=0.7, R=0.7, T=0.7)

# D + F + 0.6 L + 0.6 Lr + 0.6 S + 0.6 R + 0.6 H + 0.6 W + 0.6 T
idx += 1
service['LC'+str(idx)] = LoadFactors(D=1, F=1, L=0.6, Lr=0.6, S=0.6, R=0.6, H=0.6, W=0.6, T=0.6)


