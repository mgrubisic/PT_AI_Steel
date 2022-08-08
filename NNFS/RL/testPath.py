import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from NNFS.RL.steelProfilesRL import getProfileList

print(sys.path)
profiler = getProfileList(IPE=True, HEA=True, HEB=True, HEM=True, KVHUP=True, REKHUP=True)
profiler.sort(key=lambda x: x.getEPD())
sL = pickle.load(open("stateList.p", "rb"))
aL = pickle.load(open("actionList.p", "rb"))
'''
plt.plot(aL, 'ro')
plt.xlabel([x.name for x in profiler])
plt.show()
print(aL)
#print(max(aL))
print(aL[9])
print("Profil: " + str(profiler[9]))
print([x.name for x in profiler])
'''
SL = sL[0][6000:]
telleListe = np.zeros([359])
for numb in SL:
    telleListe[int(numb)] += 1
print(telleListe)
print(max(telleListe))