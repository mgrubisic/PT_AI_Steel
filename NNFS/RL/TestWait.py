import time
import sys
from waiting import wait

def isReadReady():
    print("Hei")
    return True

if __name__ == '__main__':
    print(sys.version)
    i = 0
    startT = time.time()
    while i < 200:
        i += 1
        wait(lambda: isReadReady())
        #print(time.time() - startT)
