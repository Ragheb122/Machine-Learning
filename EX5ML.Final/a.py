import numpy as np
arr = [["15.wav"],["5.wav"],["19.wav"],["25.wav"],["17.wav"]]
arr.sort(key=lambda x: int(x[0].split('.')[0]))
print(arr)
