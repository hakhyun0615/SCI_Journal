import numpy as np

# '['~', '~',,,]' -> [~,~,~,,,]
def to_list(x):
  x = x.replace('[','').replace(']','').replace('  ','').split(' ')
  return np.array(x,dtype=np.float64)