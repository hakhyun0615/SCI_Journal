import numpy as np
import torch

# '['~', '~',,,]' -> [~,~,~,,,]
def to_list(x):
  x = x.replace('[','').replace(']','').replace('  ','').split(' ')
  return np.array(x,dtype=np.float64)


def combine_tensors(real_estate_tensor, economy_tensor, real_estate_weighted_average, how):
    if how == 'concat':
        combined_tensor = torch.cat((real_estate_tensor, economy_tensor))
    elif how == 'sum':
        combined_tensor = real_estate_weighted_average * real_estate_tensor + (1 - real_estate_weighted_average) * economy_tensor

    return combined_tensor