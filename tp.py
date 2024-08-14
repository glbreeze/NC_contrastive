import torch
mean0 = 0.0
mean2 = 5.0
std = 1.0  # Standard deviation

label = torch.tensor([0, 2, 0, 2])

input = torch.concatenate((torch.normal(mean0, std, (1, 3)),
                           torch.normal(mean2, std, (1, 3)),
                           torch.normal(mean0, std, (1, 3)),
                           torch.normal(mean2, std, (1, 3)),
), dim=0)


sum_ = torch.zeros((torch.unique(label).size(0), 3))   # [B, C, H, W], sum over Batch
sum_ = torch.zeros((3, 3))   # [B, C, H, W], sum over Batch
sum_.index_add_(dim=0, index=label, source=input)    # [K, C, H, W]
cnt_ = torch.bincount(label)
avg_feat = sum_/cnt_[:, None]        # [K, C, H, W]  class-wise mean feat
            mean = avg_feat.mean(dim=(0, 2, 3), keepdim=True)  # channel mean (equal weight for all classes)

matrix = np.array([[1.000999, 2], [3, 4]])

# Flatten the matrix and print the values in one row
flattened_matrix = matrix.flatten()
print(' '.join(map(str, flattened_matrix)))

K = 100
S = 5
Z = torch.concatenate((
    torch.tensor([1.0]),
    torch.tensor([-1/(K-1)] * (K-1))
)).view(1, -1)

S = 10
torch.softmax(Z*S, dim=1)

import numpy as np

# Sample numpy array and mapping dictionary
my_array = np.array([1, 2, 3, 4, 5])
mapping_dict = {1: 2, 2: 4, 3: 6, 4: 8, 5: 10}

# Create a vectorized mapping function
vectorized_map = np.vectorize(mapping_dict.get)

# Map elements using the vectorized function
mapped_array = vectorized_map(my_array)
print(mapped_array)
