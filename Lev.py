import numpy as np
from tqdm import tqdm
import torch
import math
import time

class LevSelector:
    def __init__(self, ratio_list, batch_size, step_size, device_id, lmbda=1e-5):
        """
        ratio_list: a list that contains different selection ratio in [0, 100]
        batch_size: the current data pool size to compute leverage scores
        step_size: the number of selection at each step
        """
        self.ratio_list = sorted(ratio_list)
        self.batch_size = batch_size
        self.step_size = step_size
        self.top_k_list=[math.ceil(math.ceil(a/100*self.batch_size)/step_size) for a in self.ratio_list]
        self.device_id = device_id
        self.lmbda = lmbda

    def gram_update(self, kernel_matrix, data):
        if kernel_matrix is None:
            return torch.matmul(data.T, data)
        else:
            return kernel_matrix + torch.matmul(data.T, data)

    def hunger_select(self, feature):

        rng = np.random.default_rng(seed=0)
        top_k = self.top_k_list[-1]
        batch_size = self.batch_size
        step_size = self.step_size
        gpu="cuda:"+str(self.device_id)
        feature = torch.tensor(feature).to(gpu)

        results=[] # selected_indexs for different selection ratios
        ckpt=0

        selected_indexs=[]
        kernel_matrix=None
        constant = self.lmbda * torch.eye(feature.shape[1]).to(gpu)

        for i in range(top_k):
            if i==0:
                new_index = rng.choice(batch_size, size=step_size, replace=False).astype(int).astype(object).tolist()
                selected_indexs += new_index
                kernel_matrix = self.gram_update(None, feature[new_index])
            else:
                sampled_n = i * step_size
                Sigma = 1 / sampled_n * kernel_matrix + constant # Sigma: (d, d)
                L = torch.linalg.cholesky(Sigma)       # L: (d, d), lower=True by default
                XT = feature.T             # (d, n)
                Z = torch.linalg.solve_triangular(L, XT, upper=False)
                score_list = torch.sum(Z*Z, dim=0)

                sort_ind = torch.argsort(-score_list)
                new_index = []
                for x in sort_ind:
                    if x.item() in selected_indexs:
                        continue
                    new_index.append(x.item())
                    if len(new_index)==step_size:
                        break

                selected_indexs += new_index
                kernel_matrix = self.gram_update(kernel_matrix, feature[new_index])
            
            if i==self.top_k_list[ckpt]-1:
                results.append(selected_indexs[:math.ceil(self.ratio_list[ckpt]/100*self.batch_size)].copy())
                ckpt+=1

        return results

