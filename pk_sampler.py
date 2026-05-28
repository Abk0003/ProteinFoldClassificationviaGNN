import random
from collections import defaultdict
from torch.utils.data import Sampler

class PKSampler(Sampler):
    def __init__(self,labels,P=8,K=4,num_batches=200):
        self.labels = labels
        self.P = P
        self.K = K
        self.num_batches = num_batches
        self.fold_dict = defaultdict(list)
        for i,label in enumerate(labels):
            self.fold_dict[label].append(i)
        self.fold_dict = {fold: indices for fold, indices in self.fold_dict.items()
                          if len(indices) >= K}
        self.folds = list(self.fold_dict.keys())

    def __iter__(self):
        for i in range(self.num_batches):
            batch = []
            x = random.sample(self.folds,self.P)
            for fold in x:
                y = random.sample(self.fold_dict[fold],self.K)
                batch.extend(y)
            yield batch

if __name__ == "__main__":
    labels = [i // 15 for i in range(300)]
    sampler = PKSampler(labels, P=4, K=4, num_batches=10)
    for batch in sampler:
        print(f"batch size: {len(batch)}")   # should be 16
        print(f"indices: {batch[:8]}")
        break









