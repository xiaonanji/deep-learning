'''
To work with pytorch, it is always good to use data loader.
pytorch's dataloader requires a dataset to get the data from. It accepts two types of datasets: https://pytorch.org/docs/stable/data.html
Here we use map-style datasets. It is a class with two must-have methods: __getitem__ and __len__.
__len__ must return the total number of records. __getitem__ must return the data in particular index 
'''

import torch

class IMDBDataset:
    def __init__(self, reviews, targets):
        self.reviews = reviews
        self.targets = targets
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, item):
        review = self.reviews[item, :]
        target = self.targets[item]
        return {
            "review": torch.tensor(review, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float)
        }