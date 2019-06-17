import os

from gcommand_loader import GCommandLoader
import torch
dataset = GCommandLoader('./data/valid')

test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=None,
        num_workers=0, pin_memory=True, sampler=None)

print(len(test_loader.dataset))
dataset = GCommandLoader('./data/train')

test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=None,
        num_workers=0, pin_memory=True, sampler=None)
print(len(test_loader.dataset))
dataset = GCommandLoader('./data/test')

test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=None,
        num_workers=0, pin_memory=True, sampler=None)
print(len(test_loader.dataset))
