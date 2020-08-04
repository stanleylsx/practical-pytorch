import torch
import torch.nn as nn

sample = torch.ones(2, 2)
target = torch.tensor([[0, 1], [2, 3]])

# L1-loss
print('L1-loss:')
criterion_1 = nn.L1Loss(reduction='mean')
loss_1 = criterion_1(sample, target)
print(loss_1)
print('*************')


# SmoothL1-Loss
print('SmoothL1-Loss:')
criterion_2 = nn.SmoothL1Loss()
loss_2 = criterion_2(sample, target)
print(loss_2)
print('*************')

# MSE-loss
print('MSE-loss:')
criterion_3 = nn.MSELoss(reduction='mean')
loss_3 = criterion_3(sample, target)
print(loss_3)
print('*************')

sample = torch.tensor([[2.0, 3.0, 1.0], [1., 3., 2.2]])
target = torch.tensor([0, 2])

# CE-loss
print('CE-loss:')
criterion_4 = nn.CrossEntropyLoss(reduction='mean')
loss_4 = criterion_4(sample, target)
print(loss_4)
print('*************')

sample = torch.tensor([[2.0, 3], [1., 3]])
target = torch.tensor([[1., 0.], [0., 1.]])

# BCE-loss
print('BCE-loss:')
sigmoid_input = torch.sigmoid(sample)
criterion_5 = nn.BCELoss()
loss_5 = criterion_5(sigmoid_input, target)
print(loss_5)
print('*************')

# NLL-loss
print('NLL-loss:')
torch.manual_seed(0)
sample = torch.randn(3, 5)
target = torch.tensor([1, 0, 4])
print('sample', sample, sample.shape)
print('target', target.shape)

m = nn.LogSoftmax(dim=1)
loss = nn.NLLLoss()
output = loss(m(sample), target)
print('NLL loss', output)
# CE-loss v.s. NLL-loss
loss = nn.CrossEntropyLoss()
output = loss(sample, target)
print('CE loss', output)
print('*************')

# BCEWithLogits-loss
print('BCEWithLogits-loss:')
sigmoid = nn.Sigmoid()
torch.manual_seed(0)
sample = torch.randn(3, 2)
torch.manual_seed(3)
# target one-hot type, such as tensor([0., 1.]).
target = torch.empty(3, 2).random_(2)
sigmoid_input = sigmoid(sample)
criterion = nn.BCELoss()
print('BCE', criterion(sigmoid_input,target))

# BCE_logit-loss
criterion = nn.BCEWithLogitsLoss()
print('BCE_logit', criterion(sample, target))
print('*************')


