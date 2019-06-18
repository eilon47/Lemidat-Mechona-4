from torch import nn
import torch.nn.functional as F
import torch


class NNModel(nn.Module):
    def __init__(self, input_size=(100, 1, 161, 101), in1=1, out1=20, in2=20, out2=20, kernel_size1=5, kernel_size2=5, hidden_dim=1000, out_dim=30):
        super(NNModel, self).__init__()
        self.conv1 = nn.Conv2d(in1, out1, kernel_size=kernel_size1)
        hout, wout = self.compute_out_dim(input_size[2], input_size[3], kernel_size1)

        self.conv2 = nn.Conv2d(in2, out2, kernel_size=kernel_size2)
        hout2, wout2 = self.compute_out_dim(hout, wout, kernel_size2)

        self.conv2_drop = nn.Dropout2d()

        linear_in = out2 * hout2 * wout2
        hidden_dim2 = int(hidden_dim/2)
        self.fc1 = nn.Linear(linear_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, out_dim)

    def compute_out_dim(self, hin, win, kernel,dilation=1, stride=1,padding=0):
        hout = ((hin + 2 * padding - dilation * (kernel-1)) / 2)
        wout = ((win + 2 * padding - dilation * (kernel-1)) / 2)
        return int(hout), int(wout)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)

