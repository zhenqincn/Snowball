import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, hidden_size=16):
        super(GRUModel, self).__init__()
        self.conv1 = nn.GRU(input_size=300, hidden_size=hidden_size, bidirectional=True, batch_first=True, bias=False, num_layers=1)
        self.fc1 = nn.Linear(hidden_size * 2, 256, bias=False)
        self.fc2 = nn.Linear(256, 2, bias=False)

    def forward(self, embedded_text):
        output_, h_n = self.conv1(embedded_text)
        x = torch.relu(self.fc1(torch.concat((h_n[0], h_n[1]), dim=1)))
        x = self.fc2(x)
        return x