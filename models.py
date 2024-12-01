import torch
import torch.nn as nn
import utils

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, bias=False),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, bias=False),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            
            nn.Linear(64 * 4 * 4, 512, bias=False),
            nn.ReLU(),
            
            nn.Linear(512, 10, bias=False)
        )
        
    def forward(self, x):
        return self.model(x)
    
class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.model = nn.Sequential(

            nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            
            nn.Flatten(),
            
            nn.Linear(8192, 1024, bias=False),
            nn.ReLU(),
            nn.Linear(1024, num_classes, bias=False),
        )
        
    def forward(self, x):
        return self.model(x)

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, num_layers=1, output_size=1):
        super(LSTM, self).__init__()

        self.model = nn.Sequential(
            nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=False),
            utils.SelectLastStep(),
            nn.Linear(hidden_size, output_size, bias=False)
        )

    def forward(self, x):
        return self.model(x)