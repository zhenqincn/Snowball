from torch import nn
import torch.nn.functional as F


class CifarNet(nn.Module):
    def __init__(self, num_classes):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False), nn.ReLU())
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.globalavgpool = nn.AvgPool2d(8, 8)
        self.drop10 = nn.Dropout(0.1)
        self.drop50 = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.drop10(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = self.drop10(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.globalavgpool(x)
        x = self.drop50(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


class CNN1(nn.Module):
    def __init__(self, num_classes=10, client_id=None):
        super(CNN1, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        if client_id is not None:
            channel_dif = (5 - client_id) * 2
            channel_num = [32 + channel_dif * 1, 64 + channel_dif  * 2, 128 + channel_dif  * 4]
        else:
            channel_num = [32, 64, 128]
        
        self.conv1 = nn.Sequential(nn.Conv2d(3, channel_num[0], 3), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(channel_num[0], channel_num[1], 3), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(channel_num[1], channel_num[2], 3), nn.ReLU())
        
        self.fc = nn.Linear(channel_num[2] * 4 * 4, num_classes)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNN1_BN(nn.Module):
    def __init__(self, num_classes=10, client_id=None):
        super(CNN1_BN, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        if client_id is not None:
            channel_dif = (5 - client_id) * 2
            channel_num = [32 + channel_dif * 1, 64 + channel_dif  * 2, 128 + channel_dif  * 4]
        else:
            channel_num = [32, 64, 128]
        
        self.conv1 = nn.Sequential(nn.Conv2d(3, channel_num[0], 3), nn.BatchNorm2d(channel_num[0]), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(channel_num[0], channel_num[1], 3), nn.BatchNorm2d(channel_num[1]), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(channel_num[1], channel_num[2], 3), nn.BatchNorm2d(channel_num[2]), nn.ReLU())
        
        self.fc = nn.Linear(channel_num[2] * 4 * 4, num_classes)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNN1_BN_DROP(nn.Module):
    def __init__(self, num_classes=10, client_id=None):
        super(CNN1_BN_DROP, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        if client_id is not None:
            channel_dif = (5 - client_id) * 2
            channel_num = [32 + channel_dif * 1, 64 + channel_dif  * 2, 128 + channel_dif  * 4]
        else:
            channel_num = [32, 64, 128]
        
        self.conv1 = nn.Sequential(nn.Conv2d(3, channel_num[0], 3), nn.BatchNorm2d(channel_num[0]), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(channel_num[0], channel_num[1], 3), nn.BatchNorm2d(channel_num[1]), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(channel_num[1], channel_num[2], 3), nn.BatchNorm2d(channel_num[2]), nn.ReLU())
        
        self.fc = nn.Linear(channel_num[2] * 4 * 4, num_classes)
        
        self.drop10 = nn.Dropout(0.1)
        self.drop50 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.drop10(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.drop50(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class CNN2(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN2, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 5, padding=1, bias=False), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5, padding=1, bias=False), nn.ReLU())
        self.global_pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2304, 128, bias=False)
        self.fc2 = nn.Linear(128, num_classes, bias=False)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN3(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN3, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, 3, bias=False), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, bias=False), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, bias=False), nn.ReLU())
        self.fc1 = nn.Linear(256, 128, bias=False)
        self.fc2 = nn.Linear(128, num_classes, bias=False)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CNN2_BN(nn.Module):
    def __init__(self, num_classes=10, client_id=None):
        super(CNN2_BN, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        
        if client_id is not None:
            channel_dif = (5 - client_id) * 2 - 1
            print(channel_dif)
            channel_num = [64 + channel_dif * 1, 128 + channel_dif  * 2, 256 + channel_dif  * 4]
        else:
            channel_num = [64, 128, 256]
        
        self.conv1 = nn.Sequential(nn.Conv2d(3, channel_num[0], 3), nn.BatchNorm2d(channel_num[0]), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(channel_num[0], channel_num[1], 3), nn.BatchNorm2d(channel_num[1]), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(channel_num[1], channel_num[2], 3), nn.BatchNorm2d(channel_num[2]), nn.ReLU())
        
        self.fc = nn.Linear(channel_num[2] * 4 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    
class CNN2_BN_DROP(nn.Module):
    def __init__(self, num_classes=10, client_id=None):
        super(CNN2_BN_DROP, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        
        if client_id is not None:
            channel_dif = (5 - client_id) * 2
            channel_num = [64 + channel_dif * 1, 128 + channel_dif  * 2, 256 + channel_dif  * 4]
        else:
            channel_num = [64, 128, 256]
        
        self.conv1 = nn.Sequential(nn.Conv2d(3, channel_num[0], 3), nn.BatchNorm2d(channel_num[0]), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(channel_num[0], channel_num[1], 3), nn.BatchNorm2d(channel_num[1]), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(channel_num[1], channel_num[2], 3), nn.BatchNorm2d(channel_num[2]), nn.ReLU())
        
        self.fc = nn.Linear(channel_num[2] * 4 * 4, num_classes)
        
        self.drop10 = nn.Dropout(0.1)
        self.drop50 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.drop10(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.drop50(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x