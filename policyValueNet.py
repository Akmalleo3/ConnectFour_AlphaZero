
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

class PolicyValueNet(nn.Module):
    def __init__(self, boardWidth, boardHeight):
        super(PolicyValueNet,self).__init__()
        inputChannels = 10 #TODO 
        self.conv1 = nn.Conv2d(inputChannels, 32, kernel_size=2, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        #policy head
        self.conv3 = nn.Conv2d(64,128, kernel_size=3, stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.ll1 = nn.Linear(128*(boardHeight+1)*(boardWidth+1),boardWidth)

        #value head
        self.conv4 = nn.Conv2d(64,1, kernel_size=1, stride=1)
        self.ll3 = nn.Linear((boardHeight+1)*(boardWidth+1), 1)
        
    def forward(self,x):
        in_x = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = f.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = f.relu(x)
        #x = x + in_x #TODO possibly

        #policy
        p = f.relu(self.conv3(x))
        #p = self.bn3(p)
        #p = f.relu(p)
        n,c,w,h = p.shape

        p = p.view(n,-1)
        
        p = self.ll1(p)
        policy = f.log_softmax(p)
        #print(f"policy {policy}")
        
        #value 
        v = f.relu(self.conv4(x))
        n,c,w,h = v.shape
        v = v.view(n,-1)
        v = f.tanh(self.ll3(v))
        #print(f"value: {v}")
        return policy.float() ,v.float()

