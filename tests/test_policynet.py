import pytest
import policyValueNet
import numpy as np
import torch
from mcts import play_game
from connectFour import ConnectFour, historyToImage
import shutil

@pytest.mark.skip
def testforward():
    width = 5
    height = 5
    net = policyValueNet.PolicyValueNet(width,height)

    print("Start game")
    game = ConnectFour(5,5,True, True, "testMCTS")
    play_game(game,net,False,1)
    print("end game")
    shutil.rmtree("testMCTS")


    net_in = historyToImage(game.history, width, height,5)
    net_in = net_in.cpu()
    #what if I just took 10 channels 
    onlyTen = net_in[0:10,:,:]
    # start with batchsize of 1 
    #net_in = torch.unsqueeze(onlyTen, 0)
    print(net_in)
    #import pdb
    #pdb.set_trace()
    out = net(net_in)
    print(out)
    device = torch.device('cuda')
    net.load_state_dict(torch.load("test.pth",map_location=device))
  
    net.eval()
    
    x = net(net_in)

