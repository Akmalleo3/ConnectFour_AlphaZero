from policyValueNet import PolicyValueNet
from mcts import play_game
from connectFour import ConnectFour
import itertools
import numpy
import torch
from torch import optim
from torch import nn

batchSize =  512
width = 5
height = 5
def collectGameData(network, useNetwork,T):
    #play games and then sample uniformly from the aggregated data for training
    game_images = []
    game_targets = []
    for _ in range(50):
        game = ConnectFour(width,height,True)
        images, targets = play_game(game, network, useNetwork,T)
        game_images.append(images)
        game_targets.append(targets)
    flattened_images = list(itertools.chain.from_iterable(game_images))
    flattened_targets = list(itertools.chain.from_iterable(game_targets))
    batchSize = min(len(flattened_images), 512)
    sample_indices = numpy.random.choice(range(len(flattened_images)),batchSize)
    sample_images = [flattened_images[i] for i in sample_indices]
    sample_targets = [flattened_targets[i] for i in sample_indices]

    return sample_images, sample_targets
    #return list(zip(sample_images, sample_targets))

def train(states,targets,model, optimizer,T):
    cuda = torch.device('cuda')
    mse = nn.MSELoss()
    states = torch.cat(states).cuda()
    target_policy = [target[0] for target in targets]
    target_val = [target[1] for target in targets]
    log_policy, value = model(states)
    target_policy = torch.tensor(target_policy, dtype=torch.float).cuda()
    target_val = torch.tensor(target_val, dtype=torch.float).cuda()

    optimizer.zero_grad()
    value_loss = mse(value, target_val)
    crossEntropy = torch.mean(torch.sum(target_policy*log_policy,1))

    policy_loss = - crossEntropy
    loss = value_loss + policy_loss
    print(f"policy loss: {policy_loss}")
    print(f"val loss: {value_loss}")
    print(f"total loss: {loss}")
    loss.backward()
    optimizer.step()
   
    return model, optimizer

def run():
    T = 4
    device ='cuda'
    model = PolicyValueNet(width, height,2*T)
    model.load_state_dict(torch.load("test_longhaul2.pth",map_location=device))

    #optimizer = optim.SGD(model.parameters(), lr=2e-2, momentum=0.9,weight_decay=1e-4)
    #optimizer = optim.SGD(model.parameters(), lr=2e-1 )
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[20,60])
    model.cuda()
    optimizer = optim.Adagrad(model.parameters())

    batchidx = 0
    useNetwork = False

    while batchidx < 50:
        model.eval()
        print("Simulating  games")
        states, targets = collectGameData(model, useNetwork,T)
        print("Training")
        model.train()
        model,optimizer = train(states,targets,model,optimizer,T)
        #scheduler.step()
        #print(f"Learning rate: {scheduler.get_last_lr()}")
        print(f"Saving batch {batchidx}")
        torch.save(model.state_dict(), "test_longhaul2.pth")
        batchidx +=1

run()