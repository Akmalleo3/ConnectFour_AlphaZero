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
    for _ in range(20):
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

    return list(zip(sample_images, sample_targets))

def train(batch,model, optimizer,T):
    cuda = torch.device('cuda')
    mse = nn.MSELoss()
    value_loss = torch.tensor(0)
    policy_loss = torch.tensor(0)
    # TODO whole batch at once    
    optimizer.zero_grad()

    for (image, (target_policy, target_val)) in batch:
        
        image = image.unsqueeze(0)
        log_policy, value = model(image)
        target_policy = torch.tensor(target_policy, dtype=torch.float).cuda()
        target_val = torch.tensor(target_val, dtype=torch.float).cuda()
        #print(f"target p: {target_policy}")
        #print(f"target v: {target_val}")
        #print(f"policy: {torch.exp(log_policy)}")
        #print(f"val: {value}")
        # Cross entropy is p^T log(q)
        #print(torch.log(policy.squeeze()))
        crossEntropy = torch.dot(target_policy.float(), log_policy)
        #print(f"CE: {crossEntropy}")
        value_loss = value_loss + mse(value.squeeze(), target_val)
        policy_loss = policy_loss - crossEntropy
    #policy_loss = policy_loss.div(len(batch))
    #value_loss = value_loss.div(len(batch))
    print(f"Avg policy loss: {policy_loss}")
    print(f"MSE loss for batch {value_loss}")
    loss = value_loss + policy_loss
    print(f"total loss: {loss}")
    loss.backward()
    optimizer.step()
    return model, optimizer

def run():
    T = 5
    device ='cuda'
    model = PolicyValueNet(width, height,2*T)
    #optimizer = optim.SGD(model.parameters(), lr=2e-2, momentum=0.9,weight_decay=1e-4)
    optimizer = optim.SGD(model.parameters(), lr=2e-1 , weight_decay=1e-4 )
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[3,5,8])
    model.cuda()
    batchidx = 0
    useNetwork = False
    #import pdb
    #pdb.set_trace()
    while batchidx < 10:
        model.eval()
        print("Simulating  games")
        batch = collectGameData(model, useNetwork,T)
        print("Training")
        model.train()
        model,optimizer = train(batch,model,optimizer,T)
        #scheduler.step()
        #print(f"Learning rate: {scheduler.get_last_lr()}")
        print(f"Saving batch {batchidx}")
        torch.save(model.state_dict(), "baseline_nonetwork.pth")
        batchidx +=1

run()