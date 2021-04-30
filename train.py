from policyValueNet import PolicyValueNet
from mcts import play_game
from connectFour import ConnectFour
import itertools
import numpy
import torch
from torch import optim
from torch import nn

batchSize = 1024
width = 5
height = 5
def collectGameData(network, useNetwork):
    #play 5 games and then sample uniformly from the aggregated data for training
    game_images = []
    game_targets = []
    for _ in range(25):
        game = ConnectFour(width,height,True)
        images, targets = play_game(game, network, useNetwork)
        game_images.append(images)
        game_targets.append(targets)
    flattened_images = list(itertools.chain.from_iterable(game_images))
    flattened_targets = list(itertools.chain.from_iterable(game_targets))
    batchSize = min(len(flattened_images), 1024)
    sample_indices = numpy.random.choice(range(len(flattened_images)),batchSize)
    sample_images = [flattened_images[i] for i in sample_indices]
    sample_targets = [flattened_targets[i] for i in sample_indices]

    return list(zip(sample_images, sample_targets))

def train(batch,model, optimizer):
    cuda = torch.device('cuda')
    mse = nn.MSELoss()
    loss = 0
    T = 10 # number of time steps of history
    ### ??? 0 pad if the history snapshot shorter than 10??
    for (image, (target_policy, target_val)) in batch:
        (timeSteps, w,h) = image.shape
        diff = T-timeSteps
        if diff > 0:
            image = torch.cat([image, torch.zeros(diff,w,h,device=cuda)])
        elif diff < 0:
            image = image[timeSteps-T-1:timeSteps-1,:]
        #print(f"image {image}")

        image = image.unsqueeze(0)
        log_policy, value = model(image)
        target_policy = torch.tensor(target_policy, dtype=torch.float).cuda()
        target_val = torch.tensor(target_val, dtype=torch.float).cuda()
        #print(f"target p: {target_policy}")
        #print(f"target v: {target_val}")
        #print(f"policy: {log_policy}")
        #print(f"val: {value}")
        # Cross entropy is p^T log(q)
        #print(torch.log(policy.squeeze()))
        crossEntropy = torch.dot(target_policy.float(), log_policy.squeeze())
        #print(f"CE: {crossEntropy}")
        loss += (mse(value.squeeze(), target_val) -  crossEntropy).cpu()
    print(f"loss: {loss}")
    loss.backward()
    #for name, param in model.named_parameters():
    #    print(name, torch.isfinite(param.grad).all())
    optimizer.step()
    return model, optimizer

def run():
    device ='cuda'
    model = PolicyValueNet(width, height)
    optimizer = optim.SGD(model.parameters(), lr=2e-2, momentum=0.9,weight_decay=1e-4)
    model.cuda()
    batchidx = 0
    useNetwork = False
    while batchidx < 10:
        if batchidx > 3:
            useNetwork = True
        model.eval()
        print("Simulating  games")
        batch = collectGameData(model, useNetwork)
        print("Training")
        model.train()
        model,optimizer = train(batch,model,optimizer)
        #if batchidx % 10 == 0:
        print(f"Saving batch {batchidx}")
        torch.save(model.state_dict(), "baseline.pth")
        batchidx +=1

run()