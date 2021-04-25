from policyValueNet import PolicyValueNet
from mcts import play_game
from connectFour import ConnectFour
import itertools
import numpy
import torch
from torch import optim
from torch import nn

batchSize = 100
width = 5
height = 5
def collectGameData(network):
    #play 5 games and then sample uniformly from the aggregated data for training
    game_images = []
    game_targets = []
    for _ in range(10):
        game = ConnectFour(width,height,True)
        images, targets = play_game(game, network)
        game_images.append(images)
        game_targets.append(targets)
    flattened_images = list(itertools.chain.from_iterable(game_images))
    flattened_targets = list(itertools.chain.from_iterable(game_targets))
    batchSize = min(len(flattened_images), 100)
    sample_indices = numpy.random.choice(range(len(flattened_images)),batchSize)
    sample_images = [flattened_images[i] for i in sample_indices]
    sample_targets = [flattened_targets[i] for i in sample_indices]

    return list(zip(sample_images, sample_targets))

def train(batch,model, optimizer):
    mse = nn.MSELoss()
    loss = 0
    T = 10 # number of time steps of history
    ### ??? 0 pad if the history snapshot shorter than 10??
    for (image, (target_policy, target_val)) in batch:
        (timeSteps, w,h) = image.shape
        diff = T-timeSteps
        if diff > 0:
            image = torch.cat([image, torch.zeros(diff,w,h)])
        elif diff < 0:
            image = image[0:10, :,:]

        image = image.unsqueeze(0)
        policy, value = model(image)
        target_policy = torch.tensor(target_policy, dtype=torch.float)
        target_val = torch.tensor(target_val, dtype=torch.float)

        # Cross entropy is p^T log(q)
        crossEntropy = torch.dot(target_policy.float(), torch.log(policy.squeeze()))
        loss += (mse(value, target_val) -  crossEntropy)
    loss.backward()
    optimizer.step()


def run():
    model = PolicyValueNet(width, height)
    optimizer = optim.SGD(model.parameters(), lr=2e-2, momentum=0.9,weight_decay=1e-4)

    batchidx = 0
    while batchidx < 50:
        model.eval()
        batch = collectGameData(model)
        model.train()
        train(batch,model,optimizer)
        if batchidx % 10 == 0:
            torch.save(model.state_dict(), "Sunday.pth")
        batchidx +=1

run()