from policyValueNet import PolicyValueNet
from mcts import play_game
from connectFour import ConnectFour
import itertools
import numpy
import torch
from torch import optim
from torch import nn

from evaluate_net import eval_network,eval_network_v_mcts, eval_network_v_random

batchSize =512
width = 7
height = 6

from torch.multiprocessing import Process, set_start_method, Queue, Barrier
try:
     set_start_method('spawn')
except RuntimeError:
    pass

def collectGameData(b, play, network, useNetwork, T, width, height, image_q, target_q):
    images, targets = play(network,useNetwork, T, width, height)
    image_q.put(images)
    target_q.put(targets)
    b.wait()

import copy
#play games and then sample uniformly from the aggregated data for training        
def collectGameDataParallel(network, useNetwork,T, width, height):
    totalGames = 0
    game_images = []
    game_targets = []
    while totalGames < 80:
        images = Queue()
        targets = Queue()
        ngames = 5
        barrier = Barrier(ngames +1)

        processes=[Process(target=collectGameData, args=(barrier,play_game, network,\
                             useNetwork, T, width,height, images, targets)) \
                                for _ in range(ngames)]
        for p in processes:
            p.start()


        for _ in range(ngames):
            im = images.get()
            game_images.append(copy.deepcopy(im))
            del im
            t = targets.get()
            game_targets.append(copy.deepcopy(t))
            del t
        barrier.wait()

        for p in processes:
            p.join()
        totalGames += ngames
    flattened_images = list(itertools.chain.from_iterable(game_images))
    flattened_targets = list(itertools.chain.from_iterable(game_targets))
    batchSize = min(len(flattened_images), 2048)
    sample_indices = numpy.random.choice(range(len(flattened_images)),batchSize)
    sample_images = [flattened_images[i] for i in sample_indices]
    sample_targets = [flattened_targets[i] for i in sample_indices]

    return sample_images, sample_targets

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


def validate(net, T, width):
    print(f"Validating against mcts player")
    eval_network_v_mcts(net)
    print(f"Validating against random player")
    eval_network_v_random(net)

def run():
    T = 1
    device ='cuda'
    model = PolicyValueNet(width, height,2*T)
    #model.load_state_dict(torch.load("testing.pth",map_location=device))

    #optimizer = optim.SGD(model.parameters(), lr=2e-2, momentum=0.9,weight_decay=1e-4)
    #optimizer = optim.SGD(model.parameters(), lr=2e-1 )
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[20,60])
    model.cuda()
    model.share_memory()
    optimizer = optim.Adagrad(model.parameters(), lr=1e-3)

    batchidx =0
    useNetwork = False
    while batchidx < 401:
        if batchidx == 200:
           useNetwork = True
        model.eval()
        print("Simulating  games")
        states, targets = collectGameDataParallel(model, useNetwork,T, width, height)
        print("Training")
        model.train()
        model,optimizer = train(states,targets,model,optimizer,T)
        if batchidx % 40 == 0 and batchidx != 0:
            validate(model, T, width)
        #scheduler.step()
        #print(f"Learning rate: {scheduler.get_last_lr()}")
        print(f"Saving batch {batchidx}")
        torch.save(model.state_dict(), "testing.pth")
        batchidx +=1

if __name__ == '__main__':
    run()
