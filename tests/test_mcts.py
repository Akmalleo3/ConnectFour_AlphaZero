import pytest
from connectFour import ConnectFour
from mcts import GameNode, MCTS
import shutil


def test_update_tree():
    mcts = MCTS()
    root = GameNode({},1)
    prior = 0.5
    c1 = GameNode(root, prior)
    root.children[1] = c1
    root.player = 1
    c1.player = 2

    c2 = GameNode(c1, 0.2)
    c1.children[1] = c2
    c2.player = 1

    c3 = GameNode(c2,0.8) 
    c2.children[1] = c3
    c3.player = 2

    c4 = GameNode(c3,0.3)
    c3.children[1] = c4
    c4.player = 1
    root = mcts.update_tree(c3,1.4,1)
    print("update 1")
    mcts.print_tree(root)
    print("update 2")
    root = mcts.update_tree(c4, 1.5,1)

    mcts.print_tree(root)

def test_Q_U():
    root = GameNode({},1)
    prior = 0.5
    c1 = GameNode(root, prior)
    root.children[1] = c1
    root.player = 1
    c1.player = 2
    assert c1.Q() == 0
    assert c1.U() == 0

@pytest.mark.skip
def test_run_sim():
    
    mcts = MCTS()
    game = ConnectFour(4,4, True,True, "testMCTS")
    root = mcts.run_sim(game )
    print("tree after sim")
    mcts.print_tree(root)
    shutil.rmtree("testMCTS")

