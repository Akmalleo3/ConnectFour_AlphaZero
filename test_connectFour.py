import pytest 
from connectFour import ConnectFour
import shutil

def test_connectedSets():
    c = ConnectFour(5,5,"test1")
    c.move(0)
    c.move(4)
    c.move(4)
    c.move(3)
    shutil.rmtree("test1")

def test_case1():
    c = ConnectFour(5,5,"test2")
    c.move(0)
    c.move(3)
    c.move(4)
    c.move(0)
    shutil.rmtree("test2")

def test_case2():
    c = ConnectFour(5,5,"test3")
    c.move(2)
    c.move(4)
    c.move(1)
    c.move(0)
    shutil.rmtree("test3")

