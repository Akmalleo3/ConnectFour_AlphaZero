from ConnectedSetsManager import ConnectedSetsManager
import pytest

@pytest.fixture
def CSM():
    CSM = ConnectedSetsManager(5,5)
    return CSM

def test_empty(CSM):
    assert CSM.maxConnect([]) == 0

def test_type_checking(CSM):
    cs = []
    set1 = set([4,8,12])
    set2 = set([0,5,10])
    set3 = set([10,11,12])
    cs.append(set1)
    cs.append(set2)
    cs.append(set3)

    return cs
    assert not CSM.isHorizontal(set1)
    assert not CSM.isVertical(set1)
    assert CSM.isDiagonal(set1)

    assert not CSM.isHorizontal(set2)
    assert  CSM.isVertical(set2)
    assert not CSM.isDiagonal(set2)

    assert CSM.isHorizontal(set3)
    assert not CSM.isVertical(set3)
    assert not CSM.isDiagonal(set3)

def test_connected(CSM):
    cs = []
    set1 = set([4,8,12])
    set2 = set([0,5,10])
    set3 = set([10,11,12])
    cs.append(set1)
    cs.append(set2)
    cs.append(set3)

    return cs
    assert CSM.liesVertical(15,set2)
    assert CSM.connected(15,set2)
    assert not CSM.connected(15,set3)
    assert not CSM.connected(15,set1)

    assert CSM.liesHorizontal(13,set3)
    assert CSM.connected(13, set3)

    assert CSM.liesDiag(16,set1)
    assert CSM.connected(16,set1)

def test_insert_tile(CSM):
    cs = []
    set1 = set([4,8,12])
    set2 = set([0,5,10])
    set3 = set([10,11,12])
    cs.append(set1)
    cs.append(set2)
    cs.append(set3)

    csprime = CSM.insertTile(16,cs)
    set3.add(16)
    assert set3 in csprime

def test_diag(CSM):
    cs = set([4,8,12])
    assert CSM.liesDiag(16,cs)
    cs = set([4])

    assert not CSM.doesntWrapLeft(4)
    assert not CSM.liesDiag(0,cs)
    assert not CSM.liesHorizontal(0,cs)
    assert not CSM.liesVertical(0,cs)

    cs = set([0,6,12,18])
    assert CSM.isRightDiagonal(cs)

def test_merge_sets(CSM):
    cs = []
    set1 = set([0,1])
    set2 = set([3,4])
    set3 = set([13,14])
    cs.append(set1)
    cs.append(set2)
    cs.append(set3)
    csprime = CSM.insertTile(2,cs)
    fullset = set([0,1,2,3,4])
    assert fullset in csprime

    csprime = CSM.insertTile(6,csprime)
    fullset.add(6)
    assert fullset not in csprime

def test_left_diag(CSM):
    a = set([3,7])
    b = set([11,7])
    assert CSM.isLeftDiagonal(a)
    assert CSM.isLeftDiagonal(b)
    assert CSM.shouldMerge(a,b)