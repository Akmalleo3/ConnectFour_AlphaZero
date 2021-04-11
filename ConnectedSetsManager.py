import math

class ConnectedSetsManager():
    def __init__(self, width, height):
        self.width = width
        self.height = height 

    def maxConnect(self, sets):
        if not len(sets):
            return 0
        sizes = map(len, sets)
        return max(sizes)

    def liesVertical(self,tile,set):
        oneBelow = tile-self.width
        oneAbove = tile+self.width
        return (oneAbove in set) or (oneBelow in set)

    def doesntWrapLeft(self,tile):
        return ((tile+1) % (self.width)) != 0
    def doesntWrapRight(self,tile):
        return (tile % (self.width)) != 0

    def liesHorizontal(self,tile,set):
        if tile == 0:
            return (1 in set)
        if  self.doesntWrapLeft(tile-1) \
            and  ((tile-1) in set): return True
        if self.doesntWrapRight(tile+1) \
            and ((tile+1) in set): return True
        return False 

    def liesDiag(self,tile,set):
        oneBelow = tile-self.width
        oneAbove = tile+self.width

        if self.doesntWrapLeft(oneBelow-1) \
            and (oneBelow - 1) in set: return True
        if self.doesntWrapRight(oneBelow+1) \
            and (oneBelow + 1) in set: return True
        if self.doesntWrapLeft(oneAbove-1) \
            and (oneAbove -1) in set: return True
        if self.doesntWrapRight(oneAbove + 1) \
            and (oneAbove + 1) in set: return True
        return False

    def liesRightDiag(self,tile,set):
        oneBelow = tile-self.width
        oneAbove = tile+self.width

        if self.doesntWrapLeft(oneBelow-1) \
            and (oneBelow - 1) in set: return True
        if self.doesntWrapRight(oneAbove + 1) \
            and (oneAbove + 1) in set: return True
        return False

    def liesLeftDiag(self,tile,set):
        oneBelow = tile-self.width
        oneAbove = tile+self.width

        if self.doesntWrapRight(oneBelow+1) \
            and (oneBelow + 1) in set: return True
        if self.doesntWrapLeft(oneAbove-1) \
            and (oneAbove -1) in set: return True

        return False

    def isHorizontal(self, set):
        item = set.pop()
        set.add(item)
        return ((item+1) in set) or ((item-1) in set)
            
    def isVertical(self, set):
        item = set.pop()
        set.add(item)
        return ((item+self.width) in set) or ((item-self.width) in set)

    def isDiagonal(self,set):
        item = set.pop()
        set.add(item)
        return ((item + self.width-1)  in set) or ((item+self.width+1) in set)\
        or ((item - self.width -1) in set) or ((item -self.width + 1) in set)

    def isRightDiagonal(self,set):
        item = set.pop()
        set.add(item)
        return  ((item+self.width+1) in set) or ((item - self.width -1) in set) 

    def isLeftDiagonal(self,set):
        item = set.pop()
        set.add(item)
        return ((item + self.width-1)  in set) or ((item -self.width + 1) in set)

    def connected(self,tile,set):
        if len(set) == 1:
            return self.liesVertical(tile,set)\
            or self.liesHorizontal(tile,set) or self.liesRightDiag(tile,set) \
                or self.liesLeftDiag(tile,set)
        elif self.isHorizontal(set):
            return self.liesHorizontal(tile,set)
        elif self.isVertical(set):
            return self.liesVertical(tile,set)
        elif self.isRightDiagonal(set):
            return self.liesRightDiag(tile,set)
        else:
            return self.liesLeftDiag(tile,set)

    def shouldMerge(self,set1,set2):
        if self.isHorizontal(set1) and self.isHorizontal(set2):
            return True
        if self.isVertical(set1) and self.isVertical(set2):
            return True
        if self.isRightDiagonal(set1) and self.isRightDiagonal(set2):
            return True
        if self.isLeftDiagonal(set1) and self.isLeftDiagonal(set2):
            return True
        return False

    def insertTile(self,tileIndex,sets):
        if sets == []:
            newSet = set([tileIndex])
            sets.append(newSet)
            return sets

        newConnectedSets = []

        maybeJoin = set()
        for i,s in enumerate(sets):
            if self.connected(tileIndex, s):
                newConnectedSets.append(s.union([tileIndex]))
                maybeJoin.add(i)
        
        for i in maybeJoin:
            for j in maybeJoin:
                if i != j:
                    if self.shouldMerge(sets[i].union([tileIndex]),\
                                        sets[j].union([tileIndex])):
                        merged = sets[i].union(sets[j])
                        merged.add(tileIndex)
                        newConnectedSets.append(merged)

        for i in range(len(sets)):
            newConnectedSets.append(sets[i])

        newConnectedSets.append(set([tileIndex]))

        return newConnectedSets

                


