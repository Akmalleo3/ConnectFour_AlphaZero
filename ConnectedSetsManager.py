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

    def isHorizontal(self, set):
        item = set.pop()
        set.add(item)
        return (item+1 in set) or (item-1 in set)
            
    def isVertical(self, set):
        item = set.pop()
        set.add(item)
        return (item+self.width in set) or (item-self.width in set)

    def isDiagonal(self,set):
        item = set.pop()
        set.add(item)
        return (item + self.width-1  in set) or (item+self.width+1 in set)\
        or (item - self.width -1 in set) or (item -self.width + 1 in set)

    def connected(self,tile,set):
        if len(set) == 1:
            return self.liesVertical(tile,set)\
            or self.liesHorizontal(tile,set) or self.liesDiag(tile,set)
        elif self.isHorizontal(set):
            return self.liesHorizontal(tile,set)
        elif self.isVertical(set):
            return self.liesVertical(tile,set)
        else:
            return self.liesDiag(tile,set)

    def shouldMerge(self,set1,set2):
        if self.isHorizontal(set1) and self.isHorizontal(set2):
            return True
        if self.isVertical(set1) and self.isVertical(set2):
            return True
        if self.isDiagonal(set1) and self.isDiagonal(set2):
            item = set1.pop()
            for t in set2:
                if abs(item-t)%self.width != 0:
                    return False
            return True
        return False


    def insertTile(self,tileIndex,sets, recDepth=0):
        print(f"insertTile: {sets}")
        if sets == []:
            newSet = set([tileIndex])
            sets.append(newSet)
            return sets

        newConnectedSets = []

        maybeJoin = set()
        toKeep = set()
        loner = True
        for i,s in enumerate(sets):
            if self.connected(tileIndex, s):
                s.add(tileIndex)
                loner=False
                maybeJoin.add(i)
            toKeep.add(i)

        if loner and (recDepth == 0):
            newConnectedSets.append(set([tileIndex]))
            new = [set([tileIndex])]
            # if there are more than two tiles total:
            if len(sets) > 1 or len(sets[0]) >1 :
                for i,s in enumerate(sets): 
                    for t in s:
                        new = self.insertTile(t,new,recDepth+1)
                for s in new:
                    newConnectedSets.append(s)
        

        # Which sets can be merged?
        defJoin = set()
        for i in maybeJoin:
            for j in maybeJoin:
                if i != j:
                    if self.shouldMerge(sets[i],sets[j]):
                        defJoin.add(i)
                        defJoin.add(j)
                        if i in toKeep:
                            toKeep.remove(i)
                        if j in toKeep:
                            toKeep.remove(j)
                        
        mergedSet=set()
        for i in defJoin:
            mergedSet.update(sets[i])

        if len(mergedSet):
            newConnectedSets.append(mergedSet)

        # Return list of the new sets
        for i in range(len(sets)):
            if i in toKeep:
                newConnectedSets.append(sets[i])
        
        return newConnectedSets

                


