import copy
from functools import cmp_to_key
from types import FunctionType
from typing import Mapping, MutableSequence, Sequence

from config import ABSOLUTE_MAXIMUM_INSTANT_ACCELERATION, ABSOLUTE_MAXIMUM_INSTANT_DISTANCE, ABSOLUTE_MAXIMUM_INSTANT_SPEED
from personnage import PersonnageData
from utils import vectorDelta, vectorNorme


class PersonnageDistance:

    def __init__(self, p1: PersonnageData, p2: PersonnageData):
        """Asymetric distance: from new personnage (p1) to historical personnage (p2)."""
        self.p1: PersonnageData = p1
        self.p2: PersonnageData = p2
        self.frame: int = p1.frame - p2.frame
        #self.p1SmoothedCoordinate = smoothCartesianCoordinate([p1.coordinate] + p2.coordinatesHistory, 0, p1.smoothingWindow)
        #self.p2SmoothedCoordinate = p2.smoothedCoordinatesHistory[0]
        self.instantDistanceVector = vectorDelta(p1.coordinate, p2.smoothedCoordinatesHistory[0], 1)
        #self.smoothedDistanceVector = vectorDelta(self.p1SmoothedCoordinate, self.p2SmoothedCoordinate, 1)
        self.history = abs(len(p1.frameHistory) - len(p2.frameHistory))
        self.color: float = None

        # Non symetric distance history: only distance from p1 to p2 history
        self.instantSpeedVector = None
        #self.smoothedSpeedVector = None
        self.instantAccelerationVector = None
        #self.smoothedAccelerationVector = None

        if self.frame > 0:
            self.instantSpeedVector = vectorDelta(p1.coordinate, p2.smoothedCoordinatesHistory[0], self.frame)
            #self.smoothedSpeedVector = vectorDelta(self.p1SmoothedCoordinate, self.p2SmoothedCoordinate, self.frame)
            #self.distanceVectorHistory = [ vectorDelta(p1.coordinate, p2Coordinate, self.frame) for p2Coordinate in p2.smoothedCoordinatesHistory[:historySize] ]
        if self.instantSpeedVector is not None and len(p2.smoothedSpeedHistory) > 0:
            self.instantAccelerationVector = vectorDelta(self.instantSpeedVector, p2.smoothedSpeedHistory[0], self.frame)
            #self.smoothedAccelerationVector = vectorDelta(self.smoothedSpeedVector, p2.smoothedSpeedHistory[0], self.frame)
            #self.speedVectorHistory = [ vectorDelta(p1HypotheticSpeed, p2Speed, 1) for p2Speed in p2.smoothedSpeedHistory[:historySize] ]

        #self.accelerationVectorHistory = [ vectorDelta(p1.coordinate, p2Accel, 1) for p2Accel in p2.smoothedAccelerationHistory[:historySize] ]

    def __str__(self) -> str:
        return f"[Distance vector={self.instantDistanceVector} norme={self.cartesianDistance()}]"

    def isPossible(self) -> bool:
        if self.frame == 0:
            return False

        # A distance is possible if distance and speed are maxed + if acceleration is maxed
        # isPossible = abs(vectorNorme(self.smoothedSpeedVector)) < ABSOLUTE_MAXIMUM_SMOOTHED_SPEED and abs(vectorNorme(self.smoothedDistanceVector)) < ABSOLUTE_MAXIMUM_SMOOTHED_DISTANCE
        # if self.smoothedAccelerationVector is not None:
        #     return isPossible and abs(vectorNorme(self.smoothedAccelerationVector)) < ABSOLUTE_MAXIMUM_SMOOTHED_ACCELERATION

        isPossible = abs(vectorNorme(self.instantSpeedVector)) < ABSOLUTE_MAXIMUM_INSTANT_SPEED and abs(vectorNorme(self.instantDistanceVector)) < ABSOLUTE_MAXIMUM_INSTANT_DISTANCE
        if self.instantAccelerationVector is not None:
            return isPossible and abs(vectorNorme(self.instantAccelerationVector)) < ABSOLUTE_MAXIMUM_INSTANT_ACCELERATION
        return isPossible

    def cartesianDistance(self) -> float:
        return vectorNorme(self.instantDistanceVector)

    def speedDistance(self) -> float:
        ancestorSpeedVector = (0, 0, 0)
        if len(self.p2.smoothedSpeedHistory) > 1 and self.p2.smoothedSpeedHistory[0] is not None:
             ancestorSpeedVector = self.p2.smoothedSpeedHistory[0]
        return vectorNorme(vectorDelta(self.instantSpeedVector, ancestorSpeedVector, self.frame))

    def accelerationDistance(self) -> float:
        ancestorAccelerationVector = (0, 0, 0)
        if len(self.p2.smoothedAccelerationHistory) > 1 and self.p2.smoothedAccelerationHistory[0] is not None:
             ancestorAccelerationVector = self.p2.smoothedAccelerationHistory[0]
        return vectorNorme(vectorDelta(self.instantAccelerationVector, ancestorAccelerationVector, self.frame))

    # Deprecated
    def historicalCartesianDistance(self) -> float:
        historySize = len(self.instantDistanceVector)
        cartesianDistSum = 0
        weightSum = 0
        for k in range(len(self.distanceVectorHistory)):
            weight = -2*k/((historySize + 1)**2) + 2/(historySize + 1)
            weightSum += weight
            cartesianDistance = vectorNorme(self.distanceVectorHistory[k])
            cartesianDistSum += cartesianDistance * weight

        #print("DEBUG: historySize=%d weightSum=%f historical cartesian dist=%f" % (historySize, weightSum, cartesianDistSum))
        if weightSum == 0:
            return cartesianDistSum
        return cartesianDistSum / weightSum

class DistanceMatrix:

    def __init__(self, rows: Sequence[PersonnageData], columns: Sequence[PersonnageData], distanceScorer: FunctionType):
        self.__matrix: Mapping[int, Mapping[int, PersonnageDistance]] = {}
        self.__allDistances: MutableSequence[PersonnageDistance] = []
        self.__sortedDistances = False

        freshnessSumPa = 1
        # for pA in rows:
        #     freshnessSumPa += pA.freshness
        self.rowsFreshnessSum = freshnessSumPa

        freshnessSumPb = 1
        # for pB in columns:
        #     freshnessSumPb += pB.freshness
        self.columnsFreshnessSum = freshnessSumPb

        for pA in rows:
            self.__matrix[id(pA)] = {}
            for pB in columns:
                distance = PersonnageDistance(pA, pB)
                self.__matrix[id(pA)][id(pB)] = distance
                if distance.isPossible():
                    self.__allDistances.append(distance)

        def distanceScoreComparator(dist1: PersonnageDistance, dist2: PersonnageDistance):
            return distanceScorer(dist1, self) - distanceScorer(dist2, self)

        self.__distanceComparator = distanceScoreComparator


    def clone(self):
        clone = copy.copy(self) # shallow copy
        clone.__allDistances = self.__allDistances[:] # Copy into a new list
        clone.__matrix = {} # Init a new dict
        for (key, row) in self.__matrix.items():
            clone.__matrix[key] = dict(self.__matrix[key]) # Copy into a new dict
        clone.__sortedDistances = None
        return clone

    def getDistance(self, column: PersonnageData, row: PersonnageData) -> PersonnageDistance:
        distance = self.__matrix[id(column)][id(row)]
        return distance

    def getSortedDistances(self) -> Sequence[PersonnageDistance]:
        if not self.__sortedDistances:
            self.__allDistances.sort(key=cmp_to_key(self.__distanceComparator))
            self.__sortedDistances = True
            #print("DEBUG: Sorted ditances: [%s]" % ' '.join([ str(x) for x in self.__allDistances ]))
        return self.__allDistances

    def getNthMinimalDistance(self, nth: int = 0) -> PersonnageDistance: 
        """Return nth minimal distance in matrix."""
        sortedDistances = self.getSortedDistances()
        #print("DEBUG: Getting ditance #%d from: [%s]" % (nth, ' '.join([ str(x) for x in self.__allDistances ])))
        if nth < len(sortedDistances):
            return sortedDistances[nth]
        else:
            return None

    def _reduceMatrix(self, row: PersonnageData, column: PersonnageData):
        """Build a new matrix without supplied column and row """
        #print("DEBUG: reducing matrix containing %d distance(s) in %d row(s) ..." % (len(self.__allDistances), len(self.__matrix)))
        clone = self.clone()

        for rowKey in list(clone.__matrix.keys()):
            for colKey in list(clone.__matrix[rowKey].keys()):
                dist = clone.__matrix[rowKey][colKey]
                if dist.p1 == row or dist.p2 == column:
                    if dist in clone.__allDistances:
                        clone.__allDistances.remove(dist)
                    del clone.__matrix[rowKey][colKey]

        del clone.__matrix[id(row)]

        #print("DEBUG: reduced matrix contains %d distance(s) in %d row(s) ..." % (len(clone.__allDistances), len(clone.__matrix)))
        if len(clone.__allDistances) == 0:
            return None
        return clone

    def reduceMatrix(self, distance: PersonnageDistance):
        return self._reduceMatrix(distance.p1, distance.p2)


class DistancePathIterator:
    """ Iterate over a DistanceMatrix returning possibles distance pathes """

    # Path schemes: take all first minimal distances
    # All pathes for a 3x3 matrix are :
    # 0 0 0
    # 1 0 0
    # ...
    # 8 0 0
    # 0 1 0
    # 1 1 0
    # ...
    # 8 1 0
    # ...
    # 8 3 0

    def __init__(self, matrix: DistanceMatrix) -> None:
        self.matrix = matrix
        self.currentPathScheme = [0]

    def __iter__(self):
        return self

    def __next__(self):
        distancePath = []
        currentDistanceMatrix = self.matrix
        currentPathIndex = 0

        #print("DEBUG: currentPathIndex=%d currentPathScheme=%s." % (currentPathIndex, str(self.currentPathScheme)))

        # First iteration
        while True:
            n = self.currentPathScheme[currentPathIndex]
            dist = currentDistanceMatrix.getNthMinimalDistance(n)
            if n == 0 and dist is None:
                # Empty matrix
                #print("DEBUG: Empty Matrix.")
                raise StopIteration
            if dist is None:
                # Done exploring currentPathIndex
                if len(self.currentPathScheme) > currentPathIndex + 1:
                    # We can propagate the carry
                    #print("DEBUG: Carry propagation.")
                    self.currentPathScheme[currentPathIndex] = 0
                    self.currentPathScheme[currentPathIndex+1] += 1
                else:
                    # Cannot propagate the carry => end of iteration
                    #print("DEBUG: End of iteration with last path: [%s]." % (str(self.currentPathScheme)))
                    raise StopIteration
                continue
            else:
                # Found a distance. Increment currentPathIndex
                distancePath.append(dist)

                currentPathIndex += 1
                currentDistanceMatrix = currentDistanceMatrix.reduceMatrix(dist)
                if currentDistanceMatrix is None:
                    # Done exploring current path
                    #print("DEBUG: End exploring path.")
                    break
                else:
                    #print("DEBUG: Exploring reduced matrix.")
                    # A reduce matrix need to be explored
                    if currentPathIndex == len(self.currentPathScheme):
                        # We need to add next distance path in scheme initialized with 0
                        #print("DEBUG: Appending new scheme level.")
                        self.currentPathScheme.append(0)
            #print("DEBUG: infinite loop.")
        
        #print("DEBUG: DistancePathIterator path: [%s] => distances: [%s]." % (str(self.currentPathScheme), str(distancePath)))

        self.currentPathScheme[0] += 1
        return distancePath

def minimalDistanceOverallPathFinder2(persoListA: Sequence[PersonnageData], persoListB: Sequence[PersonnageData], distanceScorer: FunctionType) -> Sequence[PersonnageDistance]:
    fullDistanceMatrix = DistanceMatrix(persoListA, persoListB, distanceScorer)

    minimalDistancePath = None
    minimalDistanceScore = None
    matrixIterator = DistancePathIterator(fullDistanceMatrix)
    for distancePath in matrixIterator:
        #print("DEBUG: Exploring path: %s." % (distancePath))
        distanceScore = 0
        for distance in distancePath:
            score = distanceScorer(distance, fullDistanceMatrix)
            distanceScore += score

        if minimalDistanceScore is None or minimalDistanceScore > distanceScore:
            #print("DEBUG: Found new minimal path score: %f." % (currentDistanceScore))
            minimalDistancePath = distancePath
            minimalDistanceScore = distanceScore

    if minimalDistancePath is None:
        minimalDistancePath = []

    #print("DEBUG: Minimal distance path found: %s" % ' '.join([ str(x) for x in minimalDistancePath ]))
    return minimalDistancePath

