

"""
Reconnaissance des personnages à partir d'un json

Vous ne devez modifier que ce fichier
Installation: voir le readme

TODO:
- Vérifier l'algo de recherche de la distance minimum
- Forger un vrai bon score de fraicheur/historique
- Vérifier les calculs de vitesse et d'accélération

"""

from functools import cmp_to_key
import os
import sys
import json
import copy
from time import time, sleep
from types import FunctionType
from typing import Mapping, MutableSequence, Sequence, Tuple

import numpy as np
import cv2


#MAX_FRAME_PASSE_A_SCANNER = 200
# FIXME: DISTANCE_DEPLACEMENT_MAXIMUM_PAR_FRAME should be the max 2D distance
#DISTANCE_DEPLACEMENT_MAXIMUM_PAR_FRAME = 500
SMOOTHING_WINDOW_SIZE = 10
ABSOLUTE_MAXIMUM_SMOOTHED_DISTANCE = 600
ABSOLUTE_MAXIMUM_SMOOTHED_SPEED = 200
ABSOLUTE_MAXIMUM_SMOOTHED_ACCELERATION = 800
ABSOLUTE_MAXIMUM_INSTANT_DISTANCE = 10000
ABSOLUTE_MAXIMUM_INSTANT_SPEED = 10000
ABSOLUTE_MAXIMUM_INSTANT_ACCELERATION = 1000
MAX_MISSING_FRAME_BEFORE_DECAY = 5
MAX_MISSING_FRAME_BEFORE_DEAD = 200
MIN_SAMPLE_CONFIRMED_THRESHOLD = 5

COLORS = [(0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 255), (0, 0, 127), (0, 127, 0), (127, 127, 0), (127, 0, 127), (127, 0, 0), (0, 127, 127), (127, 127, 127)]

def vectorNorme(vector: Tuple[float, float, float]) -> float:
    if vector is None:
        return 0
    return ((vector[0])**2 + (vector[1])**2 + (vector[2])**2)**0.5

def vectorDelta(vector1: Tuple[float, float, float], vector2: Tuple[float, float, float], frameDelta = 1) -> Tuple[float, float, float]:
    if frameDelta == 0:
        #FIXME: is it a good response ?
        return (0, 0, 0)
    return ((vector1[0]-vector2[0]) / frameDelta, (vector1[1]-vector2[1]) / frameDelta, (vector1[2]-vector2[2]) / frameDelta)

def smoothCartesianCoordinate(coordinatesList: Sequence[Tuple[float, float, float]], history:int, windowSize:int) -> Tuple[float, float, float]:
    # coordinate smoothing algorithm: averaging data
    xSmoothed = 0
    ySoothed = 0
    zSmoothed = 0

    if len(coordinatesList) > 0:
        coordsCount = 0
        for coords in coordinatesList[history:history + windowSize]:
            xSmoothed += coords[0]
            ySoothed += coords[1]
            zSmoothed += coords[2]
            coordsCount += 1

        xSmoothed /= coordsCount
        ySoothed /= coordsCount
        zSmoothed /= coordsCount
    
    return (xSmoothed, ySoothed, zSmoothed)

def weightedScore(*scores: float) -> float:
    scoreCount = len(scores)
    if scoreCount == 0:
        return None
    scoreSum = 0
    weightSum = 0
    for k, score in enumerate(scores):
        weight = -2*k/((scoreCount + 1)**2) + 2/(scoreCount + 1)
        weightSum += weight
        scoreSum += score * weight
    return scoreSum / weightSum

class PersonnageData:

    def __init__(self, frame, x, y, z, smoothingWindow=SMOOTHING_WINDOW_SIZE):
        self.frame = frame
        self.coordinate = (x, y, z) # Cartesian coordinate
        self.uid: int = None
        self.color: tuple = None
        self.frameAppearanceCount = 1
        # Score qui indique l'activite recente du personnage. Proche de 0 si peu actif. Proche de 1 si tres actif.
        self.freshness: float = 0.01
        self.frameHistory = [frame]
        self.coordinatesHistory: Sequence[Tuple[float, float, float]] = [self.coordinate]
        self.smoothingWindow = smoothingWindow
        self.smoothedCoordinatesHistory: Sequence[Tuple[float, float, float]] = [self.coordinate]
        self.confirmed = False
        self.smoothedSpeedHistory = []
        self.smoothedAccelerationHistory = []

    def __str__(self):
        data = ""
        if len(self.smoothedSpeedHistory) > 0:
            data += f"speed={self.smoothedSpeedHistory[0]} "
        if len(self.smoothedAccelerationHistory) > 0:
            data += f"accel={self.smoothedAccelerationHistory[0]} "

        return f"[Personnage #{self.uid} instantCoord={self.coordinate} coord={self.smoothedCoordinatesHistory[0]} {data}]"

    def _increaseFreshness(self):
        self.freshness += 0.01
        if self.freshness > 1:
            self.freshness = 1

    def _decreaseFreshness(self):
        self.freshness -= 0.01
        if self.freshness < 0.01:
            self.freshness = 0.01

    def initializeUid(self, uid):
        if self.uid is None:
            self.uid = uid

    def getSoothedCoordinates(self, history:int=0) -> Tuple[float, float, float]:
        return smoothCartesianCoordinate(self.coordinatesHistory, history, self.smoothingWindow)

    def succeedTo(self, p):
        if p is not None:
            self.uid = p.uid
            self.frameAppearanceCount = p.frameAppearanceCount + 1
            self.freshness = p.freshness
            self._increaseFreshness()

            # Historical data
            self.frameHistory = self.frameHistory + p.frameHistory
            #print("DEBUG: Frame history: %s." % (self.frameHistory))
            self.coordinatesHistory = self.coordinatesHistory + p.coordinatesHistory

            smoothedCoordinate = smoothCartesianCoordinate(self.coordinatesHistory, 0, self.smoothingWindow)
            self.smoothedCoordinatesHistory.insert(0, smoothedCoordinate)

            if (len(p.coordinatesHistory) >= p.smoothingWindow):
                frameDelta = self.frameHistory[0] - self.frameHistory[1]
                smoothedSpeed = vectorDelta(self.coordinate, p.smoothedCoordinatesHistory[0], frameDelta)
                self.smoothedSpeedHistory = [smoothedSpeed] + p.smoothedSpeedHistory # insert at list start
                #print("DEBUG: smoothedSpeedHistory: %s." % (self.smoothedSpeedHistory))
            
            if (len(p.coordinatesHistory) > p.smoothingWindow):
                smoothedAcceleration = vectorDelta(self.smoothedSpeedHistory[0], p.smoothedSpeedHistory[0], frameDelta)
                self.smoothedAccelerationHistory = [smoothedAcceleration] + p.smoothedAccelerationHistory # insert at list start

            #FIXME: confirmed based on frame size + not to hold frames
            self.confirmed = len(self.frameHistory) > MIN_SAMPLE_CONFIRMED_THRESHOLD

            #print("DEBUG: Perso succession: %s." % (self))

    def decayFreshness(self, iteration):
        if iteration > self.frame + MAX_MISSING_FRAME_BEFORE_DECAY:
            self._decreaseFreshness()

    def isAlive(self, iteration):
        notTimeout = self.freshness > 0.1 or iteration - self.frame < MAX_MISSING_FRAME_BEFORE_DEAD
        
        # Filter short appearance
        isGhost = iteration - self.frame > 10 and self.freshness <= 0.01
        if isGhost:
                print("DEBUG: Detected ghost perso #%d." % (self.uid))

        return notTimeout and not isGhost

    def label(self):
        return "Personnage_%d" % self.uid


class PersonnageDistance:

    def __init__(self, p1: PersonnageData, p2: PersonnageData):
        """Asymetric distance: from new personnage (p1) to historical personnage (p2)."""
        self.p1: PersonnageData = p1
        self.p2: PersonnageData = p2
        self.frame: int = p1.frame - p2.frame
        self.p1SmoothedCoordinate = smoothCartesianCoordinate([p1.coordinate] + p2.coordinatesHistory, 0, p1.smoothingWindow)
        self.p2SmoothedCoordinate = p2.smoothedCoordinatesHistory[0]
        self.instantDistanceVector = vectorDelta(p1.coordinate, p2.smoothedCoordinatesHistory[0], 1)
        self.smoothedDistanceVector = vectorDelta(self.p1SmoothedCoordinate, self.p2SmoothedCoordinate, 1)
        self.history = abs(len(p1.frameHistory) - len(p2.frameHistory))
        self.color: float = None

        # Non symetric distance history: only distance from p1 to p2 history
        self.instantSpeedVector = None
        self.smoothedSpeedVector = None
        self.instantAccelerationVector = None
        self.smoothedAccelerationVector = None

        if self.frame > 0:
            self.instantSpeedVector = vectorDelta(p1.coordinate, p2.smoothedCoordinatesHistory[0], self.frame)
            self.smoothedSpeedVector = vectorDelta(self.p1SmoothedCoordinate, self.p2SmoothedCoordinate, self.frame)
            #self.distanceVectorHistory = [ vectorDelta(p1.coordinate, p2Coordinate, self.frame) for p2Coordinate in p2.smoothedCoordinatesHistory[:historySize] ]
        if self.instantSpeedVector is not None and len(p2.smoothedSpeedHistory) > 0:
            self.instantAccelerationVector = vectorDelta(self.instantSpeedVector, p2.smoothedSpeedHistory[0], self.frame)
            self.smoothedAccelerationVector = vectorDelta(self.smoothedSpeedVector, p2.smoothedSpeedHistory[0], self.frame)
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
        return vectorNorme(self.smoothedDistanceVector)

    def speedDistance(self) -> float:
        ancestorSpeedVector = (0, 0, 0)
        if self.smoothedSpeedVector is not None and len(self.p2.smoothedSpeedHistory) > 0:
             ancestorSpeedVector = self.p2.smoothedSpeedHistory[0]
        return vectorNorme(vectorDelta(self.smoothedSpeedVector, ancestorSpeedVector, self.frame))

    def accelerationDistance(self) -> float:
        ancestorAccelerationVector = (0, 0, 0)
        if self.smoothedAccelerationVector is not None and len(self.p2.smoothedAccelerationHistory) > 0:
             ancestorAccelerationVector = self.p2.smoothedAccelerationHistory[0]
        return vectorNorme(vectorDelta(self.smoothedAccelerationVector, ancestorAccelerationVector, self.frame))

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

        freshnessSumPa = 0
        for pA in rows:
            freshnessSumPa += pA.freshness
        self.rowsFreshnessSum = freshnessSumPa

        freshnessSumPb = 0
        for pB in columns:
            freshnessSumPb += pB.freshness
        self.columnsFreshnessSum = freshnessSumPb

        for pA in rows:
            self.__matrix[id(pA)] = {}
            for pB in columns:
                #relativeConfidance = (pA.freshness / freshnessSumPa + pB.freshness / freshnessSumPb) / 2 + 0.1
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


def naive2dDistanceScorer(dist: PersonnageDistance, matrix: DistanceMatrix):
    return dist.cartesianDistance()

def confident2dDistanceScorer(dist: PersonnageDistance, matrix: DistanceMatrix):
    relativeConfidance = (dist.p2.freshness / matrix.columnsFreshnessSum) + 0.1
    return dist.cartesianDistance() / relativeConfidance

def historical2dDistanceScorer(dist: PersonnageDistance, matrix: DistanceMatrix):
    return dist.cartesianDistance() / (dist.history + 1)

def multidimensionalScorer(dist: PersonnageDistance, matrix: DistanceMatrix):
    cartesianDist = dist.cartesianDistance()
    speedDist = dist.speedDistance()
    historySize = len(dist.p1.frameHistory) + len(dist.p2.frameHistory)
    historyScore = max(0, 1000 - np.log2(historySize)*100)
    freshness = 0
    for frame in dist.p2.frameHistory[:100]:
        if frame > dist.p1.frame - 100:
            freshness += 1
    freshnessScore = max(0, 1000 - freshness*10)
    #score = weightedScore(historyScore, freshnessScore, cartesianDist, speedDist)
    score = weightedScore(cartesianDist, speedDist, historyScore, freshnessScore, )
    print(f"DEBUG: score={score} from dist={cartesianDist} speed={speedDist} history={historyScore}, freshness={freshnessScore}")
    return score

def minimalDistanceOverallPathFinder2(persoListA: Sequence[PersonnageData], persoListB: Sequence[PersonnageData]) -> Sequence[PersonnageDistance]:
    distanceScorer = historical2dDistanceScorer
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


def minimalDistanceOverallPathFinder(persoListA: Sequence[PersonnageData], persoListB: Sequence[PersonnageData]) -> Sequence[PersonnageDistance]:
    distanceScorer = naive2dDistanceScorer
    fullDistanceMatrix = DistanceMatrix(persoListA, persoListB, distanceScorer)

    # Main path
    minimalDistancePath = None
    minimalDistanceScore = None
    pathSize = min(len(persoListA), len(persoListB))

    # Initial path scheme: take all first minimal distances
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
    currentPathScheme = [0] * pathSize
    
    exploredAllPath = pathSize == 0
    while not exploredAllPath:
        #print("DEBUG: Exploring path: %s." % (currentPathScheme))

        distanceMatrix = fullDistanceMatrix
        currentDistancePath = []
        pathFound = False
        currentDistanceScore = 0

        # Walk a path of minimal distance to score it
        nthDistance = None
        pathIndex = 0
        exploredPath = []
        while pathIndex < pathSize:
            # Search for next nth distance

            n = currentPathScheme[pathIndex]
            nthDistance = distanceMatrix.getNthMinimalDistance(n)
            if nthDistance is None:
                pathFound = False
                #print("DEBUG: Found empty [%d] #%dth distance." % (pathIndex, n))
                # Done testing distance on scheme pathIndex position
                if n == 0:
                    # No minimal distance exists
                    exploredAllPath = True
                    break
                currentPathScheme[pathIndex] = 0
                if pathIndex + 1 < pathSize:
                    # Carry propagation
                    currentPathScheme[pathIndex + 1] += 1
                else: 
                    # Done exploring all schemes
                    exploredAllPath = True
                    #print("DEBUG: Done exploring all schemes.")
                    break
                continue
            else:
                pathFound = True
                exploredPath.append(n)
                #print("DEBUG: Found valid [%d] #%dth distance: %s." % (pathIndex, n, nthDistance))
                # Found a valid distance
                score = distanceScorer(nthDistance, fullDistanceMatrix)
                currentDistanceScore += score
                if minimalDistanceScore is not None and currentDistanceScore > minimalDistanceScore:
                    # If currentDistanceScore greater than minimalDistanceScore we can throw away this path.
                    pathFound = False
                    break

                currentDistancePath.append(nthDistance)

                distanceMatrix = distanceMatrix.reduceMatrix(nthDistance)
                # Increment path index to complete the path
                pathIndex += 1

        # Increment scheme path for next path scoring loop
        if len(currentPathScheme) > 0:
            currentPathScheme[0] += 1

        if pathFound:
            #print("DEBUG: Explored path: %s." % (exploredPath))
            if minimalDistanceScore is None or minimalDistanceScore > currentDistanceScore:
                #print("DEBUG: Found new minimal path score: %f." % (currentDistanceScore))
                minimalDistancePath = currentDistancePath
                minimalDistanceScore = currentDistanceScore

    if minimalDistancePath is None:
        minimalDistancePath = []

    #print("DEBUG: Minimal distance path found: %s" % ' '.join([ str(x) for x in minimalDistancePath ]))
    return minimalDistancePath


class PersonnagesCoordinatesRepo:

    def __init__(self, maxPersonnages: int):
        self.reset()
        self.__maxPersonnages = maxPersonnages

    def reset(self):
        self.__repo = {}
        self.__seenPersoCounter = 0
        self.__persoTracker: Mapping[int, PersonnageData] = {}
        self.__currentFrame = 0
        self.__persoUids = set()

    def _getIterationRepo(self, iteration: int) -> dict:
        repo = self.__repo.get(iteration, {})
        if not repo:
            self.__repo[iteration] = repo
        return repo

    def _getIterationPersonnageData(self, iteration: int) -> list:
        repo = self._getIterationRepo(iteration)
        personnages = repo.get("personnages", [])
        if not personnages:
            repo["personnages"] = personnages
        return personnages

    def _addIterationPersonnageData(self, iteration: int, persoData: PersonnageData):
        persoRepo = self._getIterationPersonnageData(iteration)
        persoRepo.append(persoData)
        self.__persoTracker[persoData.uid] = persoData 

    def _newPersoUid(self):
        self.__seenPersoCounter += 1
        uid = 1
        while uid in self.__persoUids:
            uid += 1
        self.__persoUids.add(uid)
        return uid

    def _recordNewPerso(self, perso: PersonnageData):
        newUid = self._newPersoUid()
        perso.initializeUid(newUid)
        self._addIterationPersonnageData(self.__currentFrame, perso)
        print("DEBUG: Recorded new perso %s at frame: %d." % (perso, self.__currentFrame))

    def _recordPersoUpdate(self, newPerso: PersonnageData, previousPerso: PersonnageData):
        newPerso.succeedTo(previousPerso)
        self._addIterationPersonnageData(self.__currentFrame, newPerso)
        #print("DEBUG: Recorded perso update %s => %s." % (previousPerso, newPerso))

    def _forgetPerso(self, perso: PersonnageData):
        print("DEBUG: Removing perso #%d." % (perso.uid))
        self.__persoUids.discard(perso.uid)
        self.__persoTracker.pop(perso.uid)

    def _frameCleaning(self, iteration):
        self.__currentFrame = iteration
        for persoUid in list(self.__persoTracker.keys()):
            perso = self.__persoTracker[persoUid]

            if not perso.isAlive(iteration):
                self._forgetPerso(perso)


    # def newIteration(self, iteration):
    #     self._currentFrame = iteration

    def getAllPersonages(self) -> Sequence[PersonnageData]:
        return list(self.__persoTracker.values())
    
    def getCurrentIterationPersonages(self) -> Sequence[PersonnageData]:
        personnages = []
        for perso in self.getAllPersonages():
            if perso.frame == self.__currentFrame:
                personnages.append(perso)
        return personnages

    def addNewFrameCoordinates(self, iteration: int, coordinates: list):
        self._frameCleaning(iteration)

        #print("DEBUG: recording frame #%d coordinates [%s] ..." % (iteration, coordinates))

        newPersos: Sequence[PersonnageData] = []
        for coord in coordinates:
            (x, y, z) = coord
            newPerso = PersonnageData(iteration, x, y, z)
            newPersos.append(newPerso)
        
        #k = 1
        unknowPersoRemaining = True
        #while(unknowPersoRemaining and iteration - k > 0 and k < MAX_FRAME_PASSE_A_SCANNER):
        #previousPersos = self._getIterationPersonnageData(iteration - k)
        previousPersos = self.getAllPersonages()

        betterPath = minimalDistanceOverallPathFinder2(newPersos, previousPersos)
        for dist in betterPath:
            newPerso = dist.p1
            previousPerso = dist.p2
            self._recordPersoUpdate(newPerso, previousPerso)
            newPersos.remove(newPerso)

        if len(newPersos) == 0:
            unknowPersoRemaining = False
        
        if unknowPersoRemaining:
            #  Some perso cannot be matched, infer a not seen previously perso
            for perso in newPersos:
                self._recordNewPerso(perso)

            print("DEBUG: Coordinates which lead to new personage recording: [%s]." % (coordinates))

        for perso in self.getAllPersonages():
            perso.decayFreshness(iteration)


class Personnages3D:

    _repo: PersonnagesCoordinatesRepo

    def __init__(self, FICHIER):
        """Unité = mm"""
        self._repo = PersonnagesCoordinatesRepo(12)

        self.json_data = read_json(FICHIER)

        # Fenêtre pour la vue du dessus
        cv2.namedWindow('vue du dessus', cv2.WND_PROP_FULLSCREEN)
        self.black = np.zeros((720, 1280, 3), dtype = "uint8")
        cv2.line(self.black, (0, 360), (1280, 360), (255, 255, 255), 2)

        screenScale = 160/1000
        # draw horizontal lines
        for k in range(0, 720, int(1000*screenScale)):
            cv2.line(self.black, (0, k), (1280, k), (255, 255, 255), thickness=1)

        # draw vertical lines
        for k in range(0, 1280, int(1000*screenScale)):
            cv2.line(self.black, (k, 0), (k, 720), (255, 255, 255), thickness=1)

        self.loop = True
        # Numéro de la frame
        self.frame = 0

    def draw_all_personnages(self):
        """Dessin des centres des personnages dans la vue de dessus,
        représenté par un cercle
        """
        # # self.black = np.zeros((720, 1280, 3), dtype = "uint8")
        
        for perso in self._repo.getCurrentIterationPersonages():
            #print("DEBUG: coord #%d : %s" % (perso.uid, perso.coordinate))
            x = 360 + int(perso.coordinate[0]*160/1000)
            # # if x < 0: x = 0
            # # if x > 1280: x = 1280
            y = int(perso.coordinate[1]*160/1000)
            # # if y < 0: y = 0
            # # if y > 720: y = 720
            cv2.circle(self.black, (y, x), 4, (100, 100, 100), -1)
            color = (128,128,128)
            if perso.confirmed:
                color = COLORS[(perso.uid - 1) % 7]
            cv2.circle(self.black, (y, x), 6, color, thickness=2)

    def run(self):
        """Boucle infinie, quitter avec Echap dans la fenêtre OpenCV"""

        while self.loop:

            if self.frame < len(self.json_data):
                skelet_3D = self.json_data[self.frame]
                #self.main_frame_test(skelet_3D)
                if skelet_3D:
                    coordinates = []
                    for s in skelet_3D:
                        center = get_center(s)
                        x = center[0]
                        y = center[1]
                        #z = center[2]
                        #print("DEBUG: coord (%f, %f)" % (x, y))
                        coordinates.append((x, y, 0))

                    #print("DEBUG: all coordinates [%s]" % coordinates)
                    self._repo.addNewFrameCoordinates(self.frame + 1, coordinates)

                    self.draw_all_personnages()
            else:
                self.loop = False

            cv2.imshow('vue du dessus', self.black)
            self.frame += 1

            k = cv2.waitKey(76)
            if k == 27:  # Esc
                break

        cv2.destroyAllWindows()



def read_json(fichier):
    try:
        with open(fichier) as f:
            data = json.load(f)
    except:
        data = None
        print("Fichier inexistant ou impossible à lire:")
    return data

def get_center(points_3D):
    """Le centre est le centre de vue du dessus,
    c'est la moyenne des coordonées des points du squelette d'un personnage,
    sur x et z
    """

    center = []
    if points_3D:
        for i in [0, 2]:
            center.append(get_moyenne(points_3D, i))

    return center


def get_moyenne(points_3D, indice):
    """Calcul la moyenne d'une coordonnée des points, d'un personnage
    la profondeur est le 3 ème = z, le y est la verticale
    indice = 0 pour x, 1 pour y, 2 pour z
    """

    somme = 0
    n = 0
    for i in range(17):
        if points_3D[i]:
            n += 1
            somme += points_3D[i][indice]
    if n != 0:
        moyenne = int(somme/n)
    else:
        moyenne = None

    return moyenne



if __name__ == '__main__':

    for i in range(7, 8):
        FICHIER = './json/cap_' + str(i) + '.json'
        p3d = Personnages3D(FICHIER)
        p3d.run()

    # FICHIER = './json/cap_7.json'
    # p3d = Personnages3D(FICHIER)
    # p3d.run()
