

"""
Reconnaissance des personnages à partir d'un json

Vous ne devez modifier que ce fichier
Installation: voir le readme

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
DISTANCE_DEPLACEMENT_MAXIMUM_PAR_FRAME = 1000
ABSOLUTE_MAXIMUM_DISTANCE_DEPLACEMENT = 3000
MAX_MISSING_FRAME_BEFORE_DECAY = 5
MAX_MISSING_FRAME_BEFORE_DEAD = 200
MALUS_DISTANCE_SCORE = 1000000

COLORS = [(0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 255), (0, 0, 127), (0, 127, 0), (127, 127, 0), (127, 0, 127), (127, 0, 0), (0, 127, 127), (127, 127, 127)]

def speedBetweenPoints(p1, p2) -> Tuple[float, float, float]:
    speed = None
    if p1 and p2:
        frameDelta = np.abs(p1.frame - p2.frame)
        if frameDelta > 0:
            xSpeed = (p1.x - p2.x) / frameDelta
            ySpeed = (p1.y - p2.y) / frameDelta
            zSpeed = (p1.z - p2.z) / frameDelta
            speed = (xSpeed, ySpeed, zSpeed)
    return speed

def distance2dBeetweenPoints(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

class PersonnageData:

    def __init__(self, frame, x, y, z):
        self.frame = frame
        self.x = x
        self.y = y
        self.z = z
        self.uid: int = None
        self.speed: tuple = None
        self.color: tuple = None
        #self.absoluteSpeed: float = None
        #self.direction: float = None
        self.ancestorSpeed: tuple = None
        self.frameAppearanceCount = 1
        # Score qui indique l'activite recente du personnage. Proche de 0 si peu actif. Proche de 1 si tres actif.
        self.freshness: float = 0.01
        self.xHistory: Sequence[float] = []
        self.yHistory: Sequence[float] = []
        self.zHistory: Sequence[float] = []

    def __str__(self):
        return f"[Personnage #{self.uid} (#{id(self)}) x={self.x} y={self.y} speed={self.speed} freshness={self.freshness}]"

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

    def successTo(self, p):
        if p is not None:
            if p.speed is not None:
                self.ancestorSpeed = p.speed
            self.uid = p.uid
            self.speed = speedBetweenPoints(self, p)
            self.frameAppearanceCount = p.frameAppearanceCount + 1
            self.freshness = p.freshness
            self._increaseFreshness()
            self.xHistory = p.xHistory + [p.x]
            self.yHistory = p.yHistory + [p.y]
            self.zHistory = p.zHistory + [p.z]

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


    # Deprecated
    def scoreBetween(self, p):
        if p is None: return sys.maxsize

        distance2d = distance2dBeetweenPoints((self.x, self.y), (p.x, p.y))
        speedDistance = 0
        hypoteticSpeed = speedBetweenPoints(self, p)
        if hypoteticSpeed and p.ancestorSpeed:
            speedDistance = distance2dBeetweenPoints((hypoteticSpeed[0], hypoteticSpeed[1]), (p.ancestorSpeed[0], p.ancestorSpeed[1]))
        frameDistance = self.frame - p.frame

        #confidance = np.log10(p.frameAppearanceCount + 1)
        #confidance = 0.1 + p.freshness
        #confidance = 1.0 / (1.0 - np.log(p.freshness))
        # Idee un indice de confiance relatif à la confiance des autre points
        confidance = 1
        
        if frameDistance > 0:
            distance = (distance2d + speedDistance / frameDistance) / confidance
        else:
            distance = (distance2d) / confidance
        #print("DEBUG: Distance between %s and %s => %f" % (self, p, distance))
        #print("DEBUG: Confidance=%f ; Distances => 2d=%f ; speed=%f ; frame=%f" % (confidance, distance2d, speedDistance, frameDistance))
        return distance

    def label(self):
        return "Personnage_%d" % self.uid


class PersonnageDistance:

    def __init__(self, p1: PersonnageData, p2: PersonnageData):
        self.p1: PersonnageData = p1
        self.p2: PersonnageData = p2
        self.frame: int = p1.frame - p2.frame
        self.x: float = p1.x - p2.x
        self.y: float = p1.y - p2.y
        self.z: float = p1.z - p2.z
        if self.frame > 0 and p1.speed and p2.speed:
            self.xSpeed: float = p1.speed[0] - p2.speed[0]
            self.ySpeed: float = p1.speed[1] - p2.speed[1]
            self.zSpeed: float = p1.speed[2] - p2.speed[2]
        self.color: float = None

    def __str__(self) -> str:
        #return f"[p1: #{id(self.p1)} p2: #{id(self.p2)} x={self.x} y={self.y} 3d={self.cartesianDistance()}]"
        return f"[x={self.x} y={self.y} 3d={self.cartesianDistance()}]"

    def cartesianDistance(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2)**0.5


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
                    clone.__allDistances.remove(dist)
                    del clone.__matrix[rowKey][colKey]

        del clone.__matrix[id(row)]

        #print("DEBUG: reduced matrix contains %d distance(s) in %d row(s) ..." % (len(clone.__allDistances), len(clone.__matrix)))
        return clone

    def reduceMatrix(self, distance: PersonnageDistance):
        return self._reduceMatrix(distance.p1, distance.p2)

def naive2dDistanceScorer(dist: PersonnageDistance, matrix: DistanceMatrix):
    return dist.cartesianDistance()

def confident2dDistanceScorer(dist: PersonnageDistance, matrix: DistanceMatrix):
    relativeConfidance = (dist.p2.freshness / matrix.columnsFreshnessSum) + 0.1
    if dist.cartesianDistance() > ABSOLUTE_MAXIMUM_DISTANCE_DEPLACEMENT:
        # cartesianDistance is greater than we expect => Return a malus
        return MALUS_DISTANCE_SCORE
    if dist.cartesianDistance() > DISTANCE_DEPLACEMENT_MAXIMUM_PAR_FRAME * dist.frame:
        # cartesianDistance is greater than we expect => Return a malus
        return MALUS_DISTANCE_SCORE
    return dist.cartesianDistance() / relativeConfidance


# Deprecated
# Return (PersonnageDataA, PersonnageDataB, distance)
def nearestPersonnages(persoListA, persoListB) -> Tuple[PersonnageData, PersonnageData, float]:
    distanceMap = {}
    for pA in persoListA:
        for pB in persoListB:
            dist = PersonnageDistance(pA, pB).cartesianDistance()
            distKey = "%d-%d" % (id(pA), id(pB))
            distanceMap[distKey] = (pA, pB, dist)

    minDist = None
    minDistTuple = None
    for distTuple in distanceMap.values():
        dist = distTuple[2]
        if minDist is None or minDist > dist:
            minDist = dist
            minDistTuple = distTuple

    if minDistTuple is None:
        minDistTuple = (None, None, None)

    return minDistTuple

# Deprecated
# Relative confidance : Compare each perso freshness to sum of freshnesses
def nearestPersonnagesWithFreshness(persoListA: Sequence[PersonnageData], persoListB: Sequence[PersonnageData]) -> Tuple[PersonnageData, PersonnageData, float]:
    distanceMap = {}
    freshnessSumPa = 0
    for pA in persoListA:
        freshnessSumPa += pA.freshness

    freshnessSumPb = 0
    for pB in persoListB:
        freshnessSumPb += pB.freshness

    for pA in persoListA:
        for pB in persoListB:
            #relativeConfidance = (pA.freshness / freshnessSumPa + pB.freshness / freshnessSumPb) / 2 + 0.1
            relativeConfidance = (pB.freshness / freshnessSumPb) + 0.1
            #distance = pA.scoreBetween(pB) / relativeConfidance
            distance = PersonnageDistance(pA, pB)
            score = distance.cartesianDistance() / relativeConfidance
            distKey = "%d-%d" % (id(pA), id(pB))
            distanceMap[distKey] = (pA, pB, score)

    minDist = None
    minDistTuple = None
    for distTuple in distanceMap.values():
        dist = distTuple[2]
        if minDist is None or minDist > dist:
            minDist = dist
            minDistTuple = distTuple

    if minDistTuple is None:
        minDistTuple = (None, None, None)

    return minDistTuple

# Relative confidance : Compare each perso freshness to sum of freshnesses
def nearestPersonnagesWithMatrix(persoListA: Sequence[PersonnageData], persoListB: Sequence[PersonnageData]) -> Tuple[PersonnageData, PersonnageData, float]:
    #print("DEBUG: building distance matrix of (%d x %d) size ..." % (len(persoListA), len(persoListB)))
    distanceMatrix = DistanceMatrix(persoListA, persoListB, confident2dDistanceScorer)
    minDistance = distanceMatrix.getNthMinimalDistance(0)

    minDistTuple = (None, None, None)
    if minDistance is not None:
        distance = minDistance.cartesianDistance()
        minDistTuple = (minDistance.p1, minDistance.p2, distance)

    return minDistTuple

def minimalDistanceOverallPathFinder(persoListA: Sequence[PersonnageData], persoListB: Sequence[PersonnageData]) -> Sequence[PersonnageDistance]:
    distanceScorer = confident2dDistanceScorer
    fullDistanceMatrix = DistanceMatrix(persoListA, persoListB, distanceScorer)

    # Main path
    n = 0
    minimalDistancePath = None
    minimalDistanceScore = None
    
    distanceMatrix = fullDistanceMatrix
    currentDistancePath = []
    currentDistanceScore = 0
    while True:
        nthDistance = distanceMatrix.getNthMinimalDistance(n)
        if nthDistance is None:
            break

        score = distanceScorer(nthDistance, fullDistanceMatrix)
        if score < MALUS_DISTANCE_SCORE:
            currentDistanceScore += score
            currentDistancePath.append(nthDistance)

        distanceMatrix = distanceMatrix.reduceMatrix(nthDistance)

    if minimalDistanceScore is None or minimalDistanceScore > currentDistanceScore:
        minimalDistancePath = currentDistancePath

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
        newPerso.successTo(previousPerso)
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

        betterPath = minimalDistanceOverallPathFinder(newPersos, previousPersos)
        for dist in betterPath:
            newPerso = dist.p1
            previousPerso = dist.p2
            self._recordPersoUpdate(newPerso, previousPerso)
            newPersos.remove(newPerso)

        # for i in range(len(newPersos)):
        #     (newPerso, previousPerso, distance) = nearestPersonnagesWithMatrix(newPersos, previousPersos)
        #     #(newPerso, previousPerso, distance) = nearestPersonnages(newPersos, previousPersos)
        #     if distance is not None:
        #         #print("DEBUG: Minimal distance found: [%f] of perso #%d." % (distance, previousPerso.uid))
        #         pass
        #     if distance is not None and distance <= DISTANCE_DEPLACEMENT_MAXIMUM_PAR_FRAME:
        #         # Same perso
        #         self._recordPersoUpdate(newPerso, previousPerso)
        #         #print("DEBUG: Removing personnages #%d and #%d ..." % (id(newPerso), id(previousPerso)))
        #         newPersos.remove(newPerso)
        #         previousPersos.remove(previousPerso)
        #     else:
        #         # No perso are near enough to match
        #         #k += 1
        #         print("DEBUG: some perso cannot be match: [%d]." % (len(newPersos)))
        #         break
            
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

        self.loop = True
        # Numéro de la frame
        self.frame = 0

    def draw_all_personnages(self):
        """Dessin des centres des personnages dans la vue de dessus,
        représenté par un cercle
        """
        # # self.black = np.zeros((720, 1280, 3), dtype = "uint8")
        cv2.line(self.black, (0, 360), (1280, 360), (255, 255, 255), 2)
        for perso in self._repo.getCurrentIterationPersonages():
            #print("DEBUG: coord #%d : (%f, %f)" % (perso.uid, perso.x, perso.y))
            x = 360 + int(perso.x*160/1000)
            # # if x < 0: x = 0
            # # if x > 1280: x = 1280
            y = int(perso.y*160/1000)
            # # if y < 0: y = 0
            # # if y > 720: y = 720
            cv2.circle(self.black, (y, x), 4, (100, 100, 100), -1)
            cv2.circle(self.black, (y, x), 6, COLORS[(perso.uid - 1) % 7], thickness=2)

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

    for i in range(7, 9):
        FICHIER = './json/cap_' + str(i) + '.json'
        p3d = Personnages3D(FICHIER)
        p3d.run()

    # FICHIER = './json/cap_7.json'
    # p3d = Personnages3D(FICHIER)
    # p3d.run()
