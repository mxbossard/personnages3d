

"""
Reconnaissance des personnages à partir d'un json

Vous ne devez modifier que ce fichier
Installation: voir le readme

"""

import os
import sys
import json
from time import time, sleep
from typing import Mapping, Sequence, Tuple

import numpy as np
import cv2


#MAX_FRAME_PASSE_A_SCANNER = 200
# FIXME: DISTANCE_DEPLACEMENT_MAXIMUM_PAR_FRAME should be the max 2D distance
DISTANCE_DEPLACEMENT_MAXIMUM_PAR_FRAME = 4000
MAX_MISSING_FRAME_BEFORE_DECAY = 5
MAX_MISSING_FRAME_BEFORE_DEAD = 200
COLORS = [(0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 255)]

def speedBetweenPoints(p1, p2) -> Tuple[float, float, float]:
    speed = None
    if p1 and p2:
        frameDelta = p1.frame - p2.frame
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
    frame: int = 0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    uid: int = None
    speed: tuple = None
    #absoluteSpeed: float = None
    #direction: float = None
    ancestorSpeed: tuple = None
    frameAppearanceCount = 1
    # Score qui indique l'activite recente du personnage. Proche de 0 si peu actif. Proche de 1 si tres actif.
    freshness: float = 0.01

    def __init__(self, frame, x, y, z):
        self.frame = frame
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"[Personnage #{self.uid} x={self.x} y={self.y} speed={self.speed} freshness={self.freshness}]"

    def _increaseFreshness(self):
        self.freshness += 0.01
        if self.freshness > 1:
            self.freshness = 1

    def _decreaseFreshness(self):
        self.freshness -= 0.01
        if self.freshness < 0:
            self.freshness = 0

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

    def decayFreshness(self, iteration):
        if iteration > self.frame + MAX_MISSING_FRAME_BEFORE_DECAY:
            self._decreaseFreshness()

    def isAlive(self, iteration):
        notTimeout = self.freshness > 0.1 or iteration - self.frame < MAX_MISSING_FRAME_BEFORE_DEAD
        
        # Filter short appearance
        isGhost = iteration - self.frame > 10 and self.freshness == 0
        if isGhost:
                print("DEBUG: Detected ghost perso #%d." % (self.uid))

        return notTimeout and not isGhost

    def distanceTo(self, p):
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
        
        distance = (distance2d + speedDistance / frameDistance) / confidance
        #print("DEBUG: Distance between %s and %s => %f" % (self, p, distance))
        #print("DEBUG: Confidance=%f ; Distances => 2d=%f ; speed=%f ; frame=%f" % (confidance, distance2d, speedDistance, frameDistance))
        return distance

    def label(self):
        return "Personnage_%d" % self.uid

# Return (PersonnageData, distance)
def nearestPersonnage(newPerso, persoList) -> tuple:
    nearestPerso = None
    minDistance = None
    for p in persoList:
        dist = p.distanceTo(newPerso)
        if (minDistance is None or dist < minDistance):
            minDistance = dist
            nearestPerso = p

    return (nearestPerso, minDistance)


# Return (PersonnageDataA, PersonnageDataB, distance)
def nearestPersonnages(persoListA, persoListB) -> Tuple[PersonnageData, PersonnageData, float]:
    distanceMap = {}
    for pA in persoListA:
        for pB in persoListB:
            dist = pA.distanceTo(pB)
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
            dist = pA.distanceTo(pB) / relativeConfidance
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

class PersonnagesCoordinatesRepo:
    _maxPersonnages = 0
    _repo = {}
    _seenPersoCounter = 0
    _persoTracker: Mapping[int, PersonnageData] = {}
    _currentFrame = 0
    _persoUids = set()

    def __init__(self, maxPersonnages: int):
        self.reset()
        self._maxPersonnages = maxPersonnages

    def _getIterationRepo(self, iteration: int) -> dict:
        repo = self._repo.get(iteration, {})
        if not repo:
            self._repo[iteration] = repo
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
        self._persoTracker[persoData.uid] = persoData 

    def _newPersoUid(self):
        self._seenPersoCounter += 1
        uid = 1
        while uid in self._persoUids:
            uid += 1
        self._persoUids.add(uid)
        return uid

    def _recordNewPerso(self, perso: PersonnageData):
        newUid = self._newPersoUid()
        perso.initializeUid(newUid)
        self._addIterationPersonnageData(self._currentFrame, perso)
        print("DEBUG: Recorded new perso %s at frame: %d." % (perso, self._currentFrame))

    def _recordPersoUpdate(self, newPerso: PersonnageData, previousPerso: PersonnageData):
        newPerso.successTo(previousPerso)
        self._addIterationPersonnageData(self._currentFrame, newPerso)
        #print("DEBUG: Recorded perso update %s => %s." % (previousPerso, newPerso))

    def _forgetPerso(self, perso: PersonnageData):
        print("DEBUG: Removing perso #%d." % (perso.uid))
        self._persoUids.discard(perso.uid)
        self._persoTracker.pop(perso.uid)

    def _frameCleaning(self, iteration):
        self._currentFrame = iteration
        for persoUid in list(self._persoTracker.keys()):
            perso = self._persoTracker[persoUid]

            if not perso.isAlive(iteration):
                self._forgetPerso(perso)

    def reset(self):
        self._maxPersonnages = 0
        self._repo = {}
        self._seenPersoCounter = 0

    # def newIteration(self, iteration):
    #     self._currentFrame = iteration

    def getAllPersonages(self) -> Sequence[PersonnageData]:
        return list(self._persoTracker.values())
    
    def getCurrentIterationPersonages(self) -> Sequence[PersonnageData]:
        personnages = []
        for perso in self.getAllPersonages():
            if perso.frame == self._currentFrame:
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

        for i in range(len(newPersos)):
            (newPerso, previousPerso, distance) = nearestPersonnagesWithFreshness(newPersos, previousPersos)
            if distance is not None:
                #print("DEBUG: Minimal distance found: [%f] of perso #%d." % (distance, previousPerso.uid))
                pass
            if distance is not None and distance <= DISTANCE_DEPLACEMENT_MAXIMUM_PAR_FRAME:
                # Same perso
                self._recordPersoUpdate(newPerso, previousPerso)
                newPersos.remove(newPerso)
                previousPersos.remove(previousPerso)
            else:
                # No perso are near enough to match
                #k += 1
                print("DEBUG: some perso cannot be match: [%d]." % (len(newPersos)))
                break
            
        if len(newPersos) == 0:
            unknowPersoRemaining = False
        
        if unknowPersoRemaining:
            #  Some perso cannot be matched, infer a not seen previously perso
            for perso in newPersos:
                self._recordNewPerso(perso)

            print("DEBUG: Coordinates which lead to new personage recording: [%s]." % (coordinates))

        for perso in self.getAllPersonages():
            perso.decayFreshness(iteration)


class Personnage:
    """Permet de stocker facilement les attributs d'un personnage,
    et de les reset-er.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.who = None
        self.points_3D = None
        # Les centres sont dans le plan horizontal
        self.center = [100000]*2
        # Numéro de la frame de la dernière mise à jour
        self.last_update = 0
        self.stable = 0



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
            cv2.circle(self.black, (y, x), 6, COLORS[perso.uid % 7], thickness=2)

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

    # # for i in range(8):

        # # FICHIER = './json/cap_' + str(i) + '.json'

        # # p3d = Personnages3D(FICHIER)
        # # p3d.run()

    FICHIER = './json/cap_7.json'
    p3d = Personnages3D(FICHIER)
    p3d.run()
