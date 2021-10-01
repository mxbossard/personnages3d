

"""
Reconnaissance des personnages à partir d'un json

Vous ne devez modifier que ce fichier
Installation: voir le readme

"""

import os
import json
from time import time, sleep

import numpy as np
import cv2


COLORS = [(0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255), (255, 0, 0), (255, 255, 255), (0, 255, 255)]


def distance2dBeetweenPoints(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

MAX_FRAME_PASSE_A_SCANNER = 200
DISTANCE_DEPLACEMENT_MAXIMUM_PAR_FRAME = 10000

class PersonnageData:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    uid: int = None
    label: str = None

    def __init__(self, uid, x, y, z):
        self.uid = uid
        self.x = x
        self.y = y
        self.z = z

    def distanceTo(self, p):
        return distance2dBeetweenPoints((self.x, self.y), (p.x, p.y))

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
def nearestPersonnages(persoListA, persoListB) -> tuple:
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

class PersonnagesCoordinatesRepo:
    _maxPersonnages = 0
    _repo = {}
    _seenPersoCounter = 0
    _persoTracker = {}

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
        return self._seenPersoCounter

    def reset(self):
        self._maxPersonnages = 0
        self._repo = {}
        self._seenPersoCounter = 0

    def getPersonages(self):
        return self._persoTracker.values()



    def addCoordinates(self, iteration: int, coordinates: list):
        newPersos = []
        for coord in coordinates:
            (x, y) = coord
            newPerso = PersonnageData(None, x, y, 0)
            newPersos.append(newPerso)
        
        k = 1
        unknowPersoRemaining = True
        while(unknowPersoRemaining and iteration - k > 0 and k < MAX_FRAME_PASSE_A_SCANNER):
            previousPersos = self._getIterationPersonnageData(iteration - k)

            for p in newPersos:
                (newPerso, previousPerso, distance) = nearestPersonnages(newPersos, previousPersos)
                if distance is not None and distance <= DISTANCE_DEPLACEMENT_MAXIMUM_PAR_FRAME * k:
                    # Same perso
                    newPerso.uid = previousPerso.uid
                    self._addIterationPersonnageData(iteration, newPerso)
                    newPersos.remove(newPerso)
                    previousPersos.remove(previousPerso)
                else:
                    # No perso are near enough to match
                    k += 1
                    break
                
            if len(newPersos) == 0:
                unknowPersoRemaining = False
        
        if unknowPersoRemaining:
            #  Some perso cannot be matched, infer a not seen previously perso
            for perso in newPersos:
                perso.uid = self._newPersoUid()
                self._addIterationPersonnageData(iteration, perso)


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

        # Distance de rémanence pour attribution des squelettes
        self.distance = 500

        self.json_data = read_json(FICHIER)

        self.whos = [0]*4

        # Fenêtre pour la vue du dessus
        cv2.namedWindow('vue du dessus', cv2.WND_PROP_FULLSCREEN)
        self.black = np.zeros((720, 1280, 3), dtype = "uint8")

        # Toutes les datas des personnages dans un dict self.personnages
        self.personnages = []
        for i in range(4):
            self.personnages.append(Personnage())
        self.skelet_nbr = 0
        self.new_centers = None

        self.loop = 1
        # Numéro de la frame
        self.nbr = 0

    def main_frame_test(self, skelet_3D):
        """skelet_3D sont les squelettes en 2D et 3D"""

        if skelet_3D:
            # Nombre de squelettes
            self.skelet_nbr = len(skelet_3D)

            self.who_is_who(skelet_3D)
            self.apply_to_personnages(skelet_3D)
            self.draw_all_personnages()

        else:
            print(f"\nPas de squelette  ------------------------>",
               f"             Frame --->",
               f"{self.nbr}")

    def who_is_who(self, skelet_3D):

        # Préliminaire
        self.update_centers(skelet_3D)
        print(f"\nNombre de squelette  ------------------------>",
               f"{self.skelet_nbr}         Frame --->",
               f"{self.nbr}")

        # Parcours des squelettes pour calculer les distances par rapport
        # aux centres des personnages
        # dists[0] = liste des distance entre: squelette 0 et les personnages
        dists = {}
        for skel in range(self.skelet_nbr):
            dists[skel] = []
            # Recherche des perso proche de ce skelelet
            for perso in range(4):
                dist = get_distance(self.new_centers[skel], self.last_centers[perso])
                if dist > 100000:
                    dist = 100000
                dists[skel].append(dist)

        print("distances:", dists)  # {0: [41, 41, 41, 41]}

        # Attibution avec le perso le plus proche du squelette
        whos, TODO = self.attribution_with_nearest(dists)
        self.whos = self.default_attribution(whos, TODO)

    def update_centers(self, skelet_3D):
        """
        last_centers = liste des centres tirée des centres des personnages
        self.new_centers = liste des centres des squelettes de la frame
        """

        self.last_centers = []
        for i in range(4):
            self.last_centers.append(self.personnages[i].center)

        self.new_centers = []
        for i in range(self.skelet_nbr):
            self.new_centers.append(get_center(skelet_3D[i]))

    def attribution_with_nearest(self, dists):
        """ Attribution avec le plus près
        Nombre de squelette  ------------------------> 2
        distances: {0: [2, 1091, 1557, 100000], 1: [1092, 3, 1415, 100000]}
        whos: [0, 1, None, None] TODO: 0
        whos final [0, 1, None, None]

        Nombre de squelette  ------------------------> 2
        distances: {0: [1091, 2, 1413, 100000], 1: [3, 1096, 1556, 100000]}
        whos: [1, 0, None, None] TODO: 0
        whos final [1, 0, None, None]
        """
        whos = [None]*4
        # Nombre de squelette qui reste à attribuer
        TODO = self.skelet_nbr
        for i in range(self.skelet_nbr):
            if i in dists:
                # Le mini dans la liste
                mini = min(dists[i])
                # Position du mini dans la liste
                index = dists[i].index(mini)
                if mini < self.distance:
                    whos[index] = i
                    TODO -= 1

        return whos, TODO

    def default_attribution(self, whos, TODO):
        """ Attribution par défaut si pas attribué avant

        3 squelttes
        whos: [1, None, None, None] --> TODO: 2
        On doit trouver --> [1, 0, 2, None]

        liste des déjà attribués: done = [1]
        à attribuer 0 et 2:
            possible = [0, 2, 3]
            moins whos
            liste des numéros à attribuer: dispo = [0, 2]
        len(dispo) = TODO
        """

        done = [x for x in whos if x is not None]
        dispo = [x for x in range(4) if x not in whos]

        print("whos avec nearest:", whos, "TODO:", TODO, "done:", done, "dispo", dispo)

        d = 0
        while TODO > 0:
            for i, who in enumerate(whos):
                if who is None:
                    whos[i] = dispo[d]
                    TODO -= 1
                    d += 1
                    break

        print("whos final:", whos)
        return whos

    def apply_to_personnages(self, skelet_3D):
        """ whos du type [1, 0, None, 2]
                                1 attribué au perso 0
                                0 attribué au perso 1 ... etc ...
        """

        for i in range(4):
            # Data valide
            if self.whos[i] is not None:
                self.personnages[i].who = self.whos[i]
                self.personnages[i].points_3D = skelet_3D[self.whos[i]]
                c = get_center(skelet_3D[self.whos[i]])
                self.personnages[i].center = c
                self.personnages[i].last_update = self.nbr
                self.personnages[i].stable += 1
                if self.personnages[i].stable > 3:
                    self.personnages[i].stable = 3

            # Pas de data sur cette frame
            else:
                self.personnages[i].who = None
                self.personnages[i].points_3D = None
                self.personnages[i].center = [100000]*2
                self.personnages[i].stable -= 1

        # Reset si pas d' attribution pendant 5 frames
        for i in range(4):
            if self.nbr - self.personnages[i].last_update > 4:
                self.personnages[i].reset()

    def draw_all_personnages(self):
        """Dessin des centres des personnages dans la vue de dessus,
        représenté par un cercle
        """
        # # self.black = np.zeros((720, 1280, 3), dtype = "uint8")
        cv2.line(self.black, (0, 360), (1280, 360), (255, 255, 255), 2)
        for perso in self._repo.getPersonages():
            print("MBD: coord #%d : (%f, %f)" % (perso.uid, perso.x, perso.y))
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

            if self.nbr < len(self.json_data):
                skelet_3D = self.json_data[self.nbr]
                #self.main_frame_test(skelet_3D)
                if skelet_3D:
                    coordinates = []
                    for s in skelet_3D:
                        center = get_center(s)
                        x = center[0]
                        y = center[1]
                        #z = center[2]
                        #print("MBD: coord (%f, %f)" % (x, y))
                        coordinates.append((x, y))

                    print("MBD: all coordinates [%s]" % coordinates)
                    self._repo.addCoordinates(self.nbr + 1, coordinates)

                    self.draw_all_personnages()
            else:
                self.loop = 0

            cv2.imshow('vue du dessus', self.black)
            self.nbr += 1

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


def get_distance(p1, p2):
    """Distance entre les points p1 et p2, dans le plan horizontal,
    sans prendre en compte le y qui est la verticale.
    """

    if p1 and p2:
        if None not in p1 and None not in p2:
            d = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
            return int(d)
    return 100000


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
