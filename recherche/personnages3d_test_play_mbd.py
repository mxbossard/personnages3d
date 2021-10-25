

"""
Reconnaissance des personnages à partir d'un json

Vous ne devez modifier que ce fichier
Installation: voir le readme

TODO:
- Vérifier les calculs de vitesse et d'accélération
- Forger un vrai bon score de fraicheur/historique
- Remplacer les smooth coodinates par les prediction du kalman filter   TO_CHECK
- Square size proportional to uncertainty ?
- Effacer les vieux tracés      DONE
- Pause/Resume avec espace      DONE
- Right arrow => pause + one step forward   DONE
- Change speed with +- keys     DONE
- Display uid   DONE
- Left arrow => pause - one step backward

"""

from collections import deque
import json
from datetime import datetime, timedelta
from typing import Deque, Mapping, MutableSequence, Sequence, Tuple
from kalman_filter import *
import numpy as np
import cv2

from personnage_repo import PersonnagesCoordinatesRepo
from scorer import multidimensionalScorer


COLORS = [(0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 255), (0, 0, 127), (0, 127, 0), (127, 127, 0), (127, 0, 127), (127, 0, 0), (0, 127, 127), (127, 127, 127)]

class Personnages3D:

    _repo: PersonnagesCoordinatesRepo

    def __init__(self, FICHIER):
        """Unité = mm"""
        self._repo = PersonnagesCoordinatesRepo(12, multidimensionalScorer)

        self.json_data = read_json(FICHIER)

        self.pointCountHistory = 60
        self.overlaysCount = 10

        self.pause = False
        self.period = 6

        # Fenêtre pour la vue du dessus
        cv2.namedWindow('vue du dessus', cv2.WND_PROP_FULLSCREEN)
        black = np.zeros((720, 1280, 3), dtype = "uint8")
        
        # First create the image with alpha channel
        rgba = cv2.cvtColor(black, cv2.COLOR_RGB2RGBA)
        # Then assign the mask to the last channel of the image
        rgba[:, :, 3] = 0

        self.background = rgba.copy()
        self.transparentOverlay = rgba.copy()
        self.overlays = deque(maxlen=self.overlaysCount)

        # draw center line
        cv2.line(self.background, (0, 360), (1280, 360), (255, 255, 255), 2)

        screenScale = 160/1000
        # draw horizontal lines
        for k in range(0, 720, int(1000*screenScale)):
            cv2.line(self.background, (0, k), (1280, k), (255, 255, 255), thickness=1)

        # draw vertical lines
        for k in range(0, 1280, int(1000*screenScale)):
            cv2.line(self.background, (k, 0), (k, 720), (255, 255, 255), thickness=1)

        

        self.loop = True
        # Numéro de la frame
        self.frame = 0

    def _adapt_coordinates(self, coords):
        x = 360 + int(coords[0]*160/1000)
        y = int(coords[1]*160/1000)
        return (y, x)

    def _get_current_overlay(self):
        return self.overlays[0]

    def _draw_circle(self, coords, radius, color, thickness):
        overlay = self._get_current_overlay()
        circle = cv2.circle(overlay, coords, radius, color, thickness)
        return circle

    def _draw_square(self, coords, halfSide, color, thickness):
        overlay = self._get_current_overlay()
        p1 = (coords[0] - halfSide, coords[1] - halfSide)
        p2 = (coords[0] + halfSide, coords[1] + halfSide)
        rect = cv2.rectangle(overlay, p1, p2, color, thickness)
        return rect

    def draw_all_personnages(self):
        """Dessin des centres des personnages dans la vue de dessus,
        représenté par un cercle
        """

        for perso in self._repo.getCurrentIterationPersonages():
            overlay = self._get_current_overlay()

            #print("DEBUG: coord #%d : %s" % (perso.uid, perso.coordinate))
            color = (128,128,128)
            if perso.confirmed:
                color = COLORS[(perso.uid - 1) % 7]
            coords = self._adapt_coordinates(perso.coordinate)
            cv2.circle(overlay, coords, 3, (100, 100, 100), cv2.FILLED)
            self._draw_circle(coords, 3, color, 1)

            rectangleHalfSize = 8
            rectangleBorderSize = 1
            if perso.kfSmoothedCoordinates is not None:
                coords = self._adapt_coordinates(perso.kfSmoothedCoordinates)
                rect = self._draw_square(coords, int(rectangleHalfSize), color, rectangleBorderSize)
                cv2.putText(overlay, str(perso.uid), (coords[0]-rectangleHalfSize-rectangleBorderSize, coords[1]-rectangleHalfSize-rectangleBorderSize-1), 1, 1, color)

            rectangleHalfSize = 5
            if perso.kfPredictedCoordinates is not None:
                coords = self._adapt_coordinates(perso.kfPredictedCoordinates)
                self._draw_square(coords, rectangleHalfSize, color, 1)

    def run(self):
        """Boucle infinie, quitter avec Echap dans la fenêtre OpenCV"""

        while self.loop:
            if self.frame % int(self.pointCountHistory / self.overlaysCount) == 0:
                # Create new transparent overlay
                self.overlays.appendleft(self.transparentOverlay.copy())

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

            #cv2.imshow('vue du dessus', self.background)
            #mixedImage = cv2.addWeighted(self.background, 0.5, self.overlay, 1, 0)

            mixedImage = self.background
            for overlay in self.overlays:
                #mixedImage = cv2.addWeighted(mixedImage, 1, overlay, 1, 0)
                mixedImage = mixedImage + overlay
            
            # mixedImage = cv2.addWeighted(self.background, 0.5, self.overlays[0], 1, 0)    
            cv2.imshow('vue du dessus', mixedImage)

            self.frame += 1

            if self.pause:
                self._waitPeriodKeyboardAware(2**30)
            else:
                self._waitPeriodKeyboardAware(2**self.period)

        cv2.destroyAllWindows()

    def _waitPeriodKeyboardAware(self, periodInMs):
        startTime = datetime.now()
        period = timedelta(milliseconds=abs(periodInMs))
        k = None
        while startTime + period > datetime.now():
            k = cv2.waitKey(1)
            if k == 32: # Space key
                self.pause = not self.pause
                break
                
            elif k == 81:   # Left arrow key
                self.pause = True
                break

            elif k == 83:   # Right arrow key
                self.pause = True
                break

            elif k == 45:   # Minus key
                # Reduce speed
                self.period = min(self.period+1, 10)

            elif k == 43:   # Plus key
                # Increase speed
                self.period = max(self.period-1, 1)

            if k == 27: # Escape key
                self.loop = False
                break
        return k



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
