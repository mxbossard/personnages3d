

"""
Reconnaissance des personnages à partir d'un json

Vous ne devez modifier que ce fichier
Installation: voir le readme

TODO:
- Forger un vrai bon score de activity/historique   DONE
- Effacer les vieux tracés      DONE
- Pause/Resume avec espace      DONE
- Right arrow => pause + one step forward   DONE
- Change speed with +- keys     DONE
- Display uid   DONE
- Display legend of personnages with properties (confirmed, activity, ...)  DONE

- Vérifier les calculs de vitesse et d'accélération
- Remplacer les smooth coodinates par les prediction du kalman filter   TO_CHECK
- Implement max history data retained
- Left arrow => pause - one step backward
- Square size proportional to uncertainty ???

"""

import asyncio
from asyncio.streams import StreamReader, StreamWriter
from asyncio.tasks import FIRST_COMPLETED
from collections import deque
import json
from datetime import datetime, timedelta
import sys
from typing import Deque, Mapping, MutableSequence, Sequence, Tuple
from kalman_filter import *
import threading
import numpy as np
import cv2

from personnage_repo import PersonnagesCoordinatesRepo
from network_utils import runSkelet3dFileNetPusher, runSkelet3dNetReader, startNewBackgroundEventLoop
from utils import get_center
from scorer import multidimensionalScorer


COLORS = [(0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 255), (0, 0, 127), (0, 127, 0), (127, 127, 0), (127, 0, 127), (127, 0, 0), (0, 127, 127), (127, 127, 127)]

class Personnages3D:

    _repo: PersonnagesCoordinatesRepo

    def __init__(self):
        """Unité = mm"""
        self._repo = PersonnagesCoordinatesRepo(12, multidimensionalScorer)

        self.displayedPointCountHistory = 60
        self.overlaysCount = 10

        self.pause = False
        self.waitPeriodFactor = 0

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

        self.newOverlay = False
        self.mutex = threading.Lock()

    def _adapt_coordinates(self, coords):
        x = 360 + int(coords[0]*160/1000)
        y = int(coords[1]*160/1000)
        return (y, x)

    def _get_current_overlay(self):
        return self.overlays[0]

    def _draw_square(self, coords, halfSide, color, thickness):
        overlay = self._get_current_overlay()
        p1 = (coords[0] - halfSide, coords[1] - halfSide)
        p2 = (coords[0] + halfSide, coords[1] + halfSide)
        rect = cv2.rectangle(overlay, p1, p2, color, thickness)
        return rect

    def _draw_perso_stats(self):
        cv2.rectangle(self.background, (0, 0), (1280, 20), (0, 0, 0), cv2.FILLED)
        shift = 0
        for p in self._repo.getAllPersonages():
            color = (128,128,128)
            if p.confirmed:
                color = COLORS[(p.uid - 1) % 7]
            message=f"#{p.uid} {round(p.activity, 2)}"
            cv2.putText(self.background, message, (5 + shift, 15), 1, 1, color)
            shift += 80

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
            #cv2.circle(overlay, coords, 3, (100, 100, 100), cv2.FILLED)
            cv2.circle(overlay, coords, 2, color, cv2.FILLED)

            rectangleHalfSize = 6
            rectangleBorderSize = 1
            if perso.kfSmoothedCoordinates is not None:
                coords = self._adapt_coordinates(perso.kfSmoothedCoordinates)
                rect = self._draw_square(coords, int(rectangleHalfSize), color, rectangleBorderSize)
                if len(perso.frameHistory) % 10 == 2:
                    textCoords = (coords[0]-rectangleHalfSize-rectangleBorderSize, coords[1]-rectangleHalfSize-rectangleBorderSize-1)
                    cv2.putText(overlay, str(perso.uid), textCoords, 1, 1, color)

            # rectangleHalfSize = 3
            # if perso.kfPredictedCoordinates is not None:
            #     coords = self._adapt_coordinates(perso.kfPredictedCoordinates)
            #     self._draw_square(coords, rectangleHalfSize, color, 1)


    def recordSkelet3D(self, skelet_3D):
        try:
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

            self.frame += 1
            self.newOverlay = False
        except Exception as e:
            print(f"Unable to add coordinates: [{skelet_3D}] ! Error was: {e}", file=sys.stderr)

    def run(self):
        while self.loop:
            #print(f"Personnages3D loop running ...")

            self.mutex.acquire()
            if not self.newOverlay and self.frame % int(self.displayedPointCountHistory / self.overlaysCount) == 0:
                # Create new transparent overlay
                self.overlays.appendleft(self.transparentOverlay.copy())
                self.newOverlay = True
            self.mutex.release()

            #print(f"Drawing personnages ...")
            self.draw_all_personnages()

            #print(f"Drawing stats ...")
            self._draw_perso_stats()

            #print(f"Mixing layers ...")
            mixedImage = self.background
            for overlay in self.overlays:
                #mixedImage = cv2.addWeighted(mixedImage, 1, overlay, 1, 0)
                mixedImage = mixedImage + overlay
            
            #print(f"Displaying image ...")
            # mixedImage = cv2.addWeighted(self.background, 0.5, self.overlays[0], 1, 0)  
            cv2.imshow('vue du dessus', mixedImage)

            if self.pause:
                self._waitPeriodKeyboardAware(2**30)
            else:
                self._waitPeriodKeyboardAware(2**self.waitPeriodFactor)

        cv2.destroyAllWindows()
        

    async def waitLoopEnd(self):
        while self.loop:
            #print(f"Waiting Personnages3D loop to end ...")
            await asyncio.sleep(0.4)

    def _waitPeriodKeyboardAware(self, periodInMs):
        #print(f"Waiting {periodInMs} ms ...")
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
                self.waitPeriodFactor = min(self.waitPeriodFactor+1, 10)

            elif k == 43:   # Plus key
                # Increase speed
                self.waitPeriodFactor = max(self.waitPeriodFactor-1, 0)

            if k == 27: # Escape key
                self.loop = False
                break
        return k
