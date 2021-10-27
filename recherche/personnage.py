

from collections import deque
from typing import Mapping, Sequence, Tuple
from config import ACTIVITY_HISTORY_SIZE, MAX_FRAME_TO_CONFIRM_ELSE_GHOST, MAX_MISSING_FRAME_BEFORE_DEAD, MIN_SAMPLE_CONFIRMED_THRESHOLD
from kalman_filter import init2dKalmanFilter, predict2dKF, update2dKF
from utils import vectorDelta, weightedScore


class PersonnageData:

    def __init__(self, frame, x, y, z):
        self.frame = frame
        self.coordinate = (x, y, z) # Cartesian coordinate
        self.uid: int = None
        self.color: tuple = None
        self.frameAppearanceCount = 1
        # Score qui indique l'activite recente du personnage. Proche de 0 si peu actif. Proche de 1 si tres actif.
        self.activity = 0
        # Queue containing 1 for each recent active frame and 0 for each inactive frame
        self.activityQueue = deque([0] * ACTIVITY_HISTORY_SIZE, maxlen=ACTIVITY_HISTORY_SIZE)
        self.frameHistory = [frame]
        self.coordinatesHistory: Sequence[Tuple[float, float, float]] = [self.coordinate]
        self.smoothedCoordinatesHistory: Sequence[Tuple[float, float, float]] = [self.coordinate]
        self.confirmed = False
        self.smoothedSpeedHistory = []
        self.smoothedAccelerationHistory = []
        self.kf = init2dKalmanFilter((x, y), (0, 0), 1, 1, 1)
        self.kfSmoothedCoordinates = None
        self.kfPredictedCoordinates = None

    def __str__(self):
        data = ""
        if len(self.smoothedSpeedHistory) > 0:
            data += f"speed={self.smoothedSpeedHistory[0]} "
        if len(self.smoothedAccelerationHistory) > 0:
            data += f"accel={self.smoothedAccelerationHistory[0]} "

        return f"[Personnage #{self.uid} instantCoord={self.coordinate} coord={self.smoothedCoordinatesHistory[0]} {data}]"

    def initializeUid(self, uid):
        if self.uid is None:
            self.uid = uid

    def getSoothedCoordinates(self, history:int=0) -> Tuple[float, float, float]:
        #return smoothCartesianCoordinate(self.coordinatesHistory, history, self.smoothingWindow)
        return self.smoothedCoordinatesHistory[0]

    def succeedTo(self, p):
        if p is not None:
            self.uid = p.uid
            self.frameAppearanceCount = p.frameAppearanceCount + 1
            self.kf = p.kf
            kfSmoothedCoordinates = update2dKF(self.kf, (self.coordinate[0], self.coordinate[1]))
            self.kfSmoothedCoordinates = (kfSmoothedCoordinates[0], kfSmoothedCoordinates[1], 0)
            kfPredictedCoordinates = predict2dKF(self.kf)
            self.kfPredictedCoordinates = (kfPredictedCoordinates[0], kfPredictedCoordinates[1], 0)

            # Historical data
            self.frameHistory = self.frameHistory + p.frameHistory
            #print("DEBUG: Frame history: %s." % (self.frameHistory))
            self.coordinatesHistory = self.coordinatesHistory + p.coordinatesHistory

            #smoothedCoordinate = smoothCartesianCoordinate(self.coordinatesHistory, 0, self.smoothingWindow)
            smoothedCoordinate = self.kfSmoothedCoordinates
            self.smoothedCoordinatesHistory.insert(0, smoothedCoordinate)

            if (len(p.smoothedCoordinatesHistory) > 1):
                frameDelta = self.frameHistory[0] - self.frameHistory[1]
                smoothedSpeed = vectorDelta(self.coordinate, p.smoothedCoordinatesHistory[0], frameDelta)
                self.smoothedSpeedHistory = [smoothedSpeed] + p.smoothedSpeedHistory # insert at list start
                #print("DEBUG: smoothedSpeedHistory: %s." % (self.smoothedSpeedHistory))
            
            if (len(p.smoothedSpeedHistory) > 1):
                smoothedAcceleration = vectorDelta(self.smoothedSpeedHistory[0], p.smoothedSpeedHistory[0], frameDelta)
                self.smoothedAccelerationHistory = [smoothedAcceleration] + p.smoothedAccelerationHistory # insert at list start

            self.activityQueue = p.activityQueue

            #FIXME: confirmed based on frame size + not to hold frames
            self.confirmed = len(self.frameHistory) > MIN_SAMPLE_CONFIRMED_THRESHOLD

            #print("DEBUG: Perso succession: %s." % (self))

    def refreshActivity(self, iteration):
        if self.frame == iteration:
            # Add a 1 in activity queue
            self.activityQueue.appendleft(1)
        else:
            self.activityQueue.appendleft(0)
        self.activity = weightedScore(*self.activityQueue) # expand list to function args
        # print("DEBUG: Perso activity: %s." % (self.activityQueue))
        # print("DEBUG: Perso activity: %f." % (self.activity))

    def isAlive(self, iteration):
        notTimeout = iteration - self.frame < MAX_MISSING_FRAME_BEFORE_DEAD
        
        # Filter Ghost
        isGhost = not self.confirmed and self.frameHistory[len(self.frameHistory)-1] + MAX_FRAME_TO_CONFIRM_ELSE_GHOST < iteration
        if isGhost:
                print("DEBUG: Detected ghost perso #%d." % (self.uid))

        return notTimeout and not isGhost

    def label(self):
        return "Personnage_%d" % self.uid
