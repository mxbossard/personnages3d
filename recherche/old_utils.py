
# Deprecated
from typing import Sequence, Tuple


def smoothCartesianCoordinate(coordinatesList: Sequence[Tuple[float, float, float]], history:int, windowSize:int) -> Tuple[float, float, float]:
    """ coordinate smoothing algorithm: averaging data """
    xSmoothed = 0
    ySoothed = 0
    zSmoothed = 0
    avgPoint = (xSmoothed, ySoothed, zSmoothed)

    if len(coordinatesList) > 0:
        coordsCount = 0
        coordinates = coordinatesList[history:history + windowSize]
        for coords in coordinates:
            xSmoothed += coords[0]
            ySoothed += coords[1]
            zSmoothed += coords[2]
            coordsCount += 1

        xSmoothed /= coordsCount
        ySoothed /= coordsCount
        zSmoothed /= coordsCount

        avgPoint = (xSmoothed, ySoothed, zSmoothed)
    
    return avgPoint
