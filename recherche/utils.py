
from typing import Tuple


def vectorNorme(vector: Tuple[float, float, float]) -> float:
    if vector is None:
        return 0
    return ((vector[0])**2 + (vector[1])**2 + (vector[2])**2)**0.5

def vectorDelta(vector1: Tuple[float, float, float], vector2: Tuple[float, float, float], frameDelta = 1) -> Tuple[float, float, float]:
    if frameDelta == 0:
        #FIXME: is it a good response ?
        return (0, 0, 0)
    return ((vector1[0]-vector2[0]) / frameDelta, (vector1[1]-vector2[1]) / frameDelta, (vector1[2]-vector2[2]) / frameDelta)

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
