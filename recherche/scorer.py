import numpy as np

from distances import DistanceMatrix, PersonnageDistance
from config import NOT_CONFIRMED_MALUS_DISTANCE
from utils import weightedScore

def naive2dDistanceScorer(dist: PersonnageDistance, matrix: DistanceMatrix):
    return dist.cartesianDistance()

def confident2dDistanceScorer(dist: PersonnageDistance, matrix: DistanceMatrix):
    relativeConfidance = (dist.p2.freshness / matrix.columnsFreshnessSum) + 0.1
    return dist.cartesianDistance() / relativeConfidance

def historical2dDistanceScorer(dist: PersonnageDistance, matrix: DistanceMatrix):
    return dist.cartesianDistance() / (dist.history + 1)

def ghostPruner(dist: PersonnageDistance, matrix: DistanceMatrix):
    score = 0
    # p1 is never confirmed by design
    if not dist.p2.confirmed:
        score += NOT_CONFIRMED_MALUS_DISTANCE
    return score

def multidimensionalScorer(dist: PersonnageDistance, matrix: DistanceMatrix):
    cartesianDist = naive2dDistanceScorer(dist, matrix)
    ghostScore = ghostPruner(dist, matrix)
    speedDist = dist.speedDistance()
    historySize = len(dist.p1.frameHistory) + len(dist.p2.frameHistory)
    historyScore = max(0, 1000 - np.log2(historySize)*100)

    #score = weightedScore(historyScore, freshnessScore, cartesianDist, speedDist)
    #score = weightedScore(cartesianDist, speedDist, historyScore, freshnessScore, )
    score = weightedScore(cartesianDist, speedDist)
    score += ghostScore
    #print(f"DEBUG: score={score} from dist={cartesianDist} speed={speedDist} history={historyScore}, freshness={freshnessScore}")
    
    return score
