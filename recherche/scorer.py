import numpy as np

from distances import DistanceMatrix, PersonnageDistance
from config import MAX_ACTIVITY_SCORE, MAX_HISTORY_SCORE, NOT_CONFIRMED_MALUS_DISTANCE
from utils import weightedScore

def naive2dDistanceScorer(dist: PersonnageDistance, matrix: DistanceMatrix):
    """ Return a small score if 2d distance is small. """
    return dist.cartesianDistance()

def naive2dSpeedScorer(dist: PersonnageDistance, matrix: DistanceMatrix):
    """ Return a small score if 2d speed distance is small. """
    return dist.speedDistance()

def historyScorer(dist: PersonnageDistance, matrix: DistanceMatrix):
    """ Return a small score if distance history is high. """
    step = MAX_HISTORY_SCORE / 10
    # if distance is doubled => add step points
    return max(0, MAX_HISTORY_SCORE - np.log2(dist.history) * step)

def activityScorer(dist: PersonnageDistance, matrix: DistanceMatrix):
    """ Return a small score if recent activity is high. """
    return MAX_ACTIVITY_SCORE * (1 - dist.activity)

def ghostPruner(dist: PersonnageDistance, matrix: DistanceMatrix):
    score = 0
    # p1 is never confirmed by design
    if not dist.p2.confirmed:
        score += NOT_CONFIRMED_MALUS_DISTANCE
    return score

def multidimensionalScorer(dist: PersonnageDistance, matrix: DistanceMatrix):
    cartesianDist = naive2dDistanceScorer(dist, matrix)
    ghostScore = ghostPruner(dist, matrix)
    speedDist = naive2dSpeedScorer(dist, matrix)
    historyScore = historyScorer(dist, matrix)
    activityScore = activityScorer(dist, matrix)

    score = weightedScore(cartesianDist, speedDist, activityScore, historyScore)
    #score += ghostScore

    #print(f"DEBUG: score={score} from dist={cartesianDist} speed={speedDist} activity={activityScore} history={historyScore}")
    
    return score
