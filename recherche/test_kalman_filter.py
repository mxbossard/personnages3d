import pytest
import random
from kalman_filter import *

def test2dKalmanFilter():
    #positions = [(0,0), (1,1), (3,3), (7,6), (10, 11)]
    #positions = [(i,i) for i in range(20)]
    positions = [(0, 0)] + [(i + random.randint(-100, 100)/100, i + random.randint(-100, 100)/100) for i in range(30)]

    kf = init2dKalmanFilter(positions[0], (0,0), 1, 2, 1)
    assert (0, 0) == predict2dKF(kf)

    for coord in positions[1:]:
        predicted = predict2dKF(kf)
        print(f"Predicted coordinates: [{predicted}]")

        corrected = update2dKF(kf, coord)
        print(f"New coordinates: [{coord}] => corrected coordinates: [{corrected}]")
        