from typing import Tuple
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

# Explainations about Kalman Filter: http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
# Use an example to configure : https://github.com/andrewadare/kalman-tracker/blob/db411a408277d420e398844e5b5ee40165a68a7d/tracked_point.py
def init2dKalmanFilter(initialPosition: Tuple[float, float], initialSpeed:Tuple[float, float], uncertainty: float, measurementNoise: float, dt: float = 1) -> KalmanFilter:
    """ See the documentation: https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html """

    # dim_x=4 => position and velocity are accounted in 2 dimensions
    # dim_z=2 => position and velocity are exprimed in 2 dimensions
    dim_x = 4 
    dim_z = 2
    noise_variance = 0.13
    kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z, dim_u=0)
    
    # Assign initial value. (dim_x, 1), default = [0,0,0…0].
    kf.x = np.array([initialPosition[0], initialSpeed[0], initialPosition[1], initialSpeed[1]])

    # Define state transition matrix. (dim_x, dim_x).
    kf.F = np.array([[1, dt, 0, 0],  # new_position_0 = 1*old_position_0 + dt*old_speed_0
                    [0, 1, 0, 0],   # new_speed_0 = 0*old_position_0 + 1*old_speed_0
                    [0, 0, 1, dt],  # new_position_1 = 1*old_position_1 + dt*old_speed_1
                    [0, 0, 0, 1]],  # new_speed_1 = 0*old_position_1 + 1*old_speed_1
                    dtype=float)

    # Define the measurement function. (dim_z, dim_x)
    kf.H = np.zeros([kf.dim_z, kf.dim_x])
    kf.H[0, 0] = 1 # Keep position_0 as first data
    kf.H[1, 2] = 1 # Keep position_1 as second data

    # Define the covariance matrix. (dim_x, dim_x), default eye(dim_x). Take advantage of the fact that P already contains an identity matrix.
    kf.P *= uncertainty

    # Assign the measurement noise. (dim_z, dim_z), default eye(dim_x).
    kf.R *= measurementNoise**2

    # Assign the process noise. (dim_x, dim_x), default eye(dim_x).
    #kf.Q = Q_discrete_white_noise(dim=dim_x, dt=dt, var=noise_variance)
    q = Q_discrete_white_noise(dim=2, dt=dt, var=noise_variance**2)
    kf.Q = block_diag(q, q)

    return kf

def position2dKF(kf: KalmanFilter) -> Tuple[float, float]:
    return (kf.x[0], kf.x[2])

def predict2dKF(kf: KalmanFilter) -> Tuple[float, float]:
    kf.predict()
    return position2dKF(kf)

def update2dKF(kf: KalmanFilter, position: Tuple[float, float]) -> Tuple[float, float]:
    if position is not None:
        kf.update([position[0], position[1]])
    else:
        kf.update(None)
    return position2dKF(kf)