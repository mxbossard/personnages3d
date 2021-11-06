


import argparse
import sys
import time
from typing import List, Tuple
import random

MIN_X = - int(360 * 1000/160)
MAX_X = - MIN_X
MIN_Y = -1000
MAX_Y = 1000
MIN_Z = 100
MAX_Z = int(1280 * 1000/160)

def genRandomCoords() -> List[int]:
    x =  random.randint(MIN_X, MAX_X)
    y =  random.randint(MIN_Y, MAX_Y)
    z =  random.randint(MIN_Z, MAX_Z)
    return [x, y, z]

def getRandoCoordsAround(center: List[int], radius: int) -> List[int]:
    minX = center[0] - radius
    maxX = center[0] + radius
    minY = center[1] - radius
    maxY = center[1] + radius
    minZ = center[2] - radius
    maxZ = center[2] + radius
    x =  random.randint(minX, maxX)
    y =  random.randint(minY, maxY)
    z =  random.randint(minZ, maxZ)
    return [x, y, z]


def genSkeletonPosenetCoords() -> List:
    """ Generate an array of 17 coordinates for one skeleton. """
    # The 17 coordinates must be around a single point.
    center = genRandomCoords()
    radius = 500
    skeletonCoords = [center]
    for i in range(0, 16):
        skeletonCoords.append(getRandoCoordsAround(center, radius))
    return skeletonCoords


def main():

    parser = argparse.ArgumentParser(description='Start a graphical coordinates tracker.')
    parser.add_argument('count', nargs='?', default=5, type=int, help="Skeletons count")
    parser.add_argument('--period', dest="period", type=float, default=0.1, help='Generating period in seconds')
    
    args = parser.parse_args()
    periodInSec = args.period
    skeletonCount = args.count

    if len(sys.argv) > 1:
        skeletonCount = int(sys.argv[1])
    
    while True:
        skeletons = []
        for i in range(0, skeletonCount):
            skeleton = genSkeletonPosenetCoords()
            skeletons.append(skeleton)

        #print(f"Generated skeletons: {skeletons}", file=sys.stderr)
        print(f"{skeletons}")
        sys.stdout.flush()
        time.sleep(periodInSec)


if __name__ == '__main__':
    main()
