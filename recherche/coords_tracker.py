

"""
Tracking and display of move from mutliple simultaneous coordinates.
"""

import argparse
import asyncio
import sys

from network_utils import runSkelet3dFileNetPusher, runSkelet3dNetReader, startNewBackgroundEventLoop
from personnages3d import Personnages3D
from utils import read_json_file

async def runFileInputTracker(periodInSec, jsonFile):
    print(f"Starting Tracker with file input.", file=sys.stderr)
    p3d = Personnages3D()

    async def processFile():
        while p3d.loop:
            for skelet3d in read_json_file(jsonFile):
                print(f"Recording skelet3d from file: {skelet3d}", file=sys.stderr)
                if p3d.loop:
                    p3d.recordSkelet3D(skelet3d)
                await asyncio.sleep(delay=periodInSec)

    loop = startNewBackgroundEventLoop()

    asyncio.run_coroutine_threadsafe(processFile(), loop)

    # futures = [
    #     #asyncio.run_coroutine_threadsafe(p3d.waitLoopEnd(), loop),
    #     asyncio.run_coroutine_threadsafe(runSkelet3dNetReader('localhost', 55550, onSkeletonReception), loop),
    #     asyncio.run_coroutine_threadsafe(runSkelet3dFileNetPusher('localhost', 55550, jsonFile, periodInSec), loop),
    # ]

    p3d.run()

async def runNetworkInputTracker(host, port):
    print(f"Starting Tracker with network input.", file=sys.stderr)
    p3d = Personnages3D()

    def onSkeletonReception(skelet3d):
        if p3d.loop:
            p3d.recordSkelet3D(skelet3d)

    loop = startNewBackgroundEventLoop()

    futures = [
        asyncio.run_coroutine_threadsafe(runSkelet3dNetReader(host, port, onSkeletonReception), loop),
    ]

    p3d.run()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Start a graphical coordinates tracker.')
    parser.add_argument('file', nargs='?', default='./json/cap_7.json', type=str, help="File to send line by line")
    parser.add_argument('--host', dest="host", type=str, default="localhost" ,help='net address')
    parser.add_argument('--port', dest="port", type=int, default=55555, help='net port')
    parser.add_argument('--period', dest="period", type=float, default=0.01, help='Sending period in seconds')
    parser.add_argument('--network', dest="network", action='store_const', default=False, const=True, help='Read coordinates from network')
    
    args = parser.parse_args()
    host = args.host
    port = args.port
    periodInSec = args.period
    textFile = args.file

    if textFile == "-":
        textFile = sys.stdin

    if args.network:
        asyncio.run(runNetworkInputTracker(host, port))
    else:
        asyncio.run(runFileInputTracker(periodInSec, textFile))