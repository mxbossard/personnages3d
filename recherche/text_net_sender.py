
import asyncio
import argparse
import sys

from network_utils import runSkelet3dFileNetPusher


async def main(host, port, periodInSec, textFile):
    await runSkelet3dFileNetPusher(host, port, textFile, periodInSec)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Start network client and send text data.')
    parser.add_argument('file', nargs=1, type=str, help="File to send line by line")
    parser.add_argument('--host', dest="host", type=str, default="localhost" ,help='net address')
    parser.add_argument('--port', dest="port", type=int, default="55555", help='net port')
    parser.add_argument('--period', dest="period", type=float, default="0.5", help='Sending period in seconds.')
    
    args = parser.parse_args()
    host = args.host
    port = args.port
    periodInSec = args.period
    textFile = args.file[0]

    if textFile == "-":
        textFile = sys.stdin

    asyncio.run(main(host, port, periodInSec, textFile))
