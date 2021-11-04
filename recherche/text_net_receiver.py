
import asyncio
import argparse
import json

from network_utils import runSkelet3dNetReader


async def main(host, port):
    def jsonPrinter(skelet3d):
        print(json.dumps(skelet3d))

    await runSkelet3dNetReader(host, port, jsonPrinter)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Start network daemon and print received text data.')
    parser.add_argument('--host', dest="host", type=str, default="localhost" ,help='net address')
    parser.add_argument('--port', dest="port", type=int, default="55555", help='net port')
    
    args = parser.parse_args()
    host = args.host
    port = args.port

    asyncio.run(main(host, port))
