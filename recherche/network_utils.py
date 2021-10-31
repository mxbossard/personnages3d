import json
import asyncio
from asyncio.streams import StreamReader, StreamWriter

from utils import read_json


async def run_server(client_connected_cb, host='localhost', port=55555):
    server = await asyncio.start_server(client_connected_cb, host, port)
    async with server:
        await server.serve_forever()

async def build_client(host='localhost', port=55555, retries=-1):
    """ Build a client a network client with retrying capability. """
    attempt = 0
    while attempt != retries:
        try:
            attempt += 1
            print("Attempt connection ...")
            return await asyncio.open_connection(host, port)
        except ConnectionRefusedError as e:
            if attempt -1 != retries:
                await asyncio.sleep(1)
            else:
                raise e
    raise Exception('Unable to connect')


async def runSkelet3dFileNetPusher(host, port, jsonFile, periodInSec):
    """ Read all skelet3d from a json file and push it to a network stream at a specified frequency. """
    writer: StreamWriter
    reader, writer = await build_client(host, port)
    print(f"Reading json file: {jsonFile}")
    json_data = read_json(jsonFile)

    if json_data and len(json_data) > 0:
        for skelet_3D in json_data:
            if skelet_3D:
                data = json.dumps(skelet_3D) + '\n'
                #print(f"Sending data: {data}")
                writer.write(data.encode('ascii'))
                await asyncio.sleep(periodInSec)

def printData(obj):
    print(f"Received object: {obj}")

async def runSkelet3dNetReader(host, port, callback):
    """ Launch a daemon which read skelet3d from json file and push it to a network stream. """
    async def onConnect(reader: StreamReader, writer: StreamWriter):
        print(f"New client connected")
        while True:
            data = await reader.readline()
            obj = json.loads(data)
            callback(obj)

    print("Starting network server")
    await run_server(onConnect, host, port)
