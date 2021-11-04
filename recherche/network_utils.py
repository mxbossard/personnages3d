from asyncio.tasks import Task
import json
import asyncio
from asyncio.streams import StreamReader, StreamWriter
import sys
import threading

from utils import read_json_file


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
            print(f"Attempting connection to {host}:{port} ...", file=sys.stderr)
            return await asyncio.open_connection(host, port)
        except ConnectionRefusedError as e:
            if attempt -1 != retries:
                await asyncio.sleep(1)
            else:
                raise e
    raise Exception('Unable to connect')

async def runSkelet3dFileNetPusher(host, port, jsonFile, periodInSec):
    """ Read all skelet3d from a json file and push it to a network stream at a specified frequency. """
    reader, writer = await build_client(host, port)
    
    json_data = read_json_file(jsonFile)

    try:
        if json_data and len(json_data) > 0:
            for skelet_3D in json_data:
                data = json.dumps(skelet_3D) + '\n'
                #print(f"Sending data: {data}")
                sent = False
                while not sent:
                    try:
                        if writer.is_closing():
                            raise Exception
                        writer.write(data.encode('ascii'))
                        sent = True
                    except:
                        print("Connection error ! Trying to reconnect.", file=sys.stderr)
                        writer.close
                        reader, writer = await build_client(host, port)
                await asyncio.sleep(periodInSec)
    finally:
        writer.close()

async def runSkelet3dNetReader(host, port, callback):
    """ Launch a daemon which read skelet3d from json file and push it to a network stream. """
    async def onConnect(reader: StreamReader, writer: StreamWriter):
        print(f"New client connected", file=sys.stderr)
        try:
            while True:
                data = await asyncio.wait_for(reader.readline(), timeout=10.0)
                if data == b'':
                    print("Reached EOF.", file=sys.stderr)
                    break
                elif data:
                    obj = json.loads(data)
                    callback(obj)
        except asyncio.TimeoutError:
            print("Timeout detected.", file=sys.stderr)
        finally:
            print("Connection closed.", file=sys.stderr)
            writer.close()

    print(f"Starting network server listening on {host}:{port}", file=sys.stderr)
    await run_server(onConnect, host, port)


def startBackgroundLoop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()

def startNewBackgroundEventLoop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=startBackgroundLoop, args=(loop,), daemon=True)
    thread.start()
    return loop