import io
import json
from json.decoder import JSONDecodeError
import sys
from typing import Tuple
import typing


def vectorNorme(vector: Tuple[float, float, float]) -> float:
    if vector is None:
        return 0
    return ((vector[0])**2 + (vector[1])**2 + (vector[2])**2)**0.5

def vectorDelta(vector1: Tuple[float, float, float], vector2: Tuple[float, float, float], frameDelta = 1) -> Tuple[float, float, float]:
    if frameDelta == 0:
        #FIXME: is it a good response ?
        return (0, 0, 0)
    return ((vector1[0]-vector2[0]) / frameDelta, (vector1[1]-vector2[1]) / frameDelta, (vector1[2]-vector2[2]) / frameDelta)

def weightedScore(*scores: float) -> float:
    scoreCount = len(scores)
    if scoreCount == 0:
        return None
    scoreSum = 0
    weightSum = 0
    for k, score in enumerate(scores):
        weight = -2*k/((scoreCount + 1)**2) + 2/(scoreCount + 1)
        weightSum += weight
        scoreSum += score * weight
    return scoreSum / weightSum

def read_json_stream(fichier):
    print(f"Attempting to read json stream from file: {fichier}", file=sys.stderr)
    data = None
    lines = None
    if type(fichier) == str:
        with open(fichier) as f:
            lines = f.readlines()
    elif isinstance(fichier, io.IOBase):
        lines = fichier.readlines()

    if lines:
        data = [ json.loads(line) for line in lines ]
    
    print(f"Finished reading json stream from file: {fichier}", file=sys.stderr)

    return data

def read_json_file(fichier):
    data = None
    try:
        if type(fichier) == str:
            print(f"Attempting read all json file: {fichier}", file=sys.stderr)
            with open(fichier) as f:
                try:
                    # Try to load all file
                    data = json.load(f)
                except JSONDecodeError as e:
                    # Error decoding, the file may be a json stream
                    data = read_json_stream(fichier)
                
        elif isinstance(fichier, io.IOBase):
            print(f"Attempting to real all json file from stdin", file=sys.stderr)
            try:
                # Try to load all file
                data = json.load(fichier)
            except JSONDecodeError as e:
                # Error decoding, the file may be a json stream
                data = read_json_stream(fichier)
        else:
            raise Exception(f"Unable to read file {fichier}")

    except Exception as e:
        print("Fichier inexistant ou impossible à lire: %s" % e)
    return data


def get_center(points_3D):
    """Le centre est le centre de vue du dessus,
    c'est la moyenne des coordonées des points du squelette d'un personnage,
    sur x et z
    """

    center = []
    if points_3D:
        for i in [0, 2]:
            center.append(get_moyenne(points_3D, i))

    return center


def get_moyenne(points_3D, indice):
    """Calcul la moyenne d'une coordonnée des points, d'un personnage
    la profondeur est le 3 ème = z, le y est la verticale
    indice = 0 pour x, 1 pour y, 2 pour z
    """

    somme = 0
    n = 0
    for i in range(17):
        if points_3D[i]:
            n += 1
            somme += points_3D[i][indice]
    if n != 0:
        moyenne = int(somme/n)
    else:
        moyenne = None

    return moyenne
