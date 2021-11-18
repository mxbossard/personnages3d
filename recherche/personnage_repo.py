

from types import FunctionType
from typing import Mapping, Sequence, Tuple
from distances import minimalDistanceOverallPathFinder2, naiveMinimalDistancePathFinder
from personnage import PersonnageData

class PersonnagesCoordinatesRepo:

    def __init__(self, maxPersonnages: int, scorer: FunctionType):
        self.reset()
        self.__maxPersonnages = maxPersonnages
        self.scorer = scorer

    def reset(self):
        self.__repo = {}
        self.__seenPersoCounter = 0
        self.__persoTracker: Mapping[int, PersonnageData] = {}
        self.__currentFrame = 0
        self.__persoUids = set()

    def _getIterationRepo(self, iteration: int) -> dict:
        repo = self.__repo.get(iteration, {})
        if not repo:
            self.__repo[iteration] = repo
        return repo

    def _getIterationPersonnageData(self, iteration: int) -> list:
        repo = self._getIterationRepo(iteration)
        personnages = repo.get("personnages", [])
        if not personnages:
            repo["personnages"] = personnages
        return personnages

    def _addIterationPersonnageData(self, iteration: int, persoData: PersonnageData):
        persoRepo = self._getIterationPersonnageData(iteration)
        persoRepo.append(persoData)
        self.__persoTracker[persoData.uid] = persoData 

    def _newPersoUid(self):
        self.__seenPersoCounter += 1
        uid = 1
        while uid in self.__persoUids:
            uid += 1
        self.__persoUids.add(uid)
        return uid

    def _recordNewPerso(self, perso: PersonnageData):
        newUid = self._newPersoUid()
        perso.initializeUid(newUid)
        self._addIterationPersonnageData(self.__currentFrame, perso)
        print("DEBUG: Recorded new perso %s at frame: %d." % (perso, self.__currentFrame))

    def _recordPersoUpdate(self, newPerso: PersonnageData, previousPerso: PersonnageData):
        newPerso.succeedTo(previousPerso)
        self._addIterationPersonnageData(self.__currentFrame, newPerso)
        #print("DEBUG: Recorded perso update %s => %s." % (previousPerso, newPerso))

    def _forgetPerso(self, perso: PersonnageData):
        print("DEBUG: Removing perso #%d." % (perso.uid))
        self.__persoUids.discard(perso.uid)
        self.__persoTracker.pop(perso.uid)

    def _frameCleaning(self, iteration):
        self.__currentFrame = iteration
        for persoUid in list(self.__persoTracker.keys()):
            perso = self.__persoTracker[persoUid]

            if not perso.isAlive(iteration):
                self._forgetPerso(perso)


    # def newIteration(self, iteration):
    #     self._currentFrame = iteration

    def getAllPersonages(self) -> Sequence[PersonnageData]:
        return list(self.__persoTracker.values())
    
    def getCurrentIterationPersonages(self) -> Sequence[PersonnageData]:
        personnages = []
        for perso in self.getAllPersonages():
            if perso.frame == self.__currentFrame:
                personnages.append(perso)
        return personnages

    def addNewFrameCoordinates(self, iteration: int, coordinates: list):
        self._frameCleaning(iteration)

        #print("DEBUG: recording frame #%d coordinates [%s] ..." % (iteration, coordinates))

        newPersos: Sequence[PersonnageData] = []
        for coord in coordinates:
            (x, y, z) = coord
            newPerso = PersonnageData(iteration, x, y, z)
            newPersos.append(newPerso)
        
        #k = 1
        unknowPersoRemaining = True
        #while(unknowPersoRemaining and iteration - k > 0 and k < MAX_FRAME_PASSE_A_SCANNER):
        #previousPersos = self._getIterationPersonnageData(iteration - k)
        previousPersos = self.getAllPersonages()

        betterPath = minimalDistanceOverallPathFinder2(newPersos, previousPersos, self.scorer)
        #betterPath = naiveMinimalDistancePathFinder(newPersos, previousPersos, self.scorer)

        for dist in betterPath:
            newPerso = dist.p1
            previousPerso = dist.p2
            self._recordPersoUpdate(newPerso, previousPerso)
            newPersos.remove(newPerso)

        if len(newPersos) == 0:
            unknowPersoRemaining = False
        
        if unknowPersoRemaining:
            #  Some perso cannot be matched, infer a not seen previously perso
            for perso in newPersos:
                self._recordNewPerso(perso)

            print("DEBUG: Coordinates which lead to new personage recording: [%s]." % (coordinates))

        for perso in self.getAllPersonages():
            perso.refreshActivity(iteration)
            # if perso.frame != iteration:
            #     print("DEBUG: One tracked Personnage was not updated: [%s]." % (perso))
            #     #update2dKF(perso.kf, None)
            # # perso.decayFreshness(iteration)
