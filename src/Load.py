from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
import re
import numpy as np
import pathlib
import openseespy.opensees as ops
import matplotlib.pyplot as plt

from . import OpsObject
from . import GlobalData
from . import Comp
from . import Part
from . import UtilTools
from .log import *

class StaticLoads(Comp.Loads, metaclass = ABCMeta):
    @abstractmethod
    def __init__(self, name=""):
        super().__init__(name)
        self._type += '->StaticLoads'

    @abstractmethod
    def _OpsLoadBuild(self):
        ...
    
class PointLoads(StaticLoads):
    def __init__(self, node:Part.BridgeNode, fx=0.0, fy=0.0, fz=0.0, mx=0.0, my=0.0, mz=0.0, name=""):
        super().__init__(name)
        self._type += '->PointLoads'
        self._Load = (fx, fy, fz, mx, my, mz)
        self._Node:Part.BridgeNode = node
    
    def _OpsLoadBuild(self):
        OpsObject.OpsNodeLoad(self._Load, self._Node.OpsNode)
    
    @property
    def val(self):
        return [self._Load, self._Node]

class ElementsLoads(StaticLoads):
    def __init__(self, ele:Part.Segment, wx:float=0.0, wy:float=0.0, wz:float=0.0, GlobalCoordSys:bool=True, name=""):
        super().__init__(name)
        self._Elements = ele

        newZAxis = self._Elements._localZAxis
        x = self._Elements._BridgeNodeI.point
        y = self._Elements._BridgeNodeJ.point
        newXAxis = UtilTools.PointsTools.VectorSub(y, x)

        load = (wx, wy, wz)
        if GlobalCoordSys:
            load = UtilTools.PointsTools.ProjectVectsToNewCoordSYS(load, newXAxis, newZAxis)
        self._Load = load
    
    def _OpsLoadBuild(self):
        OpsObject.OpsEleLoad(self._Elements.ELeList, *self._Load)

    @property
    def val(self):
        return [self._Load, self._Elements]

class Gravity(StaticLoads):
    def __init__(self, segList:list[Part.Segment]=None, CuboList:list[Part.Cuboid]=None, name=''):
        super().__init__(name)
        self._type += '->Gravity'
        self._segList = segList
        self._CuboList = CuboList

    def _OpsLoadBuild(self):
        if self._segList:
            for seg in self._segList:
                ele_ = seg.ELeList
                mass_ = seg.MassList
                eleLength_ = seg.EleLength
                for ele, mass, length in zip(ele_, mass_, eleLength_):
                    wz = -mass * GlobalData.DEFVAL._G_ / length
                    OpsObject.OpsEleLoad([ele], wz=wz)

        if self._CuboList:
            for cuob in self._CuboList:
                for i, xval in enumerate(cuob._BridgeNodes):
                    for j, yval in enumerate(xval):
                        for k, val in enumerate(yval):
                            node:Part.BridgeNode = val
                            grav = cuob._Masses[i, j, k] * GlobalData.DEFVAL._G_
                            load = [0, 0, -grav, 0, 0, 0]
                            OpsObject.OpsNodeLoad(load, node.OpsNode)
                    
            
    @property
    def val(self):
        return [self._segList]

class DispLoad(StaticLoads):
    def __init__(self, node:Part.BridgeNode, dx:float=0.0, dy:float=0.0, dz:float=0.0, name=""):
        super().__init__(name)
        self._node = node
        self._disp = (dx, dy, dz)
    
    def _OpsLoadBuild(self):
        for i, d in enumerate(self._disp):
            if not UtilTools.Util.TOL_EQ(d, 0):
                OpsObject.OpsSP(self._node.OpsNode, i+1, d)

    @property
    def val(self):
        return [self._node, self._disp]

class DynamicLoads(Comp.Loads, metaclass = ABCMeta):
    @abstractmethod
    def __init__(self, name=""):
        super().__init__(name)
        self._type += '->DynamicLoads'

    @abstractmethod
    def _OpsLoadBuild(self):
        ...
    
@dataclass
class WaveInformation:
    station:str
    place:str
    time:str

    dt:float
    npts:int

class SeismicWave(Comp.Component):
    def __init__(self, dt, accX:list[float]=[], factorX:float=1, accY:list[float]=[], factorY:float=1, accZ:list[float]=[], factorZ:float=1, accInformation:WaveInformation=None, name="") -> None:
        super().__init__(name)
        self._information = accInformation
        accX:np.ndarray = np.array(accX).reshape(1,-1)
        accY:np.ndarray = np.array(accY).reshape(1,-1)
        accZ:np.ndarray = np.array(accZ).reshape(1,-1)

        maxlen = max(accX.shape[1], accY.shape[1], accZ.shape[1])

        if maxlen == 0:
            message = 'accX, accY, accZ can not be 0 at same time'
            StandardLogger.fatal(message)
            raise Exception(message)

        message = "Param: {} is ignored as its length are less than maxinum, {} is also ingnored"
        self._factors= [factorX, factorY, factorZ]

        if maxlen == accX.shape[1]:
            self._accX = accX
        else:
            StandardLogger.warning(message.format('accX', 'factorX'))
            self._accX = None

        if maxlen == accZ.shape[1]:
            self._accY = accY
        else:
            StandardLogger.warning(message.format('accY', 'factorY'))
            self._accY = None

        if maxlen == accZ.shape[1]:
            self._accZ = accZ
        else:
            StandardLogger.warning(message.format('accZ', 'factorZ'))
            self._accZ = None
        
        self._times = [x*dt for x in range(maxlen)]

        self._OpsGroundMotions = self._OpsGroundMotionBuild()
    
    def _OpsGroundMotionBuild(self):
        # accX = OpsObject.OpsPathTimeSerise(self._times, self._accX)
        # vel = OpsObject.OpsPathTimeSerise(self._times, self._accY)
        # disp = OpsObject.OpsPathTimeSerise(self._times, self._accZ)
        GMs:list[OpsObject.OpsPlainGroundMotion] = []
        for acc, f in zip([self._accX, self._accY, self._accZ], self._factors):
            if acc is not None:
                accTimes = OpsObject.OpsPathTimeSerise(self._times, acc)
                GMs.append(OpsObject.OpsPlainGroundMotion(accTimeHist=accTimes, factor=f))
            else:
                GMs.append(None)
        return GMs
    
    def plotWave(self):
        t = self._times
        accx = self._accX
        accy = self._accY
        accz = self._accZ

        plt.plot(t, accx[0])
        plt.plot(t, accy[0])
        plt.plot(t, accz[0])

    
    # @staticmethod
    # def LoadRecordFromPEERFile(filePath):

    #     path = pathlib.Path(filePath)
    #     if not path.exists():
    #         logger.warning("Path:{} is not exists".format(path))
    #         return None
        

    
    @staticmethod
    def LoadACCFromPEERFile(filePath):
        path = pathlib.Path(filePath)
        if not path.exists():
            StandardLogger.warning("Path:{} is not exists".format(path))
            return None
    
        with open(filePath) as f:

            f.readline()

            x = f.readline()
            x = x.split(',')
            recordStation= x[0]
            recordTime = x[1]
            recordPlace = x[2]

            x = f.readline()
            x = x.split(' ')
            recordType = x[0]
            recordUnit = x[-1][:-1]
            if recordType != 'ACCELERATION' and recordUnit != 'G':
                StandardLogger.warning('Record type: {}, record unit: {} is not supported now, return None'.format(recordType, recordUnit))
                return None

            x = f.readline()
            npts = re.findall(r'(?:NPTS=[ ]+?)(\d+)', x)
            npts = int(npts[0])
            dt = re.findall(r'(?:DT=[ ]+?)(\.\d+)', x)
            dt = float(dt[0])

            x = f.read()
            acc = re.findall(r'[-]?\.\d+E[+|-]\d+', x)
            acc =np.array(acc).reshape((1, -1)).astype(np.float32)
            
            information = WaveInformation(recordStation, recordPlace, recordTime, dt, npts)
            if acc.shape[1] != npts:
                StandardLogger.warning('npts {} is not equal to the length of accdata {}, the npts is ignored'.format(npts, len(acc)))
    
        return acc, information 

class EarthquakeLoads(Comp.Loads):
    def __init__(self, node:Part.BridgeNode, seismicWave:SeismicWave, name=""):
        super().__init__(name)
        self._node = node
        self._seismicWave = seismicWave

    def _OpsLoadBuild(self):
        for dof, opsGM in enumerate(self._seismicWave._OpsGroundMotions):
            if opsGM:
                ops.imposedMotion(self._node.OpsNode.uniqNum, dof+1, opsGM.uniqNum)
                OpsCommandLogger.info('ops.imposedMotion({}, {}, {})'.format(self._node.OpsNode.uniqNum, dof+1, opsGM.uniqNum))

            
    
    @property
    def val(self):
        return [self._node, self._seismicWave]