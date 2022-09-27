from abc import ABCMeta, abstractmethod
from sys import flags
from time import time

from . import OpsObject
from . import GlobalData
from . import Comp
from . import Part
from . import UtilTools

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
        self._Node = node
    
    def _OpsLoadBuild(self):
        OpsObject.OpsNodeLoad(self._Load, self._Node.OpsNode)

class ElementsLoads(StaticLoads):
    def __init__(self, ele:Part.Segment, wx:float=0.0, wy:float=0.0, wz:flags=0.0, GlobalCoordSys:bool=True, name=""):
        super().__init__(name)
        self._Elements = ele

        newZAxis = self._Elements._localZAxis
        x = self._Elements._Ni.point
        y = self._Elements._Nj.point
        newXAxis = UtilTools.PointsTools.vectSub(y, x)

        load = (wx, wy, wz)
        if GlobalCoordSys:
            load = UtilTools.PointsTools.ProjectVectsToNewCoordSYS(load, newXAxis, newZAxis)
        self._Load = load
    
    def _OpsLoadBuild(self):
        OpsObject.OpsEleLoad(self._Elements.ELeList, *self._Load)

class Gravity(StaticLoads):
    def __init__(self, seg:Part.Segment, name=''):
        super().__init__(name)
        self._type += '->Gravity'
        self._seg = seg

    def _OpsLoadBuild(self):
        
        ele_ = self._seg.ELeList
        mass_ = self._seg.MassList
        eleLength_ = self._seg.EleLength
        for ele, mass, length in zip(ele_, mass_, eleLength_):
            wz = -mass * GlobalData.DEFVAL._G_ / length
            OpsObject.OpsEleLoad(ele, wz=wz)

class DispLoad(StaticLoads):
    def __init__(self, node:Part.BridgeNode, dx:float=0.0, dy:float=0.0, dz:float=0.0, name=""):
        super().__init__(name)
        self._node = node
        self._disp = (dx, dy, dz)
    
    def _OpsLoadBuild(self):
        for i, d in enumerate(self._disp):
            if not UtilTools.Util.TOLEQ(d, 0):
                OpsObject.OpsSP(self._node.OpsNode, i+1, d)

class DynamicLoads(Comp.Loads, metaclass = ABCMeta):
    @abstractmethod
    def __init__(self, name=""):
        super().__init__(name)
        self._type += '->DynamicLoads'

    @abstractmethod
    def _OpsLoadBuild(self):
        ...

class EarthquakeLoads(Comp.Loads):
    def __init__(self, node, times:list[float], acc:list[float], vecl:list[float], disp:list[float], name=""):
        super().__init__(name)
        self._times = times
        self._acc = acc
        self._vecl = vecl
        self._disp = disp
    
    def _OpsLoadBuild(self):
        acc = OpsObject.OpsPathTimeSerise(self._times, self._acc)
        vel = OpsObject.OpsPathTimeSerise(self._times, self._vecl)
        disp = OpsObject.OpsPathTimeSerise(self._times, self._disp)
        OpsObject.OpsPlaneGroundMotion(disp, vel, acc)
    