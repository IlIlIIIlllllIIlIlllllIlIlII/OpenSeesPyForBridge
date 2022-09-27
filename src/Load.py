from abc import ABCMeta, abstractmethod
from . import Comp
from . import Part

class StaticLoads(Comp.Loads, metaclass = ABCMeta):
    @abstractmethod
    def __init__(self, name=""):
        super().__init__(name)
        self._type += '->StaticLoads'

    @abstractmethod
    def _OpsLoadBuild(self):
        ...
    
class PointLoads(StaticLoads):
    @Comp.CompMgr()
    def __init__(self, load:float, node:Part.BridgeNode, name=""):
        super().__init__(name)
        self._type += '->PointLoads'
        self._Load = load
        self._Node = node
        self._OpsPointLoad = self._OpsLoadBuild()
    
    def _OpsLoadBuild(self):
        ...

class ElementLoads(StaticLoads):
    ...


class Gravity(StaticLoads):
    @abstractmethod
    def __init__(self, name=''):
        super().__init__(name)
        self._type += '->Gravity'

class DynamicLoads(Comp.Loads, metaclass = ABCMeta):
    @abstractmethod
    def __init__(self, name=""):
        super().__init__(name)
        self._type += '->DynamicLoads'
