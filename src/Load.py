from abc import ABCMeta, abstractmethod
from . import Comp

class StaticLoads(Comp.Loads, metaclass = ABCMeta):
    @abstractmethod
    def __init__(self, name=""):
        super().__init__(name)
        self._type += '->StaticLoads'

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
