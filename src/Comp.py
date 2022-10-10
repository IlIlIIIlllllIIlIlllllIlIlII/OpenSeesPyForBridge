import numpy as np
import openseespy.opensees as ops
from abc import ABCMeta, abstractmethod
from .GlobalData import DEFVAL
from .log import *



# * 构件类 基础类，所有的类都有该类派生
class Component(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, name: str = ""):
        self._type = "component"
        self._uniqNum = None
        self._name = name
        self._linkedComp:self = None
        # self._argsHash = -1
        # self._kwargsHash = -1
    def __eq__(self, __o) -> bool:
        if type(self) == type(__o) and self.val == __o.val:
            return True
        else:
            return False

    # def __repr__(self):
    #     dic = ""
    #     for name, val in vars(self).items():
    #        dic += "{}:{}\n".format(name, val)
    #     return dic

    def __hash__(self) -> int:
        return hash(repr(self))

    @property
    def val(self):
        ...

    @property
    def attr(self):
        return [self._type, self._name, self._uniqNum]
# class CompCategory:

class CompMgr:
    _uniqNum = 0
    _compDic: dict = {}
    _allUniqComp = []
    _allOtherComp: list[Component] = []
    _allOpsObject:list = []
    _allPart:list = []
    OpsCommandLogger.info('ops.model(\'{}\', \'{}\', {}, \'{}\', {})'.format("basic", "-ndm", 3, "-ndf", 6))
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    @classmethod
    def __call__(cls, func):
        def Wrapper(*args, **kwargs):
            comp:Component = args[0]
            func(*args, **kwargs)

            _, exists_comp = cls.FindSameValueComp(comp)

            if exists_comp is None:
                # cls._allComp.append(comp)
                cls.StoreComp(comp)
                comp._uniqNum = cls._uniqNum
                cls._uniqNum += 1
                cls._allUniqComp.append(comp)
            else:
                comp._uniqNum = exists_comp._uniqNum
                comp._linkedComp = exists_comp
                # if isinstance(comp, OpsObj):
                #     comp._linkedComp = exists_comp 
                    # cls._allOpsObject.append(comp)
                cls.StoreComp(comp)
                
            if comp._name != "" and comp._name != exists_comp:
                cls.addCompName(comp._name, comp._uniqNum)
        return Wrapper

    @classmethod
    def StoreComp(cls, Comp):
        if isinstance(Comp, OpsObj):
            cls._allOpsObject.append(Comp)
        elif isinstance(Comp, Parts):
            cls._allPart.append(Comp)
        else:
            cls._allOtherComp.append(Comp)


    @classmethod
    def FindSameValueComp(cls, tarComp: Component):
        if isinstance(tarComp, OpsObj):
            CertainList = cls._allOpsObject
        elif isinstance(tarComp, Parts):
            CertainList = cls._allPart
        else:
            CertainList = cls._allOtherComp
        return cls.FindSameValeCompInCertainList(tarComp, CertainList)
        
        # if len(cls._allOtherComp) == 0:
        #     return (None, None)
        # for index, Comp in enumerate(cls._allOtherComp):
        #     if Comp.__class__ == tarComp.__class__:
        #         # if Comp.val == tarComp.val:
        #         if CompMgr.CompareCompVal(Comp.val, tarComp.val):
        #             return index, Comp 
        #     else:
        #         pass

        # return (None, None)
        ...
        

    @classmethod
    def FindSameValeCompInCertainList(cls, tarComp:Component, CertainList:list[Component]):

        if len(CertainList) == 0:
            return (None, None) 
        
        for idx, Comp in enumerate(CertainList):
            if CompMgr.CompareCompVal(Comp.val, tarComp.val):
                return idx, Comp

        return None, None

    @staticmethod
    def CompareCompVal(val1, val2):
        flag = True
        for x, y in zip(val1, val2):
            if isinstance(x, np.ndarray):
                flag = np.all(x==y)
            else:
                flag = (x == y)
            
            if not flag:
                return flag

        return flag

    @classmethod
    def getUniqNum(cls):
        return cls._uniqNum

    @classmethod
    def removeComp(cls, comp:Component):
        i, exists_Comp = cls.FindSameValueComp(comp)
        cls._allOtherComp.pop(i)
        if comp._name in cls._compDic.keys():
            cls._compDic.pop(comp._name)

    @classmethod
    def compNameChanged(cls, oldName, newName):
        cls._compDic[newName] = cls._compDic.pop(oldName)

    @classmethod
    def addCompName(cls, name, uniqNum):
        cls._compDic[name] = uniqNum
    
    @classmethod
    def clearComp(cls):
        cls._allOtherComp:list[Component] = []
        cls._compDic = {}
        cls._uniqNum = 0
        OpsCommandLogger.info('ops.wipe()'.format())
        ops.wipe()

    @classmethod
    @property
    def State(cls):
        return str("当前共有组件:{}个\n组件字典为:{}".format(len(cls._allOtherComp), cls._compDic))

    @classmethod
    def getCompbyName(cls, name, compClass:Component):
        for comp in cls._allOtherComp:
            if isinstance(comp, compClass):
                if comp._name == name:
                    return True, comp
            
        return False, None

    @classmethod
    def getCompByUniqNum(cls, uniqNum, compClass:Component):
        if issubclass(compClass, OpsObj):
            CertainList = cls._allOpsObject
        elif issubclass(compClass, Parts):
            CertainList = cls._allPart
        else:
            CertainList = cls._allOpsObject

        return cls.getCompByUniqNumInCertainList(uniqNum, CertainList)
    
    @classmethod
    def getCompByUniqNumInCertainList(cls, uniqNum, CertainList:list[Component]):
        for comp in CertainList:
            if comp._uniqNum == uniqNum:
                return True, comp

        return False, None

    @classmethod
    @property
    def allComponent(cls):
        return cls._allOpsObject + cls._allPart + cls._allOtherComp

class OpsObj(Component, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, name=""):
      super(OpsObj, self).__init__(name)
      self._type += "->OpsObj"
      self._built = False
      self._linkedComp:OpsObj = None

    @abstractmethod
    def _create(self):
        ...

    @property
    def uniqNum(self):
        if not self._linkedComp:
            if not self._built:
                self._create()
                self._built = True
            return self._uniqNum
        else:
            self._built = True
            return self._linkedComp.uniqNum

# * 参数类，派生出主梁截面参数类，桥墩截面参数类......
class Paras(Component, metaclass=ABCMeta):
    @abstractmethod    
    def __init__(self, name=""):
        super(Paras, self).__init__(name)
        self._type += "paras"

    def check(self):
        for val in vars(self).values():
            if type(val) is float:
                if float(val - 0) < DEFVAL._TOL_:
                    return False
        return True

    # @property
    # def val(self):
    #     ...

class Parts(Component, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, name=""):
        super(Parts, self).__init__(name)
        self._type += "->Parts"
        
    # @abstractmethod
    # def _SectReBuild(self):
    #     ...

class Loads(Component, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, name=""):
        super().__init__(name)
        self._type += '->Laods'

    @abstractmethod
    def _OpsLoadBuild(self):
        ...

    def ApplyLoad(self):
        self._OpsLoadBuild()
# class HRectSect():
#     ...

class Boundary(Component, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, name: str = ""):
        super().__init__(name)
        self._type += '->Bounday'
        self._activated = False


    @abstractmethod
    def _activate(self): ...


    def activate(self):
        if self._activated:
            return
        if not self._linkedComp:
            if not self._activated:
                self._activate()
                
        else:
            self._linkedComp.activate()

