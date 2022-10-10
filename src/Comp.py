from ast import arg
from http.client import FOUND
import numpy as np
import openseespy.opensees as ops
from abc import ABCMeta, abstractmethod
from .GlobalData import DEFVAL
from .log import *



# * 构件类 基础类，所有的类都有该类派生
class Component(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, name: str = ""):
        self._type = "Component"
        self._uniqNum = None
        self._name = name
        # self._linkedComp:self = None
        self._argsHash = -1
        self._kwargsHash = -1
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
class State:
    Found = 1
    Simailar = 2
    NotFound = 3
    
class CompMgr:
    _uniqNum = 0
    _compDic: dict[int:Component] = {}
    _FLAG_INSTANTIATED = State.NotFound
    _allUniqComp = {}

    _allOtherComp: list[Component] = []
    _allOpsObject:list = []
    _allPart:list = []

    OpsCommandLogger.info('ops.model(\'{}\', \'{}\', {}, \'{}\', {})'.format("basic", "-ndm", 3, "-ndf", 6))
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    @classmethod
    def _new(cls, func):
        # * 用于装饰__new__ 函数, 当_allUniqComp中存在相同实例是,返回该实例,并将_FLAG_INSTANTIATED置为True
        def warpper(*args, **kwargs):
            c = args[0]
            h_args = hash(str(args[1:]))
            h_kwargs = hash(str(kwargs))
            if h_args in cls._allUniqComp:
                comp:Component = cls._allUniqComp[h_args]
                if comp._kwargsHash == h_kwargs:
                    cls._FLAG_INSTANTIATED = State.Found
                    return cls._allUniqComp[h_args]
                else:
                    cls._FLAG_INSTANTIATED = State.Simailar
                    return cls._allUniqComp[h_args]
                
            else:
                cls._FLAG_INSTANTIATED = State.NotFound
                return func(c)
        
        return warpper
                


    @classmethod
    def __call__(cls, func):
        # * 用于装饰__init__函数, 根据_FLAG_INSTANTIATED的不同 确定对应的方法 
        def Wrapper(*args, **kwargs):
            comp:Component = args[0]

            h_args = hash(str(args[1:]))
            h_kwargs = hash(str(kwargs))

            if cls._FLAG_INSTANTIATED == State.Found:
                cls._FLAG_INSTANTIATED = State.NotFound

            elif cls._FLAG_INSTANTIATED == State.NotFound:
                func(*args, **kwargs)
                comp._argsHash = h_args
                comp._kwargsHash = h_kwargs
                comp._uniqNum = cls.getUniqNum()
                cls.StoreComp(comp)
                cls._allUniqComp[h_args] = comp
            
            elif cls._FLAG_INSTANTIATED == State.Simailar:
                uniqNum = comp._uniqNum
                func(*args, **kwargs)
                comp._argsHash = h_args
                comp._kwargsHash = h_kwargs
                comp._uniqNum = uniqNum
                # cls.StoreComp(comp)
                
                cls._FLAG_INSTANTIATED = State.NotFound


            # func(*args, **kwargs)

            # _, exists_comp = cls.FindSameValueComp(comp)

            # if exists_comp is None:
            #     # cls._allComp.append(comp)
            #     cls.StoreComp(comp)
            #     comp._uniqNum = cls._uniqNum
            #     cls._uniqNum += 1
            #     cls._allUniqComp.append(comp)
            # else:
            #     comp._uniqNum = exists_comp._uniqNum
            #     comp._linkedComp = exists_comp
            #     # if isinstance(comp, OpsObj):
            #     #     comp._linkedComp = exists_comp 
            #         # cls._allOpsObject.append(comp)
            #     cls.StoreComp(comp)
                
            # if comp._name != "" and comp._name != exists_comp:
            #     cls.addCompName(comp._name, comp._uniqNum)
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
        cls._uniqNum += 1
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
        # cls._allOtherComp:list[Component] = []
        # cls._compDic = {}
        # cls._uniqNum = 0
        # OpsCommandLogger.info('ops.wipe()'.format())
        # ops.wipe()
        cls._uniqNum = 0
        cls._compDic: dict[int:Component] = {}
        cls._FLAG_INSTANTIATED = State.NotFound
        cls._allUniqComp = {}

        cls._allOtherComp: list[Component] = []
        cls._allOpsObject:list = []
        cls._allPart:list = []

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
    #   self._linkedComp:OpsObj = None

    @CompMgr._new
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    @abstractmethod
    def _create(self):
        ...

    @property
    def uniqNum(self):
        if self._built:
            return self._uniqNum
        else:
            self._create()
            self._built = True
            return self._uniqNum
        

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
    @CompMgr._new
    def __new__(cls):
        return super().__new__(cls)

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

    @CompMgr._new
    def __new__(cls):
        return super().__new__(cls)


    @abstractmethod
    def _activate(self): ...


    def activate(self):
        if self._activated:
            return
        else:
            self.activate()
            self._activated
            return

