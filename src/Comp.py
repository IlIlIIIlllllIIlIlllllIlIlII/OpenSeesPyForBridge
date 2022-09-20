import openseespy.opensees as ops
from abc import ABCMeta, abstractmethod
from .GlobalData import DEFVAL

# * 构件类 基础类，所有的类都有该类派生
class Component(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, name: str = ""):
        self._type = "component"
        self._uniqNum = None
        self._name = name
    
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

    @property
    def val(self):
        ...

    @property
    def attr(self):
        return [self._type, self._name, self._uniqNum]

class CompMgr:
    _uniqNum = 0
    _compDic: dict = {}
    _allComp: list[Component] = []
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    @classmethod
    def __call__(cls, func):
        def Wrapper(*args, **kwargs):
            comp:Component = args[0]
            func(*args, **kwargs)

            _, exists_comp = cls.findSameValueComp(comp)

            if exists_comp is None:
                cls._allComp.append(comp)
                comp._uniqNum = cls._uniqNum
                cls._uniqNum += 1
            else:
                comp._uniqNum = exists_comp._uniqNum
                if isinstance(comp, OpsObj):
                   comp._linkedOpsObj = exists_comp 
                cls._allComp.append(comp)

            if comp._name != "" and comp._name != exists_comp:
                cls.addCompName(comp._name, comp._uniqNum)
        return Wrapper

    @classmethod
    def findSameValueComp(cls, tarComp: Component):
        if len(cls._allComp) == 0:
            return (None, None)
        for index, Comp in enumerate(cls._allComp):
            if Comp.__class__ == tarComp.__class__:
                if Comp.val == tarComp.val:
                    return index, Comp 
            else:
                pass

        return (None, None)

    @classmethod
    def getUniqNum(cls):
        return cls._uniqNum

    @classmethod
    def removeComp(cls, comp:Component):
        i, exists_Comp = cls.findSameValueComp(comp)
        cls._allComp.pop(i)
        if comp._name in cls._compDic.keys():
            cls._compDic.pop(comp._name)

    @classmethod
    def compNameChanged(cls, oldName, newName):
        cls._compDic[newName] = cls._compDic.pop(oldName)

    @classmethod
    def addCompName(cls, name, uniqNum):
        cls._compDic[name] = uniqNum

    @classmethod
    @property
    def State(cls):
        return str("当前共有组件:{}个\n组件字典为:{}".format(len(cls._allComp), cls._compDic))

    @classmethod
    def getCompbyName(cls, name):
        if name in cls._compDic.keys():
            index = cls._compDic[name]
            return cls._allComp[index - 1]
        else:
            print("No Component")
            return None

class OpsObj(Component, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, name=""):
      super(OpsObj, self).__init__(name)
      self._type += "->OpsObj"
      self._built = False
      self._linkedOpsObj:OpsObj = None

    @abstractmethod
    def _create(self):
        ...

    @property
    def uniqNum(self):
        if self._linkedOpsObj == None:
            if not self._built:
                self._create()
                self._built = True
            return self._uniqNum
        else:
            return self._linkedOpsObj.uniqNum

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

    @property
    def val(self):
        return super().val

class Parts(Component, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, name=""):
        super(Parts, self).__init__(name)
        self._type += "->Parts"
        
    @abstractmethod
    def _SectReBuild(self):
        ...

class HRectSect():
    ...