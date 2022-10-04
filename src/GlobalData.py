from enum import Enum

from . import Unit
from .log import *
from types import DynamicClassAttribute
# from src import UtilTools
class CoordinateSystem:
    @staticmethod
    def Show():
        x = """                              
                           z^                 
                            |      ^y        
                            |     /          
                            |    /           
                            |   /            
                            |  /             
                            | /              
                            |/______________\ x
            """

        print(x)
class DEFVAL:

    @classmethod
    @property
    
    def _LENGTH_UNIT_(cls):
        return "mm"
    
    # * 全局容差
    @classmethod
    @property
    def _TOL_(cls):
        return 1e-10

    @classmethod
    @property
    def _MAXVAL_(cls):
        return 1e10

    @classmethod
    @property
    def _MINVAL_(cls):
        return -1e10

    @classmethod
    @property
    def _G_(cls):
        return Unit.ConvertToBaseUnit(9.80665, 'm*m/s')

    # * 默认钢筋间距
    @classmethod
    @property
    def _REBAR_D_DEF(cls):
        return Unit.ConvertToBaseUnit(100, 'mm')

    # * 默认的剪切模量弹性模量之比
    @classmethod
    @property
    def _G_E_RATIO_DEF(cls):
        return 0.4

    # * 默认保护层厚度
    @classmethod
    @property
    def _COVER_DEF(cls):
        return Unit.ConvertToBaseUnit(50, 'mm')

    @classmethod
    @property
    def _ROUND_SECT_STEP_DEF(cls):
        return Unit.ConvertToBaseUnit(20, 'cm')

class ReBarArea(Enum):
    d40 = 1256.6
    d36 = 1017.9
    d32 = 804.2
    d28 = 615.8
    d25 = 490.9
    @DynamicClassAttribute
    def value(self):
        """The value of the Enum member."""
        return Unit.ConvertToBaseUnit(self._value_, 'mm*mm')

    @classmethod
    def getMaxItem(cls):
        return cls.d40
    @classmethod
    def getMinItem(cls):
        return cls.d25

    @classmethod
    def getMinValue(cls):
        return cls.getMinItem().value
    @classmethod
    def getMaxValue(cls):
        return cls.getMaxItem().value

    @classmethod
    def listAllItem(cls):
        res = []
        for _, m in cls.__members__.items():
            res.append(m)
        return res

class ConcreteType(Enum):
    C30 = "C30"
    C35 = "C35"
    C40 = "C40"
    C45 = "C45"
    C50 = "C50"
    C55 = "C55"


class ReBarType(Enum):
    HRB400 = "HPB400"
    HRB500 = "HRB500"
    HRBF400 = "HRBF400" 
    HRBF500 = "HRBF500"


class MaterialDataBase:
    @classmethod    
    def Rebar(cls, type:ReBarType):
        # * fy, e0, b, *params, a1=a2*fy/e0, a2=1.0, a3=a4*fy/e0, a4=1.0, siginit=0.0
        fy, e0, b, R0, R1, R2 =  0, 0, 0.01, 15, 0.925, 0.15 
        if type == ReBarType.HRB500 or type == ReBarType.HRBF500:
            fy = Unit.ConvertToBaseUnit(435, 'N/mm/mm')
            e0 = Unit.ConvertToBaseUnit(2.00, '10')
        elif type == ReBarType.HRB400 or type == ReBarType.HRBF400:
            fy = Unit.ConvertToBaseUnit(370, 'N/mm/mm')
            e0 = Unit.ConvertToBaseUnit(2.00, '1e5*N/mm/mm')
        else:
            StandardLogger.warning("RebarType:{} are not finished".format(type.value))
            raise Exception("RebarType:{} are not finished".format(type.value))

        return {"fy": fy, "E0": e0, "b": b, "R0": R0, "R1": R1, "R2": R2}

    
    @classmethod
    def Concrete(cls, type:ConcreteType):
        fpc, epsc0, fpcu, epsu, Lambda, ft, ets, dens = 0, 0, 0, 0, 0, 0, 0, 0
        if type == ConcreteType.C30:
            fck = Unit.ConvertToBaseUnit(-30, 'mpa')
            E = Unit.ConvertToBaseUnit(3.00, '1e4*N/mm/mm')
            dens = Unit.ConvertToBaseUnit(2.385, '1e3*kg/m/m/m')

        elif type == ConcreteType.C35:
            fck = Unit.ConvertToBaseUnit(35, 'mpa')
            E = Unit.ConvertToBaseUnit(3.15, '1e4*N/mm/mm')
            dens = Unit.ConvertToBaseUnit(2.39, '1e3*kg/m/m/m')

        elif type == ConcreteType.C40:
            fck = Unit.ConvertToBaseUnit(40, 'mpa')
            E = Unit.ConvertToBaseUnit(3.25, '1e4*N/mm/mm')
            dens = Unit.ConvertToBaseUnit(2.4, '1e3*kg/m/m/m')

        elif type == ConcreteType.C45:
            fck = Unit.ConvertToBaseUnit(45, 'mpa')
            E = Unit.ConvertToBaseUnit(3.35, '1e4*N/mm/mm')
            dens = Unit.ConvertToBaseUnit(2.41, '1e4kg/m/m/m')

        elif type == ConcreteType.C50:
            fck = Unit.ConvertToBaseUnit(50, 'mpa')
            E = Unit.ConvertToBaseUnit(3.45, '1e4*N/mm/mm')
            dens = Unit.ConvertToBaseUnit(2.42, '1e3*kg/m/m/m')

        elif type == ConcreteType.C55:
            fck = Unit.ConvertToBaseUnit(55, 'mpa')
            E = Unit.ConvertToBaseUnit(3.55, '1e4*N/mm/mm')
            dens = Unit.ConvertToBaseUnit(2.43, '1e3*kg/m/m/m')

        else:
            StandardLogger.warning("RebarType:{} are not finished".format(type.value))
            raise Exception("RebarType:{} are not finished".format(type.value))
        
        fpc = 0.79 * fck
        epsc0 = 2 * fpc / E
        fpcu = 0.2 * fpc
        epsu = 0.004
        Lambda = 0.1
        ft = -0.15*fpc
        ets = ft/0.002

        return {
                "fpc": fpc,
                "epsc0": epsc0,
                "fpcu": fpcu,
                "epsu": epsu,
                "Lambda": Lambda,
                "ft": ft,
                "ets": ets,
                "dens": dens
            }

