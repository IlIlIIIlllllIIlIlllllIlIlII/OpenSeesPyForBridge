from asyncio.log import logger
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
    HPB300 = "HPB300"
    HRB400 = "HRB400"
class MaterialDataBase:
    @classmethod    
    @property
    def Rebar(cls, type:ReBarType):
        # * fy, e0, b, *params, a1=a2*fy/e0, a2=1.0, a3=a4*fy/e0, a4=1.0, siginit=0.0
        fy, e0, b, R0, R1, R2 =  0, 0, 0.01, 15, 0.925, 0.15 
        if type == ReBarType.HPB300:
            fy = Unit.ConvertToBaseUnit(10000, '1000000*kg/s')
            e0 = Unit.ConvertToBaseUnit(10000, '1000000*kg/s')
        elif type == ReBarType.HRB400:
            fy = Unit.ConvertToBaseUnit(10000, '1000000*kg/s')
            e0 = Unit.ConvertToBaseUnit(10000, '1000000*kg/s')
        else:
            logger.warning("RebarType:{} are not finished".format(type.value))
            raise Exception("RebarType:{} are not finished".format(type.value))

        return {"fy": fy, "E0": e0, "b": b, "R0": R0, "R1": R1, "R2": R2}

    
    @classmethod
    @property
    def Concrete(cls, type:ConcreteType):
        fpc, epsc0, fpcu, epsu, Lambda, ft, ets, dens = 0, 0, 0, 0, 0, 0, 0
        if type == ConcreteType.C30:
            fpc = Unit.ConvertToBaseUnit(10000, '1000000*kg/s')
            epsc0 = 0.001
            fpcu = Unit.ConvertToBaseUnit(10000, '1000000*kg/s')
            epsu = 0.001
            Lambda = 0.001
            ft = Unit.ConvertToBaseUnit(10000, '1000000*kg/s')
            ets = Unit.ConvertToBaseUnit*(10000, '1000000*kg/s')
            dens = Unit.ConvertToBaseUnit(10000, 'kg/m/m/m')
        elif type == ConcreteType.C35:
            fpc = Unit.ConvertToBaseUnit(10000, '1000000*kg/s')
            epsc0 = 0.001
            fpcu = Unit.ConvertToBaseUnit(10000, '1000000*kg/s')
            epsu = 0.001
            Lambda = 0.001
            ft = Unit.ConvertToBaseUnit(10000, '1000000*kg/s')
            ets = Unit.ConvertToBaseUnit*(10000, '1000000*kg/s')
            dens = Unit.ConvertToBaseUnit(10000, 'kg/m/m/m')
        elif type == ConcreteType.C40:
            fpc = Unit.ConvertToBaseUnit(10000, '1000000*kg/s')
            epsc0 = 0.001
            fpcu = Unit.ConvertToBaseUnit(10000, '1000000*kg/s')
            epsu = 0.001
            Lambda = 0.001
            ft = Unit.ConvertToBaseUnit(10000, '1000000*kg/s')
            ets = Unit.ConvertToBaseUnit*(10000, '1000000*kg/s')
            dens = Unit.ConvertToBaseUnit(10000, 'kg/m/m/m')
        elif type == ConcreteType.C45:
            fpc = Unit.ConvertToBaseUnit(10000, '1000000*kg/s')
            epsc0 = 0.001
            fpcu = Unit.ConvertToBaseUnit(10000, '1000000*kg/s')
            epsu = 0.001
            Lambda = 0.001
            ft = Unit.ConvertToBaseUnit(10000, '1000000*kg/s')
            ets = Unit.ConvertToBaseUnit*(10000, '1000000*kg/s')
            dens = Unit.ConvertToBaseUnit(10000, 'kg/m/m/m')
        elif type == ConcreteType.C50:
            fpc = Unit.ConvertToBaseUnit(10000, '1000000*kg/s')
            epsc0 = 0.001
            fpcu = Unit.ConvertToBaseUnit(10000, '1000000*kg/s')
            epsu = 0.001
            Lambda = 0.001
            ft = Unit.ConvertToBaseUnit(10000, '1000000*kg/s')
            ets = Unit.ConvertToBaseUnit*(10000, '1000000*kg/s')
            dens = Unit.ConvertToBaseUnit(10000, 'kg/m/m/m')
        elif type == ConcreteType.C55:
            fpc = Unit.ConvertToBaseUnit(10000, '1000000*kg/s')
            epsc0 = 0.001
            fpcu = Unit.ConvertToBaseUnit(10000, '1000000*kg/s')
            epsu = 0.001
            Lambda = 0.001
            ft = Unit.ConvertToBaseUnit(10000, '1000000*kg/s')
            ets = Unit.ConvertToBaseUnit*(10000, '1000000*kg/s')
            dens = Unit.ConvertToBaseUnit(10000, 'kg/m/m/m')
        else:
            logger.warning("RebarType:{} are not finished".format(type.value))
            raise Exception("RebarType:{} are not finished".format(type.value))
        
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

