from configparser import ConverterMapping
from enum import Enum
from types import DynamicClassAttribute

from . import Unit
from .log import *


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

    # @classmethod
    # @property
    
    # def _LENGTH_UNIT_(cls):
    #     return "mm"
    
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
        return -DEFVAL._MAXVAL_
    @classmethod
    @property
    def _COOROFFSET_(cls):
        return 1
        # return Unit.ConvertToBaseUnit(1, 'mm')

    @classmethod
    @property
    def _G_(cls):
        return Unit.ConvertToBaseUnit(9.80665, 'm/s/s')

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

class SandType(Enum):
    LooseSand = 'LooseSand'
    MediumSand = 'MediumSand'
    MediumDenseSand = 'MediumDenseSand'
    DenseSand = 'DenseSand'

class ClayType(Enum):
    SoftClay = 'SoftClay'
    MediumClay = 'MediumClay'
    StiffClay = 'StiffClay'

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

    @classmethod
    def Sand(cls, sandType:SandType):
        nd = 3
        rho, refShearModul, refBulkModul, frictionAng, peakShearStra, refPress, pressDependCoe, PTAng, contrac, dilat, liquefac = [0] * 11

        if sandType is SandType.LooseSand:
            rho = Unit.ConvertToBaseUnit(1.7, 't/m/m/m')
            refShearModul = Unit.ConvertToBaseUnit(5.5e4, 'kPa')
            refBulkModul = Unit.ConvertToBaseUnit(1.5e5, 'kpa')
            frictionAng = 29
            peakShearStra = 0.1
            refPress = Unit.ConvertToBaseUnit(80, 'kpa')
            pressDependCoe = 0.5
            PTAng = 29
            contrac = 0.21
            dilat = [0, 0]
            liquefac = [
                Unit.ConvertToBaseUnit(10, 'kpa'),
                0.02,
                1
            ]
        if sandType is SandType.MediumSand:
            rho = Unit.ConvertToBaseUnit(1.9, 't/m/m/m')
            refShearModul = Unit.ConvertToBaseUnit(7.5e4, 'kPa')
            refBulkModul = Unit.ConvertToBaseUnit(2.0e5, 'kpa')
            frictionAng = 33
            peakShearStra = 0.1
            refPress = Unit.ConvertToBaseUnit(80, 'kpa')
            pressDependCoe = 0.5
            PTAng = 27
            contrac = 0.07
            dilat = [0.4, 2]
            liquefac = [
                Unit.ConvertToBaseUnit(10, 'kpa'),
                0.01,
                1
            ]
    
        if sandType is SandType.MediumDenseSand:
            rho = Unit.ConvertToBaseUnit(2, 't/m/m/m')
            refShearModul = Unit.ConvertToBaseUnit(1e5, 'kPa')
            refBulkModul = Unit.ConvertToBaseUnit(3.0e5, 'kpa')
            frictionAng = 37
            peakShearStra = 0.1
            refPress = Unit.ConvertToBaseUnit(80, 'kpa')
            pressDependCoe = 0.5
            PTAng = 27
            contrac = 0.05
            dilat = [0.6, 3]
            liquefac = [
                Unit.ConvertToBaseUnit(5, 'kpa'),
                0.003,
                1
            ]

        if sandType is SandType.DenseSand:
            rho = Unit.ConvertToBaseUnit(2.1, 't/m/m/m')
            refShearModul = Unit.ConvertToBaseUnit(1.3e5, 'kPa')
            refBulkModul = Unit.ConvertToBaseUnit(3.9e5, 'kpa')
            frictionAng = 40
            peakShearStra = 0.1
            refPress = Unit.ConvertToBaseUnit(80, 'kpa')
            pressDependCoe = 0.5
            PTAng = 27
            contrac = 0.03
            dilat = [0.8, 5]
            liquefac = [
                Unit.ConvertToBaseUnit(0, 'kpa'),
                0.000,
                0
            ]

        return {
            'nd':nd,
            'rho':rho,
            'refShearModul':refShearModul,
            'refBulkModul':refBulkModul,
            'frictionAng':frictionAng,
            'peakShearStra':peakShearStra,
            'refPress':refPress,
            'pressDependCoe':pressDependCoe,
            'PTAng':PTAng,
            'contrac':contrac,
            'dilat':dilat,
            'liquefac':liquefac
        }

    @classmethod
    def Clay(cls, clayType:ClayType):
        nd = 3
        rho, refShearModul, refBulkModul, cohesi, peakShearStra = [0] * 5
        if clayType is ClayType.SoftClay:
            rho = Unit.ConvertToBaseUnit(1.3, 't/m/m/m')
            refShearModul = Unit.ConvertToBaseUnit(1.3e4, 'kpa')
            refBulkModul = Unit.ConvertToBaseUnit(6.5e4, 'kpa')
            cohesi = Unit.ConvertToBaseUnit(18, 'kpa')
            peakShearStra = 0.1
        if clayType is ClayType.MediumClay:
            rho = Unit.ConvertToBaseUnit(1.5, 't/m/m/m')
            refShearModul = Unit.ConvertToBaseUnit(6.0e4, 'kpa')
            refBulkModul = Unit.ConvertToBaseUnit(3.0e5, 'kpa')
            cohesi = Unit.ConvertToBaseUnit(37, 'kpa')
            peakShearStra = 0.1
        if clayType is ClayType.StiffClay:
            rho = Unit.ConvertToBaseUnit(1.8, 't/m/m/m')
            refShearModul = Unit.ConvertToBaseUnit(1.5e5, 'kpa')
            refBulkModul = Unit.ConvertToBaseUnit(7.5e4, 'kpa')
            cohesi = Unit.ConvertToBaseUnit(75, 'kpa')
            peakShearStra = 0.1

        return {
            'nd':nd,
            'rho':rho,
            'refShearModul':refShearModul,
            'refBulkModul':refBulkModul,
            'cohesi':cohesi,
            'peakShearStra':peakShearStra

        }