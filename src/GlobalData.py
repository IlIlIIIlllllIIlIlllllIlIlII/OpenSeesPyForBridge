from enum import Enum

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
        return 9800

    # * 默认钢筋间距
    @classmethod
    @property
    def _REBAR_D_DEF(cls):
        return 100

    # * 默认的剪切模量弹性模量之比
    @classmethod
    @property
    def _G_E_RATIO_DEF(cls):
        return 0.4

    # * 默认保护层厚度
    @classmethod
    @property
    def _COVER_DEF(cls):
        return 50

    @classmethod
    @property
    def _ROUND_SECT_STEP_DEF(cls):
        return 20

class ReBarArea(Enum):
    d40 = 1256.6
    d36 = 1017.9
    d32 = 804.2
    d28 = 615.8
    d25 = 490.9

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

class MaterialDataBase:
    @classmethod    
    @property
    def Rebar(cls):

    # * fy, e0, b, *params, a1=a2*fy/e0, a2=1.0, a3=a4*fy/e0, a4=1.0, siginit=0.0
        return {
            "HPB300": {"fy": 100000, "e0": 1000, "b": 0.01, "r0": 15, "r1": 0.925, "r2": 0.15},
            "HRB400": {"fy": 100000, "e0": 1000, "b": 0.02, "r0": 15, "r1": 0.925, "r2": 0.15},
        }
    
    @classmethod
    @property
    def Concrete(cls):
        return {
            "C30": {
                "fpc": 1000,
                "epsc0": 1000,
                "fpcu": 1000,
                "epsu": 1000,
                "lambda": 10000,
                "ft": 10000,
                "ets": 1000,
                "dens": 1000,
            },
            "C35": {
                "fpc": 1000,
                "epsc0": 1000,
                "fpcu": 1000,
                "epsu": 1000,
                "lambda": 10000,
                "ft": 10000,
                "ets": 1000,
                "dens": 1000,
            },
            "C40": {
                "fpc": 1000,
                "epsc0": 1000,
                "fpcu": 1000,
                "epsu": 1000,
                "lambda": 10000,
                "ft": 10000,
                "ets": 1000,
                "dens": 1000,
            },
            "C45": {
                "fpc": 1000,
                "epsc0": 1000,
                "fpcu": 1000,
                "epsu": 1000,
                "lambda": 10000,
                "ft": 10000,
                "ets": 1000,
                "dens": 1000,
            },
            "C50": {
                "fpc": 1000,
                "epsc0": 1000,
                "fpcu": 1000,
                "epsu": 1000,
                "lambda": 10000,
                "ft": 10000,
                "ets": 1000,
                "dens": 1000,
            },
            "C55": {
                "fpc": 1000,
                "epsc0": 1000,
                "fpcu": 1000,
                "epsu": 1000,
                "lambda": 10000,
                "ft": 10000,
                "ets": 1000,
                "dens": 1000,
            },
        }

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
