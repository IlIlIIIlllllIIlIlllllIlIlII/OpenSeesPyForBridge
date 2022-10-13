from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from . import GlobalData
from .Comp import Paras


class SectParas(Paras):
    @abstractmethod
    def __init__(self, name=""):
        super(SectParas, self).__init__(name)
        self._type += "->SectParas"
    
    # @property
    # def val(self):
    #     ...

# * 箱梁截面参数类
class BoxSectParas(SectParas):
    __slots__ = []
    def __init__(
        self,
        upper_width: float,
        down_width: float,
        height: float,
        upper_thick: float,
        down_thick: float,
        web_thick: float,
        name="",
    ):
        super(BoxSectParas, self).__init__(name)
        self._type += "->BoxSectParas"
        self._upper_width = upper_width
        self._down_width = down_width
        self._height = height
        self._upper_thick = upper_thick
        self._down_thick = down_thick
        self._web_thick = web_thick

    @property
    def upper_W(self):
        return self._upper_width
    @upper_W.setter
    def upper_w(self, newVal):
        if type(newVal) is int or type(newVal) is float:
            self._upper_width = newVal
        else:
            raise Exception("Wrong Para")

    @property
    def down_W(self):
        return self._down_width
    @down_W.setter
    def down_W(self, newVal):
        if type(newVal) is int or type(newVal) is float:
            self._down_width = newVal
        else:
            raise Exception("Wrong Para")
    
    @property
    def H(self):
        return self._height
    @H.setter
    def H(self, newVal):
        if type(newVal) is int or type(newVal) is float:
            self._height = newVal
        else:
            raise Exception("Wrong Para")
        
    @property
    def upper_T(self):
        return self._upper_thick
    @upper_T.setter
    def upper_T(self, newVal):
        if type(newVal) is int or type(newVal) is float:
            self._upper_thick= newVal
        else:
            raise Exception("Wrong Para")

    @property
    def down_T(self):
        return self._down_thick
    @down_T.setter
    def down_T(self, newVal):
        if type(newVal) is int or type(newVal) is float:
            self._down_thick= newVal
        else:
            raise Exception("Wrong Para")

    @property
    def web_T(self):
        return self._web_thick
    @web_T.setter
    def web_T(self, newVal):
        if type(newVal) is int or type(newVal) is float:
            self._web_thick= newVal
        else:
            raise Exception("Wrong Para")

    @property
    def val(self):
        """
            return [self.upper_width, self.down_width, self.height, 
                    self.upper_thick, self.down_thick, self.web_thick]
        """
        return [self._upper_width, self._down_width, self._height, 
                self._upper_thick, self._down_thick, self._web_thick]

class SectRebarDistrParas(Paras):
    __slots__ = ['_type', '_uniqNum', '_name', '_r', '_Ns', '_barArea']
    def __init__(self, r: float, Ns: list[int], barArea: list[GlobalData.ReBarArea], name=""):
        super(SectRebarDistrParas, self).__init__(name)
        self._type += "->Section Rebars distribution Parameters"
        self._r = r
        self._Ns = Ns
        self._barArea = barArea
    
    @property
    def SteelRatio(self):
        return self._r
    @SteelRatio.setter
    def SteelRatio(self, newVal):
        if type(newVal) is type(self._r):
            self._r = newVal
        else:
            raise Exception("Wrong Paras")
    
    @property
    def Ns(self):
        return self._Ns
    @Ns.setter
    def Ns(self, newVal):
        if type(newVal) is type(self._Ns):
            self._Ns = newVal
        else:
            raise Exception("Wrong Paras")
    
    @property
    def BarArea(self):
        return self._barArea
    @BarArea.setter
    def BarArea(self, newVal):
        if type(newVal) is type(self._bar):
            self._bar = newVal
        else:
            raise Exception("Wrong Paras")

    @property
    def val(self):
        """
        return [self.r, self.Ns, self.barArea]
        """
        return [self._r, self._Ns, self._barArea]

class SRoundSectParas(SectParas):
    def __init__(self, R:float, cover:float = GlobalData.DEFVAL._COVER_DEF, name=""):
        super().__init__(name)
        self._type += "->SRoundSectParas"
        self._R = R
        self._C = cover

    @property
    def R(self):
        return self._R
    @R.setter
    def R(self, newVal):
        if type(newVal) is type(self._R):
            self._R = newVal
        else:
            raise Exception("Wrong Paras")
    
    @property
    def C(self):
        return self._C
    @C.setter
    def C(self, newVal):
        if type(newVal) is type(self._C):
            self._C = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def val(self):
        return [self._R, self._C]
    

class HRoundSectParas(SectParas):
    __slots__ = ["_R", "_Thick", "_Cover"]

    def __init__(self, r:float, t:float, c:float=(GlobalData.DEFVAL._COVER_DEF), name=""):
        super().__init__(name)
        self._R = r
        self._T = t
        self._C = c

    @property
    def Rout(self):
        return self._R
    @Rout.setter
    def Rout(self, newVal):
        if type(newVal) is type(self._R):
            self._R = newVal
        else:
            raise Exception("Wrong Paras")

    @property
    def Rin(self):
        return self._R - self._T

    @property
    def T(self):
        return self._T
    @T.setter
    def T(self, newVal):
        if type(newVal) is type(self._T):
            self._T = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def C(self):
        return self._C
    @C.setter
    def C(self, newVal):
        if type(newVal) is type(self._C):
            self._C = newVal
        else:
            raise Exception("Wrong Paras")
    
    @property
    def val(self):
        return [self._R, self._T, self._C]

# * 空心矩形桥墩
class HRectSectParas(SectParas):
    __slots__ = ['_type', '_uniqNum', '_name', '_type', '_uniqNum', '_name', "_width", "_length", "_thick", "_cover"]

    def __init__(self, w: float, l: float, t: float, cover: float=(GlobalData.DEFVAL._COVER_DEF), name=""):
        super(HRectSectParas, self).__init__(name)
        self._type += "PierSectParas"
        self._width = w
        self._length = l
        self._thick = t
        self._cover = cover
    
    @property
    def W(self):
        return self._width
    @W.setter
    def W(self, newVal):
        if type(newVal) is int or type(newVal) is float:
            self._width = newVal
        else:
            raise Exception("Wrong Para")

    @property
    def L(self):
        return self._length
    @L.setter
    def L(self, newVal):
        if type(newVal) is int or type(newVal) is float:
            self._length = newVal
        else:
            raise Exception("Wrong Para")

    @property
    def T(self):
        return self._thick
    @T.setter
    def T(self, newVal):
        if type(newVal) is int or type(newVal) is float:
            self._thick = newVal
        else:
            raise Exception("Wrong Para")

    @property
    def C(self):
        return self._cover
    @C.setter
    def C(self, newVal):
        if type(newVal) is int or type(newVal) is float:
            self._cover = newVal
        else:
            raise Exception("Wrong Para")

    @property
    def val(self):
        """
        return [self.width, self.length, self.thick, self.cover]
        """
        return [self._width, self._length, self._thick, self._cover]

class MaterialParas(Paras, metaclass=ABCMeta):
    def __init__(self, name: str = ""):
        super(MaterialParas, self).__init__(name)
        self._type += "MaterialParas"

    @property
    def val(self):
        ...

class ConcreteParas(MaterialParas):
    # uniaxialMaterial('Concrete02', matTag, fpc, epsc0, fpcu, epsU, lambda, ft, Ets)
    __slots__ = ['_type', '_uniqNum', '_name', "_fpc", "_epsc0", "_fpcu", "_epsu", "_lambda", 
                "_ft", "_ets", "_dens"]
    def __init__(
        self,
        conType: str,
        fpc: float,
        epsc0: float,
        fpcu: float,
        epsu: float,
        Lambda: float,
        ft: float,
        ets: float,
        dens:float,
        flag_Core=False,
        name: str = "",
    ):
        super(ConcreteParas, self).__init__(name)
        self._conType = conType
        self._fpc = fpc
        self._epsc0 = epsc0
        self._fpcu = fpcu
        self._epsu = epsu
        self._lambda = Lambda
        self._ft = ft
        self._ets = ets
        self._dens = dens

    @property
    def ConType(self):
        return self._conType
    @ConType.setter
    def ConType(self, newVal):
        if type(newVal) is type(self.conType):
            self._conType = newVal
        else:
            raise Exception("Wrong Paras")

    @property
    def fpc(self):
        return self._fpc
    @fpc.setter
    def fpc(self, newVal):
        if type(newVal) is type(self._fpc):
            self._fpc = newVal
        else:
            raise Exception("Wrong Paras")

    @property
    def epsc0(self):
        return self._epscu0
    @epsc0.setter
    def epsc0(self, newVal):
        if type(newVal) is type(self._epsc0):
            self._epsc0 = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def fpcu(self):
        return self._fpcu
    @fpcu.setter
    def fpcu(self, newVal):
        if type(newVal) is type(self._fpcu):
            self._fpcu = newVal
        else:
            raise Exception("Wrong Paras") 

    @property
    def epsu(self):
        return self._epsu
    @epsu.setter
    def epsu(self, newVal):
        if type(newVal) is type(self._epsu):
            self._epsu = newVal
        else:
            raise Exception("Wrong Paras")

    @property
    def Lambda(self):
        return self._lambda
    @Lambda.setter
    def Lambda(self, newVal):
        if type(newVal) is type(self._lambda):
            self._lambda = newVal
        else:
            raise Exception("Wrong Paras")

    @property
    def ft(self):
        return self._ft
    @ft.setter
    def ft(self, newVal):
        if type(newVal) is type(self._ft):
            self._ft = newVal
        else:
            raise Exception("Wrong Paras")
    
    @property
    def ets(self):
        return self._ets
    @ets.setter
    def ets(self, newVal):
        if type(newVal) is type(self._ets):
            self._ets = newVal
        else:
            raise Exception("Wrong Paras")
        
    @property
    def densty(self):
        return self._dens
    @densty.setter
    def densty(self, newVal):
        if type(newVal) is type(self._dens):
            self._dens = newVal
        else:
            raise Exception("Wrong Paras")
    
    
    @property
    def val(self):
        """
        return [self.fpc, self.epsc0, self.fpcu, self.epsu, self.Lambda, self.ft, self.ets]
        """
        return [self._fpc, self._epsc0, self._fpcu, self._epsu, self._lambda, self._ft, self._ets]

    # def __repr__(self):
    #     a = ","
    #     b = [str(i) for i in self.val]
    #     return a.join(b)

class SteelParas(MaterialParas):
    __slots__ = ['_type', '_uniqNum', '_name', "_fy", "_E0", "_b", "_R0", "_R1", "_R2"]

    def __init__(self, rebarType: str, fy, E0, b, R0=15, R1=0.925, R2=0.15, name=""):
        super(SteelParas, self).__init__(name)
        self._type = rebarType
        self._fy = fy
        self._E0 = E0
        self._b = b
        self._R0 = R0
        self._R1 = R1
        self._R2 = R2

    @property
    def fy(self):
        return self._fy
    @fy.setter
    def fy(self, newVal):
        if type(newVal) is type(self._fy):
            self._fy = newVal
        else:
            raise Exception("Wrong Paras")

    @property
    def E0(self):
        return self._E0
    @E0.setter
    def E0(self, newVal):
        if type(newVal) is type(self._E0):
            self.E0 = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def b(self):
        return self._b
    @b.setter
    def b(self, newVal):
        if type(newVal) is type(self._b):
            self._b = newVal
        else:
            raise Exception("Wrong Paras")
    
    @property
    def R0(self):
        return self._R0
    @R0.setter
    def R0(self, newVal):
        if type(newVal) is type(self._R0):
            self._R0 = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def R1(self):
        return self._R1
    @R1.setter
    def R1(self, newVal):
        if type(newVal) is type(self._R1):
            self._R1 = newVal
        else:
            raise Exception("Wrong Paras")
     
    @property
    def R2(self):
        return self._R2
    @R2.setter
    def R2(self, newVal):
        if type(newVal) is type(self._R2):
            self._R2 = newVal
        else:
            raise Exception("Wrong Paras")
    
    @property
    def val(self):
        """
        [self.fy, self.E0, self.b, self.R0, self.R1, self.R2]
        """
        return [self._fy, self._E0, self._b, self._R0, self._R1, self._R2]

# class ClayParas(Paras):
#     def __init__(self, clayType:str, nd, rho, refShearModul, refBulkModul, cohesi, peakShearStra, name=""):
#         super().__init__(name)
#         self._nd = nd
class SoilParas(Paras):
    def __init__(self, name=""):
        super().__init__(name)
        self._type += 'SoilParas'

@dataclass
class SandParas(SoilParas):
    sandType:str
    nd:int
    rho:float 
    refShearModul:float 
    refBulkModul:float 
    frictionAng:float 
    peakShearStra:float
    refPress:float
    pressDependCoe:float 
    PTAng:float 
    contrac:float 
    dilat:float 
    liquefac:float 
    
    @property
    def val(self):
        return [self.nd, self.rho, self.refShearModul, self.refBulkModul, self.frictionAng, self.peakShearStra, self.refPress, self.pressDependCoe, self.PTAng, self.contrac, self.dilat, self.liquefac]

@dataclass
class ClayParas:
    clayType:str
    nd:int
    rho:float
    refShearModul:float
    refBulkModul:float
    cohesi:float
    peakShearStra:str
    
    @property
    def val(self):
        return [self.nd, self.rho, self.refShearModul, self.refBulkModul, self.cohesi, self.peakShearStra]

class Concrete:
    @classmethod
    @property
    def C30(cls):
        return ConcreteParas("C30", **GlobalData.MaterialDataBase.Concrete(GlobalData.ConcreteType.C30))

    @classmethod
    @property
    def C35(cls):
        return ConcreteParas("C35", **GlobalData.MaterialDataBase.Concrete(GlobalData.ConcreteType.C35))

    @classmethod
    @property
    def C40(cls):
        return ConcreteParas("C40", **GlobalData.MaterialDataBase.Concrete(GlobalData.ConcreteType.C40))

    @classmethod
    @property
    def C45(cls):
        return ConcreteParas("C45", **GlobalData.MaterialDataBase.Concrete(GlobalData.ConcreteType.C45))

    @classmethod
    @property
    def C50(cls):
        return ConcreteParas("C50", **GlobalData.MaterialDataBase.Concrete(GlobalData.ConcreteType.C50))

    @classmethod
    @property
    def C55(cls):
        return ConcreteParas("C55", **GlobalData.MaterialDataBase.Concrete(GlobalData.ConcreteType.C55))

class ReBar:
    @classmethod
    @property
    def HRB500(cls):
        return SteelParas(rebarType="HPB300", **GlobalData.MaterialDataBase.Rebar(GlobalData.ReBarType.HRB500))

    @classmethod
    @property
    def HRBF500(cls):
        return SteelParas(rebarType="HRBF500", **GlobalData.MaterialDataBase.Rebar(GlobalData.ReBarType.HRBF500))

    @classmethod
    @property
    def HRB400(cls):
        return SteelParas(rebarType="HRB400", **GlobalData.MaterialDataBase.Rebar(GlobalData.ReBarType.HRB400))

    @classmethod
    @property
    def HRBF400(cls):
        return SteelParas(rebarType="HRBF400", **GlobalData.MaterialDataBase.Rebar(GlobalData.ReBarType.HRBF400))

class Clay:
    @classmethod
    @property
    def Soft(cls):
        return ClayParas(clayType='SoftClay', **GlobalData.MaterialDataBase.Clay(GlobalData.ClayType.SoftClay))

    @classmethod
    @property
    def Medium(cls):
        return ClayParas(clayType='MediumClay', **GlobalData.MaterialDataBase.Clay(GlobalData.ClayType.MediumClay))

    @classmethod
    @property
    def Stiff(cls):
        return ClayParas(clayType='StiffClay', **GlobalData.MaterialDataBase.Clay(GlobalData.ClayType.StiffClay))

class Sand:
    @classmethod
    @property
    def LooseSand(cls):
        return SandParas(sandType='LooseSand', **GlobalData.MaterialDataBase.Sand(GlobalData.SandType.LooseSand))

    @classmethod
    @property
    def MediumSand(cls):
        return SandParas(sandType='MediumSand', **GlobalData.MaterialDataBase.Sand(GlobalData.SandType.MediumSand))

    @classmethod
    @property
    def MediumDenseSand(cls):
        return SandParas(sandType='MediumDenseSand', **GlobalData.MaterialDataBase.Sand(GlobalData.SandType.MediumDenseSand))

    @classmethod
    @property
    def DenseSand(cls):
        return SandParas(sandType='DenseSand', **GlobalData.MaterialDataBase.Sand(GlobalData.SandType.DenseSand))

#TODO
# class BridgeParas(Paras):
#     def __init__(self, name: str = ""):
#         super(BridgeParas, self).__init__(name)
#         self.ParasDict = {
#             "L_s": 0.0,
#             "L_m": 0.0,
#             "L_0": 0.0,
#             "W_g_u": 0.0,
#             "W_g_d": 0.0,
#             "H_m": 0.0,
#             "H_p": 0.0,
#             "T_u_s": 0.0,
#             "T_d_s": 0.0,
#             "T_w_s": 0.0,
#             "T_u_m": 0.0,
#             "T_d_m": 0.0,
#             "T_w_m": 0.0,
#             "W_p_u": 0.0,
#             "L_p_u": 0.0,
#             "T_p_u": 0.0,
#             "W_p_d": 0.0,
#             "L_p_d": 0.0,
#             "T_p_d": 0.0,
#         }
#         self._type += "BridgeParas"