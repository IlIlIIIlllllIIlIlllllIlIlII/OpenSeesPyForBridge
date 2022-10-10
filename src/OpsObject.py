# * OpsObj类不需要有Getter和setter
from abc import ABC, ABCMeta, abstractmethod
from asyncio import protocols
from dataclasses import make_dataclass
from symbol import factor
from tkinter import dialog
from xml.dom.minidom import Element
import numpy as np
import openseespy.opensees as ops
from enum import Enum

from src import Paras

from . import Comp
from . import GlobalData
from . import UtilTools
from .log import *

# * 桥梁节点类 表示有限元节点
# class OpsO(Comp.OpsObj):
#     def __init__(self, name=""):
#         super().__init__(name)

    
class OpsNode(Comp.OpsObj):
    # __slots__ = ["_type", "_uniqNum", "_name", "_xyz"]
    @Comp.CompMgr()
    def __init__(self, xyz: tuple, name=""):
        super(OpsNode, self).__init__(name)
        # self = self._VALCOMPLETE(xyz)
        self._type += "->Ops Node"

        self._xyz = (float(xyz[0]), float(xyz[1]), float(xyz[2]))

    def _create(self):
        OpsCommandLogger.info('ops.node({}, *{})'.format(self._uniqNum, self._xyz))
        ops.node(self._uniqNum, *self._xyz)
    # @Comp.CompMgr()
    # def _VALCOMPLETE(self, xyz):
    #     self._xyz = xyz
    #     return self

    @property
    def xyz(self):
        return self._xyz

    @property
    def val(self):
        return [self._xyz]

    # def __repr__(self):
    #     return "({}, {}, {})".format(self._xyz[0], self._xyz[1], self._xyz[2])

class OpsMass(Comp.OpsObj):
    def __init__(self, node:OpsNode, mass:float, dof:list[int]=[0,0,-1,0,0,0], name=""):
        super(OpsMass, self).__init__(name)
        self._type += '->Ops Mass'
        self._Node = node
        self._mass = float(mass)
        if len(dof) == 6 and UtilTools.Util.isOnlyHas(dof, [0, 1, -1], flatten=True):
            self._massDof = dof
        else:
            self._massDof = [1,1,1,0,0,0]
        
        self._create()
    
    def _create(self):
        OpsCommandLogger.info('ops.mass({}, {}, {}, {}, {}, {}, {})'.format(self._Node.uniqNum, self._mass*self._massDof[0], self._mass*self._massDof[1], self._mass*self._massDof[2], self._mass*self._massDof[3], self._mass*self._massDof[4], self._mass*self._massDof[5]))

        ops.mass(self._Node.uniqNum, self._mass*self._massDof[0], self._mass*self._massDof[1], self._mass*self._massDof[2], self._mass*self._massDof[3], self._mass*self._massDof[4], self._mass*self._massDof[5])

    @property
    def val(self):
        return [self._Node, self._mass, self._massDof]

class OpsBoundary(Comp.OpsObj):
    @abstractmethod
    def __init__(self, node:OpsNode, name=""):
        super(OpsBoundary, self).__init__(name)
        self._node:OpsNode = node
    
    @abstractmethod
    def _create(self):
        ...

class OpsFix(OpsBoundary):
    __slots__ = ["_type", "_uniqNum", "_name", "_node", "_fix"]
    # @Comp.CompMgr()
    def __init__(self, node: OpsNode, fixlist:list[int] , name=""):
        super(OpsFix, self).__init__(node, name)
        self._type = '->Ops Fixed Boundary'
        if len(fixlist) != 6 and not UtilTools.Util.isOnlyHas(fixlist, [1, 0], flatten=True):
            raise Exception("Wrong Paras:{}".format(fixlist))

        self._fix = fixlist
        self._create()

    @property
    def val(self):
        """
        fixlist
        """
        return [self._node, self._fix]

    def _create(self):
        OpsCommandLogger.info('ops.fix({}, *{})'.format(self._node.uniqNum, self._fix))
        ops.fix(self._node.uniqNum, *self._fix)

class OpsEqualDOF(Comp.OpsObj):
    def __init__(self, nodeI:OpsNode, nodeJ:OpsNode, dofs:list[int], name=""):
        super().__init__(name)
        self._type += '-> Ops Equal DOF'
        self._NodeI = nodeI
        self._NodeJ = nodeJ
        self._dofs = dofs
        self._create()
    
    def _create(self):
        OpsCommandLogger.info('ops.equalDOF({}, {}, *{})'.format(self._NodeI.uniqNum, self._NodeJ.uniqNum, self._dofs))
        ops.equalDOF(self._NodeI.uniqNum, self._NodeJ.uniqNum, *self._dofs)
    
    @property
    def val(self):
        return [self._NodeI, self._NodeJ, self._dofs]
class OpsGemoTrans(Comp.OpsObj):
    @abstractmethod
    def __init__(self, vecz:tuple[float], name=""):
        super().__init__(name)
        self._type += '->OpsGemoTrans'
        self._vecz = np.array(vecz).astype(np.float64).tolist()
    
    @abstractmethod
    def _create(self):
        ...

class OpsLinearTrans(OpsGemoTrans):
    __slots__ = ["_type", "_uniqNum", "_name", "_vecz"]
    @Comp.CompMgr()
    def __init__(self, vecz:tuple[int, ...], name=""):
        super(OpsLinearTrans, self).__init__(vecz, name)
        self._type += '->OpsLinearTrans'

    @property
    def val(self):
        """
        return vecZ
        """
        return [self._vecz]

    def _create(self):
        OpsCommandLogger.info('ops.geomTransf("Linear", {}, *{})'.format(self._uniqNum, self._vecz))
        ops.geomTransf("Linear", self._uniqNum, *self._vecz)

class OpsPDletaTrans(OpsGemoTrans):
    @Comp.CompMgr()
    def __init__(self, vecz:tuple[float, ...], name=""):
        super().__init__(vecz, name)
        self._type += '->OpsPDletaTrans'
        
    @property
    def val(self):
        return [self._vecz]

    def _create(self):
        OpsCommandLogger.info('ops.geomTransf("PDelta", {}, *{})'.format(self._uniqNum, self._vecz))
        # geomTransf('PDelta', transfTag, *vecxz, '-jntOffset', *dI, *dJ)
        ops.geomTransf("PDelta", self._uniqNum, *self._vecz)

class OpsMaterial(Comp.OpsObj, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, name=""):
        super(OpsMaterial, self).__init__(name)
        self._type += "->Ops Material"

    @abstractmethod
    def _create(self):
        ...
class OpsNDMaterial(OpsMaterial, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, name=""):
        super().__init__(name)
        self._type += '->Ops nDMateial'
    
    @abstractmethod
    def _create(self):
        ...
class OpsPIMYMaterial(OpsNDMaterial):
    @Comp.CompMgr()
    def __init__(self, nd, rho, refShearModul, refBulkModul, cohesi, peakShearStra,  name=""):
        super().__init__(name)
        self._type += 'Pressure Indecent Multi Yield (clay) Material'
        self._nd = nd
        self._rho = rho
        self._refShearModul = refShearModul
        self._refBulkModul = refBulkModul
        self._cohesi = cohesi
        self._peakShearStra = peakShearStra 

    def _create(self):
        ops.nDMaterial('PressureIndependMultiYield', self._uniqNum, self._nd, self._rho, self._refShearModul, self._refBulkModul, self._cohesi, self._peakShearStra)
        OpsCommandLogger.info('ops.nDMaterial("PressureIndependMultiYield", {}, {}, {}, {}, {}, {}, {})'.format(self._uniqNum, self._nd, self._rho, self._refShearModul, self._refBulkModul, self._cohesi, self._peakShearStra))

    @property
    def val(self):
        return [self._nd, self._rho, self._refShearModul, self._refBulkModul, self._cohesi, self._peakShearStra]

class OpsClayMaterial(OpsPIMYMaterial):
    def __init__(self, nd, rho, refShearModul, refBulkModul, cohesi, peakShearStra, name=""):
        super().__init__(nd, rho, refShearModul, refBulkModul, cohesi, peakShearStra, name)
    def _create(self):
        return super()._create()

    @property
    def val(self):
        return super().val

class OpsPDMYMaterial(OpsNDMaterial):
    @Comp.CompMgr()
    def __init__(self, nd, rho, refShearModul, refBulkModul, frictionAng, peakShearStra, refPress, pressDependCoe, PTAng, contrac, dilat, liquefac, name=""):
        super().__init__(name)
        self._type += 'Pressure Depend Multi Yield (sand) Material'
        self._nd = nd
        self._rho = rho
        self._refShearModul = refShearModul
        self._refBulkModul = refBulkModul
        self._frictionAng = frictionAng
        self._peakShearStra = peakShearStra
        self._refPress = refPress
        self._pressDependCoe = pressDependCoe
        self._PTAng = PTAng
        self._contrac = contrac
        self._dilat = dilat
        self._liquefac = liquefac

    def _create(self):
        ops.nDMaterial('PressureDependMultiYield', self._uniqNum, self._nd, self._rho, self._refShearModul, self._refBulkModul, self._frictionAng, self._peakShearStra, self._refPress, self._pressDependCoe, self._PTAng, self._contrac, *self._dilat, *self._liquefac)

    @property
    def val(self):
        return [self._nd, self._rho, self._refBulkModul, self._refShearModul, self._frictionAng, self._peakShearStra, self._refPress, self._pressDependCoe, self._PTAng, self._contrac, self._dilat, self._liquefac]

class OpsSandMaterial(OpsPDMYMaterial):
    def __init__(self, nd, rho, refShearModul, refBulkModul, frictionAng, peakShearStra, refPress, pressDependCoe, PTAng, contrac, dilat, liquefac, name=""):
        super().__init__(nd, rho, refShearModul, refBulkModul, frictionAng, peakShearStra, refPress, pressDependCoe, PTAng, contrac, dilat, liquefac, name)
    
    def _create(self):
        return super()._create()

    @property
    def val(self):
        return super().val

class OpsUniaxialMaterial(OpsMaterial, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, name=""):
        super().__init__(name)
        self._type += 'Ops uniaxial Material'
    
    @abstractmethod
    def _create(self):
        ...

    @property
    def val(self):
        ...

class OpsPySimpleMaterial(OpsUniaxialMaterial):
    @Comp.CompMgr()
    def __init__(self, pult:float, Y50:float, Cd:float, soilType:int=2, name=''):
        super().__init__(name)
        self._type += 'Ops P-y Material'
        self._pult = pult
        self._y50 = Y50
        self._cd = Cd
        self._soilType = soilType

    def _create(self):
        ops.uniaxialMaterial('PySimple1', self._uniqNum, self._soilType, self._pult, self._y50, self._cd, c=0.0)

        OpsCommandLogger.info('ops.uniaxialMaterial("PySimple1", {}, {}, {}, {}, {}, c=0.0)'.format(self._uniqNum, self._soilType, self._pult, self._y50, self._cd))

    @property
    def val(self):
        return [self._pult, self._y50, self._cd, self._soilType]

class OpsTzSimpleMaterial(OpsUniaxialMaterial):
    @Comp.CompMgr()
    def __init__(self, tult:float, Z50:float, soilType:int=2, name=''):
        super().__init__(name)
        self._type += 'Ops P-y Material'
        self._tult = tult
        self._z50 = Z50
        self._soilType = soilType

    def _create(self):
        ops.uniaxialMaterial('TzSimple1', self._uniqNum, self._soilType, self._tult, self._z50, c=0.0)
        OpsCommandLogger.info('ops.uniaxialMaterial("PySimple1", {}, {}, {}, {}, c=0.0)'.format(self._uniqNum, self._soilType, self._tult, self._z50))

    @property
    def val(self):
        return [self._tult, self._z50, self._soilType]

class OpsQzSimpleMaterial(OpsUniaxialMaterial):
    @Comp.CompMgr()
    def __init__(self, qult:float, q50:float, SoilType:int=2, name=''):
        super().__init__(name)
        self._type += 'Ops P-y Material'
        self._qult = qult
        self._z50 = q50
        self._qzType = SoilType

    def _create(self):
        ops.uniaxialMaterial('QzSimple1', self._uniqNum, self._qzType, self._qult, self._z50, suction=0.0, c=0.0)
        OpsCommandLogger.info('ops.uniaxialMaterial("PySimple1", {}, {}, {}, {}, suction=0.0, c=0.0)'.format(self._uniqNum, self._qzType, self._qult, self._z50))

    @property
    def val(self):
        return [self._qult, self._z50, self._qzType]
class OpsElasticPPMaterial(OpsUniaxialMaterial):
    @Comp.CompMgr()
    def __init__(self, E, epsT, name=""):
        super().__init__(name)
        self._type += 'Elastic-Perfectly Plastic Material'
        self._E = E
        self._epsT = epsT
    
    def _create(self):
        ops.uniaxialMaterial('ElasticPP', self._uniqNum, self._E, self._epsT)
        OpsCommandLogger.info('ops.uniaxialMaterial("ElasticPP", {}, {}, {})'.format(self._uniqNum, self._E, self._epsT))
    
    @property
    def val(self):
        return [self._E, self._epsT]
        
class OpsConcrete02(OpsUniaxialMaterial):
    """
    使用Ops的Concrete02模型
    uniaxialMaterial('Concrete02', matTag, fpc, epsc0, fpcu, epsU, lambda, ftype(x) == Componenttype(x) == Componentt, Ets)
    fpc:28天抗压强度
    epsc0:混凝土最大应变
    fpcu:混凝土极限抗压强度
    epsU:混凝土极限压应变
    lambda:斜率之比,opensees默认为0.1
    ft:混凝土抗拉强度, opensees默认为0.1的fpc
    Ets:混凝土抗拉刚度, opensees默认为0.1*fpc/epsc0
    """
    __slots__ = ['_type', '_uniqNum', '_name', '_fpc', '_epsc0', '_fpcu', '_epsu', '_lambda', '_ft',
                 '_ets', '_E', '_G']
    @Comp.CompMgr()
    def __init__(self, Fpc:float, Epsc0:float, Fpcu:float, EpsU:float, 
                    Lambda:float=0.1, Ft:float=None, Ets:float=None, name=""):
        super(OpsConcrete02, self).__init__(name)
        self._type += "->Concrete02"
        if Ft == None:
            Ft = 0.1 * Fpc
        if Ets == None:
            Ets = 0.1 * Fpc * Epsc0
        self._fpc = Fpc
        self._epsc0 = Epsc0
        self._fpcu = Fpcu
        self._epsu = EpsU
        self._lambda = Lambda
        self._ft = Ft
        self._ets = Ets
        self._E = self._fpc/self._epsc0
        self._G = (GlobalData.DEFVAL._G_E_RATIO_DEF) * self._E
    @property
    def val(self):
        """
        return [self._fpc, self._epsc0, self._fpcu, self._epsu, 
                    self._lambda, self._ft, self._ets, self._E, self._G]
        """
        return [self._fpc, self._epsc0, self._fpcu, self._epsu, 
                    self._lambda, self._ft, self._ets, self._E, self._G]
                    
    def _create(self):
        OpsCommandLogger.info('ops.uniaxialMaterial("Concrete02", {}, {}, {}, {}, {}, {}, {}, {})'.format(self._uniqNum, self._fpc, self._epsc0, self._fpcu, self._epsu, self._lambda, self._ft, self._ets))
        ops.uniaxialMaterial( "Concrete02", self._uniqNum, self._fpc, self._epsc0, self._fpcu, self._epsu, self._lambda, self._ft, self._ets)

class OpsSteel02(OpsUniaxialMaterial):
    """
    采用openseespy中的steel02材料模型
    uniaxialMaterial('Steel02', matTag, Fy, E0, b, *params, a1=a2*Fy/E0, a2=1.0, a3=a4*Fy/E0, a4=1.0, sigInit=0.0)
    Fy:钢筋的屈服强度
    E0:钢筋的切线弹性模量
    b:应变强化系数
    params [R0, cR1, cR2]: 弹性向塑性过度的控制参数,10<R0<20, cR1=0.925, cR2=0.15
    a1, a2, a3, a4:可选参数
    """
    __slots__  = ['_type', '_uniqNum', '_name', '_fy', '_e0', '_b', '_r0', '_cr1', '_cr2']
    @Comp.CompMgr()
    def __init__(self, Fy:float, E0:float, b:float, R0:float, cR1:float, cR2:float, name=""):
        super(OpsSteel02, self).__init__(name)
        self._type += "->Steel02"
        self._fy = Fy
        self._e0 = E0
        self._b = b
        self._r0 = R0
        self._cr1 = cR1
        self._cr2 = cR2
    
    @property
    def val(self):
        """
        return [self._fy, self._e0, self._b, self._r0, self._cr1, self._cr2]
        """
        return [self._fy, self._e0, self._b, self._r0, self._cr1, self._cr2]

    def _create(self):
        OpsCommandLogger.info('ops.uniaxialMaterial("Steel02", {}, {}, {}, {}, {}, {}, {})'.format(self._uniqNum, self._fy, self._e0, self._b, self._r0, self._cr1, self._cr2))
        ops.uniaxialMaterial( "Steel02", self._uniqNum, self._fy, self._e0, self._b, self._r0, self._cr1, self._cr2)

class OpsSection(Comp.OpsObj, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, name=""):
        super(OpsSection, self).__init__(name)
        self._type += "->Bridge Cross Section"

    @abstractmethod
    def _create(self):
        ...

    @staticmethod
    def RectRebarFiber(p1: tuple, p2: tuple, m:OpsUniaxialMaterial, area: GlobalData.ReBarArea, n: int):
        if n == 1:
            return
        np1 = np.array(p1)
        np2 = np.array(p2)
        d = (np1 - np2) / n
        np2 = np2 - d
        np1 = list(np1)
        np2 = list(np2)
        OpsCommandLogger.info('ops.layer({}, {}, {}, {}, {}, {}, {})'.format(m.uniqNum, n, area.value, float(np1[0]), float(np1[1]), np2[0], np2[1]))
        ops.layer( "straight", m.uniqNum, n, area.value, float(np1[0]), float(np1[1]), np2[0], np2[1])

    @staticmethod
    def RoundRebarFiberBuild(r:float, m:OpsUniaxialMaterial, area:GlobalData.ReBarArea, n:int):
        ops.layer("circ", m.uniqNum, n, area.value, 0, 0, r)
        message = 'ops.layer("circ", {}, {}, {}, 0, 0, {})'.format(m.uniqNum, n, area.value, r)
        OpsCommandLogger.info(message)

    @staticmethod
    def HRectConcreteFiberBuild(w:float, l:float, t:float, m:OpsUniaxialMaterial, fibersize:tuple[int, ...]):
        
        # patch('rect', matTag, numSubdivY, numSubdivZ, *crdsI, *crdsJ)
        p1 = (w / 2, l / 2)
        p2 = (w / 2, -l / 2)
        p3 = (-w / 2, -l / 2)
        p4 = (-w / 2, l / 2)
        w1 = w - 2 * t
        l1 = l - 2 * t
        p11 = (w1 / 2, l1 / 2)
        p22 = (w1 / 2, -l1 / 2)
        p33 = (-w1 / 2, -l1 / 2)
        p44 = (-w1 / 2, l1 / 2)
        
        message = 'ops.patch("rect", {}, {}, {}, {})'

        ops.patch("rect", m.uniqNum, *fibersize, *p1, *p22)
        message = message.format(m.uniqNum, *fibersize, *p1, *p22)
        OpsCommandLogger.info(message)

        ops.patch("rect", m.uniqNum, *fibersize, *p2, *p33)
        message = message.format(m.uniqNum, *fibersize, *p2, *p33)
        OpsCommandLogger.info(message)

        ops.patch("rect", m.uniqNum, *fibersize, *p3, *p44)
        message = message.format(m.uniqNum, *fibersize, *p3, *p44)
        OpsCommandLogger.info(message)

        ops.patch("rect", m.uniqNum, *fibersize, *p4, *p11)
        message = message.format(m.uniqNum, *fibersize, *p4, *p11)
        OpsCommandLogger.info(message)

    @staticmethod
    def RoundConcreteFiberBuild(Rin:float, Rout:float, m:OpsUniaxialMaterial, fiberSize:tuple[int, ...]):
        # patch('circ', matTag, numSubdivCirc, numSubdivRad, *center, *rad, *ang)
        Circ, Rad = fiberSize
        nRad = int(round((Rout-Rin)/Circ, 0))
        nCirc = int(round((np.pi*2*Rout + np.pi*2*Rin) / 2 / Rad, 0))
        if nRad < 1:
            nRad = 1
        if nCirc < 4:
            nCirc = 4
        ops.patch("circ", m.uniqNum, nCirc, nRad, 0, 0, Rin, Rout, 0, 360)
        message = 'ops.patch("circ", {}, {}, {}, 0, 0, {}, {}, 0, 360)'.format(m.uniqNum, nCirc, nRad, Rin, Rout)
        OpsCommandLogger.info(message)


class OpsBoxSection(OpsSection):
    __slots__ = ['_type', '_uniqNum', '_name', '_attr', '_material']
    @Comp.CompMgr()
    def __init__(
        self, area:float, Ix:float, Iy:float, Ij:float, m:OpsConcrete02, name=""
    ):
        super(OpsBoxSection, self).__init__(name)
        self._type += 'Box Section'
        self._attr = {"area":area, "inertia_x":Ix, "interia_y":Iy, "interia_j":Ij}
        self._material = m

    @property
    def val(self):
        return [self._attr, self._material]

    def _create(self):
        # section('Elastic', secTag, E_mod, A, Iz, Iy, G_mod, Jxx, alphaY=None, alphaZ=None)
        if self._built is not True:
            OpsCommandLogger.info('ops.section( "Elastic", {}, {}, {}, {}, {}, {}, {})'.format(self._uniqNum, self._material._E, self._attr["area"], self._attr['inertia_x'], self._attr['interia_y'], self._material._G, self._attr['interia_j']))
            ops.section( "Elastic", self._uniqNum, self._material._E, self._attr["area"], self._attr['inertia_x'], self._attr['interia_y'], self._material._G, self._attr['interia_j'])

class OpsHRoundFiberSection(OpsSection):

    """
    """
    @Comp.CompMgr()
    def __init__(
        self,
        cover:float,
        R_out:float,
        R_in:float,
        sectAttr:dict,
        rebarsDistr:Paras.SectRebarDistrParas,
        conCover:OpsConcrete02,
        conCore:OpsConcrete02 = None,
        rebar:OpsSteel02 = None,
        fiberSize:tuple = (100, 100),
        name = ""
    ):
        super().__init__(name)
        self._type += '->Hollow Round FiberSection'
    
        self._Rin = R_in
        self._Rout = R_out
        self._C = cover
        self._SectAttr = sectAttr
        self._RebarDistr = rebarsDistr
        if conCover == None:
            conCover = conCore
        self._CoverCon = conCover
        self._CoreCon = conCore
        self._Rebar = rebar
        self._FiberSize = fiberSize
    
    @property
    def val(self):
        return [self._Rin, self._Rout, self._C, self._SectAttr, self._RebarDistr, self._CoreCon, self._CoverCon, self._Rebar, self._FiberSize]
        
    def _create(self):

        J = self._SectAttr["inertia_j"]
        G = self._CoreCon._G
        Rin = self._Rin + self._C
        Rout = self._Rout - self._C

        OpsCommandLogger.info('ops.section("Fiber", {}, \'{}\', {})'.format(self._uniqNum, "-GJ", G * J))
        ops.section("Fiber", self._uniqNum, "-GJ", G * J)

        # * 普通钢筋纤维部分
        for count, (As_, Ns_) in enumerate(zip(self._RebarDistr.BarArea, self._RebarDistr.Ns)):
            Rin += GlobalData.DEFVAL._REBAR_D_DEF * count
            Rout -= GlobalData.DEFVAL._REBAR_D_DEF * count
            self.RoundRebarFiberBuild(Rin, self._Rebar, As_[0], Ns_[0])
            self.RoundRebarFiberBuild(Rout, self._Rebar, As_[1], Ns_[1])

        # * 混凝土纤维部分
        Rout = self._Rout
        Rin = Rout - self._C
        self.RoundConcreteFiberBuild(Rin, Rout, self._CoverCon, self._FiberSize)
        
        Rout = Rin
        Rin = self._Rin + self._C
        self.RoundConcreteFiberBuild(Rin, Rout, self._CoreCon, self._FiberSize)

        Rout = Rin
        Rin = self._Rin
        self.RoundConcreteFiberBuild(Rin, Rout, self._CoverCon, self._FiberSize)

class OpsSRoundFiberSection(OpsSection):
    @Comp.CompMgr()
    def __init__(self, 
        cover:float,
        R:float,
        sectAttr:dict,
        rebarsDistr:Paras.SectRebarDistrParas,
        conCore:OpsConcrete02,
        conCover:OpsConcrete02,
        rebar:OpsSteel02,
        fiberSize:tuple[int,...],
        name=""):
        
        super().__init__(name)
        self._type += "->OpsSRoundFiberSection"
        self._R = R
        self._C = cover
        self._SectAttr = sectAttr
        self._RebarDistr = rebarsDistr
        self._CoreCon = conCore
        self._CoverCon = conCover
        self._Rebar = rebar
        self._FiberSize = fiberSize

    @property
    def val(self):
        return [self._R, self._C, self._SectAttr, self._RebarDistr, self._CoreCon, self._CoverCon, self._Rebar, self._FiberSize]

    def _create(self):
        
        J = self._SectAttr["inertia_j"]
        G = self._CoreCon._G
        R = self._R - self._C

        OpsCommandLogger.info('ops.section("Fiber", {}, \'{}\', {})'.format(self._uniqNum, "-GJ", G * J))
        ops.section("Fiber", self._uniqNum, "-GJ", G * J)

        # * 普通钢筋纤维部分
        for count, (As_, Ns_) in enumerate(zip(self._RebarDistr.BarArea, self._RebarDistr.Ns)):
            R -= GlobalData.DEFVAL._REBAR_D_DEF * count
            self.RoundRebarFiberBuild(R, self._Rebar, As_[0], Ns_[0])

        # * 混凝土纤维部分
        Rout = self._R
        Rin = Rout - self._C
        self.RoundConcreteFiberBuild(Rin, Rout, self._CoverCon, self._FiberSize)
        
        Rout = Rin
        Rin = 0
        self.RoundConcreteFiberBuild(Rin, Rout, self._CoreCon, self._FiberSize)

class OpsHRectFiberSection(OpsSection):
    """
    桥墩纤维截面对象, opensees.py命令为
    ops.Section()
    """

    @Comp.CompMgr()
    def __init__(
        self,
        cover:float,
        width:float,
        length:float,
        thick:float,
        sectAttr:dict,
        rebarsDistr:Paras.SectRebarDistrParas,
        conCore: OpsConcrete02,
        conCover: OpsConcrete02 = None,
        rebar: OpsSteel02 = None,
        fiberSize: tuple = (100, 100),
        name="",
    ):
        super(OpsHRectFiberSection, self).__init__(name)
        self._type += "->Hollow Rectangle Section"
        self._c = cover
        self._l = length
        self._w = width
        self._t = thick
        self._sectAttr = sectAttr
        self._rebarDistr = rebarsDistr
        self._CoreCon = conCore
        if conCover == None:
            conCover = conCore
        self._CoverCon = conCover
        self._rebar = rebar
        self._fiberSize = fiberSize

    @property
    def val(self):
        """
        return [self._c, self._l, self._w, self._t, self._a1, self._a2, 
                self._CoreCon, self._CoverCon, self._rebar, self._fiberSize]
        """
        return [self._c, self._l, self._w, self._t, self._rebarDistr, 
                self._CoreCon, self._CoverCon, self._rebar, self._fiberSize]


    def _create(self):
        J = self._sectAttr["inertia_j"]
        G = self._CoreCon._G
        c = self._c
        t = self._t - 2 * c
        w = self._w - 2 * c
        l = self._l - 2 * c

        OpsCommandLogger.info('ops.section("Fiber", {}, \'{}\', {})'.format(self._uniqNum, "-GJ", G * J))
        ops.section("Fiber", self._uniqNum, "-GJ", G * J)

        # * 普通钢筋纤维部分
        for count, (As_, Ns_) in enumerate(zip(self._rebarDistr.BarArea, self._rebarDistr.Ns)):
            x = (w - GlobalData.DEFVAL._REBAR_D_DEF * count) / 2
            y = (l - GlobalData.DEFVAL._REBAR_D_DEF * count) / 2
            p11 = (+x, +y)
            p12 = (+x, -y)
            p21 = (-x, +y)
            p22 = (-x, -y)

            x = w - 2 * t + GlobalData.DEFVAL._REBAR_D_DEF * count
            y = l - 2 * t + GlobalData.DEFVAL._REBAR_D_DEF * count
            p33 = (+x, +y)
            p34 = (+x, -y)
            p43 = (-x, +y)
            p44 = (-x, -y)
            
            p1_ = [p11, p12, p22, p21, p33, p34, p44, p43]
            p2_ = [p12, p22, p21, p11, p34, p44, p43, p33]
            for p1, p2, As, Ns in zip(p1_, p2_, As_, Ns_):
                self.RectRebarFiber(p1, p2, self._rebar, As, Ns)

        # * 混凝土纤维部分
        self.HRectConcreteFiberBuild(w, l, c, self._CoverCon)
        w1 = w - 2 * c
        l1 = l - 2 * c
        t1 = t - 2 * c
        self.HRectConcreteFiberBuild(w1, l1, t1, self._CoreCon)
        w2 = w - 2 * t + 2 * c
        l2 = l - 2 * t + 2 * c
        self.HRectConcreteFiberBuild(w2, l2, c, self._CoverCon)

class OpsElement(Comp.OpsObj, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, name=""):
        super().__init__(name)
        self._type += '->Ops Element'

    @abstractmethod
    def _create(self):...
    
    @property
    def val(self):...

class OpsLineElement(OpsElement, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, node1:OpsNode, node2:OpsNode, sect:OpsSection, transf:OpsLinearTrans, name=""):
        super(OpsLineElement, self).__init__(name)
        self._type += "->Ops Line Element"
        self._Node1 = node1
        self._Node2 = node2
        self._Sect = sect
        self._Transf = transf

    @property
    def NodeI(self):
        return self._Node1
    @property
    def NodeJ(self):
        return self._Node2
    @property
    def Sect(self):
        return self._Sect
    @abstractmethod
    def _create(self):
        ...

class OpsZLElement(OpsLineElement):
    __slots__ = []
    @Comp.CompMgr()
    def __init__(self, node1:OpsNode, node2:OpsNode, mats:list[OpsUniaxialMaterial], dirs:list[int], name=""):
        super().__init__(node1, node2, None, None, name)

        self._type += '->Zero Length Element'

        if len(mats) != len(dirs):
            raise Exception("wrong paras:{} and {}".format(mats, dirs))
        self._m = mats
        self._dirs = dirs

    @property
    def val(self):
        """
        return [self._node1, self._node2, self._m, self._dirs]
        """
        return [self._Node1, self._Node2, self._m, self._dirs]

    def _create(self):
        OpsCommandLogger.info('ops.element(\'{}\', {}, {}, {}, \'{}\', *{}, \'{}\', *{})'.format('zeroLength', self._uniqNum, self._Node1.uniqNum, self._Node2.uniqNum, '-mat', self._m, '-dir', self._dirs))
        ops.element('zeroLength', self._uniqNum, self._Node1.uniqNum, self._Node2.uniqNum, '-mat', *self._m, '-dir', *self._dirs)

    @property
    def NodeI(self):
        return self._Node1
    @property
    def NodeJ(self):
        return self._Node2

class OpsEBCElement(OpsLineElement):
    """
    ElasticBeamColumn单元,使用的opensees.py命令为:
    element('elasticBeamColumn', eleTag, *eleNodes, secTag, transfTag, <'-mass', mass>, <'-cMass'>)
    """
    __slots__ = []
    @Comp.CompMgr()
    def __init__(
        self, node1:OpsNode, node2: OpsNode, sec: OpsSection, localZ: tuple, name=""
    ):
        super(OpsEBCElement, self).__init__(node1, node2, sec, OpsLinearTrans(localZ), name)

        self._type += "->Elastic Beam Column Element"

    @property
    def val(self):
        """
        return [self._Node1, self._Node2, self._Sect, self._localZ, self._transf]
        """
        return [self._Node1, self._Node2, self._Sect,  self._Transf]

    @property
    def NodeI(self):
        return self._Node1
    @property
    def NodeJ(self):
        return self._Node2
    @property
    def Sect(self):
        return self._Sect

    def _create(self):
        OpsCommandLogger.info('ops.element({}, {}, {}, {}, {}, {})'.format("elasticBeamColumn", self._uniqNum, self._Node1.uniqNum, self._Node2.uniqNum, self._Sect.uniqNum, self._Transf.uniqNum))
        ops.element("elasticBeamColumn", self._uniqNum, self._Node1.uniqNum, self._Node2.uniqNum, self._Sect.uniqNum, self._Transf.uniqNum)

class OpsNBCElement(OpsLineElement):
    __slots__ = []
    @Comp.CompMgr()
    def __init__(self, node1:OpsNode, node2:OpsNode, sect:OpsBoxSection, localZ:tuple[int],IntgrNum:int=5, maxIter=10, tol:float=1e-12, mass:float=0.0, IntgrType:str="Lobatto", name=""):
        super(OpsNBCElement, self).__init__(node1, node2, sect, OpsLinearTrans(localZ), name)
        self._type += "->Nonlinear Beam Column Element"

        self._intgrN = IntgrNum
        self._maxIter = maxIter
        self._tol = tol
        self._mass = mass
        self._intgrType = IntgrType

    @property
    def NodeI(self):
        return self._Node1
    @property
    def NodeJ(self):
        return self._Node2
    @property
    def Sect(self):
        return self._Sect

    @property
    def val(self):
        """
        return [self._node1, self._node2, self._sect, self._localZ, self._trans] 
        """
        return [self._Node1, self._Node2, self._Sect,  self._Transf] 

    def _create(self):
        # element('nonlinearBeamColumn', eleTag, *eleNodes, numIntgrPts, 
        # secTag, transfTag, '-iter', maxIter=10, tol=1e-12, '-mass', mass=0.0, 
        # '-integration', intType)
        ops.element('nonlinearBeamColumn', self._uniqNum, self._Node1.uniqNum, self._Node2.uniqNum, self._intgrN, self._Sect.uniqNum, self._Transf.uniqNum, '-iter', self._maxIter, self._tol, '-mass', self._mass, '-itegration', self._intgrType)

        OpsCommandLogger.info('ops.element("nonlinearBeamColumn", {}, {}, {}, {}, {}, {}, "-iter", {}, {}, "-mass", {}, "-itegration", \'{}\')'.format(self._uniqNum, self._Node1.uniqNum, self._Node2.uniqNum, self._intgrN, self._Sect.uniqNum, self._Transf.uniqNum, self._maxIter, self._tol,  self._mass,self._intgrType))

# class OpsPlaneElement(OpsElement):
#     def __init__(self, node1: OpsNode, node2: OpsNode, sect: OpsSection, transf: OpsLinearTrans, name=""):
#         super().__init__(node1, node2, sect, transf, name)
class OpsBrickElement(OpsElement, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, node1:OpsNode,  node2:OpsNode, node3:OpsNode, node4:OpsNode, node5:OpsNode,node6:OpsNode, node7:OpsNode, node8:OpsNode, name=""):
        super().__init__(name)
        self._node1 = node1
        self._node2 = node2
        self._node3 = node3
        self._node4 = node4
        self._node5 = node5
        self._node6 = node6
        self._node7 = node7
        self._node8 = node8
    
    @abstractmethod
    def _create(self):
        ...
    
    @property
    def val(self): ...

class OpsStanderBrickElement(OpsBrickElement):
    @Comp.CompMgr()
    def __init__(self, node1: OpsNode, node2: OpsNode, node3: OpsNode, node4: OpsNode, node5: OpsNode, node6: OpsNode, node7: OpsNode, node8: OpsNode, material:OpsSandMaterial, name=""):
        super().__init__(node1, node2, node3, node4, node5, node6, node7, node8, name)
        self._material = material
        self._eleNodes = [self._node1.uniqNum, self._node2.uniqNum, self._node3.uniqNum, self._node4.uniqNum, self._node5.uniqNum, self._node6.uniqNum, self._node7.uniqNum, self._node8.uniqNum]
    
    def _create(self):
        ops.element('stdBrick', self._uniqNum, *self._eleNodes, self._material.uniqNum)
        OpsCommandLogger.info('ops.element("stdBrick", {}, *{}, {})'.format(self._uniqNum, self._eleNodes, self._material.uniqNum))
    
    @property
    def val(self):
        return [self._eleNodes, self._material]
       
class OpsTimeSeries(Comp.OpsObj):
    @abstractmethod
    def __init__(self, name=""):
        super().__init__(name)
        self._type += '->TimeSerise'

    @abstractmethod
    def _create(self):
        ...

class OpsConstTimeSeries(OpsTimeSeries):
    @Comp.CompMgr()
    def __init__(self, name=""):
        super().__init__(name)
        self._type += '->ConstantTimeSeries'
    
    def _create(self):
        OpsCommandLogger.info('ops.timeSeries({}, {})'.format('Constant', self._uniqNum))
        # timeSeries('Constant', tag, '-factor', factor=1.0)
        ops.timeSeries('Constant', self._uniqNum)

class OpsLinearTimeSerise(OpsTimeSeries):
    @Comp.CompMgr()
    def __init__(self, name=""):
        super().__init__(name)
        self._type += '->LinearTimeSeries'
    
    def _create(self):
    # timeSeries('Linear', tag, '-factor', factor=1.0, '-tStart', tStart=0.0)
        OpsCommandLogger.info('ops.timeSeries(\'{}\', {})'.format('Linear', self._uniqNum))
        ops.timeSeries('Linear', self._uniqNum)

class OpsPathTimeSerise(OpsTimeSeries):
    @Comp.CompMgr()
    def __init__(self, times:list[float], values:list[float], name=""):
        super().__init__(name)
        self._type += '->OpsPathTimeSerise'
        self._times = np.array(times).tolist()
        self._values = np.array(values).flatten().tolist()
    
    def _create(self):
        OpsCommandLogger.info('ops.timeSeries(\'{}\', {}, \'{}\', *{}, \'{}\', *{})'.format('Path', self._uniqNum, '-values', self._values, '-time', self._times))
    # timeSeries('Path', tag, '-dt', dt=0.0, '-values', *values, '-time', *time, '-filePath', filePath='', '-fileTime', fileTime='', '-factor', factor=1.0, '-startTime', startTime=0.0, '-useLast', '-prependZero')
        ops.timeSeries('Path', self._uniqNum, '-time', *self._times, '-values', *self._values)
    
    @property
    def val(self):
        return [self._times, self._values]

class OpsTimeSeriesEnum(Enum):
    Linear = 'Linear'
    Const = 'Const'


class OpsPlainLoadPattern(Comp.OpsObj):
    @Comp.CompMgr()
    def __init__(self, timeseriesType:OpsTimeSeriesEnum=OpsTimeSeriesEnum.Linear, name=""):
        super().__init__(name)
        self._type += '->OpsPlainPattern'
        if timeseriesType == OpsTimeSeriesEnum.Linear:
            timeseries = OpsLinearTimeSerise()
        elif timeseriesType == OpsTimeSeriesEnum.Const:
            timeseries = OpsConstTimeSeries()
        else:
            raise Exception("unexcepted params:{}".format(timeseriesType))

        self._timeseries = timeseries
    
    def _create(self):
        OpsCommandLogger.info('ops.pattern(\'{}\', {}, {})'.format('Plain', self._uniqNum, self._timeseries.uniqNum))
        ops.pattern('Plain', self._uniqNum, self._timeseries.uniqNum)
    
    def val(self):
        return [self._timeseries]


class OpsMSELoadPattern(Comp.OpsObj):
    @Comp.CompMgr()
    def __init__(self, name=""):
        super().__init__(name)
        self._type += '->Multi-Support Excitation Pattern'
    
    def _create(self):
        OpsCommandLogger.info('ops.pattern(\'{}\', {})'.format('MultipleSupport', self._uniqNum))
        ops.pattern('MultipleSupport', self._uniqNum)


class OpsPlainLoads(Comp.OpsObj):
    @abstractmethod
    def __init__(self, name=""):
        super().__init__(name)
        self._type += '->OpsPlainLoads'
    
    @abstractmethod
    def _create(self):
        ...

class OpsNodeLoad(OpsPlainLoads):
    # @Comp.CompMgr()
    def __init__(self, load:tuple[float], node:OpsNode, name=""):
        super().__init__(name)
        self._type += '->OpsNodeLoad'
        self._Load = load
        self._Node = node
        self._create()
    
    def _create(self):
        # load(nodeTag, *loadValues)
        
        OpsCommandLogger.info('ops.load({}, *{})'.format(self._Node.uniqNum, *self._Load))
        ops.load(self._Node.uniqNum, *self._Load)
    
    @property
    def val(self):
        return [self._Load, self._Node]

class OpsEleLoad(OpsPlainLoads):
    # @Comp.CompMgr()
    def __init__(self, eles:list[OpsLineElement], wx:float=0.0, wy:float=0.0, wz:float=0.0, name=""):
        super().__init__(name)
        self._type += '->OpsElementLoad'
        self._Elements = eles

        self._Load = (wx, wy, wz)
        self._create()

    def _create(self):
        # super()._create()
        uniqNum_ = []
        for ele in self._Elements:
            uniqNum_.append(ele.uniqNum)
        # ops.eleLoad('-ele', self._Element.uniqNum, '-type', '-beamUniform', , <Wz>, Wx=0.0)
        OpsCommandLogger.info('ops.eleLoad(\'{}\', *{}, \'{}\', \'{}\', {}, {}, {})'.format('-ele', uniqNum_, '-type', '-beamUniform', self._Load[1], self._Load[2], self._Load[0]))
        ops.eleLoad('-ele', *uniqNum_, '-type', '-beamUniform', self._Load[1], self._Load[2], self._Load[0])
    
    @property
    def val(self):
        return [self._Elements, self._Load]
        
class OpsSP(OpsPlainLoads):
    # @Comp.CompMgr()
    def __init__(self, node:OpsNode, dof:int, dofValue:float, name=""):
        super().__init__(name)
        self._type += '->OpsSP'
        self._node = node
        self._dof = dof
        self._d = dofValue
        self._create()
    
    def _create(self):
        # sp(nodeTag, dof, *dofValues)
        OpsCommandLogger.info('ops.sp({}. {}, {})'.format(self._node.uniqNum, self._dof, self._d))
        ops.sp(self._node.uniqNum, self._dof, self._d)

class OpsPlainGroundMotion(Comp.OpsObj):
    @Comp.CompMgr()
    def __init__(self, dispTimeHist:OpsPathTimeSerise=None, velTimeHist:OpsPathTimeSerise=None, accTimeHist:OpsPathTimeSerise=None, factor:float=1.0, name=""):
        super().__init__(name)
        self._type += '->OpsPlain Ground Motion'

        if not dispTimeHist and not velTimeHist and not accTimeHist:
            raise Exception("disp, vel and accel can not be None at same time")

        self._disp = dispTimeHist
        self._vel = velTimeHist
        self._acc = accTimeHist
        self._factor = factor
        

    def _args(self):
        args =[]

        if self._disp:
            args += ['-disp', self._disp.uniqNum]
        if self._vel:
            args += ['-vel', self._vel.uniqNum]
        if self._acc:
            args += ['-accel', self._acc.uniqNum]

        return args


    def _create(self):
        # groundMotion(gmTag, 'Plain', '-disp', dispSeriesTag, '-vel', velSeriesTag, '-accel', accelSeriesTag, '-int', tsInt='Trapezoidal', '-fact', factor=1.0)
        OpsCommandLogger.info('ops.groundMotion({}, \'{}\', *{}, \'{}\', {})'.format(self._uniqNum, 'Plain', self._args(), '-fact', self._factor))
        ops.groundMotion(self._uniqNum, 'Plain', *self._args(), '-fact', self._factor)
    
    @property
    def val(self):
        return [self._disp, self._vel, self._acc]

class OpsImposedGroundMotion(Comp.OpsObj):
    # @Comp.CompMgr()
    def __init__(self, Node:OpsNode, dof:int, groundMotion:OpsPlainGroundMotion, name=""):
        super().__init__(name)
        self._type += '->Ops Imposed Ground Motion'
        self._node = Node
        self._dof = dof
        self._groundMotion = groundMotion
        OpsCommandLogger.info('ops.imposedMotion({}, {}, {})'.format(self._node.uniqNum, self._dof, self._groundMotion.uniqNum))
        ops.imposedMotion(self._node.uniqNum, self._dof, self._groundMotion.uniqNum)