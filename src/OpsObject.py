# * OpsObj类不需要有Getter和setter
from abc import ABCMeta, abstractmethod
import numpy as np
import openseespy.opensees as ops
from enum import Enum

from src import Paras

from . import Comp
from . import GlobalData
from . import UtilTools

# * 桥梁节点类 表示有限元节点
class OpsNode(Comp.OpsObj):
    __slots__ = ["_type", "_uniqNum", "_name", "_xyz"]
    @Comp.CompMgr()
    def __init__(self, xyz: tuple, name=""):
        super(OpsNode, self).__init__(name)
        self._type += "->OpsNode"
        self._xyz = xyz

    def _create(self):
        ops.node(self._uniqNum, *self._xyz)
    
    @property
    def val(self):
        return [self._xyz]

    def __repr__(self):
        return "({}, {}, {})".format(self._xyz[0], self._xyz[1], self._xyz[2])

class OpsMass(Comp.OpsObj):
    @Comp.CompMgr()
    def __init__(self, node:OpsNode, mass:float, massDist:list[int], name=""):
        super(OpsMass, self).__init__(name)
        self._Node = node
        self._mass = mass
        if len(massDist) == 6 and UtilTools.Util.isOnlyHas(massDist, [0, 1]):
            self._Dist = massDist
    
    def _create(self):
        ops.mass(self._uniqNum,self._Node.uniqNum, self._mass, 
                 self._mass*self._mass[0], self._mass*self._mass[1], self._mass*self._mass[2],
                 self._mass*self._mass[3],self._mass*self._mass[4],self._mass*self._mass[5])

class OpsBoundary(Comp.OpsObj):
    @abstractmethod
    def __init__(self, node:tuple[int, ...], name=""):
        super(OpsBoundary, self).__init__(name)
        self._node = node
    
    @abstractmethod
    def _create(self):
        ...

class OpsFix(OpsBoundary):
    __slots__ = ["_type", "_uniqNum", "_name", "_node", "_fix"]
    @Comp.CompMgr()
    def __init__(self, node: tuple[int, ...], fixlist:list[int] , name=""):
        super(OpsFix, self).__init__(node, name)
        if len(fixlist) != 3 \
            and fixlist[0] * fixlist[1] * fixlist[2] != 0 \
            or fixlist[0] * fixlist[1] * fixlist[2] != 1:

            raise Exception("Wrong Paras:{}".format(fixlist))

        self._fix = fixlist

    @property
    def val(self):
        """
        fixlist
        """
        return [self._node, self._fix]

    def _create(self):
        ops.fix(self._uniqNum, *self._fix)

class OpsLinearTrans(Comp.OpsObj):
    __slots__ = ["_type", "_uniqNum", "_name", "_vecz"]
    @Comp.CompMgr()
    def __init__(self, vecz:tuple[int, ...], name=""):
        super(OpsLinearTrans, self).__init__(name)
        self._vecz = vecz

    @property
    def val(self):
        """
        return vecZ
        """
        return [self._vecz]

    def _create(self):
        ops.geomTransf("Linear", self._uniqNum, *self._vecz)

class OpsMaterial(Comp.OpsObj, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, name=""):
        super(OpsMaterial, self).__init__(name)
        self._type += "->Material"

    @abstractmethod
    def _create(self):
        ...

class OpsConcrete02(OpsMaterial):
    """
    使用Ops的Concrete02模型
    uniaxialMaterial('Concrete02', matTag, fpc, epsc0, fpcu, epsU, lambda, ft, Ets)
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
        self._type += "Concrete02"
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
        ops.uniaxialMaterial(
            "Concrete02",
            self._uniqNum,
            self._fpc,
            self._epsc0,
            self._fpcu,
            self._epsu,
            self._lambda,
            self._ft,
            self._ets,
        )

class OpsSteel02(OpsMaterial):
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
        ops.uniaxialMaterial(
            "Steel02",
            self._uniqNum,
            self._fy,
            self._e0,
            self._b,
            self._r0,
            self._cr1,
            self._cr2,
        )

class OpsSection(Comp.OpsObj, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, name=""):
        super(OpsSection, self).__init__(name)
        self._type += "->BridgeCrossSection"

    @abstractmethod
    def _create(self):
        ...

    @staticmethod
    def RectRebarFiber(p1: tuple, p2: tuple, m: OpsMaterial, area: GlobalData.ReBarArea, n: int):
        if n == 1:
            return
        np1 = np.array(p1)
        np2 = np.array(p2)
        d = (np1 - np2) / n
        np2 = np2 - d
        np1 = list(np1)
        np2 = list(np2)
        ops.layer(
            "straight",
            m.uniqNum,
            n,
            area.value,
            float(np1[0]),
            float(np1[1]),
            np2[0],
            np2[1],
        )

    @staticmethod
    def RoundRebarFiberBuild(r:float, m:OpsMaterial, area:GlobalData.ReBarArea, n:int):
        ops.layer("circ", m.uniqNum, n, area, 0, 0, r)

    @staticmethod
    def HRectConcreteFiberBuild(w:float, l:float, t:float, m: OpsMaterial, fibersize:tuple[int, ...]):
        
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
        ops.patch("rect", m.uniqNum, *fibersize, *p1, *p22)
        ops.patch("rect", m.uniqNum, *fibersize, *p2, *p33)
        ops.patch("rect", m.uniqNum, *fibersize, *p3, *p44)
        ops.patch("rect", m.uniqNum, *fibersize, *p4, *p11)
    
    def RoundConcreteFiberBuild(Rin:float, Rout:float, m:OpsMaterial, fiberSize:tuple[int, ...]):
        # patch('circ', matTag, numSubdivCirc, numSubdivRad, *center, *rad, *ang)
        Circ, Rad = fiberSize
        nRad = int(round((Rout-Rin)/Circ, 0))
        nCirc = int(round((np.pi*2*Rout + np.pi*2*Rin) / 2 / Rad), 0)
        ops.patch("circ", m.uniqNum, nCirc, nRad, 0, 0, Rin, Rout, 0, 360)


class OpsBoxSection(OpsSection):
    __slots__ = ['_type', '_uniqNum', '_name', '_attr', '_material']
    @Comp.CompMgr()
    def __init__(
        self, area:float, Ix:float, Iy:float, Ij:float, m:OpsConcrete02, name=""
    ):
        super(OpsBoxSection, self).__init__(name)
        self._attr = {"area":area, "inertia_x":Ix, "interia_y":Iy, "interia_j":Ij}
        self._material = m

    @property
    def val(self):
        return [self._attr, self._material]

    def _create(self):
        # section('Elastic', secTag, E_mod, A, Iz, Iy, G_mod, Jxx, alphaY=None, alphaZ=None)
        if self._built is not True:
            ops.section(
                "Elastic",
                self._uniqNum,
                self._material._E,
                self._attr["area"],
                self._attr['inertia_x'],
                self._attr['inertia_y'],
                self._material._G,
                self._attr['inertia_j'],
                alphaY=None,
                alphaZ=None,
            )

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

    def val(self):
        return [self._R, self._C, self._SectAttr, self._RebarDistr, self._CoreCon, self._CoverCon, self._Rebar, self._FiberSize]

    def _create(self):
        
        J = self._SectAttr["inertia_j"]
        G = self._CoreCon._G
        R = self._R - self._C

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
        self._type += "->PierFiberSection"
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
    def __init__(self, node1:OpsNode, node2:OpsNode, sect:OpsSection, transf:OpsLinearTrans, name=""):
        super(OpsElement, self).__init__(name)
        self._type += "BridgeElement"
        self._Node1 = node1
        self._Node2 = node2
        self._Sect = sect
        self._Transf = transf

    @abstractmethod
    def _create(self):
        ...

class OpsZLElement(OpsElement):
    __slots__ = []
    @Comp.CompMgr()
    def __init__(self, node1:OpsNode, node2:OpsNode, mats:list[OpsMaterial], dirs:list[int], name=""):
        super().__init__(node1, node2, None, None, name)

        self._type = '->OpsZeroLengthElement'

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
        ops.element('zeroLength', self._uniqNum, self._Node1.uniqNum, self._Node2.uniqNum, 
                    '-mat', *self._m, '-dir', *self._dirs)


class OpsEBCElement(OpsElement):
    """
    ElasticBeamColumn单元,使用的opensees.py命令为:
    element('elasticBeamColumn', eleTag, *eleNodes, secTag, transfTag, <'-mass', mass>, <'-cMass'>)
    """
    __slots__ = []
    @Comp.CompMgr()
    def __init__(
        self, node1:tuple[int, ...], node2: tuple[int, ...], sec: OpsSection, localZ: tuple, name=""
    ):
        super(OpsEBCElement, self).__init__(node1, node2, sec, OpsLinearTrans(localZ), name)

        self._type += "->ElasticBeamColumnElement"

    @property
    def val(self):
        """
        return [self._Node1, self._Node2, self._Sect, self._localZ, self._transf]
        """
        return [self._Node1, self._Node2, self._Sect,  self._Transf]

    def _create(self):
        ops.element(
            "elasticBeamColumn",
            self._uniqNum,
            *(self._Node1),
            *(self._Node2),
            self._Sect.uniqNum,
            self._Transf.uniqNum
        )

class OpsNBCElement(OpsElement):
    __slots__ = []
    @Comp.CompMgr()
    def __init__(self, node1:tuple[int, ...], node2:tuple[int, ...], sect:OpsBoxSection, localZ:tuple[int],
                IntgrNum:int=5, maxIter=10, tol:float=1e-12, mass:float=0.0, IntgrType:str="Lobatto", name=""):
        super(OpsNBCElement, self).__init__(node1, node2, sect, OpsLinearTrans(localZ), name)
        self._type += "->NonlinearBeamColumnElement"

        self._intgrN = IntgrNum
        self._maxIer = maxIter
        self._tol = tol
        self._mass = mass
        self._intgrType = IntgrType

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
        ops.element('nonlinearBeamColumn', self._uniqNum, *self._Node1, *self._Node2, 
                    self._intgrN, self._Sect.uniqNum, self._Transf, '-iter', self._maxIer, self._tol,
                    '-mass', self._mass, '-itegration', self._intgrType)

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
        # timeSeries('Constant', tag, '-factor', factor=1.0)
        ops.timeSeries('Constant', self._uniqNum)

class OpsLinearTimeSerise(OpsTimeSeries):
    @Comp.CompMgr()
    def __init__(self, name=""):
        super().__init__(name)
        self._type += '->LinearTimeSeries'
    
    def _create(self):
    # timeSeries('Linear', tag, '-factor', factor=1.0, '-tStart', tStart=0.0)
        ops.timeSeries('Linear', self._uniqNum)

class OpsTimeSeriesEnum(Enum):
    Linear = OpsLinearTimeSerise()
    Const = OpsConstTimeSeries()

class OpsPlainLoads(Comp.OpsObj):
    @abstractmethod
    def __init__(self, name=""):
        super().__init__(name)
        self._type += '->OpsLoads'
    
    @abstractmethod
    def _create(self):
        # opattern('Plain', patternTag, tsTag, '-fact', fact)
        ops.pattern('Plain', self._uniqNum, OpsTimeSeriesEnum.Linear.value.uniqNum)

class OpsNodeLoad(OpsPlainLoads):
    def __init__(self, load:tuple[float], node:OpsNode, name=""):
        super().__init__(name)
        self._type += '->OpsNodeLoad'
        self._Load = load
        self._Node = node
    
    def _create(self):
        # load(nodeTag, *loadValues)
        super()._create()
        ops.load(self._Node.uniqNum, *self._Load)

class OpsEleLoad(OpsPlainLoads):
    def __init__(self, load:tuple[float], ele:OpsElement, name=""):
        super().__init__(name)
        self._type += '->OpsElementLoad'
        self._Load = load
        self._Element = ele
        self._Element._Transf._vecz
        self._Element._Node1._xyz
        self._Element._Node2._xyz
        UtilTools.PointsTools.vectSub(x, y)

    def _create(self):
        super()._create()
        # ops.eleLoad('-ele', self._Element.uniqNum, '-type', '-beamUniform', , <Wz>, Wx=0.0)
        