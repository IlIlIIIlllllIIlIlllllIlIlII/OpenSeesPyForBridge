from dataclasses import dataclass
from typing import overload

from . import UtilTools
# from src.UtilTools import Util
from . import Comp
from . import Load
from . import OpsObject
from . import Part
from enum import Enum
import openseespy.opensees as ops
from .log import *

class ConstraintsEnum(Enum):
    Plain = 'Plain'
    Penalty = 'Penalty'
    Transformaiton = 'Transformation'
    Lagrange = 'Lagrange'

class NumbererEnum(Enum): 
    Plain = 'Plain' 
    RCM = 'RCM' 
    AMD = 'AMD' 
    ParallelPlain = 'ParallelPlain'
    ParallelRCM = 'ParallelRCM'

class SystemEnum(Enum):
    BandGeneral = 'BandGen'
    BandSPD = 'BandSPD'
    ProfileSPD = 'ProfileSPD'
    SuperLU = 'SuperLU'
    Umfpack = 'UmfPack'
    FullGeneral = 'FullGeneral'
    SparseSYM = 'SparseSYM'
    Mumps = 'Mumps'

class TestEnum(Enum):
    NormUnbalance = 'NormUnbalance'
    NormDispIncr = 'NormDispIncr'


class AlgorithmEnum(Enum):
    Linear = 'Linear'
    Newton = 'Newton'
    NewtonLineSearch = 'NewtonLineSearch'
    ModifiedNewton = 'ModifiedNewton'
    # TODO

class InteratorEnum():
    class Static(Enum):
        LoadControl = 'LoadControl'
        DisplacementControl = 'DisplacementControl'
    class Transient(Enum):
        Newmark = 'Newmark'
    

@dataclass
class AnalysParams():
    constraints = []
    numberer = []
    system = []
    test = []
    algorithm = []
    integrator = []
    analyze = []

    def setConStrains(self, consEnum:ConstraintsEnum):
        self.constraints = [consEnum.value]
        

    def setNumberer(self, numbererEnum:NumbererEnum):
        self.numberer = [numbererEnum.value]

    def setSystem(self, systemEnum:SystemEnum, icntl14=20.0, icntl7=7):
        if systemEnum != SystemEnum.Mumps:
            logger.info("INFO: systemEnum is not Mumps, icntl14 and inctly is ignored")
            self.system = [systemEnum.value]
        else:
            self.system = [systemEnum.value]
            self.system += [icntl14, icntl7]

    @overload
    def setTest(self, testEnum=TestEnum.NormDispIncr, tol:float=None, Iter:int=None, pFlag:int=0, nType:int=2) -> None: ...
    @overload
    def setTest(self, testEnum=TestEnum.NormUnbalance, tol:float=None, Iter:int=None, pFlag:int=0, nType:int=2) -> None: ...

    def setTest(self, testEnum:TestEnum, *args):
        if testEnum == TestEnum.NormDispIncr:
            logger.info("INFO: testEnum is NormDispIncr, args should be [tol, Iter, pFlag, nType]")
            self.test = [testEnum.value]
            self.test += args
            if len(args) == 2:
                self.test += [0, 2]
        elif testEnum == TestEnum.NormUnbalance:
            logger.info("INFO: testEnum is NormUnbalance, args should be [tol, iter, pFlag=0, nType=2, maxIncr=maxIncr]")
            self.test = [testEnum.value]
            self.test += args
        else:
            raise Exception("Unfinshed")
    @overload
    def setAlogrithm(self, alogrithmEnum=AlgorithmEnum.Linear, secant=False, initial=False, factorOnce=False)->None:...
    @overload
    def setAlogrithm(self, alogrithmEnum=AlgorithmEnum.Newton, secant=False, initial=False, initialThenCurrent=False)->None:...
    @overload
    def setAlogrithm(self, alogrithmEnum=AlgorithmEnum.ModifiedNewton, secant=False, initial=False)->None:...
    @overload
    def setAlogrithm(self, alogrithmEnum=AlgorithmEnum.NewtonLineSearch, Bisection=False, Secant=False, RegulaFalsi=False, InitialInterpolated=False, tol=0.8, maxIter=10, minEta=0.1, maxEta=10.0)->None:...

    def setAlogrithm(self, alogrithmEnum:AlgorithmEnum, *args):
        if alogrithmEnum == AlgorithmEnum.Linear:
            logger.info("INFO: Alogrithm is Linear, args should be [secant=False, initial=False, factorOnce=False]")
            self.algorithm = [alogrithmEnum.value]
            self.algorithm += args
            if len(args) == 0:
                self.algorithm += [False, False, False]
        elif alogrithmEnum == AlgorithmEnum.Newton:
            logger.info("INFO: Alogrithm is Newton, args should be [secant=False, initial=False, initialThenCurrent=False]")
            self.algorithm = [alogrithmEnum.value]
            self.algorithm += args
            if len(args) == 0:
                self.algorithm += [False, False, False]
        elif alogrithmEnum == AlgorithmEnum.ModifiedNewton:
            logger.info("INFO: Alogrithm is ModifiedNewton, args should be [secant=False, initial=False]")
            self.algorithm = [alogrithmEnum.value]
            self.algorithm += args
            if len(args) == 0:
                self.algorithm += [False, False]
        elif alogrithmEnum == AlgorithmEnum.NewtonLineSearch:
            logger.info("INFO: Alogrithm is NewtonLineSearch, args should be [Bisection=False, Secant=False, RegulaFalsi=False, InitialInterpolated=False, tol=0.8, maxIter=10, minEta=0.1, maxEta=10.0]")
            self.algorithm = [alogrithmEnum.value]
            self.algorithm += args
            if len(args) == 0:
                self.algorithm += [False, False, False, False, 0.8, 10.0, 0.1, 10.0]
        else:
            raise Exception("Unfinshed")

    @overload
    def setIntegrator(self, interEnum=InteratorEnum.Static.DisplacementControl, nodeTag:int=None, dof:int=None, incr:int=None) -> None: ...
    @overload
    def setIntegrator(self, interEnum=InteratorEnum.Static.LoadControl, incr:int=None, numIter:int=1, minIncr:int=None, maxIncrr:int=None) -> None: ...
    @overload
    def setIntegrator(self, interEnum=InteratorEnum.Transient.Newmark, gamma:float=1/4, beta:float=1/6, argName='-form', form:str='') -> None: ...

    def setIntegrator(self, interEnum, *args):
        if interEnum == InteratorEnum.Static.DisplacementControl:
            logger.info('Integrator is DisplacementControl, args should be [nodeTag, dof, incr, numIter=1, dUmin=incr, dUmax=incr]')
            # print("INFO: Integrator is DisplacementControl, args should be [nodeTag, dof, incr, numIter=1, dUmin=incr, dUmax=incr]")
            
            self.integrator = [interEnum.value]
            self.integrator += args
        elif interEnum == InteratorEnum.Static.LoadControl:
            logger.info('INFO: Integrator is LoadControl, args should be [incr, numIter=1, minIncr=incr, maxIncr=incr]')
            # print("INFO: Integrator is LoadControl, args should be [incr, numIter=1, minIncr=incr, maxIncr=incr]")
            self.integrator = [interEnum.value]
            self.integrator += args
        elif interEnum == InteratorEnum.Transient.Newmark:
            logger.info("INFO: Integrator is Newmark, args should be [gamma, beta, '-form', form]")
            # print("INFO: Integrator is Newmark, args should be [gamma, beta, '-form', form]")
            self.integrator = [interEnum.value]
            self.integrator += [1/2, 1/4]
        else:
            raise Exception("Unfinished")

    def setAnalyz(self, N:int=1, dt:float=0.1, dtMin=0.1, dtMax=0.1, Jd=0):
        self.analyze = [N, dt, dtMin, dtMax, Jd]

class DEF_ANA_PARAM:
    @classmethod
    @property
    def DefaultGrivateAnalysParam(cls):
        dt = 0.01
        
        anaParm = AnalysParams()
        anaParm.setAlogrithm(AlgorithmEnum.Linear)
        anaParm.setAnalyz(1/dt, dt)
        anaParm.setIntegrator(InteratorEnum.Static.LoadControl, 0.1, 1, 0.1, 0.1)
        anaParm.setNumberer(NumbererEnum.RCM)
        anaParm.setSystem(SystemEnum.BandSPD)
        anaParm.setTest(TestEnum.NormDispIncr, 10-4, 10)
        anaParm.setConStrains(ConstraintsEnum.Transformaiton)

        return anaParm

    @classmethod
    @property
    def DefaultStaticAnalysParam(cls):
        return cls.DefaultGrivateAnalysParam

class NodeRst:
    def __init__(self, NodeuniqNum) -> None:
        self._uniqNum = NodeuniqNum

        self._ux = None
        self._uy = None
        self._uz = None

        self._ratx = None
        self._raty = None
        self._ratz = None
    
    @property
    def ux(self):
        if self._ux == None:
            self._ux = ops.nodeDisp(self._uniqNum, 1)
        return self._ux

    @property
    def uy(self):
        if self._uy == None:
            self._uy = ops.nodeDisp(self._uniqNum, 2)
        
        return  self._uy
    @property
    def uz(self):
        if self._uz == None:
            self._uz = ops.nodeDisp(self._uniqNum, 3)

        return self._uz
    @property
    def ratx(self):
        if self._ratx == None:
            self._ratx = ops.nodeDisp(self._uniqNum, 4)

        return self._ratx
    @property
    def raty(self):
        if self._raty == None:
            self._raty = ops.nodeDisp(self._uniqNum, 5)

        return self._raty
    @property
    def ratz(self):
        if self._ratz == None:
            self._ratz = ops.nodeDisp(self._uniqNum, 6)

        return self._ratz
class EleRst:
    def __init__(self, ele:OpsObject.OpsElement) -> None:
        ele.se
class AnalsisModel(Comp.Component):
    _StaticLoad:list[Load.StaticLoads] = []
    _DynamicLoads:list[Load.DynamicLoads] = []
    # _Gravity:list[Load.Gravity] = []

    _StaticPattern = OpsObject.OpsPlainLoadPattern()
    _DynamicPattern = OpsObject.OpsMSELoadPattern()

    _CustomPatternList:list = []
    _SegmentList:list[Part.Segment] = []

    # def InitFlag(func):
    #     @functools.wraps
    #     def wapper(cls, *args, **kwargs):
    #         cls._AnalsyFinshed = False
    #         if cls._FEMBuilt:
    #             try:
    #                 func(*args, **kwargs)
    #             except:
    #                 print("analys failed")
    #                 cls._AnalsyFinshed = False
    #             cls._AnalsyFinshed = True
    #         else:
    #             print("FEM model not built")
    #     return wapper
    # class Res        

    @classmethod
    def AnalysInit(cls):
        Comp.CompMgr.clearComp()
        ops.wipe()

        cls._StaticLoad = []
        cls._DynamicLoads = []
        cls._CustomPatternList = []
        cls._SegmentList = []
        cls._StaticPattern = OpsObject.OpsPlainLoadPattern()
        cls._DynamicPattern = OpsObject.OpsMSELoadPattern()
        cls._FEMBuilt = False
        cls._AnalsyFinshed = False

    @classmethod
    def addSegment(cls, seg:Part.Segment):
        cls._SegmentList.append(seg)
    
    @classmethod
    def addFixBoundary(cls, points:list[tuple[float, ...]], fixval:list[int]):
        if type(points) is tuple:
            points = [points]

        if not UtilTools.Util.isOnlyHas(fixval, [0, 1], flatten=True):
            raise Exception("Wrong Params:{}".format(fixval))

        for p in points:
            flag, node = cls.FindeNode(p)
            if flag:
                Part.BridgeFixedBoundary(node, fixval)
            else:
                logger.warning("point:{} can find in this Analsy model, ignored".format(p))
    
    @classmethod
    def FindeNode(cls, point:tuple):
        for seg in cls._SegmentList:
            flag, n = seg.FindPoint(point)
            if flag:
                return True, n
        return False, None

    # def FindeClosestNode(cls, point:tuple):
    #     for seg in cls._SegmentList:
    #         dis_ = []
    #         dis_.append(UtilTools.PointsTools.PointsDist(point, seg.NodeList[0]))
    #         dis_.append(UtilTools.PointsTools.PointsDist(point, seg.NodeList[1]))

    #         for i, p in enumerate(seg.NodeList[:-2]):
    #             d1 = dis_[i]
    #             d2 = dis_[+1]
    #             d3 = UtilTools.PointsTools.PointsDist(point, p)
    #             dis_.append(d3)
    class Rst:
        def __init__(self, analysType:str):
            self._type = analysType
        
        def __getitem__(self, arg):
            if type(arg) is tuple:
                flag, node = AnalsisModel.FindeNode(arg)
                if flag:
                    return NodeRst(node.OpsNode.uniqNum)
                else:
                    logger.warning("can't find node:{}".format(arg))
                    return None
            else:
                logger.warning("unspported arg:{}".format(arg))

            
    @classmethod
    def FindElement(cls, p1:tuple, p2:tuple):
        for seg in cls._SegmentList:
            flag, e = seg.FindElement(p1, p2)
            if flag:
                return True, e

        return False, None

    @classmethod
    def FindElementHasPoint(cls, point:tuple[float]):
        res = []
        for seg in cls._SegmentList:
            for ele in seg.ELeList:
                if point in (ele.NodeI.xyz, ele.NodeJ.xyz):
                    res.append(ele)

        if len(res) == 0:
            return False, None
        else:
            return True, res
    
    @classmethod
    def FindSeg(cls, p1, p2):

        for seg in cls._SegmentList:
            sp = seg.NodeList[0].point
            ep = seg.NodeList[-1].point
            if UtilTools.PointsTools.isInLine(p1, sp, ep) and UtilTools.PointsTools.isInLine(p2, sp, ep):
                return True, seg
        
        return False, None
    
    @classmethod
    def FindSegsByOnePoint(cls, p1):
        segs = []
        for seg in cls._SegmentList:
            sp = seg.NodeList[0].point
            ep = seg.NodeList[-1].point
            if UtilTools.PointsTools.isInLine(p1, sp, ep):
                segs.append(seg)

        if segs:
            return True, segs
        else:
            return False, None

    @classmethod
    def AllNodes(cls) -> list[Part.BridgeNode]:
        nodes = []
        for seg in cls._SegmentList:
            nodes += seg.NodeList
        return nodes

    @classmethod
    def AllElements(cls) -> list[OpsObject.OpsElement]:
        eles = []
        for seg in cls._SegmentList:
            eles += seg.ELeList
        return eles

    @classmethod
    def AddPattern(cls, type:str):
        if type == 'static':
            custPattern = OpsObject.OpsPlainLoadPattern()
        elif type == 'dynamic':
            custPattern = OpsObject.OpsMSELoadPattern()
        
        cls._CustomPatternList.append(custPattern)
    
    # @classmethod
    # def AddGravity(cls, g:Load.Gravity):
    #     if not cls._Gravity:
    #         cls._Gravity.append(g)
    #     else:
    #         print("already exist gravity, ignored")
    
    @classmethod
    def AddStaticLoads(cls, loads:Load.StaticLoads):
        cls._AnalsyFinshed = False
        cls._StaticLoad.append(loads)

    @classmethod
    def AddEarthquack(cls, eq:Load.SeismicWave):
        cls._AnalsyFinshed = False
        cls._DynamicLoads.append(eq)

    @classmethod
    def buildFEM(cls):
        eles = cls.AllElements()
        for ele in eles:
            ele.uniqNum
        cls._FEMBuilt = True
     
    @classmethod
    def RunGravityAnalys(cls, analysParam:AnalysParams=DEF_ANA_PARAM.DefaultGrivateAnalysParam):
        ops.wipeAnalysis()
        ops.loadConst('-time', 0.0)
        if not cls._FEMBuilt:
            cls.buildFEM()
        cls._AnalsyFinshed = False
        if cls._SegmentList:
            cls._StaticPattern.uniqNum
            g = Load.Gravity(cls._SegmentList)
            g.ApplyLoad()

        # if Analsis._Gravity:
        #     Analsis._StaticPattern.uniqNum
        #     Analsis._Gravity[0].ApplyLoad()

            ops.constraints(*analysParam.constraints)
            ops.integrator(*analysParam.integrator)
            ops.numberer(*analysParam.numberer)
            ops.system(*analysParam.system)
            ops.test(*analysParam.test)
            ops.algorithm(*analysParam.algorithm)
            ops.analysis('Static')
            ops.analyze(*analysParam.analyze)

            cls._AnalsyFinshed = True

            return AnalsisModel.Rst('Grivate')

        else:
            raise Exception("No gravity parameter is added")


    @classmethod
    def RunStaticAnalys(cls, analysParam:AnalysParams=DEF_ANA_PARAM.DefaultStaticAnalysParam, gravity=False, gravityAnaParams:AnalysParams=DEF_ANA_PARAM.DefaultGrivateAnalysParam):
        loads:list[Load.StaticLoads] = []
        ops.wipeAnalysis()
        if not cls._FEMBuilt:
            cls.buildFEM()
        
        cls._AnalsyFinshed = False

        if gravity and gravityAnaParams:
            AnalsisModel.RunGravityAnalys(gravityAnaParams)

        ops.loadConst('-time', 0.0)
        loads = AnalsisModel._StaticLoad

        if len(loads) == 0:
            raise Exception("No Loads exits")

        AnalsisModel._StaticPattern.uniqNum

        for load in loads:
            load.ApplyLoad()
            
        ops.constraints(*analysParam.constraints)
        ops.integrator(*analysParam.integrator)
        ops.numberer(*analysParam.numberer)
        ops.system(*analysParam.system)
        ops.test(*analysParam.test)
        ops.algorithm(*analysParam.algorithm)
        ops.analysis('Static')
        ops.analyze(*analysParam.analyze)

        cls._AnalsyFinshed = True
        
        return AnalsisModel.Rst('Static')
        
    @classmethod
    def RunSeismicAnalysis(cls, analysParam:AnalysParams, gravity=False, gravityAnaParams:AnalysParams=None):

        loads:list[Load.DynamicLoads] = []
        
        ops.wipeAnalysis()
        if not cls._FEMBuilt:
            cls.buildFEM()
        
        cls._AnalsyFinshed = False

        if gravity and gravityAnaParams:
            AnalsisModel.RunGravityAnalys(gravityAnaParams)

        ops.loadConst('-time', 0.0)
        load = AnalsisModel._DynamicLoads

        if len(loads) == 0:
            raise Exception("No Loads exits")

        AnalsisModel._StaticPattern.uniqNum

        for load in loads:
            load.ApplyLoad()
            
        ops.constraints(*analysParam.constraints)
        ops.numberer(*analysParam.numberer)
        ops.system(*analysParam.system)
        ops.test(*analysParam.test)
        ops.algorithm(*analysParam.algorithm)
        ops.analysis('')
        ops.analyze(*analysParam.analyze)

        cls._AnalsyFinshed = True

        return AnalsisModel.Rst('Seismic')
    