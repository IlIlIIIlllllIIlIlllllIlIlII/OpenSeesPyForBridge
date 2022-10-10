from cmath import isfinite
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from platform import node
from signal import raise_signal
from symbol import annassign
from sys import flags
# from email import message
# from multiprocessing import pool
# from sys import flags
# from tracemalloc import StatisticDiff
from typing import overload
from webbrowser import BackgroundBrowser
import numpy as np
import matplotlib.pyplot as plt

from src.Unit import ConvertToBaseUnit

from . import UtilTools
from . import Comp
from . import Load
from . import OpsObject
from . import Part
from . import GlobalData
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
    dt = 0
    damp = 0

    def setDeltaT(self, dt:float):
        self.dt = dt

    def setConStrains(self, consEnum:ConstraintsEnum):
        self.constraints = [consEnum.value]

    def setNumberer(self, numbererEnum:NumbererEnum):
        self.numberer = [numbererEnum.value]

    def setSystem(self, systemEnum:SystemEnum, icntl14=20.0, icntl7=7):
        if systemEnum != SystemEnum.Mumps:
            StandardLogger.info("INFO: systemEnum is not Mumps, icntl14 and inctly is ignored")
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
            StandardLogger.info("INFO: testEnum is NormDispIncr, args should be [tol, Iter, pFlag, nType]")
            self.test = [testEnum.value]
            self.test += args
            if len(args) == 2:
                self.test += [0, 2]
        elif testEnum == TestEnum.NormUnbalance:
            StandardLogger.info("INFO: testEnum is NormUnbalance, args should be [tol, iter, pFlag=0, nType=2, maxIncr=maxIncr]")
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
            StandardLogger.info("INFO: Alogrithm is Linear, args should be [secant=False, initial=False, factorOnce=False]")
            self.algorithm = [alogrithmEnum.value]
            self.algorithm += args
            if len(args) == 0:
                self.algorithm += [False, False, False]
        elif alogrithmEnum == AlgorithmEnum.Newton:
            StandardLogger.info("INFO: Alogrithm is Newton, args should be [secant=False, initial=False, initialThenCurrent=False]")
            self.algorithm = [alogrithmEnum.value]
            self.algorithm += args
            if len(args) == 0:
                self.algorithm += [False, False, False]
        elif alogrithmEnum == AlgorithmEnum.ModifiedNewton:
            StandardLogger.info("INFO: Alogrithm is ModifiedNewton, args should be [secant=False, initial=False]")
            self.algorithm = [alogrithmEnum.value]
            self.algorithm += args
            if len(args) == 0:
                self.algorithm += [False, False]
        elif alogrithmEnum == AlgorithmEnum.NewtonLineSearch:
            StandardLogger.info("INFO: Alogrithm is NewtonLineSearch, args should be [Bisection=False, Secant=False, RegulaFalsi=False, InitialInterpolated=False, tol=0.8, maxIter=10, minEta=0.1, maxEta=10.0]")
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
            StandardLogger.info('Integrator is DisplacementControl, args should be [nodeTag, dof, incr, numIter=1, dUmin=incr, dUmax=incr]')
            # print("INFO: Integrator is DisplacementControl, args should be [nodeTag, dof, incr, numIter=1, dUmin=incr, dUmax=incr]")
            
            self.integrator = [interEnum.value]
            self.integrator += args
        elif interEnum == InteratorEnum.Static.LoadControl:
            StandardLogger.info('INFO: Integrator is LoadControl, args should be [incr, numIter=1, minIncr=incr, maxIncr=incr]')
            # print("INFO: Integrator is LoadControl, args should be [incr, numIter=1, minIncr=incr, maxIncr=incr]")
            self.integrator = [interEnum.value]
            self.integrator += args
        elif interEnum == InteratorEnum.Transient.Newmark:
            StandardLogger.info("INFO: Integrator is Newmark, args should be [gamma, beta, '-form', form]")
            # print("INFO: Integrator is Newmark, args should be [gamma, beta, '-form', form]")
            self.integrator = [interEnum.value]
            self.integrator += [1/2, 1/4]
        else:
            raise Exception("Unfinished")

    def setAnalyz(self, N:int=1, dt:float=0.1, dtMin=0.1, dtMax=0.1, Jd=0):
        self.analyze = [N, dt]
    
    def setDamp(self, damp):
        self.damp = damp

class DEF_ANA_PARAM:
    @classmethod
    @property
    def DefaultGrivateAnalysParam(cls):
        
        anaParm = AnalysParams()
        anaParm.setDeltaT(0.01)
        anaParm.setAlogrithm(AlgorithmEnum.Newton)
        anaParm.setAnalyz(1)
        anaParm.setIntegrator(InteratorEnum.Static.LoadControl, anaParm.dt)
        anaParm.setNumberer(NumbererEnum.RCM)
        anaParm.setSystem(SystemEnum.BandSPD)
        anaParm.setTest(TestEnum.NormDispIncr, 10-4, 10)
        anaParm.setConStrains(ConstraintsEnum.Transformaiton)

        return anaParm

    @classmethod
    @property
    def DefaultStaticAnalysParam(cls):
        return cls.DefaultGrivateAnalysParam
    
    @classmethod
    @property
    def DefaultSeismicAnalysParam(cls):
        
        anaParm = AnalysParams()
        anaParm.setDeltaT(0.01)        
        anaParm.setAlogrithm(AlgorithmEnum.Newton)
        anaParm.setAnalyz(1, anaParm.dt)
        anaParm.setIntegrator(InteratorEnum.Transient.Newmark)
        anaParm.setNumberer(NumbererEnum.RCM)
        anaParm.setSystem(SystemEnum.BandSPD)
        anaParm.setTest(TestEnum.NormDispIncr, 10-1, 10)
        anaParm.setConStrains(ConstraintsEnum.Transformaiton)
        anaParm.setDamp(0.2)

        return anaParm

class NodeRst:
    def __init__(self, BridgeNode, nodeRes) -> None:
        self._rst = nodeRes
        self._node = BridgeNode

    
    @property
    def ux(self):
        return self._rst[0]

    @property
    def uy(self):
        return  self._rst[1]

    @property
    def uz(self):
        return self._rst[2]

    @property
    def ratx(self):
        return self._rst[3]

    @property
    def raty(self):
        return self._rst[1]

    @property
    def ratz(self):
        return self._rst[1]

class EleRst:
    def __init__(self, ele:OpsObject.OpsLineElement) -> None:
        ...
class AnalsisModel(Comp.Component):
    # _StaticLoad:list[Load.StaticLoads] = []
    # _SeismicLoads:list[Load.DynamicLoads] = []
    
    # # _Gravity:list[Load.Gravity] = []

    # _BoundaryNodes:list[Part.BridgeNode] = []

    # _StaticPattern = OpsObject.OpsPlainLoadPattern()

    # _DynamicPattern = OpsObject.OpsMSELoadPattern()
    # _EarthquackNode:list[Part.BridgeNode] = []

    # _CustomPatternList:list = []
    # _SegmentList:list[Part.Segment] = []

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
        OpsCommandLogger.info('ops.wipe()')
        ops.wipe()

        cls._SegmentList:list[Part.Segment] = []
        cls._CuboidsList:list[Part.Cuboid] = []
        cls._SpecialNodeList:list[Part.BridgeNode] = []
        cls._SpecialElementList:list[OpsObject.OpsLineElement] = []

        # cls._BoundaryNodes:list[Part.BridgeNode] = []
        # cls._BoundaryList:list[Part.BoundaryDescriptor] = []
        cls._BoundaryDict:dict = {}
        
        cls._StaticLoad:list[Load.StaticLoads] = []
        cls._SeismicLoads:list[Load.EarthquakeLoads] = []
        cls._CustomPatternList:list = []
        cls._StaticPattern = OpsObject.OpsPlainLoadPattern()
        cls._DynamicPattern = OpsObject.OpsMSELoadPattern()
        cls._FEMBuilt = False
        cls._AnalsyFinshed = False

    @classmethod
    def StoreBoundaryToDict(cls, key, boundary):
        if key in cls._BoundaryDict and boundary not in cls._BoundaryDict[key]:
            cls._BoundaryDict[key].append(boundary)
        else:
            cls._BoundaryDict[key] = [boundary]

    @classmethod
    def AddSegment(cls, seg:Part.Segment):
        cls._SegmentList.append(seg)


    @classmethod
    def AddBoundary(cls, boundary:Part.BoundaryDescriptor, checkFunc=lambda x:True if x else False):
        if not checkFunc:
            print("boundary {} failed the checkFunc")
            return False
        if isinstance(boundary, Part.BridgeFixedBoundary):
            cls.StoreBoundaryToDict(boundary.BridgeNode, boundary)

        elif isinstance(boundary, Part.BridgeBearingBoundary):
            nodeI = boundary._NodeI
            nodeJ = boundary._NodeJ
            
            cls.StoreBoundaryToDict(nodeI, boundary)
            cls.StoreBoundaryToDict(nodeJ, boundary)

        elif isinstance(boundary, Part.BridgeSimplePileSoilBoundary):
            nodeI = boundary._NodeI
            nodeJ = boundary._NodeJ
            cls.StoreBoundaryToDict(nodeI, boundary)
            cls.StoreBoundaryToDict(nodeJ, boundary)

        elif isinstance(boundary, Part.BridgeFullPileSoilBoundary):
            segs = boundary._pileSegs
            for seg in segs:
                cls.StoreBoundaryToDict(seg, boundary)
        elif isinstance(boundary, Part.BridgeEQDOFSBoundary):
            nodeI = boundary._nodeI
            nodeJ = boundary._nodeJ
            cls.StoreBoundaryToDict(nodeI, boundary)
            cls.StoreBoundaryToDict(nodeJ, boundary)
                

        else:
            msg = "Unsupported boundary type {}, ignored".format(boundary.__class__)
            StandardLogger.warning(msg)

            return False

        return True

    @classmethod
    def checkModel(): ...
    
    @classmethod
    def buildBoundary(cls, checkFunc=lambda k, v:True if k and v else False):
        #* Expand complex boundaries to lower level boundary
        ExpandBoudary = []
        for key, val in cls._BoundaryDict.items():
            for boundary in val:
                if isinstance(boundary, Part.BridgeFullPileSoilBoundary) and not boundary._activated:
                    x, y, z = boundary._activate()
                    cls._CuboidsList += boundary._SoilCuboids
                    ExpandBoudary += x
                    ExpandBoudary += y
                    ExpandBoudary += z
                    # for exboundary in (x+y+z):
                        # cls.AddBoundary(exboundary)
        # *Add Expand boundaries to the boundary dictionary
        for expB in ExpandBoudary:
            cls.AddBoundary(expB)


        # * The boundary dictionary is traversed to create the corresponding boundary 
        # * And the boundary of the same node can be checked to exclude conflicts by checkFunc()
        for key, val in cls._BoundaryDict.items():
            if not checkFunc(key, val):
                raise Exception("failed to pass checkFunc")
            
            for boundary in val:
                if isinstance(boundary, Part.BridgeFixedBoundary) and not boundary._activated:
                    flag, node = cls.Inquire.FindNode(boundary.BridgeNode.point)
                    if flag:
                        boundary._activate()
                    else:
                        raise Exception('can not find BridgeNode: {}'.format(boundary.BridgeNode.point))
                elif isinstance(boundary, Part.BridgeBearingBoundary) and not boundary._activated:
                    flag1, node1 = cls.Inquire.FindNode(boundary._NodeI.point)
                    flag2, node2 = cls.Inquire.FindNode(boundary._NodeJ.point)

                    if flag1 and flag2:
                        ele = boundary._activate()
                        cls._SpecialElementList.append(ele)
                elif isinstance(boundary, Part.BridgeEQDOFSBoundary) and not boundary._activated:
                    flag1, node1 = cls.Inquire.FindNode(boundary._nodeI.point)
                    flag2, node2 = cls.Inquire.FindNode(boundary._nodeJ.point)

                    if flag1 and flag2:
                        ele = boundary._activate()
                elif isinstance(boundary, Part.BridgeSimplePileSoilBoundary) and not boundary._activated:
                    ...
                elif isinstance(boundary, Part.BridgeFullPileSoilBoundary) and not boundary._activated:
                    msg = 'Unexpted BridgeFullPileSoilBoundary:{}, all this Boundary should be expanded to lower form boundary'.format(boundary._uniqNum)
                    StandardLogger.error(msg)
                    raise Exception(msg)
                elif boundary._activate:
                    msg = 'Boundary:{} {} has been expanded'.format(boundary._type, boundary._uniqNum)
                    StandardLogger.info(msg)
                else:
                    msg = 'Unspported BoundaryType:{}'.format(boundary._type)
                    raise Exception(msg)
                    
                


        
    @classmethod
    def AddFixBoundary(cls, points:list[tuple[float, ...]], fixval:list[int]):
        if type(points) is tuple:
            points = [points]

        if not UtilTools.Util.isOnlyHas(fixval, [0, 1], flatten=True):
            raise Exception("Wrong Params:{}".format(fixval))

        for p in points:
            flag, node = cls.Inquire.FindNode(p)
            if flag:
                # cls._BoundaryNodes.append(node)
                # cls._BoundaryList.append(Part.BridgeFixedBoundary(node, fixval))
                cls.StoreBoundaryToDict(node, Part.BridgeFixedBoundary(node, fixval))
            else:
                StandardLogger.warning("point:{} can find in this Analsy model, ignored".format(p))

    @classmethod
    def AddPlasticBearingBoundary(cls, p1:tuple[float], E:float=None, plasticStrain= ConvertToBaseUnit(0.02, 'm'), dirVal:list[int]=[1, 1, 0, 0, 0, 0]):
        
        if not UtilTools.Util.isOnlyHas(dirVal, [0, 1], flatten=True):
            raise Exception("Wrong Params:{}".format(dirVal))

        flag1, node1 = cls.Inquire.FindNode(p1)
        x, y, z = p1
        if flag1:
            node2 = Part.BridgeNode(x+GlobalData.DEFVAL._COOROFFSET_, y+GlobalData.DEFVAL._COOROFFSET_, z+GlobalData.DEFVAL._COOROFFSET_)
            AnalsisModel._SpecialNodeList.append(node2)
            AnalsisModel.AddFixBoundary(node2.point, [1]*6)

            cls._BoundaryNodes.append(node1)
            cls._BoundaryNodes.append(node2)
            b = Part.BridgeBearingBoundary(node1, node2, plasticStrain, E, dirVal)
            AnalsisModel._SpecialElementList.append(b._BearingElement)
            AnalsisModel._BoundaryList.append(b)
        else:
            StandardLogger.warning("point:{} can find in this Analsy model, ignored".format(p1))
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
    def AddStaticLoads(cls, load:Load.StaticLoads):
        
        cls._AnalsyFinshed = False
        cls._StaticLoad.append(load)

    @classmethod
    def AddEarthquack(cls, eq:Load.SeismicWave):
        cls._AnalsyFinshed = False
        cls._SeismicLoads.append(eq)

    @classmethod
    def buildFEM(cls):
        cls.buildBoundary()
        eles = cls.Inquire.AllElements
        for ele in eles:
            ele.uniqNum
        cls._FEMBuilt = True
     
    @classmethod
    def RunGravityAnalys(cls, res_func, points:list[tuple[float]]=None, analysParam:AnalysParams=DEF_ANA_PARAM.DefaultGrivateAnalysParam):
        OpsCommandLogger.info('ops.wipeAnalysis()')
        ops.wipeAnalysis()
        OpsCommandLogger.info('ops.loadConst(\'{}\', {})'.format('-time', 0.0))
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
            ops.reactions()
            OpsCommandLogger.info('ops.constraints(*{})'.format(analysParam.constraints))
            ops.constraints(*analysParam.constraints)
            
            OpsCommandLogger.info('ops.integrator(*{})'.format(analysParam.integrator))
            ops.integrator(*analysParam.integrator)

            OpsCommandLogger.info('ops.numberer(*{})'.format(analysParam.numberer))
            ops.numberer(*analysParam.numberer)

            OpsCommandLogger.info('ops.system(*{})'.format(analysParam.system))
            ops.system(*analysParam.system)

            OpsCommandLogger.info('ops.test(*{})'.format(analysParam.test))
            ops.test(*analysParam.test)

            OpsCommandLogger.info('ops.algorithm(*{})'.format(analysParam.algorithm))
            ops.algorithm(*analysParam.algorithm)
            # ops.algorithm('ModifiedNewton', '-initial')

            OpsCommandLogger.info('ops.analysis(\'{}\')'.format('Static'))
            ops.analysis('Static')

            # nodes:list[Part.BridgeNode] = []
            # warningmess = "can not find point:{} in this analysis model"

            # if points:
            #     for p in points:

            #         flag, node = cls.Inquire.FindeNode(p)
            #         if flag:
            #             nodes.append(node)
            #         else:
                        
            #             StandardLogger.warning(warningmess.format(p))
            #             print(warningmess.format(p))
                
            times = []
            OpsCommandLogger.info('ops.getTime()'.format())
            time = ops.getTime()
            times.append(time)

            res = []

            while(time < 1):
                
                OpsCommandLogger.info('ops.analyze(*{})'.format(analysParam.analyze))
                output = ops.analyze(*analysParam.analyze)
                if output != 0:
                    mess = "Opensees Analyze failed, return code: {}".format(output)
                    print(mess)
                    raise Exception(mess)

                OpsCommandLogger.info('ops.getTime()'.format())
                time = ops.getTime()
                times.append(time)
                

                nodes_res = []
                if points:
                    for p in points:
                        n_res = res_func(p)
                    nodes_res.append(n_res)
                    
                else:
                    nodes_res = res_func()
                
                nodes_res = np.array(nodes_res)

                res.append(nodes_res)
            
            times = times[:-1]
            
            res = np.array(res)
            res = np.transpose(res, [1, 2, 0])

            cls._AnalsyFinshed = True

            return AnalsisModel.Rst(times, points, res, 'Gravity')

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

        OpsCommandLogger.info('ops.loadConst(\'{}\', {})'.format('-time', 0.0))
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
        time = ops.getTime()
        print(time)
        while(time < 1):
            
            ops.analyze(*analysParam.analyze)

        cls._AnalsyFinshed = True
        
        return AnalsisModel.Rst('Static')
        
    @classmethod
    def RunSeismicAnalysis(cls, res_func, duraT=0.0, points:list[tuple[float]]=None, analysParam:AnalysParams=DEF_ANA_PARAM.DefaultSeismicAnalysParam):

        OpsCommandLogger.info('ops.wipeAnalysis()')
        ops.wipeAnalysis()
        OpsCommandLogger.info('ops.loadConst(\'{}\', {})'.format('-time', 0.0))
        ops.loadConst('-time', 0.0)
        freq = ops.eigen('-fullGenLapack', 1)[0] ** 0.5
        # ops.rayleigh(0, 0, 0, 2*analysParam.damp/freq)
        ops.rayleigh(0.0, 0.0, 0.0, 0.0000625)
        if not cls._FEMBuilt:
            cls.buildFEM()
        
        cls._DynamicPattern.uniqNum
        durT = 0
        if cls._SeismicLoads:
            for load in cls._SeismicLoads:
                load.ApplyLoad()

            if duraT != 0.0:
                durT = duraT
            else:
                durT = load._seismicWave._information.dt * load._seismicWave._information.npts
 

            OpsCommandLogger.info('ops.constraints(*{})'.format(analysParam.constraints))
            ops.constraints(*analysParam.constraints)
            
            OpsCommandLogger.info('ops.integrator(*{})'.format(analysParam.integrator))
            ops.integrator(*analysParam.integrator)

            OpsCommandLogger.info('ops.numberer(*{})'.format(analysParam.numberer))
            ops.numberer(*analysParam.numberer)

            OpsCommandLogger.info('ops.system(*{})'.format(analysParam.system))
            ops.system(*analysParam.system)

            OpsCommandLogger.info('ops.test(*{})'.format(analysParam.test))
            ops.test(*analysParam.test)

            OpsCommandLogger.info('ops.algorithm(*{})'.format(analysParam.algorithm))
            ops.algorithm(*analysParam.algorithm)
            # ops.algorithm('ModifiedNewton', '-initial')

            OpsCommandLogger.info('ops.analysis(\'{}\')'.format('Transient'))
            ops.analysis('Transient')

            # nodes:list[Part.BridgeNode] = []
            # warningmess = "can not find point:{} in this analysis model"

            # if points:
            #     for p in points:

            #         flag, node = cls.Inquire.FindeNode(p)
            #         if flag:
            #             nodes.append(node)
            #         else:
                        
            #             StandardLogger.warning(warningmess.format(p))
            #             print(warningmess.format(p))
                
            times = []
            OpsCommandLogger.info('ops.getTime()'.format())
            time = ops.getTime()
            times.append(time)

            res = []
            Norms = []
            while(time < durT):
                
                OpsCommandLogger.info('ops.analyze(*{})'.format(analysParam.analyze))
                output = ops.analyze(*analysParam.analyze)

                norms = ops.testNorm()
                iters = ops.testIter()

                for j in range(iters):
                    Norms.append(norms[j])

                if output != 0:
                    mess = "Opensees Analyze failed, return code: {}".format(output)
                    print(mess)
                    plt.semilogy(Norms,'k-x')
                    
                    raise Exception(mess)

                OpsCommandLogger.info('ops.getTime()'.format())
                time = ops.getTime()
                times.append(time)
                
                nodes_res = []
                if points:
                    for p in points:
                        n_res = res_func(p)
                    nodes_res.append(n_res)
                    
                else:
                    nodes_res = res_func()
                
                nodes_res = np.array(nodes_res)

                res.append(nodes_res)
            
            times = times[:-1]
            
            res = np.array(res)
            res = np.transpose(res, [1, 2, 0])

            cls._AnalsyFinshed = True

            plt.semilogy(Norms,'k-x')
            return AnalsisModel.Rst(times, points, res, 'Seismic')

        else:
            raise Exception("No Seismic parameter is added")

    class Inquire:
        @classmethod
        def FindNode(cls, point:tuple):
            for seg in AnalsisModel._SegmentList:
                flag, n = seg.FindBridgeNode(point)
                if flag:
                    return True, n

            for n in AnalsisModel._SpecialNodeList:
                if n.point == point:
                    return True, n

            for b in AnalsisModel._CuboidsList:
                flag, n = b.FindBridgeNode(point)
                if flag:
                    return True, n

            return False, None
        
        @classmethod
        def FindNodeByTag(cls, nodeTag):
            flag, node = Comp.CompMgr.getCompByUniqNum(nodeTag, Part.BridgeNode)
            return flag, node

        @classmethod
        def FindSectByTag(cls, sectTag):
            flag, sect = Comp.CompMgr.getCompByUniqNum(sectTag, Part.CrossSection)
            return flag, sect

        @classmethod
        def FindSegElement(cls, p1:tuple, p2:tuple):
            for seg in AnalsisModel._SegmentList:
                flag, e = seg.FindElement(p1, p2)
                if flag:
                    return True, e

            for ele in AnalsisModel._SpecialElementList:
                if (ele.NodeI.xyz == p1 and ele.NodeJ.xyz == p2) or (ele.NodeJ.xyz == p1 and ele.NodeI.xyz == p2):
                    return True, e

            return False, None
        
        @classmethod
        def FindCuboidElement(cls, p3):
            for b in AnalsisModel._CuboidsList:
                flag, n = b.FindElement(p3)
                if flag:
                    return True, n
            return False, None
        
        @classmethod
        def FindElementByTag(cls, eleTag):
            
            flag, ele = Comp.CompMgr.getCompByUniqNum(eleTag, OpsObject.OpsLineElement)
            return flag, ele
            

        @classmethod
        def FindElementHasPoint(cls, point:tuple[float]):
            res = []
            for seg in AnalsisModel._SegmentList:
                for ele in seg.ELeList:
                    if point in (ele.NodeI.xyz, ele.NodeJ.xyz):
                        res.append(ele)

            for ele in AnalsisModel._SpecialElementList:
                if point in (ele.NodeI.xyz, ele.NodeJ.xyz):
                    res.append(ele)

            for b in AnalsisModel._CuboidsList:
                flag, eles = b.FindElementHasPoint(point)
                if flag:
                    res += eles

            if len(res) == 0:
                return False, None
            else:
                return True, res
        
        @classmethod
        def FindSeg(cls, p1, p2):

            for seg in AnalsisModel._SegmentList:
                sp = seg.NodeList[0].point
                ep = seg.NodeList[-1].point
                if UtilTools.PointsTools.IsPointInLine(p1, sp, ep) and UtilTools.PointsTools.IsPointInLine(p2, sp, ep):
                    return True, seg
            
            return False, None
        
        @classmethod
        def FindSegsByOnePoint(cls, p1):
            segs = []
            for seg in AnalsisModel._SegmentList:
                sp = seg.NodeList[0].point
                ep = seg.NodeList[-1].point
                if UtilTools.PointsTools.IsPointInLine(p1, sp, ep):
                    segs.append(seg)

            if segs:
                return True, segs
            else:
                return False, None

        @classmethod
        @property
        def AllNodes(cls) -> list[Part.BridgeNode]:
            nodes = []
            for seg in AnalsisModel._SegmentList:
                nodes += seg.NodeList

            nodes += AnalsisModel._SpecialNodeList

            for b in AnalsisModel._CuboidsList:
                nodes += b._BridgeNodes.flatten().tolist()

            return nodes

        @classmethod
        @property
        def AllElements(cls) -> list[OpsObject.OpsLineElement]:
            eles = []
            for seg in AnalsisModel._SegmentList:
                eles += seg.ELeList

            eles += AnalsisModel._SpecialElementList

            for b in AnalsisModel._CuboidsList:
                eles += b._Elements.flatten().tolist()
            return eles
        
        @classmethod
        @property
        def AllSegments(cls) -> list[Part.Segment]:
            return AnalsisModel._SegmentList

        @classmethod
        @property
        def AllCuboids(cls) -> list[Part.Segment]:
            return AnalsisModel._CuboidsList
        
        @classmethod
        def listNodeInfo(clso, bridgeNode:Part.BridgeNode) -> None:
            info = "BridgeNode tag:{}\nlocation:{}\nmass:{}\nOpsElement tag:{}\n"
            info = info.format(bridgeNode._uniqNum, bridgeNode.point, bridgeNode.Mass, bridgeNode.OpsNode.uniqNum)

        @classmethod
        def listAllNodeInfo(cls) -> None:
            nodes = cls.AllNodes
            for node in nodes:
                cls.listNodeInfo(node)

        @classmethod
        def listAllElementsInfo(cls):
            eles = cls.AllElements
            for ele in eles:
                cls.listElementInfo(ele)
                
        @classmethod
        def listElementInfo(cls, ele:OpsObject.OpsLineElement):
            info = 'Element tag:{}\nNodeI - tag:{} - location:{}\nNodeJ - tag:{} - location:{}\nSect tag:{}\n'

            info = info.format(ele.uniqNum, ele.NodeI.uniqNum, ele.NodeI.xyz, ele.NodeI.uniqNum, ele.NodeI.xyz, ele.Sect.uniqNum)
            
            print(info)

        @classmethod
        def listSegmentInfo(cls, seg:Part.Segment):
            info ='Segment tag:{}\nSegment nodes tag:{}\nSegment elements tag:{}\nSegment Element Length List:{}\nSegment Cross-Section tags:{}\nSegment Mass list:{}\n'
            nodes = []
            eles = []
            sects = []

            for node in seg.NodeList:
                nodes.append(node._uniqNum) 
            for ele in seg.ELeList:
                eles.append(ele._uniqNum)
            for sect in seg.SectList:
                sects.append(sect._uniqNum)

            info = info.format(seg._uniqNum, nodes, eles, seg.EleLength, sects, seg.MassList)
        
        @classmethod
        def listAllSegments(cls):
            segs = cls.AllSegments
            for seg in segs:
                cls.listSegmentInfo(seg)

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
        def __init__(self, times:list[float], pointslist:list[int], res:np.ndarray, analysType:str):
            self._type = analysType
            self._nodes = pointslist
            #* res.shape = (len(nodelist), len([ux, uy, uz, ....], len(times)))
            self._res:np.ndarray = res
            self._times = np.array(times)

        def getFrame(self, time:float):
            index = np.argmin(np.abs(self._times - time))
            if self._times[index] != time:
                StandardLogger.warning("cant find time {}, return the clost time {}".format(time, self._times[index]))
            
            return self._res[:, :, index]

        def getNodeValueTimes(self, nodeIndex):
            if nodeIndex < self._res.shape[0]:
                return self._times, self._res[nodeIndex, :, :]
            else:
                StandardLogger.warning("cant find nodeIndex {}".format(nodeIndex))
        
        def getValueTimes(self, nodeIndex, valueIndex):
            if nodeIndex < self._res.shape[0] and valueIndex < self._res.shape[1]:
                return self._times, self._res[nodeIndex, valueIndex, :]

        def __getitem__(self, arg):
            if self._nodes:
                if type(arg) is tuple:
                    # flag, node = AnalsisModel.Inquire.FindeNode(arg)
                    # index = self._nodes.index(arg)
                    try:
                        index = self._nodes.index(arg)
                    except ValueError:
                        mess = "can't find node:{}".format(arg)
                        StandardLogger.warning(mess)
                        print(mess)
                        return None
                    
                    return self.getNodeValueTimes(index)
                else:
                    StandardLogger.warning("unspported arg:{}".format(arg))
            else:
                StandardLogger.warning("Missing log point information, use getNodeValueTimes()")

            

 