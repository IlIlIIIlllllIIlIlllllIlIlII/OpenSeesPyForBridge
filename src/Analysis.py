from dataclasses import dataclass
from typing import overload
from . import Comp
from . import Load
from . import OpsObject
from . import Part
from enum import Enum
import openseespy.opensees as ops

class OpsConstraintsEnum(Enum):
    Plain = 'Plain'
    Penalty = 'Penalty'
    Transformaiton = 'Transformation'
    Lagrange = 'Lagrange'

class OpsNumbererEnum(Enum): 
    Plain = 'Plain' 
    RCM = 'RCM' 
    AMD = 'AMD' 
    ParallelPlain = 'ParallelPlain'
    ParallelRCM = 'ParallelRCM'

class OpsSystemEnum(Enum):
    BandGeneral = 'BandGen'
    BandSPD = 'BandSPD'
    ProfileSPD = 'ProfileSPD'
    SuperLU = 'SuperLU'
    Umfpack = 'UmfPack'
    FullGeneral = 'FullGeneral'
    SparseSYM = 'SparseSYM'
    Mumps = 'Mumps'

class OpsTestEnum(Enum):
    NormUnbalance = 'NormUnbalance'
    NormDispIncr = 'NormDispIncr'


class OpsAlgorithmEnum(Enum):
    Linear = 'Linear'
    Newton = 'Newton'
    NewtonLineSearch = 'NewtonLineSearch'
    ModifiedNewton = 'ModifiedNewton'
    # TODO

class OpsInteratorEnum():
    class Static(Enum):
        LoadControl = 'LoadControl'
        DisplacementControl = 'DisplacementControl'
    class Transient(Enum):
        Newmark = 'Newmark'
    

@dataclass
class OpsAnasysParams():
    constraints = []
    numberer = []
    system = []
    test = []
    algorithm = []
    integrator = []
    analyze = []

    def setConStrains(self, consEnum:OpsConstraintsEnum):
        self.constraints = [consEnum.value]
        

    def setNumberer(self, numbererEnum:OpsNumbererEnum):
        self.numberer = [numbererEnum.value]

    def setSystem(self, systemEnum:OpsSystemEnum, icntl14=20.0, icntl7=7):
        if systemEnum != OpsSystemEnum.Mumps:
            print("systemEnum is not Mumps, icntl14 and inctly is ignored")
            self.system = [systemEnum.value]
        else:
            self.system = [systemEnum.value]
            self.system += [icntl14, icntl7]

    @overload
    def setTest(self, testEnum=OpsTestEnum.NormDispIncr, tol:float=None, Iter:int=None, pFlag:int=0, nType:int=2) -> None: ...
    @overload
    def setTest(self, testEnum=OpsTestEnum.NormUnbalance, tol:float=None, Iter:int=None, pFlag:int=0, nType:int=2) -> None: ...

    def setTest(self, testEnum:OpsTestEnum, *args):
        if testEnum == OpsTestEnum.NormDispIncr:
            print("testEnum is NormDispIncr, args should be [tol, Iter, pFlag, nType]")
            self.test = [testEnum.value]
            self.test += args
            if len(args) == 2:
                self.test += [0, 2]
        elif testEnum == OpsTestEnum.NormUnbalance:
            print("testEnum is NormUnbalance, args should be [tol, iter, pFlag=0, nType=2, maxIncr=maxIncr]")
            self.test = [testEnum.value]
            self.test += args
        else:
            raise Exception("Unfinshed")
    @overload
    def setAlogrithm(self, alogrithmEnum=OpsAlgorithmEnum.Linear, secant=False, initial=False, factorOnce=False)->None:...
    @overload
    def setAlogrithm(self, alogrithmEnum=OpsAlgorithmEnum.Newton, secant=False, initial=False, initialThenCurrent=False)->None:...
    @overload
    def setAlogrithm(self, alogrithmEnum=OpsAlgorithmEnum.ModifiedNewton, secant=False, initial=False)->None:...
    @overload
    def setAlogrithm(self, alogrithmEnum=OpsAlgorithmEnum.NewtonLineSearch, Bisection=False, Secant=False, RegulaFalsi=False, InitialInterpolated=False, tol=0.8, maxIter=10, minEta=0.1, maxEta=10.0)->None:...

    def setAlogrithm(self, alogrithmEnum:OpsAlgorithmEnum, *args):
        if alogrithmEnum == OpsAlgorithmEnum.Linear:
            print("Alogrithm is Linear, args should be [secant=False, initial=False, factorOnce=False]")
            self.algorithm = [alogrithmEnum.value]
            self.algorithm += args
            if len(args) == 0:
                self.algorithm += [False, False, False]
        elif alogrithmEnum == OpsAlgorithmEnum.Newton:
            print("Alogrithm is Newton, args should be [secant=False, initial=False, initialThenCurrent=False]")
            self.algorithm = [alogrithmEnum.value]
            self.algorithm += args
            if len(args) == 0:
                self.algorithm += [False, False, False]
        elif alogrithmEnum == OpsAlgorithmEnum.ModifiedNewton:
            print("Alogrithm is ModifiedNewton, args should be [secant=False, initial=False]")
            self.algorithm = [alogrithmEnum.value]
            self.algorithm += args
            if len(args) == 0:
                self.algorithm += [False, False]
        elif alogrithmEnum == OpsAlgorithmEnum.NewtonLineSearch:
            print("Alogrithm is NewtonLineSearch, args should be [Bisection=False, Secant=False, RegulaFalsi=False, InitialInterpolated=False, tol=0.8, maxIter=10, minEta=0.1, maxEta=10.0]")
            self.algorithm = [alogrithmEnum.value]
            self.algorithm += args
            if len(args) == 0:
                self.algorithm += [False, False, False, False, 0.8, 10.0, 0.1, 10.0]
        else:
            raise Exception("Unfinshed")

    @overload
    def setIntegrator(self, interEnum=OpsInteratorEnum.Static.DisplacementControl, nodeTag:int=None, dof:int=None, incr:int=None) -> None: ...
    @overload
    def setIntegrator(self, interEnum=OpsInteratorEnum.Static.LoadControl, incr:int=None, numIter:int=1, minIncr:int=None, maxIncrr:int=None) -> None: ...
    @overload
    def setIntegrator(self, interEnum=OpsInteratorEnum.Transient.Newmark, gamma:float=1/4, beta:float=1/6, argName='-form', form:str='') -> None: ...

    def setIntegrator(self, interEnum, *args):
        if interEnum == OpsInteratorEnum.Static.DisplacementControl:
            print("Integrator is DisplacementControl, args should be [nodeTag, dof, incr, numIter=1, dUmin=incr, dUmax=incr]")
            
            self.integrator = [interEnum.value]
            self.integrator += args
        elif interEnum == OpsInteratorEnum.Static.LoadControl:
            print("Integrator is LoadControl, args should be [incr, numIter=1, minIncr=incr, maxIncr=incr]")
            self.integrator = [interEnum.value]
            self.integrator += args
        elif interEnum == OpsInteratorEnum.Transient.Newmark:
            print("Integrator is Newmark, args should be [gamma, beta, '-form', form]")
            self.integrator = [interEnum.value]
            self.integrator += [1/2, 1/4]
        else:
            raise Exception("Unfinished")

    def setAnalyz(self, N:int=1, dt:float=0.1, dtMin=0.1, dtMax=0.1, Jd=0):
        self.analyze += [N, dt, dtMin, dtMax, Jd]




class Analsis(Comp.Component):
    _StaticLoad:list[Load.StaticLoads] = []
    _DynamicLoads:list[Load.DynamicLoads] = []
    # _Gravity:list[Load.Gravity] = []

    _StaticPattern = OpsObject.OpsPlainLoadPattern()
    _DynamicPattern = OpsObject.OpsMSELoadPattern()

    _CustomPatternList:list = []
    _SegmentList:list[Part.Segment] = []

    @classmethod
    def AnalysInit(cls):
        cls._StaticLoad = []
        cls._DynamicLoads = []
        cls._CustomPatternList = []
        cls._SegmentList = []
        cls._StaticPattern = OpsObject.OpsPlainLoadPattern()
        cls._DynamicPattern = OpsObject.OpsMSELoadPattern()
        Comp.CompMgr.clearComp()
        ops.wipe()

    @classmethod
    def addSegment(cls, seg:Part.Segment):
        cls._SegmentList.append(seg)

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
        cls._StaticLoad.append(loads)

    @classmethod
    def AddEarthquack(cls, eq:Load.EarthquakeLoads):
        cls._DynamicLoads.append(eq)
    
    @classmethod
    def RunGravityAnalys(cls, analysParam:OpsAnasysParams):
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

        else:
            raise Exception("No gravity parameter is added")


    @classmethod
    def RunStaticAnalys(cls, analysParam:OpsAnasysParams, gravity=False, gravityAnaParams:OpsAnasysParams=None):
        loads:list[Load.StaticLoads] = []
        ops.wipeAnalysis()

        if gravity and gravityAnaParams:
            Analsis.RunGravityAnalys(gravityAnaParams)

        ops.loadConst('-time', 0.0)
        loads = Analsis._StaticLoad

        if len(loads) == 0:
            raise Exception("No Loads exits")

        Analsis._StaticPattern.uniqNum

        for load in loads:
            load.ApplyLoad()
            
        ops.constraints(*analysParam.constraints)
        ops.numberer(*analysParam.numberer)
        ops.system(*analysParam.system)
        ops.test(*analysParam.test)
        ops.algorithm(*analysParam.algorithm)
        ops.analysis('Static')
        ops.analyze(*analysParam.analyze)
        
        
    @classmethod
    def seismicAnalysis(cls, analysParam:OpsAnasysParams, gravity=False, gravityAnaParams:OpsAnasysParams=None):

        loads:list[Load.DynamicLoads] = []
        
        ops.wipeAnalysis()

        if gravity and gravityAnaParams:
            Analsis.RunGravityAnalys(gravityAnaParams)

        ops.loadConst('-time', 0.0)
        load = Analsis._DynamicLoads

        if len(loads) == 0:
            raise Exception("No Loads exits")

        Analsis._StaticPattern.uniqNum

        for load in loads:
            load.ApplyLoad()
            
        ops.constraints(*analysParam.constraints)
        ops.numberer(*analysParam.numberer)
        ops.system(*analysParam.system)
        ops.test(*analysParam.test)
        ops.algorithm(*analysParam.algorithm)
        ops.analysis('')
        ops.analyze(*analysParam.analyze)

    