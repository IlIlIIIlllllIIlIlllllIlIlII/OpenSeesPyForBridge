from ast import arg
from cProfile import Profile
from copyreg import constructor
from dataclasses import dataclass
from . import Comp
from . import Load
from . import OpsObject
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
        self.constraints.append(consEnum.value)
        

    def setNumberer(self, numbererEnum:OpsNumbererEnum):
        self.constraints.append(numbererEnum.value)

    def setSystem(self, systemEnum:OpsSystemEnum, icntl14=20.0, icntl7=7):
        if systemEnum != OpsSystemEnum.Mumps:
            print("systemEnum is not Mumps, icntl14 and inctly is ignored")
            self.system.append(systemEnum.value)
        else:
            self.system += [systemEnum.value, icntl14, icntl7]

    def setTest(self, testEnum:OpsTestEnum, *args):
        if testEnum == OpsTestEnum.NormDispIncr:
            print("testEnum is NormDispIncr, args should be [tol, Iter, pFlag, nType]")
            self.test += [testEnum.value]
            self.test += args
            if len(args) == 2:
                self.test += [0, 2]
        elif testEnum == OpsTestEnum.NormUnbalance:
            print("testEnum is NormUnbalance, args should be [tol, iter, pFlag=0, nType=2, maxIncr=maxIncr]")
            self.test += [testEnum.value]
            self.test += args
        else:
            raise Exception("Unfinshed")

    def setAlogrithm(self, alogrithmEnum:OpsAlgorithmEnum, *args):
        if alogrithmEnum == OpsAlgorithmEnum.Linear:
            print("testEnum is Linear, args should be [secant=False, initial=False, factorOnce=False]")
            self.algorithm += [alogrithmEnum.value]
            self.algorithm += args
            if len(args) == 0:
                self.algorithm += [False, False, False]
        elif alogrithmEnum == OpsAlgorithmEnum.Newton:
            print("testEnum is Newton, args should be [secant=False, initial=False, initialThenCurrent=False]")
            self.algorithm += [alogrithmEnum.value]
            self.algorithm += args
            if len(args) == 0:
                self.algorithm += [False, False, False]
        elif alogrithmEnum == OpsAlgorithmEnum.ModifiedNewton:
            print("testEnum is ModifiedNewton, args should be [secant=False, initial=False]")
            self.algorithm += [alogrithmEnum.value]
            self.algorithm += args
            if len(args) == 0:
                self.algorithm += [False, False]
        elif alogrithmEnum == OpsAlgorithmEnum.NewtonLineSearch:
            print("testEnum is NewtonLineSearch, args should be [Bisection=False, Secant=False, RegulaFalsi=False, InitialInterpolated=False, tol=0.8, maxIter=10, minEta=0.1, maxEta=10.0]")
            self.algorithm += [alogrithmEnum.value]
            self.algorithm += args
            if len(args) == 0:
                self.algorithm += [False, False, False, False, 0.8, 10.0, 0.1, 10.0]

        else:
            raise Exception("Unfinshed")

    def setIntegrator(self, interEnum:OpsInteratorEnum.Static, *args):
        if interEnum == OpsInteratorEnum.Static.DisplacementControl:
            print("testEnum is DisplacementControl, args should be [nodeTag, dof, incr, numIter=1, dUmin=incr, dUmax=incr]")
            
            self.integrator += [interEnum.value]
            self.integrator += args
        elif interEnum == OpsInteratorEnum.Static.LoadControl:
            print("testEnum is LoadControl, args should be [incr, numIter=1, minIncr=incr, maxIncr=incr]")
            self.integrator += [interEnum.value]
            self.integrator += args
        elif interEnum == OpsInteratorEnum.Transient.Newmark:
            print("testEnum is Newmark, args should be [gamma, beta, '-form', form]")
            self.integrator += [interEnum.value]
            self.integrator += [1/2, 1/4]
        else:
            raise Exception("Unfinished")

    def setAnalyz(self, N:int=1, dt:float=0.1, dtMin=0.0, dtMax=0.0, Jd=0):
        self.analyze += [N, dt, dtMin, dtMax, Jd]




class Analsis(Comp.Component):
    _StaticLoad = []
    _DynamicLoads = []
    _Gravity:list[Load.Gravity] = []

    _StaticPattern = OpsObject.OpsPlainLoadPattern()
    _DynamicPattern = OpsObject.OpsMSELoadPattern()

    _CustomPatternList = []

    @classmethod
    def AddPattern(type:str):
        if type == 'static':
            custPattern = OpsObject.OpsPlainLoadPattern()
        elif type == 'dynamic':
            custPattern = OpsObject.OpsMSELoadPattern()
        
        Analsis._CustomPatternList.append(custPattern)
    
    @classmethod
    def AddGravity(g:Load.Gravity):
        if Analsis._Gravity:
            Analsis._Gravity.append(g)
        else:
            print("already exist gravity, ignored")
    
    @classmethod
    def AddLoads(loads:Load.StaticLoads, ):
        Analsis._StaticLoad.append(loads)

    @classmethod
    def AddEarthquack(eq:Load.EarthquakeLoads):
        Analsis._DynamicLoads.append(eq)
    
    @classmethod
    def RunGravityAnalys(analysParam:OpsAnasysParams):
        if Analsis._Gravity:
            Analsis._StaticPattern.uniqNum
            Analsis._Gravity[0].ApplyLoad()

            ops.constraints(*analysParam.constraints)
            ops.numberer(*analysParam.numberer)
            ops.system(*analysParam.system)
            ops.test(*analysParam.test)
            ops.algorithm(*analysParam.algorithm)
            ops.analysis('Static')
            ops.analyze(*analysParam.analyze)

        else:
            raise Exception("No gravity parameter is added")


    @classmethod
    def RunStaticAnalys(analysParam:OpsAnasysParams, gravity=False, gravityAnaParams:OpsAnasysParams=None):
        loads:list[Load.StaticLoads] = []
        ops.wipeAnalysis()

        if gravity and gravityAnaParams:
            Analsis.RunGravityAnalys(gravityAnaParams)

        ops.loadConst('-time', 0.0)
        load = Analsis._StaticLoad

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
    def seismicAnalysis(analysParam:OpsAnasysParams, gravity=False, gravityAnaParams:OpsAnasysParams=None):

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

    