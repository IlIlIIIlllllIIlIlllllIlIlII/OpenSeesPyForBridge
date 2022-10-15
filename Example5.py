#%%
import opsvis as opsv

from src import *
from src import Display
from src.Analysis import *
from src.Boundary import *
from src.Display import *
from src.OpsObject import OpsNode

#%%
AnalsisModel.InitAnalsis()
con = BridgeParas.Concrete.C30
rebar = BridgeParas.ReBar.HRB400
sand = BridgeParas.Sand.DenseSand
L = ConvertToBaseUnit(20, 'm')
H = ConvertToBaseUnit(20, 'm')
R = ConvertToBaseUnit(1, 'm')
mass = ConvertToBaseUnit(10, 'kg')
p1 = (0., 0., 0.)
p2 = (0., 0., H)
p3 = (0., 0., -H)
eleNum = 5
totL = UtilTools.PointsTools.PointsDist(p1, p2)
eleL = UtilTools.SegmentTools.BuildWithSettedNum(totL, eleNum)
paras = HRoundSectParas(R, 0.5*R)
PierSeg = LineHRoundSeg(p1, p2, paras, paras, LineHRoundSeg.SupportedElementType.ElasticBeamColumnElement, con, con, rebar, eleL, [0.02]*eleNum, localZ=(-1, 0, 0) )

AnalsisModel.AddSegment(PierSeg)

flag, Node2 = AnalsisModel.Inquire.FindNode(p2)
# if flag:
#     print("Add mass")
#     Node2.addMass(mass)
fixed = False

if True:
    totL = UtilTools.PointsTools.PointsDist(p3, p1)
    eleL = UtilTools.SegmentTools.BuildWithSettedNum(totL, eleNum)
    paras = SRoundSectParas(R)
    PileSeg = LineSRoundSeg(p3, p1, paras, paras, LineSRoundSeg.SupportedElementType.ElasticBeamColumnElement, con, con, rebar, eleL, [0.02]*eleNum, localZ=(-1, 0, 0))
    AnalsisModel.AddSegment(PileSeg)

    # flag, node = AnalsisModel.Inquire.FindNode(p3)
    # if flag:
    #     print("fix")
    #     AnalsisModel.AddBoundary(BridgeFixedBoundary(node, [1]*6))

    FPSI = BridgeSimplyPileSoilBoundary([PileSeg], sand)
    AnalsisModel.AddBoundary(FPSI)
else:
    flag, node = AnalsisModel.Inquire.FindNode(p1)
    if flag:
        print('fix')
        AnalsisModel.AddBoundary(BridgeFixedBoundary(node, [1]*6))
#%%
AnalsisModel.buildFEM()

#%%
def rs_func(ntag):

    res = []
    res = ops.nodeDisp(ntag, -1)
    return res
rst = AnalsisModel.RunGravityAnalys(rs_func, [Node2.OpsNode._uniqNum])
# opsv.plot_defo()
freqs = AnalsisModel.GetNaturalFrequencies(5)
#%%
waveE, info = Load.SeismicWave.LoadACCFromPEERFile(r'D:\Spectrum Response\AT\RSN1562_CHICHI_TTN006-E')
waveN, info = Load.SeismicWave.LoadACCFromPEERFile(r'D:\Spectrum Response\AT\RSN1562_CHICHI_TTN006-N')
waveV, info = Load.SeismicWave.LoadACCFromPEERFile(r'D:\Spectrum Response\AT\RSN1562_CHICHI_TTN006-V')
# %%
g = GlobalData.DEFVAL._G_
wave = Load.SeismicWave(info.dt, accX=waveE, factorX=g, accY=waveN, factorY=g, accZ=waveV, factorZ=g, accInformation=info)
# wave = wave._CutWaveFromMaxVal(30)
flag, brgNode = AnalsisModel.Inquire.FindNode(p1)
EqLoad = Load.EarthquakeLoads(brgNode, wave)
AnalsisModel.AddEarthquack(EqLoad)
flag, brgNode = AnalsisModel.Inquire.FindNode(p3)
EqLoad = Load.EarthquakeLoads(brgNode, wave)
AnalsisModel.AddEarthquack(EqLoad)
# AnalsisModel.AddEarthquack(eqLoad)
AnalsisModel.buildFEM()
# opsv.plot_model()
#%%
anaParams = DEF_ANA_PARAM.DefaultSeismicAnalysParam
anaParams.setDeltaT(0.1)
anaParams.setAlogrithm(AlgorithmEnum.NewtonLineSearch)
anaParams.setTest(TestEnum.NormDispIncr, 1e-10)
rst = AnalsisModel.RunSeismicAnalysis(rs_func, duraT=90, targets=[Node2.OpsNode.uniqNum])
# %%
dsp = Display.ModelDisplayer()

dsp.PlotModel(AnalsisModel)
# %%
