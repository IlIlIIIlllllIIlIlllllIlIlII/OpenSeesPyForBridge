#%%
# from src.OpsObject import OpsNode
import opsvis as opsv

from src import *
from src.Analysis import *
from src.Boundary import *
from src.Display import *

#%%
con = BridgeParas.Concrete.C30
rebar = BridgeParas.ReBar.HRB400
L = ConvertToBaseUnit(20, 'm')
H = ConvertToBaseUnit(20, 'm')
R = ConvertToBaseUnit(2, 'm')
p1 = (0., 0., 0.)
p2 = (0., 0., H)
p3 = (L, 0., H)
p4 = (L, 0., 0.)

FP = [p1, p2, p3]
SP = [p2, p3, p4]
LOACALZ = [(-1, 0, 0), (0, 0, 1), (-1, 0, 0)]

AnalsisModel.InitAnalsis()

paras = BridgeParas.SRoundSectParas(R)
for i, (fp, sp, localz) in enumerate(zip(FP, SP, LOACALZ)):
    totL = UtilTools.PointsTools.PointsDist(fp, sp)
    eleNum = 10
    if i==1:
        paras = BridgeParas.SRoundSectParas(R*(i+1), 0.1*R)
        eleL = UtilTools.SegmentTools.BuildWithSettedNum(totL, eleNum)
        seg = Part.LineSRoundSeg(fp, sp, paras, paras, LineSRoundSeg.SupportedElementType.NonlinearBeamColumnElement, con, con, rebar, eleL, [0.02]*eleNum, localZ=localz)
    else:
        paras = BridgeParas.HRoundSectParas(R*(i+1), 0.1*R)
        eleL = UtilTools.SegmentTools.BuildWithSettedNum(totL, eleNum)
        seg = Part.LineHRoundSeg(fp, sp, paras, paras, LineHRoundSeg.SupportedElementType.ElasticBeamColumnElement, con, con, rebar, eleL, [0.02]*eleNum, localZ=localz)

    AnalsisModel.AddSegment(seg)

#%%
flag, node = AnalsisModel.Inquire.FindNode(p1)
if flag:

    print("fix")
    AnalsisModel.AddBoundary(BridgeFixedBoundary(node, [1]*6))

flag, node = AnalsisModel.Inquire.FindNode(p4)
if flag:
    print("fix")
    AnalsisModel.AddBoundary(BridgeFixedBoundary(node, [1]*6))
#%%
AnalsisModel.buildFEM()
#%%
dsp = ModelDisplayer()
dsp.PlotModel(AnalsisModel)
opsv.plot_model()

#%%
dsp = ModelDisplayer()
# for seg in AnalsisModel._SegmentList:
print(AnalsisModel._SegmentList[0].ELeList[0].Sect._FibersDistr)
# AnalsisModel._SegmentList[0].ELeList[0].Sect.RebarsDistr
dsp.PlotFiberSect(AnalsisModel._SegmentList[0].ELeList[0])
#%%
def rs_func(p:tuple[float]):

    Norms = []
    norms = ops.testNorm()
    iters = ops.testIter()
    for j in range(iters):
         Norms.append(norms[j])
    return Norms
AnalsisModel.RunGravityAnalys(rs_func, p1)
opsv.plot_defo()
# %%
waveE, info = Load.SeismicWave.LoadACCFromPEERFile(r'D:\Spectrum Response\AT\RSN1562_CHICHI_TTN006-E')
waveN, info = Load.SeismicWave.LoadACCFromPEERFile(r'D:\Spectrum Response\AT\RSN1562_CHICHI_TTN006-N')
waveV, info = Load.SeismicWave.LoadACCFromPEERFile(r'D:\Spectrum Response\AT\RSN1562_CHICHI_TTN006-V')
# %%
g = GlobalData.DEFVAL._G_
wave = Load.SeismicWave(info.dt, accX=waveE, factorX=g, accY=waveN, factorY=g, accZ=waveV, factorZ=g, accInformation=info)
flag, brgNode = AnalsisModel.Inquire.FindNode(p1)
eqLoad = Load.EarthquakeLoads(brgNode, wave)
AnalsisModel.AddEarthquack(eqLoad)
flag, brgNode = AnalsisModel.Inquire.FindNode(p4)
eqLoad = Load.EarthquakeLoads(brgNode, wave)
AnalsisModel.AddEarthquack(eqLoad)
# %%
# AnalsisModel.AddEarthquack(eqLoad)
AnalsisModel.buildFEM()
opsv.plot_model()
#%%
anaParams = DEF_ANA_PARAM.DefaultSeismicAnalysParam
anaParams.setDeltaT(0.1)
anaParams.setAlogrithm(AlgorithmEnum.NewtonLineSearch)
anaParams.setTest(TestEnum.NormDispIncr, 1e-10)
rst = AnalsisModel.RunSeismicAnalysis(rs_func, duraT=90, points=[p3])
# %%
