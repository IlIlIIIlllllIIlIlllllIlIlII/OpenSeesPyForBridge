#%%
import opsvis as opsv

from src import *
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
p1 = (0., 0., 0.)
p2 = (0., 0., H)
p3 = (0., 0., -H)
eleNum = 3
totL = UtilTools.PointsTools.PointsDist(p1, p2)
eleL = UtilTools.SegmentTools.BuildWithSettedNum(totL, eleNum)
paras = HRoundSectParas(R, 0.5*R)
PierSeg = LineHRoundSeg(p1, p2, paras, paras, LineHRoundSeg.SupportedElementType.ElasticBeamColumnElement, con, con, rebar, eleL, [0.02]*eleNum, localZ=(-1, 0, 0) )
AnalsisModel.AddSegment(PierSeg)

totL = UtilTools.PointsTools.PointsDist(p3, p1)
eleL = UtilTools.SegmentTools.BuildWithSettedNum(totL, eleNum)
paras = SRoundSectParas(R)
PileSeg = LineSRoundSeg(p1, p3, paras, paras, LineSRoundSeg.SupportedElementType.ElasticBeamColumnElement, con, con, rebar, eleL, [0.02]*eleNum, localZ=(-1, 0, 0))
AnalsisModel.AddSegment(PileSeg)


#%%
flag, node = AnalsisModel.Inquire.FindNode(p3)
if flag:
    print("fix")
    # AnalsisModel.AddBoundary(BridgeFixedBoundary(node, [1]*6))
D = PileSeg._Secti.R * 2
eleW = eleL = [D, D, 2*D, 2*D]
eleH = [5*D, 5*D]
# PSSI = BridgeFullPileSoilBoundary([PileSeg], sand, eleW, eleL, eleH)
# AnalsisModel.AddBoundary(PSSI)

SSI = BridgeSimplyPileSoilBoundary([PileSeg], sand)
AnalsisModel.AddBoundary(SSI)

AnalsisModel.buildFEM()

#%%
# dsp = ModelDisplayer()
# dsp.PlotModel(AnalsisModel)
# opsv.plot_model()

#%%
# dsp = ModelDisplayer()
# dsp.PlotFiberSect(AnalsisModel._SegmentList[1].ELeList[0])
#%%
def rs_func(p:tuple[float]):

    Norms = []
    norms = ops.testNorm()
    iters = ops.testIter()
    for j in range(iters):
         Norms.append(norms[j])
    return Norms
AnalsisModel.RunGravityAnalys(rs_func, p1)
freqs = AnalsisModel.GetNaturalFrequencies(5)
opsv.plot_defo()

#%%
eig = ops.eigen(1)
T = 2* np.pi / eig**0.5
f = eig**0.5/2/np.pi
# %%
waveE, info = Load.SeismicWave.LoadACCFromPEERFile(r'D:\Spectrum Response\AT\RSN1562_CHICHI_TTN006-E')
waveN, info = Load.SeismicWave.LoadACCFromPEERFile(r'D:\Spectrum Response\AT\RSN1562_CHICHI_TTN006-N')
waveV, info = Load.SeismicWave.LoadACCFromPEERFile(r'D:\Spectrum Response\AT\RSN1562_CHICHI_TTN006-V')
# %%
g = GlobalData.DEFVAL._G_
wave = Load.SeismicWave(info.dt, accX=waveE, factorX=g, accY=waveN, factorY=g, accZ=waveV, factorZ=g, accInformation=info)
flag, brgNode = AnalsisModel.Inquire.FindNode(p1)
EqLoad = Load.EarthquakeLoads(brgNode, wave)
AnalsisModel.AddEarthquack(EqLoad)
flag, brgNode = AnalsisModel.Inquire.FindNode(p3)
EqLoad = Load.EarthquakeLoads(brgNode, wave)
AnalsisModel.AddEarthquack(EqLoad)
# %%
# AnalsisModel.AddEarthquack(eqLoad)
AnalsisModel.buildFEM()
# opsv.plot_model()
#%%
anaParams = DEF_ANA_PARAM.DefaultSeismicAnalysParam
anaParams.setDeltaT(0.1)
anaParams.setAlogrithm(AlgorithmEnum.NewtonLineSearch)
anaParams.setTest(TestEnum.NormDispIncr, 1e-10)
rst = AnalsisModel.RunSeismicAnalysis(rs_func, duraT=90, targets=[p3])
# %%
for key, comp in Comp.CompMgr._allUniqComp.items():
    if isinstance(comp, OpsNode):
        print(comp.xyz)
# %%
flag, b = AnalsisModel.Inquire.FindElementByTag(211)
# %%
