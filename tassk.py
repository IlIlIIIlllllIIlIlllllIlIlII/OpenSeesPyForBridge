#%%
from src import *
import allpairspy
from src.Analysis import *
import opsvis as opsv

pileLeng = [10, 15, 20, 25, 30]

pierH = Unit.ConvertToBaseUnit(10, 'm')

mass = [10, 20, 30, 40, 50]

Mode = [False, True]

soil = [BridgeParas.Sand.DenseSand, BridgeParas.Sand.MediumDenseSand, BridgeParas.Sand.MediumSand, BridgeParas.Sand.LooseSand,  BridgeParas.Clay.Soft, BridgeParas.Clay.Medium, BridgeParas.Clay.Stiff]

steelRaito = [0.01, 0.02, 0.03, 0.04]

R = Unit.ConvertToBaseUnit(1, 'm')
HRpara = HRoundSectParas(R, 0.5*R)
SRpara = SRoundSectParas(R)
eleNum = 10

parameters = [
    pileLeng,
    mass,
    Mode,
    soil,
    steelRaito,
]

con = BridgeParas.Concrete.C40
rebar = BridgeParas.ReBar.HRB400
comb = {}
all_res = []
for i, pair in enumerate(allpairspy.AllPairs(parameters)):
    print('{}:{}, {}, {}, {}'.format(i, pair[0], pair[1], pair[2], pair[3].soilType, pair[4]))
    comb[i] = pair
    
#%%

def Run(pL, mass, flag, soil, steelRatio):
    AnalsisModel.InitAnalsis()
    try:
        pL = Unit.ConvertToBaseUnit(pL, 'm')
        mass = Unit.ConvertToBaseUnit(mass, 't')
        flag = pair[2]

        p1 = (0., 0., 0.)
        p2 = (0., 0., pierH)
        p3 = (0., 0., -pL)

        s = soil
        r = steelRatio

        totL = UtilTools.PointsTools.PointsDist(p1, p2)
        eleL = UtilTools.SegmentTools.BuildWithSettedNum(totL, eleNum)
        pier = LineHRoundSeg(p1, p2, HRpara, HRpara, LineHRoundSeg.SupportedElementType.ElasticBeamColumnElement, con, con, rebar,  eleL, [r]*eleNum, localZ=(-1, 0, 0))
        
        AnalsisModel.AddSegment(pier)
        totL = UtilTools.PointsTools.PointsDist(p1, p3)
        eleL = UtilTools.SegmentTools.BuildWithSettedNum(totL, eleNum)
        pile = LineSRoundSeg(p1, p3, SRpara, SRpara, LineSRoundSeg.SupportedElementType.ElasticBeamColumnElement, con, con, rebar,  eleL, [r]*eleNum, localZ=(-1, 0, 0))
        
        AnalsisModel.AddSegment(pile)

        flag, node1 = AnalsisModel.Inquire.FindNode(p1)
        flag, node2 = AnalsisModel.Inquire.FindNode(p2)
        flag, node3 = AnalsisModel.Inquire.FindNode(p3)
        if flag:
            flag, node = AnalsisModel.Inquire.FindNode(p3)
            if flag:
                print("fix")
                AnalsisModel.AddBoundary(Boundary.BridgeFixedBoundary(node, [1]*6))

            D = pile._Secti.R * 2
            eleW = eleL = [D]
            eleH = [D]
            PSSI = Boundary.BridgeFullPileSoilBoundary([pile], None, s, eleW, eleL, eleH)
            AnalsisModel.AddBoundary(PSSI)
            # D = pile._Secti.R * 2
            # if flag:
            #     H = W = L = []
            #     b = Boundary.BridgeFullPileSoilBoundary([pile],node1, s, L, W, H)
            #     AnalsisModel.AddBoundary(b)
            
            # fix = Boundary.BridgeFixedBoundary(node3, [1]*3)
            # AnalsisModel.AddBoundary(fix)
        else:
            b = Boundary.BridgeSimplyPileSoilBoundary([pile], s)
            AnalsisModel.AddBoundary(s)
        
        AnalsisModel.buildFEM()
        # opsv.plot_model()

        # flag, ele = AnalsisModel.Inquire.FindSegElement(p1, p2)
        ele1 = pier.ELeList[0]
        def rs_func(ntag):

            res = []
            res.append(ops.nodeDisp(ntag, -1))
            res.append(ops.eleForce(ele1.uniqNum))
            return res
        rst = AnalsisModel.RunGravityAnalys(rs_func, [node2.OpsNode._uniqNum])
        # opsv.plot_defo()
        freqs = AnalsisModel.GetNaturalFrequencies(5)
        print(freqs)
        waveE, info = Load.SeismicWave.LoadACCFromPEERFile(r'D:\Spectrum Response\AT\RSN1562_CHICHI_TTN006-E')
        waveN, info = Load.SeismicWave.LoadACCFromPEERFile(r'D:\Spectrum Response\AT\RSN1562_CHICHI_TTN006-N')
        waveV, info = Load.SeismicWave.LoadACCFromPEERFile(r'D:\Spectrum Response\AT\RSN1562_CHICHI_TTN006-V')
        g = GlobalData.DEFVAL._G_
        wave = Load.SeismicWave(info.dt, accX=waveE, factorX=g, accY=waveN, factorY=g, accZ=waveV, factorZ=g, accInformation=info)
        # wave = wave._CutWaveFromMaxVal(30)
        flag, brgNode = AnalsisModel.Inquire.FindNode(p1)
        EqLoad = Load.EarthquakeLoads(brgNode, wave)
        AnalsisModel.AddEarthquack(EqLoad)
        # flag, brgNode = AnalsisModel.Inquire.FindNode(p3)
        # EqLoad = Load.EarthquakeLoads(brgNode, wave)
        # AnalsisModel.AddEarthquack(EqLoad)
        # AnalsisModel.AddEarthquack(eqLoad)
        AnalsisModel.buildFEM()
        # opsv.plot_model()
        anaParams = DEF_ANA_PARAM.DefaultSeismicAnalysParam
        anaParams.setDeltaT(0.1)
        anaParams.setAlogrithm(AlgorithmEnum.NewtonLineSearch)
        anaParams.setTest(TestEnum.NormDispIncr, 1e-10)
        rst = AnalsisModel.RunSeismicAnalysis(rs_func, duraT=90, targets=[node2.OpsNode.uniqNum])

        times, val = rst.getNodeValueTimes(0)
        # print(val[0,-1])
        # print(val[1,-1])
        # print(val[2,-1])
        print("success")
        return times, val, freqs
    except Exception as e:
        print(e)
        print("failed")
#%%
res = Run(*comb[4])
# %%
BridgeParas.Clay.Soft