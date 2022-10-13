#%%
import openseespy.opensees as ops

from src import *
from src import BridgeParas
from src.Analysis import AnalsisModel
from src.Boundary import BridgeFixedBoundary
from src.Display import ModelDisplayer
from src.Part import SoilCuboid

AnalsisModel.InitAnalsis()
sand = BridgeParas.Sand.DenseSand
cub = SoilCuboid.FromCuboidP3_P5((0, 0, 0), (1, 1, 1), sand, [1], [1], [1])
flag, node = cub.FindRangeBridgeNodes((0, 0, 0), 1, 1, 0)
if flag:
    for n in node.flatten().tolist():
        fix = BridgeFixedBoundary(n, [1]*6)
        AnalsisModel.AddBoundary(fix)

AnalsisModel.AddCuboid(cub)
AnalsisModel.buildFEM()

dsp = ModelDisplayer()
dsp.PlotModel(AnalsisModel)
# %%
def rs_func(p:tuple[float]):

    Norms = []
    norms = ops.testNorm()
    iters = ops.testIter()
    for j in range(iters):
         Norms.append(norms[j])
    return Norms
AnalsisModel.RunGravityAnalys(rs_func, (0, 0, 0))
# %%
