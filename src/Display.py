#%%
from enum import Enum
from unittest import defaultTestLoader
from src.Analysis import AnalsisModel
from src.Boundary import BridgeBearingBoundary, BridgeEQDOFSBoundary, BridgeFixedBoundary

from src.Comp import Boundary
from . import Part
import matplotlib.pyplot as plt
#! Using it to draw 3d graphics does not perform well, consider using other 3D drawing librarie or improve the performance
from mpl_toolkits.mplot3d import Axes3D
from . import UtilTools
import numpy as np
#%%

class DisplayProf:

    class CuboidPlotMode(Enum):
        Points = 0
        CornerPoints = 1
        EdgePoints = 2
        SurfaceLines = 3
        Edge = 4
        Surface = 5

    class BoundaryPlotMode(Enum):
        FIX = 'FIX'
        BEAR = 'BEAR'
        EQDOF = 'EQDOF'

    # class DEFAUT_DISP:
    DefaultPointColor = 'b'
    DefaultPointSize = 90

    DefaultLineColor = 'k'
    DefaultLineWidth = 1

    DefaultSegLineWidth = 10
    DefaultSegLineColor = 'r'

    DefaultSurfaceColor = (0.1, 0.1, 0.1, 0.3)
    
    DefaultBoundaryMakerStyle = {
        'FIX':'x',
        'EQDOF':'1',
        'BEAR':'2',
    }
    DefaultBoundaryColor = 'g'

    DefaultElementPlotMode = [CuboidPlotMode.SurfaceLines]
    DefaultBoundaryPlotMode = [BoundaryPlotMode.BEAR, BoundaryPlotMode.FIX]

    def __init__(self, ElementPlotMode=DefaultElementPlotMode, BoundaryPlotMode=DefaultBoundaryPlotMode, PointColor=DefaultPointColor, PointSize=DefaultPointSize, LineColor=DefaultLineColor, LineWidth=DefaultLineWidth, SegLineWidth=DefaultLineWidth, SegLineColor=DefaultSegLineColor, SurfaceColor=DefaultSurfaceColor, BoundaryMarker=DefaultBoundaryMakerStyle, BoundaryColor = DefaultBoundaryColor) -> None:
        self._CuboidMode = ElementPlotMode
        self._BoundaryMode = BoundaryPlotMode
        self._PointColor = PointColor
        self._PointSize = PointSize
        self._LineColor = LineColor
        self._LineWidth = LineWidth
        self._SegLineWidth = SegLineWidth
        self._SegLineColor = SegLineColor
        self._SurfaceColor = SurfaceColor
        self._BoundaryMarker =BoundaryMarker
        self._BoundaryColor = BoundaryColor

class ModelDisplayer:

    def __init__(self, displayProf=None) -> None:
        plt.ion()
        self.fig = plt.figure()
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        if not displayProf:
            displayProf = DisplayProf()
        self._DisProf = displayProf
    
    # def PlotNode(self, BridgeNode:Part.BridgeNode, shouBoundary:bool=False):
    
    def Plot3DPoint(self, points, c=None, s=None, marker=None):
        if not c:
            c = self._DisProf._PointColor
        if not s:
            s = self._DisProf._PointSize

        if not marker:
            marker = '.'

        self.ax.scatter3D(points[0], points[1], points[2], c=c, s=s, marker=marker)

    def Plot3DLine(self, p1, p2, LineW=None, LineC=None, lineStyle=None):
        new = []
        for i in zip(p1, p2):
            new.append(i)
        if not LineW:
            LineW = self._DisProf._LineWidth
        if not LineC:
            LineC = self._DisProf._LineColor
        if not lineStyle:
            lineStyle = '-'
        self.ax.plot3D(new[0], new[1], new[2], linewidth=LineW, c=LineC, linestyle=lineStyle)
        
    def PlotSurface(self, p1, p2, p3, p4, color=None):
        v1 = UtilTools.PointsTools.VectorSub(p2, p1)
        v2 = UtilTools.PointsTools.VectorSub(p3, p1)
        v3 = UtilTools.PointsTools.VectorSub(p4, p1)
        if UtilTools.PointsTools.IsVectorInPlane(v1, v2, v3):

            x = np.array([[p1[0], p4[0]],
                        [p2[0], p3[0]]])
            y = np.array([[p1[1], p4[1]],
                        [p2[1], p3[1]]])
            z = np.array([[p1[2], p4[2]],
                [p2[2], p3[2]]])
            if not color:
                color = self._DisProf._SurfaceColor                

            self.ax.plot_surface(x, y, z, color=color)
        else:
            print('Points are not in one plane')

    def PlotSegment(self, seg:Part.Segment, segLineW=None, segLineC=None):
        # ps = [n.point for n in seg.NodeList]
        # lens = len(ps)
        if not segLineC:
            segLineC = self._DisProf._SegLineColor
        if not segLineW:
            segLineW = self._DisProf._SegLineWidth
        n1 = seg.NodeList[0] 

        self.Plot3DPoint(n1.point)
        for i in range(len(seg.NodeList)-1):
            n1 = seg.NodeList[i]
            n2 = seg.NodeList[i+1]
            self.Plot3DPoint(n2.point)
            
            self.Plot3DLine(n1.point, n2.point, LineW=segLineW, LineC=segLineC)
        

    def PlotCuboid(self, Cuboid:Part.SoilCuboid, mode:set[DisplayProf.CuboidPlotMode] = None, PointColor=None, LineColor=None, SurfaceColor=None):
        if not mode:
            mode = self._DisProf._CuboidMode
        if not PointColor:
            PointColor = self._DisProf._PointColor
        if not LineColor:
            LineColor = self._DisProf._LineColor

        ps = Cuboid._PointsIndex
        x, y, z = ps.shape[:-1]

        p3 = [0, 0, 0]
        p2 = [x-1, 0, 0]
        p1 = [x-1, y-1, 0]
        p4 = [0, y-1, 0]
        p7 = [0, 0, z-1]
        p6 = [x-1, 0, z-1]
        p5 = [x-1, y-1, z-1]
        p8 = [0, y-1, z-1]

        Corner = [p1, p2, p3, p4, p5, p6, p7, p8]
        Edge = [(p2, p1), (p3, p2), (p3, p4), (p4, p1), 
                (p6, p5), (p7, p6), (p7, p8), (p8, p5),
                (p1, p5), (p2, p6), (p3, p7), (p4, p8)]

        surface = [(p1, p2, p3, p4), 
                    (p5, p6, p7, p8), 
                    (p5, p1, p2, p6), 
                    (p6, p7, p3, p2),
                    (p8, p7, p3, p4),
                    (p5, p8, p4, p1)]
        if DisplayProf.CuboidPlotMode.Points in mode:
            p_ = ps.reshape(-1, 3).tolist()
            for p in p_:
                self.Plot3DPoint(p, c=PointColor)

        if DisplayProf.CuboidPlotMode.CornerPoints in mode:
            for pi in Corner:
                self.Plot3DPoint(ps[pi[0], pi[1], pi[2]])

        if DisplayProf.CuboidPlotMode.EdgePoints in mode:
            for pi in Edge:
                pp = pi[0]
                pb = pi[1]
                EdgePs = ps[pp[0]:pb[0]+1, pp[1]:pb[1]+1, pp[2]:pb[2]+1].reshape([-1, 3]).tolist()
                for p in EdgePs:
                    self.Plot3DPoint(p)

        if DisplayProf.CuboidPlotMode.Edge in mode:

            for pi in Edge:
                pp = pi[0]
                pb = pi[1]
                self.Plot3DLine(ps[pp[0], pp[1], pp[2]], ps[pb[0], pb[1], pb[2]])
        
        if DisplayProf.CuboidPlotMode.SurfaceLines in mode:
            for area in surface:
                p1, p2, p3, p4 = area

                #  * (p2, p1)<--->(p3, p4)
                l1 = ps[p2[0]:p1[0]+1, p2[1]:p1[1]+1, p2[2]:p1[2]+1]
                l2 = ps[p3[0]:p4[0]+1, p3[1]:p4[1]+1, p3[2]:p4[2]+1]
                l1 = l1.reshape([-1, 3]).tolist()
                l2 = l2.reshape([-1, 3]).tolist()

                for l1p, l2p in zip(l1, l2):
                    self.Plot3DLine(l1p, l2p)

                # * (p4, p1) <-----> (p3, p2)

                l1 = ps[p4[0]:p1[0]+1, p4[1]:p1[1]+1, p4[2]:p1[2]+1]
                l2 = ps[p3[0]:p2[0]+1, p3[1]:p2[1]+1, p3[2]:p2[2]+1]
                l1 = l1.reshape([-1, 3]).tolist()
                l2 = l2.reshape([-1, 3]).tolist()

                for l1p, l2p in zip(l1, l2):
                    self.Plot3DLine(l1p, l2p)

        if DisplayProf.CuboidPlotMode.Surface in mode:

            for s in surface:
                self.PlotSurface(*s, SurfaceColor)
    
    def PlotBoundary(self, boundary:Boundary, plotMode:list[DisplayProf.BoundaryPlotMode]=None, color=None, mker=None):
        if not mker:
            mker = self._DisProf._BoundaryMarker
        
        if not plotMode:
            plotMode = self._DisProf._BoundaryMode

        if not color:
            color = self._DisProf._BoundaryColor

        if DisplayProf.BoundaryPlotMode.FIX in plotMode and isinstance(boundary, BridgeFixedBoundary):
            p = boundary.FixedNode.point
            self.Plot3DPoint(p, marker=mker['FIX'], c=color)
        elif DisplayProf.BoundaryPlotMode.EQDOF in plotMode and isinstance(boundary, BridgeEQDOFSBoundary):
            p1 = boundary._nodeI.point
            p2 = boundary._nodeJ.point
            self.Plot3DPoint(p1, marker=mker['EQDOF'], c=color)
            self.Plot3DPoint(p2, marker=mker['EQDOF'], c=color)
            rd = np.random.random()
            self.Plot3DLine(p1, p2, lineStyle=':', LineC=(rd, rd, rd, 1))
        elif DisplayProf.BoundaryPlotMode.BEAR in plotMode and isinstance(boundary, BridgeBearingBoundary):
            p1 = boundary._NodeI.point
            p2 = boundary._NodeJ.point
            self.Plot3DPoint(p1, marker=mker['BEAR'], c=color)
            self.Plot3DPoint(p2, marker=mker['BEAR'], c=color)
            self.Plot3DLine(p1, p2, lineStyle=':')
        else:
            ...

    def PlotModel(self, anayModel:AnalsisModel):
        
        for _, val in anayModel._BoundaryDict.items():
            for boun in val:
                self.PlotBoundary(boun, self._DisProf._BoundaryMode)

        for seg in anayModel._SegmentList:
            self.PlotSegment(seg)
        for coub in anayModel._CuboidsList:
            self.PlotCuboid(coub, self._DisProf._CuboidMode)


            



#%%


#%%
