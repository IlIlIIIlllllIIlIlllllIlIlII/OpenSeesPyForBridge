#%%
from enum import Enum
from . import Part
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from . import UtilTools
import numpy as np

#! Using it to draw 3d graphics does not perform well, consider using other 3D drawing librarie or improve the performance

#%%

class PlotMode(Enum):
    Points = 0
    CornerPoints = 1
    EdgePoints = 2
    SurfaceLines = 3
    Edge = 4
    Surface = 5

class ModelDisplay:
    class DEFAUT_DISP:
        PointColor = 'b'
        PointSize = 20

        LineColor = 'k'
        LineWidth = 1

        SegLineWidth = 5
        SegLineColor = 'b'

        SurfaceColor = (0.1, 0.1, 0.1, 0.3)


    def __init__(self) -> None:
        plt.ion()
        self.fig = plt.figure()
        self.ax = plt.axes(projection='3d')
    
    
    def Plot3dPoint(self, points, c=None, s=None):
        if not c:
            c = self.DEFAUT_DISP.PointColor
        if not s:
            s = self.DEFAUT_DISP.PointSize
        self.ax.scatter3D(points[0], points[1], points[2], c=c, s=s)

    def Plot3DLine(self, p1, p2, LineW=None, LineC=None):
        new = []
        for i in zip(p1, p2):
            new.append(i)
        if not LineW:
            LineW = self.DEFAUT_DISP.LineWidth
        if not LineC:
            LineC = self.DEFAUT_DISP.LineColor
        self.ax.plot3D(new[0], new[1], new[2], linewidth=LineW, c=LineC)
        
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
                color = self.DEFAUT_DISP.SurfaceColor                

            self.ax.plot_surface(x, y, z, color=color)
        else:
            print('Points are not in one plane')

    def PlotSegment(self, seg:Part.Segment, segLineW=None, segLineC=None):
        # ps = [n.point for n in seg.NodeList]
        # lens = len(ps)
        if not segLineC:
            segLineC = self.DEFAUT_DISP.SegLineColor
        if not segLineW:
            segLineW = self.DEFAUT_DISP.SegLineWidth
        n1 = seg.NodeList[0] 

        self.Plot3dPoint(n1.point)
        for i in range(len(seg.NodeList)-1):
            n1 = seg.NodeList[i]
            n2 = seg.NodeList[i+1]
            self.Plot3dPoint(n2.point)
            
            self.Plot3DLine(n1.point, n2.point, LineW=segLineW, LineC=segLineC)
        

    def PlotCuboid(self, be:Part.SoilCuboid, mode:set[PlotMode], CuboidColor=None):
        ps = be._PointsIndex
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
        if PlotMode.Points in mode:
            p_ = ps.reshape(-1, 3).tolist()
            for p in p_:
                self.Plot3dPoint(p)

        if PlotMode.CornerPoints in mode:
            for pi in Corner:
                self.Plot3dPoint(ps[pi[0], pi[1], pi[2]])

        if PlotMode.EdgePoints in mode:
            for pi in Edge:
                pp = pi[0]
                pb = pi[1]
                EdgePs = ps[pp[0]:pb[0]+1, pp[1]:pb[1]+1, pp[2]:pb[2]+1].reshape([-1, 3]).tolist()
                for p in EdgePs:
                    self.Plot3dPoint(p)

        if PlotMode.Edge in mode:

            for pi in Edge:
                pp = pi[0]
                pb = pi[1]
                self.Plot3DLine(ps[pp[0], pp[1], pp[2]], ps[pb[0], pb[1], pb[2]])
        
        if PlotMode.SurfaceLines in mode:
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

        if PlotMode.Surface in mode:

            for s in surface:
                self.PlotSurface(*s, CuboidColor)




#%%


#%%
# %%
