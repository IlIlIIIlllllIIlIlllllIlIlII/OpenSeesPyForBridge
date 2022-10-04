from enum import Enum
from typing import overload
import xsect
from abc import ABCMeta, abstractmethod
import numpy as np

from . import Comp
from . import GlobalData
from . import Paras
from . import OpsObject
from . import UtilTools
from .log import *


class BridgeNode(Comp.Parts):
    __slots__ = ['_type', '_uniqNum', '_name', '_point', '_mass', '_OpsNode', '_OpsMass']
    @Comp.CompMgr()
    def __init__(self, x:float, y:float, z:float, mass:float=0.0, name="") -> None:
        super(Comp.Parts, self).__init__(name)
        self._type += "BridgeNode"
        self._point = (x, y, z)
        self._mass = mass
        self._OpsNode:OpsObject.OpsNode = OpsObject.OpsNode(self._point)
        self._OpsMassList = []

        if mass - 0.0 >= abs(GlobalData.DEFVAL._TOL_):
            self._OpsMassList.append(OpsObject.OpsMass(self._OpsNode, self._mass, [1, 1, 1, 0, 0, 0]))
        
    def _SectReBuild(self):
        Comp.CompMgr.removeComp(self._OpsNode)
        Comp.CompMgr.removeComp(self._OpsMassList)
        self._OpsNode = OpsObject.OpsNode(self._point)
        self._OpsMassList = OpsObject.OpsMass(self._OpsNode.uniqNum, self._mass, [1, 1, 1, 0, 0, 0])

    @property
    def point(self):
        return self._point
    @point.setter
    def point(self, newVal):
        if type(newVal) is type(self._point):
            self._point = newVal
            self._SectReBuild()
        else:
            raise Exception("Wrong paras")
    
    @property
    def Mass(self):
        return self._mass
    @Mass.setter
    def Mass(self, newVal):
        if type(newVal) is type(self._Mass):
            self._Mass = newVal
            self._SectReBuild()
        else:
            raise Exception("Wrong Paras")
    
    @property
    def OpsNode(self):
        return self._OpsNode
    
    def addMass(self, mass:float, massDis:list=[1, 1, 1, 0, 0, 0]):
        self._mass += mass
        self._OpsMassList.append(OpsObject.OpsMass(self._OpsNode.uniqNum, mass, massDis))
        
    @property
    def val(self):
        """
        return [self._point, self._mass, self._OpsNode, self._OpsMass]
        """
        return [self._point, self._mass, self._OpsNode, self._OpsMassList]


class Boundary(Comp.Parts, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, name=""):
        super(Comp.Parts, self).__init__(name)
        self._type += "Boundary"

class BridgeFixedBoundary(Boundary):
    __slots__ = ['_type', '_uniqNum', '_name', '_Node', '_OpsFix']
    # @Comp.CompMgr()
    def __init__(self, node:BridgeNode, fixVal:list, name=""):
        super(BridgeFixedBoundary, self).__init__(name)
        self._type += "->FixedBoundary"
        self._node = node
        self._fixVal = fixVal
        self._OpsFix = OpsObject.OpsFix(self._node.OpsNode, self._fixVal)
     
    def _SectReBuild(self):
        Comp.CompMgr.removeComp(self._OpsFix)
        self._OpsFix = OpsObject.OpsFix(self._node.OpsNode, self._fixVal)

    @property
    def Node(self):
        return self._node
    @Node.setter
    def Node(self, newVal):
        if type(newVal) is type(self._node):
            self._node = newVal
            self._SectReBuild()
        else:
            raise Exception("Wrong Paras")
    
    @property
    def fixVal(self):
        return self._fixVal
    @fixVal.setter
    def fixVal(self, newVal):
        if type(newVal) is type(self._fixVal):
            self._fixVal = newVal
        else:
            raise Exception("Wrong Paras")
    
    @property
    def val(self):
        return [self._node, self._OpsFix]

# * 截面类, 继承自构件类，派生出 主梁截面、桥墩截面
class CrossSection(Comp.Parts, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, paras, fiberSize, name=""):
        """
        
        """
        super(CrossSection, self).__init__(name)

        self._type += "->CrossSect" 
        self._Paras = paras
        # self._OrigPoint = p
        # self._N_axis = N_axis
        self._fiberSize = fiberSize

        self._SectAttr:dict = None
        self._Points:list[tuple[float, ...]] = None
        self._SectAttr, self._Points = self._SectAttrBuild()

        # self._OpsSect = self._OpsSectBuild()
        

    @abstractmethod
    def _OpsSectBuild(self):
        ...

    # @abstractmethod
    # def _SectAttrBuild(self):
        ...

    @abstractmethod
    def _SectReBuild(self):
        ...

    @abstractmethod
    def plotSect(self, Axe3d):
        ...

    def check(self):
        ...
        # if (
        #     type(self._OrigPoint) != np.ndarray
        #     or type(self._N_axis) != tuple
        # ):
        #     return False
        # else:
        #     return True

# * 箱梁截面类
class RCCrossSect(CrossSection):

    @abstractmethod
    def __init__(self, paras, CoreCon:Paras.ConcreteParas, CoverCon:Paras.ConcreteParas, Rebar:Paras.SteelParas, R_flag, fiberSize:tuple[int]=(100, 100), name=""):
        super().__init__(paras, fiberSize, name)

        self._type += "->Reforce Concrete Cross-Section"
        self._CoreCon:Paras.ConcreteParas = CoreCon
        self._CoverCon:Paras.ConcreteParas = CoverCon
        self._Rebar:Paras.SteelParas = Rebar

        if type(R_flag) is float:
            self._RebarsDistr:Paras.SectRebarDistrParas = self._RebarsDistrBuild(R_flag)
        elif type(R_flag) is Paras.SectRebarDistrParas:
            self._RebarsDistr:Paras.SectRebarDistrParas = R_flag
        else:
            raise Exception("Wrong Params")
        
        self._OpsSect = self._OpsSectBuild()
    
    @abstractmethod
    def _OpsSectBuild(self):
        ...
        
    @abstractmethod
    def _SectAttrBuild(self):
        ...

    @abstractmethod
    def _RebarsDistrBuild(self, R):
        ...

    @abstractmethod
    def _SectReBuild(self):
        ...

    @abstractmethod
    def plotSect(self, Axe3d):
        ...

# * 箱形截面
class BoxSect(RCCrossSect):
    # __slots__ = []
    # __slots__ = ['_type', '_uniqNum', '_name', '_Paras', '_N_axis', '_Points', '_orig_point', '_attr', '_m', '_OpsSect']
    @Comp.CompMgr()
    def __init__(self, paras: Paras, Concrete: Paras.ConcreteParas, R_flag=0.0, fiberSize:tuple[int, ...] = (100, 100), name=""):
        # if R_flag is not None:
        print("R_flag is not used in BoxSect currently, has been set to 0")
        R_flag = 0.0
        super().__init__(paras, Concrete, None, None, R_flag, fiberSize, name)
        self._type += "->BoxSect"
        
        # self._attr, self._poins = self._SectAttrBuild()
        # self._OpsSect = self._OpsSectBuild()

    
    def _OpsSectBuild(self):
        con = OpsObject.OpsConcrete02(*self._CoreCon.val)
        return OpsObject.OpsBoxSection(self._SectAttr['area'], 
                            self._SectAttr['inertia_x'], self._SectAttr['inertia_y'], self._SectAttr['inertia_j'], 
                            con)

    def _SectAttrBuild(self):
        paras:Paras.BoxSectParas = self.Paras
        # if self.Paras.check() != True:
        #     raise Exception("Parameters Exists Zero")
        # * 根据箱梁参数、坐标原点、法向轴，建立箱梁截面
        # if len(self._OrigPoint) != 3 or len(self._N_axis) != 3:
        #     raise Exception("Error length of orig_point or N_axis, should be 3")
        # *                            ^z
        # *                            |
        # *         1__________________|___________________2
        # *         |                  |                   |
        # *        8|________7   9_____|_____10  4_________|3
        # *                  |   |     |     |   |
        # *                  |   |     |     |   |
        # *                  |   |     |     |   |
        # *                  | 12|_____|_____|11 |
        # *                 6|_________|_________|5
        # * ---------------------------|-------------------------->y
        # *                            |
        points = np.array(
            [
                [0, -paras.upper_W / 2, 0],
                [0, paras.upper_W / 2, 0],
                [0, paras.upper_W / 2, -paras.upper_T],
                [0, paras.down_W / 2, -paras.upper_T],
                [0, paras.down_W / 2, -paras.H],
                [0, -paras.down_W / 2, -paras.H],
                [0, -paras.down_W / 2, -paras.upper_T],
                [0, -paras.upper_W / 2, -paras.upper_T],
                [0, -paras.down_W / 2 + paras.web_T, -paras.upper_T],
                [0, paras.down_W / 2 - paras.web_T, -paras.upper_T],
                [0, paras.down_W / 2 - paras.web_T, -paras.H + paras.down_T],
                [0, -paras.down_W / 2 + paras.web_T, -paras.H + paras.down_T]
            ]
        )
        yz = np.hsplit(points, [1])[1]
        yz_out = yz[:8,]
        yz_in = yz[8:,]
        arr = xsect.multi_section_summary([yz_out], subtract=[yz_in])
        # # * X轴
        # x1, y1, z1 = (1.0, 0.0, 0.0)
        # # * 新X轴
        # x2, y2, z2 = (float(x) for x in self._N_axis)
        # # * 绕Z转动，计算x-y平面的角度变化
        # try:
        #     cos_theta = (x1 * x2 + y1 * y2) / math.sqrt(x2 * x2 + y2 * y2)
        #     theta = np.arccos(cos_theta)
        #     if y2 < 0:
        #         theta = 2 * np.pi - theta
        #     Trans_Zaxis = np.array(
        #         [
        #             [np.cos(theta), np.sin(theta), 0],
        #             [-np.sin(theta), np.cos(theta), 0],
        #             [0, 0, 1],
        #         ]
        #     )
        #     points = np.matmul(points, Trans_Zaxis)
        # except:
        #     pass
        # # * 绕Y转动，计算x-z平面的角度变化
        # try:
        #     cos_theta = (x1 * x2 + z1 * z2) / math.sqrt(x2 * x2 + z2 * z2)
        #     theta = -np.arccos(cos_theta)
        # except:
        #     pass
        # # * 平移到orig_point
        # points = points + self._orig_point
        # points = UtilTools.PointsTools.RotatePointsByPoints(points, (1, 0, 0), self._N_axis)
        # points += self._OrigPoint
        return arr, points

    def _RebarsDistrBuild(self, R):
        # TODO 可以写也可以不写
        return None

    def _SectReBuild(self):
        Comp.CompMgr.removeComp(self._OpsSect)
        Comp.CompMgr.removeComp(self._CoreCon)
        self._SectAttr, self._Points = self._SectAttrBuild()
        self._OpsSect = self._OpsSectBuild()
    
    @property
    def OpsSect(self):
        return self._OpsSect

    @property
    def Paras(self):
        return self._Paras
    @Paras.setter
    def Paras(self, newVal):
        if type(newVal) is type(self._Paras):
            self._Paras = newVal
            self._SectReBuild()
        else:
            raise Exception("Wrong Paras")
    
    # @property
    # def N_aixs(self):
    #     return self._N_axis
    # @N_aixs.setter
    # def N_aixs(self, newVal):
    #     if type(newVal) is type(self._N_axis):
    #         self._N_axis = newVal
    #         self._SectReBuild()
    #     else:
    #         raise Exception("Wrong Paras")
    
    # @property
    # def OrigPoint(self):
        
    #     return self._OrigPoint
    # @OrigPoint.setter
    # def OrigPoint(self, newVal):
    #     if type(newVal) is type(self._OrigPoint):
    #         self._OrigPoint = newVal
    #         self._SectReBuild()
    #     else:
    #         raise Exception("Wrong Paras")
    
    @property
    def SectAttr(self):
        return self._SectAttr

    @property
    def Points(self):
        return self._Points 
    
    @property
    def Material(self):
        return self._CoreCon
    @Material.setter
    def Material(self, newVal):
        if type(newVal) is type(self._CoreCon):
            self._CoreCon = newVal
            self._SectReBuild()
        else:
            raise Exception("Wrong Paras")

    @property
    def val(self):
        """
        return [self._Paras, self._N_axis, self._orig_point, self._Points, self._Arr, self._m, self._OpsSect]
        """
        return [self._Paras, self._Points, self._SectAttr, self._CoreCon, self._OpsSect]
        
    
    def plotSect(self, ax3d):
        # if self.check():
        x, y, z = np.hsplit(self._Points, [1, 2])
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        ax3d.plot(
            np.hstack((x[:8], x[0])),
            np.hstack((y[:8], y[0])),
            np.hstack((z[:8], z[0])),
        )
        ax3d.plot(
            np.hstack((x[8:], x[8])),
            np.hstack((y[8:], y[8])),
            np.hstack((z[8:], z[8])),
        )
        orig_point = (self._Points[0] + self._Points[1]) / 2
        ax3d.plot(
            [orig_point[0], orig_point[0] + self._N_axis[0]],
            [orig_point[1], orig_point[1] + self._N_axis[1]],
            [orig_point[2], orig_point[2] + self._N_axis[2]],
        )
        # else:
        #     raise Exception("Exist Undefined Member")

class SRoundRCSect(RCCrossSect):
    __slots__ = []
    @Comp.CompMgr()
    def __init__(self, paras: Paras.SRoundSectParas, CoreCon: Paras.ConcreteParas, CoverCon: Paras.ConcreteParas, Rebar: Paras.SteelParas, R_flag, fiberSize=(100, 100), name=""):
        super().__init__(paras, CoreCon, CoverCon, Rebar, R_flag, fiberSize,name)
        self._type += "->Soild Round Reforcment Concrete Section"

    def _SectAttrBuild(self):
        paras:Paras.SRoundSectParas = self._Paras
        d = paras.R * 2
        step = paras.R * np.pi / GlobalData.DEFVAL._ROUND_SECT_STEP_DEF

        sectAttr = xsect.round_summary(d, step = step)
        points = xsect.round_points(d, step = step)
        points = UtilTools.PointsTools.TransXsectPointTo3D(points)
        # points = UtilTools.PointsTools.RotatePoinsByVects(points, UtilTools.PointsTools.X_AXIS, self._N_axis)

        return sectAttr, points

    def _RebarsDistrBuild(self, R):
        
        # if len(self._OrigPoint) != 3 or len(self._N_axis) != 3:
        #     raise Exception("Error length of orig_point or N_axis, should be 3")

        Rebars: Paras.SectRebarDistrParas = Paras.SectRebarDistrParas(
            *(UtilTools.BarsTools.SRoundRebarDistr(self._Paras, self._SectAttr, R))
        )
        
        return Rebars

    def _OpsSectBuild(self):
        paras:Paras.SRoundSectParas = self._Paras

        cover = OpsObject.OpsConcrete02(*self._CoverCon.val)
        core = OpsObject.OpsConcrete02(*self._CoreCon.val)
        rebar = OpsObject.OpsSteel02(*self._Rebar.val)

        return OpsObject.OpsSRoundFiberSection(paras.C, paras.R, self._SectAttr, self._RebarsDistr, core, cover, rebar, self._fiberSize)
    
    @property
    def val(self):
        return [self._Paras, self._SectAttr, self._Points, self._RebarsDistr, self._CoreCon, self._CoverCon, self._Rebar, self._fiberSize, self._OpsSect]
    @property
    def SectAttr(self):
        return self._SectAttr
    @property
    def OpsSect(self):
        return self._OpsSect

    def _SectReBuild(self):
        ...
    
    def plotSect(self, Axe3d):
        ...
# * 空心圆截面桥墩
class HRoundRCSect(RCCrossSect):
    __slots__ = []

    @Comp.CompMgr()
    def __init__(self, paras: Paras.HRoundSectParas, CoreCon: Paras.ConcreteParas, CoverCon: Paras.ConcreteParas, Rebar: Paras.SteelParas, R_flag, fiberSize:tuple[int,...]=(100, 100), name=""):
        super().__init__(paras, CoreCon, CoverCon, Rebar, R_flag, fiberSize, name)
        self._type += "->Hollow Round RC Sect"
        # self._attr, self._Points = self._SectAttrBuild()
        # self._RebarsDistr = self._RebarsDistrBuild()
    
    def _SectAttrBuild(self):
        paras:Paras.HRoundSectParas = self._Paras
        d = paras.Rout * 2
        t = paras.T
        step = (d/2 - t) * np.pi / GlobalData.DEFVAL._ROUND_SECT_STEP_DEF
        attr = xsect.round_summary(d, t, step = step)
        points = xsect.round_points(d, t, step = step)
        points = UtilTools.PointsTools.TransXsectPointTo3D(points)
        # points = UtilTools.PointsTools.RotatePoinsByVects(points, UtilTools.PointsTools.X_AXIS, self._N_axis)

        return attr, points
        
    def _RebarsDistrBuild(self, R):
        # if len(self._OrigPoint) != 3 or len(self._N_axis) != 3:
        #     raise Exception("Error length of orig_point or N_axis, should be 3")

        Rebars: Paras.SectRebarDistrParas = Paras.SectRebarDistrParas(
            *(UtilTools.BarsTools.HRoundRebarDistr(self._Paras, self._SectAttr, R))
        )
        
        return Rebars

    def _OpsSectBuild(self):
        paras:Paras.HRoundSectParas = self._Paras
        
        core = OpsObject.OpsConcrete02(*self._CoreCon.val)
        cover = OpsObject.OpsConcrete02(*self._CoverCon.val)
        rebar = OpsObject.OpsSteel02(*self._Rebar.val)
        return OpsObject.OpsHRoundFiberSection(paras.C, paras.Rout, paras.Rin, self._SectAttr, self._RebarsDistr, cover, core, rebar, self._fiberSize)

    def _SectReBuild(self):
        ...

    def plotSect(self, Axe3d):
        #TODO
        ...

    @property
    def SectAttr(self):
        return self._SectAttr
    # @SectAttr.setter
    # def SectAttr(self, newVal):
    #     if type(newVal) is type(self._SectAttr):
    #         self._SectAttr = newVal
    #     else:
    #         raise Exception("Wrong Paras")
    
    @property
    def RebarDistr(self):
        return self._RebarsDistr

    @property
    def OpsSect(self):
        return self._OpsSect
    
    @property
    def val(self):
        return [self._Paras, self._SectAttr, self._Points, self._RebarsDistr, self._CoreCon, self._CoverCon, self._Rebar, self._fiberSize, self._OpsSect]
    

# * 空心矩形截面
class HRectRCSect(RCCrossSect):
    @Comp.CompMgr()
    def __init__(self, paras: Paras.HRectSectParas, CoreCon: Paras.ConcreteParas, CoverCon: Paras.ConcreteParas, Rebar: Paras.SteelParas, R_flag, fiberSize:tuple[int,...]=(100, 100), name=""):
        super().__init__(paras, CoreCon, CoverCon, Rebar, R_flag, fiberSize, name)
        self._type += "->Hollow Rectangle RC Sect"

        # self._Attr, self._Points = self._SectAttrBuild()

    def _OpsSectBuild(self):

        opsConCore = OpsObject.OpsConcrete02(*self._CoreCon.val)
        opsConCover = OpsObject.OpsConcrete02(*self._CoverCon.val)
        opsRebar = OpsObject.OpsSteel02(*self._Rebar.val)
        paras:Paras.HRectSectParas = self._Paras

        return OpsObject.OpsHRectFiberSection(paras.C, paras.W, paras.L, paras.T, self._RebarsDistr, opsConCore, opsConCover, opsRebar, self._fiberSize)

    def _SectAttrBuild(self):
        paras:Paras.HRectSectParas = self._Paras
        # *                | w
        # *       1________|________2
        # *       |  5_____|_____6  |
        # *       |  |     |     | -|--t
        # *       |  |     |     |  |
        # *       |  |     |     |  |--l
        # *-------|--|-----|-----|--|---------->x
        # *       |  |     |     |  |
        # *       |  |     |     |  |
        # *       | 8|_____|_____|7 |
        # *      4|________|________|3
        # *                |
        # * 检查输入参数是否满足要求
        if paras.check() != True:
            raise Exception("Parameters Exists Zero")
        points = np.array(
            [
                [-paras.W / 2, paras.L / 2, 0],
                [paras.W / 2, paras.W/ 2, 0],
                [paras.W / 2, -paras.L / 2, 0],
                [-paras.W / 2, -paras.L / 2, 0],
                [-paras.W / 2 + paras.T, paras.W/ 2 - paras.T, 0],
                [paras.W / 2 - paras.T, paras.L / 2 - paras.T, 0],
                [paras.W / 2 - paras.T, -paras.L / 2 + paras.T, 0],
                [-paras.W / 2 + paras.T, -paras.L / 2 + paras.T, 0],
            ]
        )
        xy = np.hsplit(points, [2])[0]
        xy_out = xy[:4]
        xy_in = xy[4:]
        arr = xsect.multi_section_summary([xy_out], subtract=[xy_in])
        # points = UtilTools.PointsTools.RotatePointsByPoints(points, (1, 0, 0), self._N_axis)
        # points += self._OrigPoint
        return arr, points
        # x1, y1, z1 = (0, 0, 1)
        # x2, y2, z2 = (float(x) for x in self._N_axis)
        # # * 绕X轴偏转, 计算y-z平面的夹角
        # try:
        #     cos_theta = (y1 * y2 + z1 * z2) / (math.sqrt(y2 * y2 + z2 * z2))
        #     theta = np.arccos(cos_theta)
        #     if y2 > 0:
        #         theta = 2 * np.pi - theta
        #     Trans_Xaxis = np.array(
        #         [
        #             [1, 0, 0],
        #             [0, np.cos(theta), np.sin(theta)],
        #             [0, -np.sin(theta), np.cos(theta)],
        #         ]
        #     )
        #     points = np.matmul(points, Trans_Xaxis)
        # except:
        #     pass
        # # * 绕y轴偏转，计算x-z平面夹角
        # try:
        #     cos_theta = (x1 * x2 + z1 * z2) / math.sqrt(x2 * x2 + z2 * z2)
        #     theta = np.arccos(cos_theta)
        #     if x2 < 0:
        #         theta = 2 * np.pi - theta
        #     Trans_Yaxis = np.array(
        #         [
        #             [np.cos(theta), 0, -np.sin(theta)],
        #             [0, 1, 0],
        #             [np.sin(theta), 0, np.cos(theta)],
        #         ]
        #     )
        #     points = np.matmul(points, Trans_Yaxis)
        #     points = points + self._OrigPoints
        # except:
        #     pass
    
        return arr, points

    def _RebarsDistrBuild(self, R):
        
        # if len(self._OrigPoints) != 3 or len(self._N_axis) != 3:
        #     raise Exception("Error length of orig_point or N_axis, should be 3")

        Rebars: Paras.SectRebarDistrParas= Paras.SectRebarDistrParas(
            *(UtilTools.BarsTools.HRectRebarDistr(self._Paras, self._SectAttr, R)))

        return Rebars
    
    def _SectReBuild(self):
        ...
        # Comp.CompMgr.removeComp(self._OpsSect)
        # self._OpsConCore = OpsObject.OpsConcrete02(*self._CoreCon.val)
        # self._OpsConCover = OpsObject.OpsConcrete02(*self._OpsConCover.val)
        # paras:Paras.HRectSectParas = self._Paras
        # # self._OpsSect = OpsObject.OpsHRectFiberSection(paras.C, paras.W, paras.L, paras.T, paras.)

    @property
    def Paras(self):
        return self._Paras
    @Paras.setter
    def Paras(self, newVal):
        if type(newVal) is type(self._Paras):
            self._Paras = newVal
            self._SectAttrBuild()
            self._SectReBuild()
        else:
            raise Exception("Wrong Paras")

    @property
    def RebarsDistr(self):
        return self._RebarsDistr
    @RebarsDistr.setter
    def RebarsDistr(self, newVal):
        if type(newVal) is float or type(newVal) is type(Paras.SectRebarDistrParas):
            self._RebarsDistr = self._RebarsDistrBuild(newVal)
            self._SectReBuild()
        else:
            raise Exception("Wrong Paras")
    
    # @property
    # def N_Axis(self):
    #     return self._N_axis
    # @N_Axis.setter
    # def N_Axis(self, newVal):
    #     if type(newVal) is type(self._N_axis):
    #         self._N_axis = newVal
    #         self._SectAttrBuild()
    #         self._SectReBuild()
    #     else:
    #         raise Exception("Wrong Paras")
    
    # @property
    # def OrigPoint(self):
    #     return self._OrigPoints
    # @OrigPoint.setter
    # def OrigPoint(self, newVal):
    #     if type(newVal) is type(self._OrigPoints):
    #         self._OrigPoints= newVal
    #         self._SectAttrBuild()
    #         self._SectReBuild()
    #     else:
    #         raise Exception("Wrong Paras")
    
    @property
    def ConCore(self):
        return self._CoreCon
    @ConCore.setter
    def ConCore(self, newVal):
        if type(newVal) is type(self._CoreCon):
            self._CoreCon = newVal
            self._SectReBuild()
        else:
            raise Exception("Wrong Paras")
    
    @property
    def ConCover(self):
        return self._CoverCon
    @ConCover.setter
    def ConCover(self, newVal):
        if type(newVal) is type(self._CoverCon):
            self._CoverCon = newVal
            self._SectReBuild()
        else:
            raise Exception("Wrong Paras")
    
    @property
    def Steel(self):
        return self._Rebar
    @Steel.setter
    def Steel(self, newVal):
        if type(newVal) is type(self._Rebar):
            self._Rebar = newVal
            self._SectReBuild()
        else:
            raise Exception("Wrong Paras")
    
    @property
    def val(self):
        """
        return [self._Paras, self._N_axis, self._Orig_p, self._Points, self._Arr, self._Rebars, 
                self._OpsConCore, self._OpsConCover, self._OpsRebar, self._OpsPierSect]
        """
        return [self._Paras, self._Points, self._Rebar, self._CoreCon, self._CoverCon, self._RebarsDistr, self._OpsSect]

    def plotSect(self, ax3d):
        ...
        # if self.check():
        #     x, y, z = np.hsplit(self.Points, [1, 2])
        #     x = x.flatten()
        #     y = y.flatten()
        #     z = z.flatten()
        #     ax3d.plot(
        #         np.hstack((x[:4], x[0])),
        #         np.hstack((y[:4], y[0])),
        #         np.hstack((z[:4], z[0])),
        #     )
        #     ax3d.plot(
        #         np.hstack((x[4:], x[4])),
        #         np.hstack((y[4:], y[4])),
        #         np.hstack((z[4:], z[4])),
        #     )
        #     orig_point = (self.Points[0] + self.Points[2]) / 2
        #     ax3d.plot(
        #         [orig_point[0], orig_point[0] + self.N_axis[0]],
        #         [orig_point[1], orig_point[1] + self.N_axis[1]],
        #         [orig_point[2], orig_point[2] + self.N_axis[2]],
        #     )
        # else:
        #     raise Exception("Exist Undefined Member")

class Segment(Comp.Parts):
    class SupportedElementType(Enum):
        NonlinearBeamColumnElement = 'NBCE'
        ElasticBeamColumnElement = 'EBCE'
        # ZeroLengthElement = 'ZLE'

    @abstractmethod
    def __init__(self, point_i:tuple[float, ...], point_j:tuple[float, ...], 
                       SectParas_i:Paras.SectParas, SectParas_j:Paras.SectParas,
                       elementType:SupportedElementType,
                       power:float,
                       localZ:tuple[float],
                       eleLengthList:list[float],
                       eleDictParams:dict ={},
                       name=""):
        super(Segment, self).__init__(name)
        self._Ni:BridgeNode = BridgeNode(*point_i)
        self._Nj:BridgeNode = BridgeNode(*point_j)

        self._Secti:Paras.SectParas = SectParas_i
        self._Sectj:Paras.SectParas = SectParas_j

        self._ElementType = elementType
        self._ElementDictParams = eleDictParams

        self._p:float = power
        self._localXAxis:np.ndarray = UtilTools.PointsTools.vectSub(self._Ni.point, self._Nj.point)
        self._localZAxis:np.ndarray = np.array(localZ)

        self._totalLen = UtilTools.PointsTools.PointsDist(self._Ni.point, self._Nj.point)

        self._eleLengthList:list[float] = eleLengthList
        
        self._BridgeNodeList:list[BridgeNode] = None
        self._SegElementList:list[OpsObject.OpsElement] = None
        self._SegMassList:list[float] = None
        self._SegSectList:list[CrossSection] = None

    def FindPoint(self, point:tuple[float]):
        for n in self._BridgeNodeList:
            if n.point == point:
                return True, n
        return False, None
        
    def FindElement(self, p1:tuple, p2:tuple):
        for e in self._SegElementList:
            if (e.NodeI.xyz == p1 and e.NodeJ.xyz == p2) or (e.NodeI.xyz == p2 and e.NodeJ.xyz == p1):
                return True, e
        return False, None

    @abstractmethod
    def _Build():
        ...
    @overload
    def BuildElement(cls, elementType:SupportedElementType, node1:OpsObject.OpsNode, node2: OpsObject.OpsNode, sec: OpsObject.OpsSection, localZ: tuple, name="") -> OpsObject.OpsEBCElement:...
    @overload
    def BuildElement(cls, elementType:SupportedElementType, node1:OpsObject.OpsNode, node2:OpsObject.OpsNode, sect:OpsObject.OpsBoxSection, localZ:tuple[int],IntgrNum:int=5, maxIter=10, tol:float=1e-12, mass:float=0.0, IntgrType:str="Lobatto", name="") -> OpsObject.OpsNBCElement:...

    # @classmethod
    def BuildElement(cls, elementType:SupportedElementType, *args, **kwargs):
        
        if elementType == cls.SupportedElementType.ElasticBeamColumnElement:
            ele = OpsObject.OpsEBCElement(*args, **kwargs)
        elif elementType == cls.SupportedElementType.NonlinearBeamColumnElement:
            ele = OpsObject.OpsNBCElement(*args, **kwargs)
        else:
            message = 'Unsupport Element Type'
            StandardLogger.error(message)
            raise Exception(message)
        
        return ele

    @property
    def ELeList(self):
        return self._SegElementList
    @ELeList.setter
    def ELeList(self, newVal):
        if type(newVal) is type(self._SegElementList):
            self._SegElementList = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def NodeList(self):
        return self._BridgeNodeList
    @NodeList.setter
    def NodeList(self, newVal):
        if type(newVal) is type(self._BridgeNodeList):
            self._BridgeNodeList = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def MassList(self):
        return self._SegMassList
    @MassList.setter
    def MassList(self, newVal):
        if type(newVal) is type(self._SegMassList):
            self._SegMassList = newVal
        else:
            raise Exception("Wrong Paras")
    
    @property
    def SectList(self):
        return self._SegSectList
    @SectList.setter
    def SectList(self, newVal):
        if type(newVal) is type(self._SegSectList):
            self._SegSectList = newVal
        else:
            raise Exception("Wrong Paras")
    
    @property
    def EleLength(self):
        return self._eleLengthList
    @EleLength.setter
    def EleLength(self, newVal):
        if type(newVal) is type(self._eleLengthList):
            self._eleLengthList = newVal
        else:
            raise Exception("Wrong Paras")

class LineBoxSeg(Segment):
    __slots__ = []
    @property
    def MassList(self):
        return self._SegMassList
    @MassList.setter
    def MassList(self, newVal):
        if type(newVal) is type(self._SegMassList):
            self._SegMassList = newVal
        else:
            raise Exception("Wrong Paras")
    
    @Comp.CompMgr()
    def __init__(self, node_i:BridgeNode, node_j:BridgeNode, 
                       SectParas_i:Paras.BoxSectParas, SectParas_j:Paras.BoxSectParas, 
                       elementType:Segment.SupportedElementType,
                       con:Paras.ConcreteParas,
                       localZ:tuple[int],
                       eleLengthList:list[float],
                       power:float, 
                       eleDictParams:dict={},
                       name=""):

        super(LineBoxSeg, self).__init__(node_i, node_j, SectParas_i, SectParas_j, elementType, power, localZ, eleLengthList, eleDictParams, name)
        self._con = con
        self._BridgeNodeList, self._SegSectList, self._SegMassList, self._SegElementList = self._Build()

    def _Build(self):
        if UtilTools.Util.TOLEQ(sum(self._eleLengthList), self._totalLen) and self._eleLengthList is not None:
            points = UtilTools.PointsTools.LinePointBuilder(self._Ni.point, self._Nj.point, self._eleLengthList)
            SectI:Paras.BoxSectParas = self._Secti
            SectJ:Paras.BoxSectParas = self._Sectj

            upper_W = UtilTools.SegmentTools.PowerParasBuilder(SectI.upper_W, SectJ.upper_W, self._totalLen, 1, self._eleLengthList)
            upper_T = UtilTools.SegmentTools.PowerParasBuilder(SectI.upper_T, SectJ.upper_T, self._totalLen, 1, self._eleLengthList)

            down_W =  UtilTools.SegmentTools.PowerParasBuilder(SectI.down_W, SectJ.down_W, self._totalLen, 1, self._eleLengthList)
            down_T =  UtilTools.SegmentTools.PowerParasBuilder(SectI.down_T, SectJ.down_T, self._totalLen, 1, self._eleLengthList)

            web_T =  UtilTools.SegmentTools.PowerParasBuilder(SectI.web_T, SectJ.web_T, self._totalLen, 1, self._eleLengthList)

            h =  UtilTools.SegmentTools.PowerParasBuilder(SectI.H, SectJ.H, self._totalLen, self._p, self._eleLengthList)
            N_axis =  UtilTools.PointsTools.vectSub(self._Nj.point, self._Ni.point)
            l_paras:list[Paras.BoxSectParas] = []
            l_node:list[BridgeNode] = []
            l_sect:list[BoxSect] = []
            l_element:list[OpsObject.OpsElement] = []
            l_mass:list[float] = []
            
            for p in points:
                node = BridgeNode(*p, 0.0)
                l_node.append(node)

            for uW, dW, _h, uT, dT, wT in zip(upper_W, down_W, h, upper_T, down_T, web_T):
                paras = Paras.BoxSectParas(uW, dW, _h, uT, dT, wT)
                l_paras.append(paras)
            
            for i in range(len(self._eleLengthList)):
                paras = UtilTools.SectTools.MeanSectParas(l_paras[i], l_paras[i+1])
                paras = Paras.BoxSectParas(*paras)
                sect = BoxSect(paras, self._con)
                l_sect.append(sect)
                mass = sect.SectAttr['area'] * self._eleLengthList[i] * self._con.densty
                l_mass.append(mass)
                l_node[i].addMass(mass/2)
                l_node[i+1].addMass(mass/2)
                ele = self.BuildElement(self._ElementType, l_node[i].OpsNode, l_node[i+1].OpsNode, l_sect[i].OpsSect, self._localZAxis, **self._ElementDictParams) 
                # ele = OpsObject.OpsEBCElement(l_node[i].OpsNode, 
                #                     l_node[i+1].OpsNode,
                #                     l_sect[i].OpsSect,
                #                     self._localZAxis
                #                     )

                l_element.append(ele) 
            
            return l_node, l_sect, l_mass, l_element

    @property
    def NodeI(self):
        return self._Ni
    @NodeI.setter
    def NodeI(self, newVal):
        if type(newVal) is type(self._Ni):
            self._Ni = newVal
        else:
            raise Exception("Wrong Paras")
     
    @property
    def NodeJ(self):
        return self._Nj
    @NodeJ.setter
    def NodeJ(self, newVal):
        if type(newVal) is type(self._Nj):
            self._Nj = newVal
        else:
            raise Exception("Wrong Paras")
     
    @property
    def SectI(self):
        return self._Secti
    @SectI.setter
    def SectI(self, newVal):
        if type(newVal) is type(self._Secti):
            self._Secti = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def SectJ(self):
        return self._Sectj
    @SectJ.setter
    def SectJ(self, newVal):
        if type(newVal) is type(self._Sectj):
            self._Sectj = newVal
        else:
            raise Exception("Wrong Paras")
    
    @property
    def Power(self):
        return self._p
    @Power.setter
    def Power(self, newVal):
        if type(newVal) is type(self._p):
            self._p = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def ElementList(self):
        return self._GirderElementList
    @ElementList.setter
    def ElementList(self, newVal):
        if type(newVal) is type(self.val):
            self._GirderElementList= newVal
        else:
            raise Exception("Wrong Paras")
    
    def _SectReBuild(self):
        ...

    @property
    def ELeList(self):
        return self._SegElementList
    @ELeList.setter
    def ELeList(self, newVal):
        if type(newVal) is type(self._SegElementList):
            self._SegElementList = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def NodeList(self):
        return self._BridgeNodeList
    @NodeList.setter
    def NodeList(self, newVal):
        if type(newVal) is type(self._BridgeNodeList):
            self._BridgeNodeList = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def MassList(self):
        return self._SegMassList
    @MassList.setter
    def MassList(self, newVal):
        if type(newVal) is type(self._SegMassList):
            self._SegMassList = newVal
        else:
            raise Exception("Wrong Paras")
    
    @property
    def SectList(self):
        return self._SegSectList
    @SectList.setter
    def SectList(self, newVal):
        if type(newVal) is type(self._SegSectList):
            self._SegSectList = newVal
        else:
            raise Exception("Wrong Paras")
    
    @property
    def val(self):
        return [self._Ni, self._Nj, self._Secti, self._Sectj, self._ElementType, self._con, self._localZAxis, self._eleLengthList, self._p]
# 
class LineHRoundSeg(Segment):
    @Comp.CompMgr()
    def __init__(self, point_i: tuple[float, ...], point_j: tuple[float, ...], SectParas_i:Paras.HRoundSectParas, SectParas_j:Paras.HRoundSectParas, elementType:Segment.SupportedElementType, ConCore:Paras.ConcreteParas, ConCover:Paras.ConcreteParas, Rebar:Paras.SteelParas, eleLengthList: list[float], RebarRList:list[float]=None, RebarDistrParasList:list[Paras.SectRebarDistrParas]=None, localZ: tuple[float]=None, elemetDictParam:dict={}, name=""):
        super().__init__(point_i, point_j, SectParas_i, SectParas_j, elementType, 1, localZ, eleLengthList, elemetDictParam, name)
        self._CoreCon = ConCore
        self._CoverCon = ConCover
        self._Rebar = Rebar

        if RebarDistrParasList and len(RebarDistrParasList) == len(eleLengthList):
            print("INFO:As 'RebarDistrParasList' was used, 'RebarRList' is ingored")
            self._R = RebarDistrParasList
        elif RebarRList and len(RebarRList) == len(eleLengthList):
            self._R = RebarRList
        else:
            raise Exception("Wrong Params, RebarDistParasList and RebarRList can not be None at same time, and the length should be same with eleLengthList.")

        self._BridgeNodeList, self._SegSectList, self._SegMassList, self._SegElementList = self._Build()

    def _Build(self):
        if UtilTools.Util.TOLEQ(sum(self._eleLengthList), self._totalLen) and self._eleLengthList is not None:
            points = UtilTools.PointsTools.LinePointBuilder(self._Ni.point, self._Nj.point, self._eleLengthList)
            SectI:Paras.HRoundSectParas = self._Secti
            SectJ:Paras.HRoundSectParas = self._Sectj

            T = UtilTools.SegmentTools.PowerParasBuilder(SectI.T, SectJ.T, self._totalLen, 1, self._eleLengthList)
            

            Rout = UtilTools.SegmentTools.PowerParasBuilder(SectI.Rout, SectJ.Rout, self._totalLen, 1, self._eleLengthList)

            # N_axis =  UtilTools.PointsTools.vectSub(self._Nj.point, self._Ni.point)
            l_paras:list[Paras.HRoundSectParas] = []
            l_node:list[BridgeNode] = []
            l_sect:list[HRoundRCSect] = []
            l_element:list[OpsObject.OpsElement] = []
            l_mass:list[float] = []
            
            for p in points:
                node = BridgeNode(*p, 0.0)
                l_node.append(node)

            for rout, t in zip(Rout, T):
                paras = Paras.HRoundSectParas(rout, t)
                l_paras.append(paras)

            for i in range(len(self._eleLengthList)):
                paras = UtilTools.SectTools.MeanSectParas(l_paras[i], l_paras[i+1])
                paras = Paras.HRoundSectParas(*paras)

                sect = HRoundRCSect(paras, self._CoreCon, self._CoverCon, self._Rebar, self._R[i])
                l_sect.append(sect)

                mass = sect.SectAttr['area'] * self._eleLengthList[i] * self._CoreCon.densty
                l_mass.append(mass)
                l_node[i].addMass(mass/2)
                l_node[i+1].addMass(mass/2)
                
                ele = self.BuildElement(self._ElementType, l_node[i].OpsNode, l_node[i+1].OpsNode, l_sect[i].OpsSect, self._localZAxis, **self._ElementDictParams)
                # ele = OpsObject.OpsEBCElement(l_node[i].OpsNode, 
                #                     l_node[i+1].OpsNode,
                #                     l_sect[i].OpsSect,
                #                     self._localZAxis
                #                     )

                l_element.append(ele) 

            return l_node, l_sect, l_mass, l_element
        else:
            raise Exception("wrong param, element totoal length is not euqal the sum of elementlengthlist")
    

    @property
    def ELeList(self):
        return self._SegElementList
    @ELeList.setter
    def ELeList(self, newVal):
        if type(newVal) is type(self._SegElementList):
            self._SegElementList = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def NodeList(self):
        return self._BridgeNodeList
    @NodeList.setter
    def NodeList(self, newVal):
        if type(newVal) is type(self._BridgeNodeList):
            self._BridgeNodeList = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def MassList(self):
        return self._SegMassList
    @MassList.setter
    def MassList(self, newVal):
        if type(newVal) is type(self._SegMassList):
            self._SegMassList = newVal
        else:
            raise Exception("Wrong Paras")
    
    @property
    def SectList(self):
        return self._SegSectList
    @SectList.setter
    def SectList(self, newVal):
        if type(newVal) is type(self._SegSectList):
            self._SegSectList = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def ELeList(self):
        return self._SegElementList
    @ELeList.setter
    def ELeList(self, newVal):
        if type(newVal) is type(self._SegElementList):
            self._SegElementList = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def NodeList(self):
        return self._BridgeNodeList
    @NodeList.setter
    def NodeList(self, newVal):
        if type(newVal) is type(self._BridgeNodeList):
            self._BridgeNodeList = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def MassList(self):
        return self._SegMassList
    @MassList.setter
    def MassList(self, newVal):
        if type(newVal) is type(self._SegMassList):
            self._SegMassList = newVal
        else:
            raise Exception("Wrong Paras")
    
    @property
    def SectList(self):
        return self._SegSectList
    @SectList.setter
    def SectList(self, newVal):
        if type(newVal) is type(self._SegSectList):
            self._SegSectList = newVal
        else:
            raise Exception("Wrong Paras")
    
    @property
    def val(self):
        return [self._Ni, self._Nj, self._Secti, self._Sectj, self._CoreCon, self._CoverCon, self._Rebar, self._localZAxis, self._eleLengthList, self._R]

class LineSRoundSeg(Segment):
    @Comp.CompMgr()
    def __init__(self, node_i: tuple[float, ...], node_j: tuple[float, ...], SectParas_i: Paras.SRoundSectParas, SectParas_j: Paras.SRoundSectParas, elementType:Segment.SupportedElementType, ConCore:Paras.ConcreteParas, ConCover:Paras.ConcreteParas, Rebar:Paras.SteelParas, eleLengthList: list[float], RebarRList:list[float]=None, RebarDistrParasList:list[Paras.SectRebarDistrParas]=None, localZ: tuple[float]=None, eleDictParams:dict={}, name=""):
        super().__init__(node_i, node_j, SectParas_i, SectParas_j, elementType, 1, localZ, eleLengthList, eleDictParams, name)
        self._CoreCon = ConCore
        self._CoverCon = ConCover
        self._Rebar = Rebar

        if RebarDistrParasList and len(RebarDistrParasList) == len(eleLengthList):
            print("INFO:As 'RebarDistrParasList' was used, 'RebarRList' is ingored")
            self._R = RebarDistrParasList
        elif RebarRList and len(RebarRList) == len(eleLengthList):
            self._R = RebarRList
        else:
            raise Exception("Wrong Params, RebarDistParasList and RebarRList can not be None at same time, and the length should be same with eleLengthList.")

        self._BridgeNodeList, self._SegSectList, self._SegMassList, self._SegElementList = self._Build()
        
    def _Build(self):
        points = UtilTools.PointsTools.LinePointBuilder(self._Ni.point, self._Nj.point, self._eleLengthList)
        SectI:Paras.SRoundSectParas = self._Secti
        SectJ:Paras.SRoundSectParas = self._Sectj


        R = UtilTools.SegmentTools.PowerParasBuilder(SectI.R, SectJ.R, self._totalLen, 1, self._eleLengthList)

        # N_axis =  UtilTools.PointsTools.vectSub(self._Nj.point, self._Ni.point)
        l_paras:list[Paras.SRoundSectParas] = []
        l_node:list[BridgeNode] = []
        l_sect:list[SRoundRCSect] = []
        l_element:list[OpsObject.OpsElement] = []
        l_mass:list[float] = []
        
        for p in points:
            node = BridgeNode(*p, 0.0)
            l_node.append(node)

        for r in R:
            paras = Paras.SRoundSectParas(r)
            l_paras.append(paras)

        for i in range(len(self._eleLengthList)):
            paras = UtilTools.SectTools.MeanSectParas(l_paras[i], l_paras[i+1])
            paras = Paras.SRoundSectParas(*paras)

            sect = SRoundRCSect(paras, self._CoreCon, self._CoverCon, self._Rebar, self._R[i])
            l_sect.append(sect)
            

            mass = sect.SectAttr['area'] * self._eleLengthList[i] * self._CoreCon.densty
            l_mass.append(mass)
            l_node[i].addMass(mass/2)
            l_node[i+1].addMass(mass/2)

            # ele = OpsObject.OpsEBCElement(l_node[i].OpsNode, 
            #                     l_node[i+1].OpsNode,
            #                     l_sect[i].OpsSect,
            #                     self._localZAxis
            #                     )
            ele = self.BuildElement(self._ElementType, l_node[i].OpsNode, l_node[i+1].OpsNode, l_sect[i].OpsSect, self._localZAxis, **self._ElementDictParams)
            # ele = OpsObject.OpsNBCElement(l_node[i].OpsNode, l_node[i+1].OpsNode, l_sect[i].OpsSect, self._localZAxis, 5)

            l_element.append(ele) 
        
        return l_node, l_sect, l_mass, l_element
    
    @property
    def ELeList(self):
        return self._SegElementList
    @ELeList.setter
    def ELeList(self, newVal):
        if type(newVal) is type(self._SegElementList):
            self._SegElementList = newVal
        else:
            raise Exception("Wrong Paras")

    @property
    def NodeList(self):
        return self._BridgeNodeList
    @NodeList.setter
    def NodeList(self, newVal):
        if type(newVal) is type(self._BridgeNodeList):
            self._BridgeNodeList = newVal
        else:
            raise Exception("Wrong Paras")

    @property
    def MassList(self):
        return self._SegMassList
    @MassList.setter
    def MassList(self, newVal):
        if type(newVal) is type(self._SegMassList):
            self._SegMassList = newVal
        else:
            raise Exception("Wrong Paras")
    
    @property
    def SectList(self):
        return self._SegSectList
    @SectList.setter
    def SectList(self, newVal):
        if type(newVal) is type(self._SegSectList):
            self._SegSectList = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def val(self):
        return [self._Ni, self._Nj, self._Secti, self._Sectj, self._CoreCon, self._CoverCon, self._Rebar, self._localZAxis, self._eleLengthList, self._R]

#TODO
class BridgeModel(Comp.Parts):
    @Comp.CompMgr()
    def __init__(self, name=""):
        super(BridgeModel, self).__init__(name)
        self._type += "BridgeParts"
        self._BridgeParas = None
        self._Materials: list = []
        self._Boundaries: list = []
        self._Girder = None
        self._Pier: list = []

    def SetBridgeAttribution(self, *args):
        for arg in args:
            if isinstance(arg, Paras.BridgeParas):
                self._BridgeParas = arg
            elif isinstance(arg, Paras.MaterialParas):
                self._Materials.append(arg)
            elif isinstance(arg, Boundary):
                self._Boundaries.append(arg)
            elif isinstance(arg, LineBoxSeg):
                self._Girder = arg
            elif isinstance(arg, LineHRoundSeg):
                self._Pier.append(arg)
            else:
                print('Wrong Type of argument: "' + arg + '"')
                pass
    def check(self):
        if (self._BridgeParas is not Paras.BoxSectParas
            or len(self._Materials) == 0
            or len(self._Boundaries) == 0
            or self._Girder is not LineBoxSeg
            or len(self._Pier) == 0
        ):
            return False
        else:
            return True
