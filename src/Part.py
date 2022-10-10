from enum import Enum
from typing import overload
import xsect
from abc import ABCMeta, abstractmethod
import numpy as np

# from src.Unit import ConvertToBaseUnit

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
        self._OpsMassList.append(OpsObject.OpsMass(self._OpsNode, mass, massDis))
        
    @property
    def val(self):
        """
        return [self._point, self._mass, self._OpsNode, self._OpsMass]
        """
        return [self._point, self._mass, self._OpsNode, self._OpsMassList]


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

        self._type += "->Reinforcement Concrete Cross-Section"
        self._CoreCon:Paras.ConcreteParas = CoreCon
        self._CoverCon:Paras.ConcreteParas = CoverCon
        self._Rebar:Paras.SteelParas = Rebar

        if type(R_flag) is float:
            self._RebarDistr:Paras.SectRebarDistrParas = self._RebarDistrBuild(R_flag)
        elif type(R_flag) is Paras.SectRebarDistrParas:
            self._RebarDistr:Paras.SectRebarDistrParas = R_flag
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
    def _RebarDistrBuild(self, R):
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
        
        # self._attr, self._points = self._SectAttrBuild()
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
        #     Trans_ZAxis = np.array(
        #         [
        #             [np.cos(theta), np.sin(theta), 0],
        #             [-np.sin(theta), np.cos(theta), 0],
        #             [0, 0, 1],
        #         ]
        #     )
        #     points = np.matmul(points, Trans_ZAxis)
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

    def _RebarDistrBuild(self, R):
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
    # def N_axis(self):
    #     return self._N_axis
    # @N_axis.setter
    # def N_axis(self, newVal):
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
        self._type += "->Solid Round Reinforcement Concrete Section"

    def _SectAttrBuild(self):
        paras:Paras.SRoundSectParas = self._Paras
        d = paras.R * 2
        step = paras.R * np.pi / GlobalData.DEFVAL._ROUND_SECT_STEP_DEF

        sectAttr = xsect.round_summary(d, step = step)
        points = xsect.round_points(d, step = step)
        points = UtilTools.PointsTools.TransXsectPointTo3D(points)
        # points = UtilTools.PointsTools.RotatePointsByVector(points, UtilTools.PointsTools.X_AXIS, self._N_axis)

        return sectAttr, points

    def _RebarDistrBuild(self, R):
        
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

        return OpsObject.OpsSRoundFiberSection(paras.C, paras.R, self._SectAttr, self._RebarDistr, core, cover, rebar, self._fiberSize)
    
    @property
    def val(self):
        return [self._Paras, self._SectAttr, self._Points, self._RebarDistr, self._CoreCon, self._CoverCon, self._Rebar, self._fiberSize, self._OpsSect]
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
        # points = UtilTools.PointsTools.RotatePointsByVector(points, UtilTools.PointsTools.X_AXIS, self._N_axis)

        return attr, points
        
    def _RebarDistrBuild(self, R):
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
        return OpsObject.OpsHRoundFiberSection(paras.C, paras.Rout, paras.Rin, self._SectAttr, self._RebarDistr, cover, core, rebar, self._fiberSize)

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
        return self._RebarDistr

    @property
    def OpsSect(self):
        return self._OpsSect
    
    @property
    def val(self):
        return [self._Paras, self._SectAttr, self._Points, self._RebarDistr, self._CoreCon, self._CoverCon, self._Rebar, self._fiberSize, self._OpsSect]
    

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

        return OpsObject.OpsHRectFiberSection(paras.C, paras.W, paras.L, paras.T, self._RebarDistr, opsConCore, opsConCover, opsRebar, self._fiberSize)

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
        #     points = np.matmul(points, Trans_XAxis)
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

    def _RebarDistrBuild(self, R):
        
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
        return self._RebarDistr
    @RebarsDistr.setter
    def RebarsDistr(self, newVal):
        if type(newVal) is float or type(newVal) is type(Paras.SectRebarDistrParas):
            self._RebarDistr = self._RebarDistrBuild(newVal)
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
        return [self._Paras, self._Points, self._Rebar, self._CoreCon, self._CoverCon, self._RebarDistr, self._OpsSect]

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

class Body(Comp.Parts, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, name=""):
        super().__init__(name)
        self._type += '->body'
        self._Elements:list[OpsObject.OpsElement] = None
        self._BridgeNodes:list[BridgeNode] = None
        self._Masses:list[float] = None
    
    @abstractmethod
    def _Build(self):
        ...

    @abstractmethod
    def FindElement(self): ...

    @abstractmethod
    def FindRangeElement(self): ...

    @abstractmethod
    def FindBridgeNode(self): ...

    @abstractmethod
    def FindRangeBridgeNodes(self): ...

    @abstractmethod
    def BuildElement(self, elementType, *args, **kwargs):...

class Segment(Body, metaclass=ABCMeta):
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
                       eleExtraDictParams:dict ={},
                       name=""):
        super(Segment, self).__init__(name)
        self._type += '->Segment'
        self._BridgeNodeI:BridgeNode = BridgeNode(*point_i)
        self._BridgeNodeJ:BridgeNode = BridgeNode(*point_j)

        self._Secti:Paras.SectParas = SectParas_i
        self._Sectj:Paras.SectParas = SectParas_j

        self._ElementType = elementType
        self._ElementExtraDictParams = eleExtraDictParams

        self._p:float = power
        self._localXAxis:np.ndarray = UtilTools.PointsTools.VectorSub(self._BridgeNodeI.point, self._BridgeNodeJ.point)
        self._localZAxis:np.ndarray = np.array(localZ)

        self._totalLen = UtilTools.PointsTools.PointsDist(self._BridgeNodeI.point, self._BridgeNodeJ.point)

        self._eleLengthList:list[float] = eleLengthList
        
        self._BridgeNodes:list[BridgeNode] = None
        self._Elements:list[OpsObject.OpsLineElement] = None
        self._Masses:list[float] = None
        self._SegSectList:list[CrossSection] = None
        

    def FindBridgeNode(self, point:tuple[float], fuzzy:bool=False, tol=None):
        tempN = None
        tempDis = GlobalData.DEFVAL._MAXVAL_
        for n in self._BridgeNodes:
            if n.point == point:
                return True, n
            elif fuzzy:
                dis = UtilTools.PointsTools.PointsDist(n.point, point)

                if tol and not UtilTools.Util.TOLLE(dis, tol):
                    dis = GlobalData.DEFVAL._MAXVAL_

                tempDis, tempN = (dis, n) if dis < tempDis else (tempDis, tempN)
        
        if fuzzy and UtilTools.Util.TOLLG(tempDis, GlobalData.DEFVAL._MAXVAL_):
            return True, tempN
        else:
            return False, None
    
    def FindRangeBridgeNodes(self, p1, p2, fuzzy=False, tol=None):
        i = j = -1
        if fuzzy:
            flag, p1 = self.FindBridgeNode(p1, fuzzy=True, tol=tol)
            if not flag:
                return False, None

            flag, p2 = self.FindBridgeNode(p2, fuzzy=True, tol=tol)
            if not flag:
                return False, None

        for index, n in enumerate(self._BridgeNodes):
            if n.point == p1:
                i = index

            if n.point == p2:
                j = index
        

        if -1 in (i, j):
            return False, None
        else:
            i,j = (i, j) if i<j else (j,i)
            return True, self._BridgeNodes[i:j+1]
        
    def FindElement(self, p1:tuple, p2:tuple, fuzzy=False, tol=None):
        if fuzzy:
            flag, p1 = self.FindBridgeNode(p1, fuzzy=True, tol=tol)
            if not flag:
                return False, None

            flag, p2 = self.FindBridgeNode(p2, fuzzy=True, tol=tol)
            if not flag:
                return False, None

        for e in self._Elements:
            if (e.NodeI.xyz == p1 and e.NodeJ.xyz == p2) or (e.NodeI.xyz == p2 and e.NodeJ.xyz == p1):
                return True, e
        return False, None

    def FindRangeElement(self, p1, p2, fuzzy=False, tol=None):
        if fuzzy:
            flag, p1 = self.FindBridgeNode(p1, fuzzy=True, tol=tol)
            if not flag:
                return False, None

            flag, p2 = self.FindBridgeNode(p2, fuzzy=True, tol=tol)
            if not flag:
                return False, None

        disP1 = UtilTools.PointsTools.PointsDist(p1, self._Elements[0].NodeI.xyz)
        disP2 = UtilTools.PointsTools.PointsDist(p2, self._Elements[0].NodeI.xyz)

        stP, edP = (p1, p2) if disP2 > disP1 else (p2, p1)

        idx_stP = idx_edP = -1

        for index, e in enumerate(self._Elements):
            if e.NodeI.xyz == stP:
                idx_stP = index

            if e.NodeJ.xyz == edP:
                idx_edp = index

        if idx_edP == -1 and idx_stP == -1:
            return False, None
        elif idx_edP == -1 or idx_stP == -1:
            msg = "Start point {} or end point {} can not find, try FindElement".format(stP, edP)
            StandardLogger.info(msg)
            print(msg)
            return True, 
    @abstractmethod
    def _Build():
        ...

    @overload
    def BuildElement(cls, elementType:SupportedElementType, node1:OpsObject.OpsNode, node2: OpsObject.OpsNode, sec: OpsObject.OpsSection, localZ: tuple, name="") -> OpsObject.OpsEBCElement:...
    @overload
    def BuildElement(cls, elementType:SupportedElementType, node1:OpsObject.OpsNode, node2:OpsObject.OpsNode, sect:OpsObject.OpsBoxSection, localZ:tuple[int],IntgrNum:int=5, maxIter=10, tol:float=1e-12, mass:float=0.0, IntgrType:str="Lobatto", name="") -> OpsObject.OpsNBCElement:...

    # @classmethod
    def BuildElement(self, elementType:SupportedElementType, *args, **kwargs):
        
        if elementType == self.SupportedElementType.ElasticBeamColumnElement:
            ele = OpsObject.OpsEBCElement(*args, **kwargs)
        elif elementType == self.SupportedElementType.NonlinearBeamColumnElement:
            ele = OpsObject.OpsNBCElement(*args, **kwargs)
        else:
            message = 'Unsupport Element Type'
            StandardLogger.error(message)
            raise Exception(message)
        
        return ele

    @property
    def ELeList(self):
        return self._Elements
    @ELeList.setter
    def ELeList(self, newVal):
        if type(newVal) is type(self._Elements):
            self._Elements = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def NodeList(self):
        return self._BridgeNodes
    @NodeList.setter
    def NodeList(self, newVal):
        if type(newVal) is type(self._BridgeNodes):
            self._BridgeNodes = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def MassList(self):
        return self._Masses
    @MassList.setter
    def MassList(self, newVal):
        if type(newVal) is type(self._Masses):
            self._Masses = newVal
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
        return self._Masses
    @MassList.setter
    def MassList(self, newVal):
        if type(newVal) is type(self._Masses):
            self._Masses = newVal
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
        self._BridgeNodes, self._SegSectList, self._Masses, self._Elements = self._Build()

    def _Build(self):
        if UtilTools.Util.TOLEQ(sum(self._eleLengthList), self._totalLen) and self._eleLengthList is not None:
            points = UtilTools.PointsTools.LinePointBuilder(self._BridgeNodeI.point, self._BridgeNodeJ.point, self._eleLengthList)
            SectI:Paras.BoxSectParas = self._Secti
            SectJ:Paras.BoxSectParas = self._Sectj

            upper_W = UtilTools.SegmentTools.PowerParasBuilder(SectI.upper_W, SectJ.upper_W, self._totalLen, 1, self._eleLengthList)
            upper_T = UtilTools.SegmentTools.PowerParasBuilder(SectI.upper_T, SectJ.upper_T, self._totalLen, 1, self._eleLengthList)

            down_W =  UtilTools.SegmentTools.PowerParasBuilder(SectI.down_W, SectJ.down_W, self._totalLen, 1, self._eleLengthList)
            down_T =  UtilTools.SegmentTools.PowerParasBuilder(SectI.down_T, SectJ.down_T, self._totalLen, 1, self._eleLengthList)

            web_T =  UtilTools.SegmentTools.PowerParasBuilder(SectI.web_T, SectJ.web_T, self._totalLen, 1, self._eleLengthList)

            h =  UtilTools.SegmentTools.PowerParasBuilder(SectI.H, SectJ.H, self._totalLen, self._p, self._eleLengthList)
            N_axis = UtilTools.PointsTools.VectorSub(self._BridgeNodeJ.point, self._BridgeNodeI.point)
            l_paras:list[Paras.BoxSectParas] = []
            l_node:list[BridgeNode] = []
            l_sect:list[BoxSect] = []
            l_element:list[OpsObject.OpsLineElement] = []
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
                ele = self.BuildElement(self._ElementType, l_node[i].OpsNode, l_node[i+1].OpsNode, l_sect[i].OpsSect, self._localZAxis, **self._ElementExtraDictParams) 
                # ele = OpsObject.OpsEBCElement(l_node[i].OpsNode, 
                #                     l_node[i+1].OpsNode,
                #                     l_sect[i].OpsSect,
                #                     self._localZAxis
                #                     )

                l_element.append(ele) 
            
            return l_node, l_sect, l_mass, l_element

    @property
    def NodeI(self):
        return self._BridgeNodeI
    @NodeI.setter
    def NodeI(self, newVal):
        if type(newVal) is type(self._BridgeNodeI):
            self._BridgeNodeI = newVal
        else:
            raise Exception("Wrong Paras")
     
    @property
    def NodeJ(self):
        return self._BridgeNodeJ
    @NodeJ.setter
    def NodeJ(self, newVal):
        if type(newVal) is type(self._BridgeNodeJ):
            self._BridgeNodeJ = newVal
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
        return self._Elements
    @ELeList.setter
    def ELeList(self, newVal):
        if type(newVal) is type(self._Elements):
            self._Elements = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def NodeList(self):
        return self._BridgeNodes
    @NodeList.setter
    def NodeList(self, newVal):
        if type(newVal) is type(self._BridgeNodes):
            self._BridgeNodes = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def MassList(self):
        return self._Masses
    @MassList.setter
    def MassList(self, newVal):
        if type(newVal) is type(self._Masses):
            self._Masses = newVal
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
        return [self._BridgeNodeI, self._BridgeNodeJ, self._Secti, self._Sectj, self._ElementType, self._con, self._localZAxis, self._eleLengthList, self._p]
# 
class LineHRoundSeg(Segment):
    @Comp.CompMgr()
    def __init__(self, point_i: tuple[float, ...], point_j: tuple[float, ...], SectParas_i:Paras.HRoundSectParas, SectParas_j:Paras.HRoundSectParas, elementType:Segment.SupportedElementType, ConCore:Paras.ConcreteParas, ConCover:Paras.ConcreteParas, Rebar:Paras.SteelParas, eleLengthList: list[float], RebarRList:list[float]=None, RebarDistrParasList:list[Paras.SectRebarDistrParas]=None, localZ: tuple[float]=None, elementDictParam:dict={}, name=""):
        super().__init__(point_i, point_j, SectParas_i, SectParas_j, elementType, 1, localZ, eleLengthList, elementDictParam, name)
        self._CoreCon = ConCore
        self._CoverCon = ConCover
        self._Rebar = Rebar

        if RebarDistrParasList and len(RebarDistrParasList) == len(eleLengthList):
            print("INFO:As 'RebarDistrParasList' was used, 'RebarRList' is ignored")
            self._R = RebarDistrParasList
        elif RebarRList and len(RebarRList) == len(eleLengthList):
            self._R = RebarRList
        else:
            raise Exception("Wrong Params, RebarDistParasList and RebarRList can not be None at same time, and the length should be same with eleLengthList.")

        self._BridgeNodes, self._SegSectList, self._Masses, self._Elements = self._Build()

    def _Build(self):
        if UtilTools.Util.TOL_EQ(sum(self._eleLengthList), self._totalLen) and self._eleLengthList is not None:
            points = UtilTools.PointsTools.LinePointBuilder(self._BridgeNodeI.point, self._BridgeNodeJ.point, self._eleLengthList)
            SectI:Paras.HRoundSectParas = self._Secti
            SectJ:Paras.HRoundSectParas = self._Sectj

            T = UtilTools.SegmentTools.PowerParasBuilder(SectI.T, SectJ.T, self._totalLen, 1, self._eleLengthList)
            

            Rout = UtilTools.SegmentTools.PowerParasBuilder(SectI.Rout, SectJ.Rout, self._totalLen, 1, self._eleLengthList)

            # N_axis =  UtilTools.PointsTools.vectSub(self._Nj.point, self._Ni.point)
            l_paras:list[Paras.HRoundSectParas] = []
            l_node:list[BridgeNode] = []
            l_sect:list[HRoundRCSect] = []
            l_element:list[OpsObject.OpsLineElement] = []
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
                
                ele = self.BuildElement(self._ElementType, l_node[i].OpsNode, l_node[i+1].OpsNode, l_sect[i].OpsSect, self._localZAxis, **self._ElementExtraDictParams)
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
        return self._Elements
    @ELeList.setter
    def ELeList(self, newVal):
        if type(newVal) is type(self._Elements):
            self._Elements = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def NodeList(self):
        return self._BridgeNodes
    @NodeList.setter
    def NodeList(self, newVal):
        if type(newVal) is type(self._BridgeNodes):
            self._BridgeNodes = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def MassList(self):
        return self._Masses
    @MassList.setter
    def MassList(self, newVal):
        if type(newVal) is type(self._Masses):
            self._Masses = newVal
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
        return self._Elements
    @ELeList.setter
    def ELeList(self, newVal):
        if type(newVal) is type(self._Elements):
            self._Elements = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def NodeList(self):
        return self._BridgeNodes
    @NodeList.setter
    def NodeList(self, newVal):
        if type(newVal) is type(self._BridgeNodes):
            self._BridgeNodes = newVal
        else:
            raise Exception("Wrong Paras")
    @property
    def MassList(self):
        return self._Masses
    @MassList.setter
    def MassList(self, newVal):
        if type(newVal) is type(self._Masses):
            self._Masses = newVal
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
        return [self._BridgeNodeI, self._BridgeNodeJ, self._Secti, self._Sectj, self._CoreCon, self._CoverCon, self._Rebar, self._localZAxis, self._eleLengthList, self._R]

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

        self._BridgeNodes, self._SegSectList, self._Masses, self._Elements = self._Build()
        
    def _Build(self):
        points = UtilTools.PointsTools.LinePointBuilder(self._BridgeNodeI.point, self._BridgeNodeJ.point, self._eleLengthList)
        SectI:Paras.SRoundSectParas = self._Secti
        SectJ:Paras.SRoundSectParas = self._Sectj


        R = UtilTools.SegmentTools.PowerParasBuilder(SectI.R, SectJ.R, self._totalLen, 1, self._eleLengthList)

        # N_axis =  UtilTools.PointsTools.vectSub(self._Nj.point, self._Ni.point)
        l_paras:list[Paras.SRoundSectParas] = []
        l_node:list[BridgeNode] = []
        l_sect:list[SRoundRCSect] = []
        l_element:list[OpsObject.OpsLineElement] = []
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
            ele = self.BuildElement(self._ElementType, l_node[i].OpsNode, l_node[i+1].OpsNode, l_sect[i].OpsSect, self._localZAxis, **self._ElementExtraDictParams)
            # ele = OpsObject.OpsNBCElement(l_node[i].OpsNode, l_node[i+1].OpsNode, l_sect[i].OpsSect, self._localZAxis, 5)

            l_element.append(ele) 
        
        return l_node, l_sect, l_mass, l_element
    
    @property
    def ELeList(self):
        return self._Elements
    @ELeList.setter
    def ELeList(self, newVal):
        if type(newVal) is type(self._Elements):
            self._Elements = newVal
        else:
            raise Exception("Wrong Paras")

    @property
    def NodeList(self):
        return self._BridgeNodes
    @NodeList.setter
    def NodeList(self, newVal):
        if type(newVal) is type(self._BridgeNodes):
            self._BridgeNodes = newVal
        else:
            raise Exception("Wrong Paras")

    @property
    def MassList(self):
        return self._Masses
    @MassList.setter
    def MassList(self, newVal):
        if type(newVal) is type(self._Masses):
            self._Masses = newVal
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
        return [self._BridgeNodeI, self._BridgeNodeJ, self._Secti, self._Sectj, self._CoreCon, self._CoverCon, self._Rebar, self._localZAxis, self._eleLengthList, self._R]

class Cuboid(Body):
    class SupportedElementType(Enum):
        StanderBrickElement = 'StanderBrickElement'

    """
                    z|H
                    |
                    |
                    |      7_______6
                    |     /|      /|
                    |    / |     / |
                    |___/__|3___/__|2____ y|W
                   /   8___/___5   /   
                  /    |  /    |  /   
                 /     | /     | /   
                /      |/______|/
               x|L     4       1
    """
    
    @abstractmethod
    def __init__(self, p1:tuple[float, ...], p2:tuple[float, ...], p3:tuple[float, ...], p4:tuple[float, ...], p5:tuple[float, ...], p6:tuple[float, ...], p7:tuple[float, ...], p8:tuple[float, ...], eleLensL=[], eleLensW=[], eleLensH=[], name=""):
        super().__init__(name)
        self._type += '->Cube'
        self._Node1 = BridgeNode(*p1)
        self._Node2 = BridgeNode(*p2)
        self._Node3 = BridgeNode(*p3)
        self._Node4 = BridgeNode(*p4)
        self._Node5 = BridgeNode(*p5)
        self._Node6 = BridgeNode(*p6)
        self._Node7 = BridgeNode(*p7)
        self._Node8 = BridgeNode(*p8)

        self._L = UtilTools.PointsTools.PointsDist(p1, p2) 
        self._W = UtilTools.PointsTools.PointsDist(p1, p4) 
        self._H = UtilTools.PointsTools.PointsDist(p1, p5) 
        self._volume = self._L * self._W * self._H

        self._eleLensL = eleLensL
        self._eleLensW = eleLensW
        self._eleLensH = eleLensH


        self._volumeMass = None

        self._BridgeNodes:np.ndarray = None
        self._Elements:np.ndarray = None
        self._PointsIndex:np.ndarray = None
        self._Masses:np.ndarray = None
        
    
    @abstractmethod
    def _Build(self): ...
    
    def FindElement(self, p3):
        flag, val = self.FindPointIndex(p3)
        if flag:
            i, j, k = val
            return True, self._Elements[i, j, k]
        else:
            return False, None
    
    def FindElementIndex(self, ele:OpsObject.OpsBrickElement):
        flag, val = self.FindPointIndex(ele._node3)
        if flag:
            return True, val
        else:
            return False, None
    
    def FindElementHasPoint(self, p):
        e1 = e2 = e3 = e4 = e5 = e6 = e7 = e8 = None
        flag, val = self.FindPointIndex(p)
        if flag:
            i, j, k = val

            if k-1<0:
                e1 = None
            else:
                e1 = self._Elements[i, j, k-1]

            e5 = self._Elements[i, j, k]

            if k-1<0 and j-1<0:
                e2 = None
            else:
                e2 = self._Elements[i, j-1, k-1]

            if j-1<0:
                e6 = None
            else:
                e6 = self._Elements[i, j-1, k]

            if i-1<0 or j-1<0 or k-1<0:
                e3 = None
            else:
                e3 = self._Elements[i-1, j-1, k-1]
            
            if i-1<0 or j-1<0:
                e7 = None
            else:
                e7 = self._Elements[i-1, j-1, k]

            if i-1<0 or k-1<0:
                e4 = None
            else:
                e4 = self._Elements[i-1, j, k-1]
            
            if i-1<0:
                e8 = None
            else:
                e8 = self._Elements[i-1, j, k]

            return True, [e1, e2, e3, e4, e5, e6, e7, e8]
        else:
            return False, None

    def FindRangeElement(self, p3, l, w, h):
        flag, val = self.FindRangeBridgeNodes(p3, l, w, h)
        if flag:
            # x, y, z = val

            x, y, z = val.shape
            # ix3, iy3, iz3 = self.FindBridgeNodeIndex(val[0,0,0])
            # ix7, iy7, iz7 = self.FindBridgeNodeIndex(val[x-1,y-1,z-1])
            flag, Nval = self.FindBridgeNodeIndex(val[0, 0, 0])
            if flag:
                ix3, iy3, iz3 = Nval
            else:
                return False, None

            flag, Nval = self.FindBridgeNodeIndex(val[x-1, y-1, z-1])

            if flag:
                ix5, iy5, iz5 = Nval
            else:
                return False, None
            if ix3 == ix5 or iy3==iy5 or iz3== iz5:
                return False, None

            return True, self._Elements[ix3:ix5, iy3:iy5, iz3:iz5]
        else:
            return False, None

            # return self._Elements[n1[0]:n7[0], n1[1]:n7[1], n1[2]:n7[2]]
            # ix = iy = iz = -1

            # for idx in range(len(self._eleLensL) - x):
            #     if sum(self._eleLensL[x:x+idx]) == l:
            #         ix = x+idx+1
            #         break
            #     elif sum(self._eleLensL[x:x+idx]) > l:
            #         ix = x+idx
            #         break
            
            # for idx in range(len(self._eleLensW) - y):
            #     if sum(self._eleLensW[y:y+idx]) == w:
            #         iy = y+idx
            #         break
            #     elif sum(self._eleLensL[y:y+idx]) > w:
            #         iy = y+idx-1
            #         break

            # for idx in range(len(self._eleLensH) - z):
            #     if sum(self._eleLensH[z:z+idx]) == h:
            #         iz = z+idx
            #         break
            #     elif sum(self._eleLensH[z:z+idx]) > h:
            #         iz = z+idx-1
            #         break

            return True, self._Elements[x:ix+1, y:iy+1, z:iz+1]

    def FindPointIndex(self, p):

        L, W, H = self._PointsIndex.shape[:3]
        flag_Found = False
        for i in range(L):
            # v1 = self._PointsIndex[i, 0, H-1]
            # v2 = self._PointsIndex[i, W-1, 0]
            v = UtilTools.PointsTools.VectorSub(p, self._PointsIndex[i, 0, 0])
            if UtilTools.Util.isOnlyHas(v, [0], flatten=True):
                return True, [i, 0, 0]
            
            if v[0] == 0:
                flag_Found = True
                break

            # if UtilTools.PointsTools.IsVectorInPlane(v, v1, v2):
            #     Found_flag = True
            #     break

        if not flag_Found:
            return False, None
        else:
            flag_Found = False

        for j in range(W):
            # if UtilTools.PointsTools.IsPointInLine(p, self._PointsIndex[i, j, 0], self._PointsIndex[i, j, H-1]):
            if p[1] == self._PointsIndex[i, j, 0][1]:
                flag_Found = True
                break

        if not flag_Found:
            return False, None
        else:
            flag_Found = False

        for k in range(H):
            if np.all(self._PointsIndex[i, j, k] == p):
                return True, [i, j, k]
        
        return False, None
    def FuzzyFindPoint(self, p, tol=None):
        flag, val = self.FuzzyFindNode(p, tol)
        if flag:
            return True, val.point
        else:
            return False, None

    def FindRangePointIndex(self, p3, l, w, h):
        flag, val = self.FindPointIndex(p3)

        if flag:
            x, y, z = val
            ix = iy = iz = -1
            
            if x == len(self._eleLensL):
                ix = x
            else:
                for idx in range(len(self._eleLensL) - x):
                    if sum(self._eleLensL[x:x+idx+1]) == l:
                        ix = x+1+idx
                        break
                    elif sum(self._eleLensL[x:x+idx+1]) > l:
                        ix = x+idx
                        break
            
            if y == len(self._eleLensW):
                iy = y
            else:
                for idx in range(len(self._eleLensW) - y):
                    if sum(self._eleLensW[y:y+idx+1]) == w:
                        iy = y+idx+1
                        break
                    elif sum(self._eleLensW[y:y+idx+1]) > w:
                        iy = y+idx
                        break

            if z == len(self._eleLensW):
                iz = z
            else:
                for idx in range(len(self._eleLensH) - z):
                    if sum(self._eleLensH[z:z+idx+1]) == h:
                        iz = z+idx+1
                        break
                    elif sum(self._eleLensH[z:z+idx+1]) > h:
                        iz = z+idx
                        break

            return True, [(x, ix+1), (y, iy+1), (z, iz+1)]
        else:
            return False, None


    def FindBridgeNode(self, p:tuple[float, ...]): 
        flag, val = self.FindPointIndex(p)
        if flag:
            i, j, k = val
            return True, self._BridgeNodes[i, j, k]
        else:
            return False, None

    def FindBridgeNodeIndex(self, briNode:BridgeNode):
        flag, val = self.FindPointIndex(briNode.point)
        if flag:
            return True, val
        else:
            return False, None

    def FuzzyFindNode(self, p, tol=None):
        ps:list[BridgeNode] = self._BridgeNodes.flatten().tolist()
        tempN = None
        tempDis = GlobalData.DEFVAL._MAXVAL_
        for n in ps:
            dis = UtilTools.PointsTools.PointsDist(n.point, p)

            if tol and not UtilTools.Util.TOL_LE(dis, tol):
                dis = GlobalData.DEFVAL._MAXVAL_

            tempDis, tempN = (dis, n) if dis < tempDis else (tempDis, tempN)
        
        if not UtilTools.Util.TOL_GE(tempDis, GlobalData.DEFVAL._MAXVAL_):
            return True, tempN
        else:
            return False, None

    def FindRangeBridgeNodes(self, p3:tuple[float], l:float, w:float, h:float):
        flag, val = self.FindRangePointIndex(p3, l, w, h)
        if flag:
            (x1, x2), (y1, y2), (z1, z2) = val
            return True, self._BridgeNodes[x1:x2, y1:y2, z1:z2]
        else:
            return False, None

    def BuildElement(self, elementType:SupportedElementType, *args, **kwargs):
        if elementType == self.SupportedElementType:
            ele = OpsObject.OpsStanderBrickElement(*args, **kwargs)
        else:
            message = 'Unsupported Element Type'
            StandardLogger.error(message)
            raise Exception(message)
        
        return ele

class SoilCuboid(Cuboid):
    # @property
    @Comp.CompMgr()
    def __init__(self, p1: tuple[float, ...], p2: tuple[float, ...], p3: tuple[float, ...], p4: tuple[float, ...], p5: tuple[float, ...], p6: tuple[float, ...], p7: tuple[float, ...], p8: tuple[float, ...], MaterialParas: Paras.ClayParas, eleLens_L:list[float], eleLens_W:list[float], eleLens_H:list[float], name=""):
        super().__init__(p1, p2, p3, p4, p5, p6, p7, p8, eleLens_L, eleLens_W, eleLens_H, name)
        self._MaterialParas = MaterialParas
        self._volumeMass = self._volume * self._MaterialParas.rho
        self._PointsIndex, self._BridgeNodes, self._Masses, self._Elements = self._Build() 

    
    def _Build(self):
        # x_points = UtilTools.PointsTools.LinePointBuilder(self._Node3.point, self._Node4.point, self._eleLensX)
        # y_points = UtilTools.PointsTools.LinePointBuilder(self._Node3.point, self._Node2.point, self._eleLensY)
        # z_points = UtilTools.PointsTools.LinePointBuilder(self._Node3.point, self._Node7.point, self._eleLensZ)
        numNode_L = len(self._eleLensL) + 1
        numNode_Y = len(self._eleLensW) + 1
        numNode_Z = len(self._eleLensH) + 1
        np_P = np.empty((numNode_L, numNode_Y, numNode_Z, 3), dtype=np.float32)
        np_N = np.empty((numNode_L, numNode_Y, numNode_Z), dtype=object)

        # for i, px in enumerate(x_points):
        #     for j, py in enumerate(y_points):
        #         for k, pz in enumerate(z_points):
        #             # np_P[i, j, k] = [px[0], py[1], pz[2]]
        #             np_N[i, j, k] = BridgeNode(px[0], py[1], pz[2])
        h1 = UtilTools.PointsTools.LinePointBuilder(self._Node1.point, self._Node5.point, self._eleLensH)
        h2 = UtilTools.PointsTools.LinePointBuilder(self._Node2.point, self._Node6.point, self._eleLensH)
        h3 = UtilTools.PointsTools.LinePointBuilder(self._Node3.point, self._Node7.point, self._eleLensH)
        h4 = UtilTools.PointsTools.LinePointBuilder(self._Node4.point, self._Node8.point, self._eleLensH)

        for k, (z1, z2, z3, z4) in enumerate(zip(h1, h2, h3, h4)):
            w1 = UtilTools.PointsTools.LinePointBuilder(z3, z2, self._eleLensW)
            w2 = UtilTools.PointsTools.LinePointBuilder(z4, z1, self._eleLensW)
            for j, (y1, y2) in enumerate(zip(w1, w2)):
                l = UtilTools.PointsTools.LinePointBuilder(y1, y2, self._eleLensL)
                for i, p in enumerate(l):
                    np_P[i,j,k] = [p[0], p[1], p[2]]
                    np_N[i,j,k] = BridgeNode(p[0], p[1], p[2])

        if type(self._MaterialParas) == Paras.ClayParas:
            m = OpsObject.OpsClayMaterial(*self._MaterialParas.val)
        elif type(self._MaterialParas) == Paras.SandParas:
            m = OpsObject.OpsSandMaterial(*self._MaterialParas.val)

        np_E = np.empty((numNode_L-1, numNode_Y-1, numNode_Z-1), dtype=object)
        np_mass = np.empty_like(np_N, dtype=np.float32)

        for i in range(numNode_L - 1):
            for j in range(numNode_Y - 1):
                for k in range(numNode_Z - 1):
                    i3 = (i, j, k)
                    i2 = (i, j+1, k)
                    i4 = (i+1, j, k)
                    i1 = (i+1, j+1, k)
                    n3:BridgeNode = np_N[i3]
                    n4:BridgeNode = np_N[i4]
                    n2:BridgeNode = np_N[i2]
                    n1:BridgeNode = np_N[i1]

                    i7 = (i, j, k+1)
                    i6 = (i, j+1, k+1)
                    i8 = (i+1, j, k+1)
                    i5 = (i+1, j+1, k+1)
                    n7:BridgeNode = np_N[i7]
                    n8:BridgeNode = np_N[i8]
                    n6:BridgeNode = np_N[i6]
                    n5:BridgeNode = np_N[i5]

                    d = UtilTools.PointsTools.VectorSub(n5.point, n3.point)

                    mass = m._rho * d[0]*d[1]*d[2] / 8

                    np_mass[i1] = np_mass[i2] = np_mass[i3] = np_mass[i4] = np_mass[i5] = np_mass[i6] = np_mass[i7] = np_mass[i8] = mass

                    np_E[i, j, k] = OpsObject.OpsStanderBrickElement(n1.OpsNode, n2.OpsNode, n3.OpsNode, n4.OpsNode, n5.OpsNode, n6.OpsNode, n7.OpsNode, n8.OpsNode, m.uniqNum)
        # x_index = [x[0] for x in x_points]
        # y_index = [x[1] for x in y_points]
        # z_index = [x[2] for x in z_points]
        return np_P, np_N, np_mass, np_E 

    @classmethod
    def FromP3_LWH(cls, p3, xl, yw, zh, MaterialParas: Paras.ClayParas, eleLensL=[], eleLensW=[], eleLensH=[]):
        p1 = (p3[0]+xl, p3[1]+yw, p3[2])
        p2 = (p3[0], p3[1]+yw, p3[2])
        p4 = (p3[0]+xl, p3[1], p3[2])

        p5 = (p3[0]+xl, p3[1]+yw, p3[2]+zh)
        p6 = (p3[0], p3[1]+yw, p3[2]+zh)
        p8 = (p3[0]+xl, p3[1], p3[2]+zh)
        p7 = (p3[0], p3[1], p3[2]+zh)

        return cls(p1, p2, p3, p4, p5, p6, p7, p8, MaterialParas, eleLensL, eleLensW, eleLensH)

    @classmethod
    def FromCuboidP3_P5(cls, p3, p5, MaterialParas: Paras.ClayParas, eleLensL=[], eleLensW=[], eleLensH=[]):
        xl = p5[0] - p3[0]
        yw = p5[1] - p3[1]
        zh = p5[2] - p3[2]

        return cls.FromP3_LWH(p3, xl, yw, zh, MaterialParas, eleLensL, eleLensW, eleLensH)
    
    @property
    def val(self):
        return [self._Node1, self._Node2, self._Node3, self._Node4, self._Node5, self._Node6, self._Node7, self._Node8]


                

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
            # elif isinstance(arg, Boundary):
                # self._Boundaries.append(arg)
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
