import xsect
import openseespy.opensees as ops
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from enum import Enum
from abc import ABC

# 类继承图
# Component---|--Paras---|--BoxSectParas  //主梁截面参数
#             |          |--PierSectParas //桥墩截面参数类
#             |          |--MaterialParas //材料参数类
#             |          |--BridgeParas   //全桥参数类
#             |          |--......
#             |
#             |--Parts---|--BoxGirderParts //主梁类
#             |          |--PierParts      //桥墩类
#             |          |--BridgeParts    //全桥类
#             |          |--CrossSection   //截面类
#             |          |--......
#             |
#             |--OpsObj--|--Node           //OpenSees 节点对象
#                        |--Element        //OpenSees 单元对象
#                        |--Section        //OpenSees 截面对象
#                        |--......

# * 全局容差
_TOL_ = 1e-10


# * 定义区 用于解析引用
# * Component的申明 用于CompMgr的定义，将被后面的的定义覆盖
class Component:
    def __init__(self, name: str = ""):
        self._type = "component"
        self._uniqNum = CompMgr.getUniqNum()
        self._name = name

    ...


# * PierSect的申明 用于UtilTools的定义
class PierSect:

    pass

class UtilTools:
    def __init__(self):
        pass

    class BarsTools:
        @staticmethod
        def TargetBarsNum(bar, r: float, area: float):
            """
            :param bar: 钢筋类型 枚举类BarsType确定
            :param r: 目标配筋率
            :param area: 截面面积
            :return: 需要的钢筋总数
            """

            N = int(area * r / bar)
            return N

        @staticmethod
        def TryBarsNum(w, l, t, d1, d2):
            N_x1 = int(w / d1)
            N_x2 = int((w - 2 * t) / d2)
            N_y1 = int(l / d1)
            N_y2 = int((l - 2 * t) / d2)

            sum = (N_x1 + N_x2 + N_y1 + N_y2) * 2

            return (sum, N_x1, N_x2, N_y1, N_y2)

        # * 计算钢筋分布
        @staticmethod
        def ReBarDis(pierSect: PierSect, r: float):
            """
            :type pierSect: PierSect
            :param pierSect:桥墩截面对象
            :param r: 目标配筋率
            :returns r:实际配筋率
            :returns N:各边的钢筋数量
            :returns BarType:钢筋类型及截面积
            """

            # """
            # r: 配筋率：0.006~0.04
            # d1,d2: 钢筋间距从10cm 12cm 14cm增加
            # 钢筋直径: 32, 28
            # phy1, phy2: 钢筋截面(804.2, 615.8)
            # N_y1,N_y2,N_x1,N_x2: 钢筋个数
            # """

            #                 ^y
            #                 | w
            #        1________|________2
            #        |  5_____|_____6  |
            #        |  |     |     | -|--t
            #        |  |     |     |  |
            #   N_y1-|  |-N_y2|     |  |--l
            # -------|--|-----|-----|--|---------->x
            #        |  |     |     |  |
            #        |  |     |N_x2 |  |
            #        | 8|_____|_|___|7 |
            #       4|________|________|3
            #                 |N_x1

            if r - 0.006 < _TOL_ or r - 0.04 > _TOL_:
                raise Exception("Wrong Re-Bar Ratio")

            w = pierSect.Paras.width
            l = pierSect.Paras.length
            t = pierSect.Paras.thick

            area = pierSect.Arr["area"]

            d1 = d2 = 10
            # N_tgt = TargetBarsNum(BarType["d32"], r, area)
            N_try = UtilTools.BarsTools.TryBarsNum(w, l, t, d1, d2)

            if r - N_try[0] * BarType.d32.value / area > _TOL_:
                raise Exception("Can not calc bars num")

            count = 0
            N_old = N_try

            while (r - (N_try[0] * BarType.d32.value / area) <= _TOL_):
                N_old = N_try
                if count % 2 == 0:
                    d1 += 2
                else:
                    d2 = d1
                N_try = UtilTools.BarsTools.TryBarsNum(w, l, t, d1, d2)

                count += 1

            r_old = N_old[0] * BarType.d32.value / area
            r_try = N_try[0] * BarType.d32.value / area
            r_28 = r_old * BarType.d28.value / BarType.d32.value

            index = np.argmin(np.abs([r_old - r, r_try - r, r_28 - r]))
            if index == 0:
                return r_old, N_old[1:], BarType.d32
            if index == 1:
                return r_try, N_try[1:], BarType.d32
            if index == 2:
                return r_28, N_old[1:], BarType.d28


# * 构件管理类 用于维护所有构件信息
class CompMgr:
    _uniqNum = 0
    _compDic = {}
    _allComp = []

    def __call__(cls, func):
        def Wrapper(*args, **kwargs):
            func(*args, **kwargs)
            comp = args[0]
            index = cls.findSameValueComp(comp)

            if index is None:
                cls._allComp.append(comp)
                cls._uniqNum = len(cls._allComp)
            else:
                cls._uniqNum = index

            comp._uniqNum = cls._uniqNum

            if comp._name != "":
                cls.addCompName(comp._name, comp._uniqNum)

        return Wrapper

    @classmethod
    def findSameValueComp(cls, tarComp: Component):
        if len(cls._allComp) == 0:
            return None
        for Comp in cls._allComp:
            if Comp.__class__ == tarComp.__class__:
                target = tarComp.__dict__.copy()
                source = Comp.__dict__.copy()

                target.pop("_type")
                target.pop("_uniqNum")
                target.pop("_name")

                source.pop("_type")
                source.pop("_uniqNum")
                source.pop("_name")

                if source == target:
                    return tarComp._uniqNum + 1
            else:
                pass

            return None

    @classmethod
    def getUniqNum(cls):
        return cls._uniqNum

    @classmethod
    def compNameChanged(cls, oldName, newName):
        cls._compDic[newName] = cls._compDic.pop(oldName)

    @classmethod
    def addCompName(cls, name, uniqNum):
        cls._compDic[name] = uniqNum

    @classmethod
    def currentState(cls):
        print("当前共有组件：{}个\n组件字典为：{}".format(len(cls._allComp), cls._compDic))


# * 构件类 基础类，所有的类都有该类派生
class Component:

    def __init__(self, name: str = ""):
        self._type = "component"
        self._uniqNum = CompMgr.getUniqNum()
        self._name = name

    def __str__(self):
        dic = ""
        for name, val in vars(self).items():
            dic += "{}:{}\n".format(name, val)
        return dic


# * 参数类，派生出主梁截面参数类，桥墩截面参数类......
class Paras(Component):

    def __init__(self, name=""):
        super(Paras, self).__init__(name)
        self._type = "paras"

    def check(self):
        for val in vars(self).values():
            if type(val) is float:
                if float(val - 0) < _TOL_:
                    return False

        return True


# 箱梁截面参数类
class BoxSectParas(Paras):
    @CompMgr()
    def __init__(self, name="", upper_width: float = 0.0, down_width: float = 0.0,
                 height: float = 0.0, upper_thick: float = 0.0, down_thick: float = 0.0, web_thick: float = 0.0):
        super(BoxSectParas, self).__init__(name)
        self._type = "BoxSectParas"
        self.upper_width = upper_width
        self.down_width = down_width
        self.height = height
        self.upper_thick = upper_thick
        self.down_thick = down_thick
        self.web_thick = web_thick


# 桥墩截面参数类
class PierSectParas(Paras):
    @CompMgr()
    def __init__(self, name="", w: float = 0, l: float = 0, t: float = 0):
        super(PierSectParas, self).__init__(name)
        self._type = "PierSectParas"

        self.width = w
        self.length = l
        self.thick = t


class MaterialParas(Paras):
    def __init__(self, name: str = "", dic: dict = {}):
        super(MaterialParas, self).__init__(name)
        self._type = "MaterialParas"
        self.MaterialAttribute = dic

    def AddMaterialParas(self, dic: dict):
        self.MaterialAttribute.update(dic)


class BridgeParas(Paras):
    @CompMgr()
    def __init__(self, name: str = ""):
        super(BridgeParas, self).__init__(name)
        self.ParasDict = {
            "L_s": 0., "L_m": 0., "L_0": 0.,
            "W_g_u": 0., "W_g_d": 0., "H_m": 0., "H_p": 0.,
            "T_u_s": 0., "T_d_s": 0., "T_w_s": 0.,
            "T_u_m": 0., "T_d_m": 0., "T_w_m": 0.,
            "W_p_u": 0., "L_p_u": 0., "T_p_u": 0.,
            "W_p_d": 0., "L_p_d": 0., "T_p_d": 0.
        }
        self._type = "BridgeParas"


class Parts(Component):
    def __init__(self, name=""):
        super(Parts, self).__init__(name)
        self._type = "Parts"


class Boundaries(Parts):
    @CompMgr()
    def __init__(self, type="BoundariesParts", name=""):
        super(Parts, self).__init__(name)
        self._type = "Boundaries"


class GirderPart(Parts):
    pass


class PierPart(Parts):
    pass


class BridgeModel(Parts):
    @CompMgr()
    def __init__(self, name=""):

        super(BridgeModel, self).__init__(name)
        self._type = "BridgeParts"
        self.BridgeParas = "None"
        self.Materials: list = []
        self.Boundaries: list = []
        self.Girder = "None"
        self.Pier: list = []

        ops.model("Basic", "-ndm", 3, "-ndf", 6)

    def SetBridgeAttribution(self, *args):
        for arg in args:
            if isinstance(arg, BridgeParas):
                self.BridgeParas = arg
            elif isinstance(arg, MaterialParas):
                self.Materials.append(arg)
            elif isinstance(arg, Boundaries):
                self.Boundaries.append(arg)
            elif isinstance(arg, GirderPart):
                self.Girder = arg
            elif isinstance(arg, PierPart):
                self.Pier.append(arg)
            else:
                print("Wrong Type of argument: \"" + arg + "\"")
                pass

    def check(self):
        if self.BridgeParas is not BoxSectParas or \
                len(self.Materials) == 0 or \
                len(self.Boundaries) == 0 or \
                self.Girder is not GirderPart or \
                len(self.Pier) == 0:
            return False
        else:
            return True


# * 截面类, 继承自构件类，派生出 主梁截面、桥墩截面
class CrossSection(Parts):

    def __init__(self, name=""):
        super(CrossSection, self).__init__(name)
        self._type = "CrossSect"

        self.Paras: Paras = None
        self.Points = 0
        self.Arr = 0
        self.N_axis = 0

    def check(self):
        if type(self.Points) != np.ndarray \
                or type(self.Arr) != dict or type(self.N_axis) != tuple:
            return False
        else:
            return True


# 箱梁截面类
class BoxSect(CrossSection):
    @CompMgr()
    def __init__(self, name=""):
        super(BoxSect, self).__init__(name)
        self._type = "BoxSect"

    def plotBoxSect(self, ax3d):

        if self.check():
            # fig = plt.figure()
            # ax3d = mplot3d.Axes3D(fig)
            x, y, z = np.hsplit(self.Points, [1, 2])

            x = x.flatten()
            y = y.flatten()
            z = z.flatten()

            ax3d.plot(np.hstack((x[:8], x[0])), np.hstack((y[:8], y[0])), np.hstack((z[:8], z[0])))
            ax3d.plot(np.hstack((x[8:], x[8])), np.hstack((y[8:], y[8])), np.hstack((z[8:], z[8])))

            orig_point = (self.Points[0] + self.Points[1]) / 2

            ax3d.plot([orig_point[0], orig_point[0] + self.N_axis[0]], [orig_point[1], orig_point[1] + self.N_axis[1]],
                      [orig_point[2], orig_point[2] + self.N_axis[2]])
        else:
            raise Exception("Exist Undefined Member")

    # 根据箱梁参数、坐标原点、法向轴，建立箱梁截面
    def BoxSectBuilder(self, paras: BoxSectParas, orig_point: tuple, N_axis: tuple):
        # * 检查输入参数是否满足要求
        if paras.check() != True:
            raise Exception("Parameters Exists Zero")

        if len(orig_point) != 3 or len(N_axis) != 3:
            raise Exception("Error length of orig_point or N_axis, should be 3")
        # *                            ^z
        # *                            |
        # *         1__________________|___________________2
        # *         |                  |                  |
        # *        8|________7   9_____|_____10  4________|3
        # *                  |   |     |     |   |
        # *                  |   |     |     |   |
        # *                  |   |     |     |   |
        # *                  | 12|_____|_____|11 |
        # *                 6|_________|_________|5
        # * ---------------------------|-------------------------->y
        # *                            |
        boxSect = BoxSect()

        self.Paras = paras
        self.N_axis = N_axis

        points = np.array([[0, -paras.upper_width / 2, 0],
                           [0, paras.upper_width / 2, 0],
                           [0, paras.upper_width / 2, -paras.upper_thick],
                           [0, paras.down_width / 2, -paras.upper_thick],
                           [0, paras.down_width / 2, -paras.height],
                           [0, -paras.down_width / 2, -paras.height],
                           [0, -paras.down_width / 2, -paras.upper_thick],
                           [0, -paras.upper_width / 2, -paras.upper_thick],
                           [0, -paras.down_width / 2 + paras.web_thick, -paras.upper_thick],
                           [0, paras.down_width / 2 - paras.web_thick, -paras.upper_thick],
                           [0, paras.down_width / 2 - paras.web_thick, -paras.height + paras.down_thick],
                           [0, -paras.down_width / 2 + paras.web_thick, -paras.height + paras.down_thick]])

        yz = np.hsplit(points, [1])[1]
        yz_out = yz[:8, ]
        yz_in = yz[8:, ]

        arr = xsect.multi_section_summary([yz_out], subtract=[yz_in])
        boxSect.Arr = arr

        # * X轴

        x1, y1, z1 = (1., 0., 0.)
        # * 新X轴

        x2, y2, z2 = (float(x) for x in N_axis)

        # * 绕Z转动，计算x-y平面的角度变化
        try:
            cos_theta = (x1 * x2 + y1 * y2) / math.sqrt(x2 * x2 + y2 * y2)
            theta = np.arccos(cos_theta)
            if y2 < 0:
                theta = 2 * np.pi - theta

            Trans_Zaxis = np.array([[np.cos(theta), np.sin(theta), 0],
                                    [-np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])

            points = np.matmul(points, Trans_Zaxis)
        except:
            pass

        # * 绕Y转动，计算x-z平面的角度变化
        try:
            cos_theta = (x1 * x2 + z1 * z2) / math.sqrt(x2 * x2 + z2 * z2)
            theta = -np.arccos(cos_theta)
            if z2 > 0:
                theta = 2 * np.pi - theta

            Trans_Yaxis = np.array([[np.cos(theta), 0, -np.sin(theta)],
                                    [0, 1, 0],
                                    [np.sin(theta), 0, np.cos(theta)]])

            points = np.matmul(points, Trans_Yaxis)
        except:
            pass

        # * 平移到orig_point

        points = points + orig_point

        self.Points = points


# 桥墩截面类
class PierSect(CrossSection):

    def __init__(self, name=""):
        super(PierSect, self).__init__(name)
        self._type = "PierSect"

        self.Paras: PierSectParas = None
        self.BarNum:tuple = None

    def plotPierSect(self, ax3d):
        if self.check():
            x, y, z = np.hsplit(self.Points, [1, 2])
            x = x.flatten()
            y = y.flatten()
            z = z.flatten()

            ax3d.plot(np.hstack((x[:4], x[0])), np.hstack((y[:4], y[0])), np.hstack((z[:4], z[0])))
            ax3d.plot(np.hstack((x[4:], x[4])), np.hstack((y[4:], y[4])), np.hstack((z[4:], z[4])))

            orig_point = (self.Points[0] + self.Points[2]) / 2

            ax3d.plot([orig_point[0], orig_point[0] + self.N_axis[0]], [orig_point[1], orig_point[1] + self.N_axis[1]],
                      [orig_point[2], orig_point[2] + self.N_axis[2]])
        else:
            raise Exception("Exist Undefined Member")

    # 桥墩截面生成
    def PierSectBuilder(self, paras: PierSectParas, orig_point, N_axis: tuple):
        """
        :param paras: 桥墩截面参数
        :param orig_point: 截面的原点
        :param N_axis: 截面的法向轴
        :return:
        """
        # * 检查输入参数是否满足要求
        if paras.check() != True:
            raise Exception("Parameters Exists Zero")

        if len(orig_point) != 3 or len(N_axis) != 3:
            raise Exception("Error length of orig_point or N_axis, should be 3")

            #                 ^y
            #                 | w
            #        1________|________2
            #        |  5_____|_____6  |
            #        |  |     |     | -|--t
            #        |  |     |     |  |
            #        |  |     |     |  |--l
            # -------|--|-----|-----|--|---------->x
            #        |  |     |     |  |
            #        |  |     |     |  |
            #        | 8|_____|_____|7 |
            #       4|________|________|3
            #                 |
        self.Paras = paras
        self.N_axis = N_axis

        points = np.array([[-paras.width / 2, paras.length / 2, 0],
                           [paras.width / 2, paras.length / 2, 0],
                           [paras.width / 2, -paras.length / 2, 0],
                           [-paras.width / 2, -paras.length / 2, 0],
                           [-paras.width / 2 + paras.thick, paras.length / 2 - paras.thick, 0],
                           [paras.width / 2 - paras.thick, paras.length / 2 - paras.thick, 0],
                           [paras.width / 2 - paras.thick, -paras.length / 2 + paras.thick, 0],
                           [-paras.width / 2 + paras.thick, -paras.length / 2 + paras.thick, 0]])

        xy = np.hsplit(points, [2])[0]
        xy_out = xy[:4]
        xy_in = xy[4:]

        arr = xsect.multi_section_summary([xy_out], subtract=[xy_in])
        pierSect.Arr = arr

        x1, y1, z1 = (0, 0, 1)
        x2, y2, z2 = (float(x) for x in N_axis)

        # 绕X轴偏转, 计算y-z平面的夹角
        try:
            cos_theta = (y1 * y2 + z1 * z2) / (math.sqrt(y2 * y2 + z2 * z2))
            theta = np.arccos(cos_theta)

            if y2 > 0:
                theta = 2 * np.pi - theta

            Trans_Xaxis = np.array([[1, 0, 0],
                                    [0, np.cos(theta), np.sin(theta)],
                                    [0, -np.sin(theta), np.cos(theta)]])

            points = np.matmul(points, Trans_Xaxis)
        except:
            pass

        # 绕y轴偏转，计算x-z平面夹角
        try:
            cos_theta = (x1 * x2 + z1 * z2) / math.sqrt(x2 * x2 + z2 * z2)
            theta = np.arccos(cos_theta)

            if x2 < 0:
                theta = 2 * np.pi - theta

            Trans_Yaxis = np.array([[np.cos(theta), 0, -np.sin(theta)],
                                    [0, 1, 0],
                                    [np.sin(theta), 0, np.cos(theta)]])

            points = np.matmul(points, Trans_Yaxis)

            points = points + orig_point
        except:
            pass

        self.Points = points


class BoxGirderElement:
    def __init__(self):
        pass


class OpsObj(Component):
    def __init__(self, name=""):
        super(OpsObj, self).__init__(name)
        self._type = "OpsObj"


# * 桥梁节点类 表示有限元节点
class BridgeNode(OpsObj):
    @CompMgr()
    def __init__(self, xyz: tuple, name=""):
        super(BridgeNode, self).__init__(name)
        self._type = "BridgeNode"
        self.xyz = xyz
        ops.node(self._uniqNum, *xyz)


class BridgeSection(OpsObj):
    def __init__(self, name=""):
        super(BridgeSection, self).__init__(name)
        self._type = "BridgeSection"


class PierFiberSection(BridgeSection):
    """
    桥墩纤维截面对象, opensees.py命令为
    ops.Section()
    """

    @CompMgr()
    def __init__(self, pierSect: PierSect, name=""):
        super(PierFiberSection, self).__init__(name)
        self._type = "PierFiberSection"
        self.SectParas = pierSect
        ops.section("Fiber", self._uniqNum, )


class BridgeElement(OpsObj):
    """该类不会生成任何OpenSees单元，仅用与派生相应的单元"""

    def __init__(self, name=""):
        super(BridgeElement, self).__init__(name)
        self._type = "BridgeElement"


class BridgeEBCElement(BridgeElement):
    """
    ElasticBeamColumn单元，使用的opensees.py命令为：
    element('elasticBeamColumn', eleTag, *eleNodes, secTag, transfTag, <'-mass', mass>, <'-cMass'>)
    """

    @CompMgr()
    def __init__(self, node1: BridgeNode, node2: BridgeNode, sec: BridgeSection, trans: tuple, name=""):
        self.Node1 = node1
        self.Node2 = node2
        self.Sect = sec
        self.Trans = trans

        super(BridgeEBCElement, self).__init__("ElasticBeamColumnElement", name)

        ops.element('elasticBeamColumn', self._uniqNum, *(self.Node1.xyz), *(self.Node2.xyz), self.Sect.uniqNum, )


class BarType(Enum):
    d32 = 804.2
    d28 = 615.8


if __name__ == "__main__":
    pierParas = PierSectParas(w=10000, l=10000, t=2000)
    pierSect = PierSect()
    pierSect.PierSectBuilder(pierParas, (0, 0, 0), (1, 0, 0))
    fig = plt.figure()
    ax3d = fig.add_subplot(projection="3d")
    pierSect.plotPierSect(ax3d)
    plt.show()
    x = UtilTools.BarsTools.ReBarDis(pierSect, 0.01)

    print(x)
