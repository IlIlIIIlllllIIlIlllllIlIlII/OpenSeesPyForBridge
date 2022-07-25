import xsect
import openseespy.opensees as ops
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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


_TOL_ = 1e-10


# * 构件类 基础类，所有的类都有该类派生
class Component:

    def __init__(self, type="comp", name=""):
        self.type = type
        self.name = self.type + "_" + name
        self.uniqNum = 0

        ManageComponents.RegistComp(self)

    def Print(self):
        for name, val in vars(self).items():
            print(name + ":", val)


# * 构件管理类 用于维护所有构件信息

class ManageComponents:
    __uniqNum = 0
    __compDic = {}
    __allComp = []

    @classmethod
    def RegistComp(cls, comp: Component):
        cls.__allComp.append(comp)
        comp.uniqNum = cls.getUnqNum()
        cls.__compDic[comp.name] = cls.__uniqNum

    @classmethod
    def getUnqNum(cls):
        cls.__uniqNum += 1
        return cls.__uniqNum

    @classmethod
    def compNameChanged(cls, oldName, newName):
        cls.__compDic[newName] = cls.__compDic.pop(oldName)

    @classmethod
    def CurrentStat(cls):
        print(cls.__uniqNum)
        print(cls.__compDic)


# * 参数类，派生出主梁截面参数类，桥墩截面参数类......
class Paras(Component):

    def __init__(self, type="Paras", name=""):
        super(Paras, self).__init__(type, name)

    def check(self):
        for val in vars(self).values():
            if type(val) is float:

                if float(val - 0) < _TOL_:
                    return False

        return True



# 箱梁截面参数类
class BoxSectParas(Paras):
    def __init__(self, name="", upper_width: float = 0.0, down_width: float = 0.0,
                 height: float = 0.0, upper_thick: float = 0.0, down_thick: float = 0.0, web_thick: float = 0.0):
        self.upper_width = upper_width
        self.down_width = down_width
        self.height = height
        self.upper_thick = upper_thick
        self.down_thick = down_thick
        self.web_thick = web_thick

        super(BoxSectParas, self).__init__("BoxSectParas", name)


# 桥墩截面参数类
class PierSectParas(Paras):
    def __init__(self, name="", w: float = 0, l: float = 0, t: float = 0):
        self.width = w
        self.length = l
        self.thick = t

        super(PierSectParas, self).__init__("PierSectParas", name)


class MaterialParas(Paras):
    def __init__(self, name: str = "", dic: dict = {}):
        self.MaterialAttribute = dic
        super(MaterialParas, self).__init__("Material", name)

    def AddMaterialParas(self, dic: dict):
        self.MaterialAttribute.update(dic)


class BridgeParas(Paras):
    def __init__(self, name: str = ""):
        self.ParasDict = {
            "L_s": 0., "L_m": 0., "L_0": 0.,
            "W_g_u": 0., "W_g_d": 0., "H_m": 0., "H_p": 0.,
            "T_u_s": 0., "T_d_s": 0., "T_w_s": 0.,
            "T_u_m": 0., "T_d_m": 0., "T_w_m": 0.,
            "W_p_u": 0., "L_p_u": 0., "T_p_u": 0.,
            "W_p_d": 0., "L_p_d": 0., "T_p_d": 0.
        }

        super(BridgeParas, self).__init__("BridgeParas", name)



class OpsObj(Component):
    def __init__(self, type="OpsObj", name=""):
        super(OpsObj, self).__init__(type, name)

class BridgeNode(OpsObj):
    def __init__(self, xyz: tuple, type="Node", name=""):
        self.xyz = xyz

        super(BridgeNode, self).__init__(type, name)

        ops.node(self.uniqNum, *xyz)

class BridgeElement(OpsObj):
    def __init__(self):
        pass

class BridgeSection(OpsObj):
    def __init__(self):
        pass


class Parts(Component):
    def __init__(self, type="Parts", name = ""):
        super(Parts, self).__init__(type, name)

class Boundaries(Parts):
    def __init__(self, type="BoundariesParts", name = ""):
        super(Parts, self).__init__(type, name)

class GirderPart(Parts):
    pass

class PierPart(Parts):
    pass

class BridgeModel(Parts):
    def __init__(self, name=""):
        self.BridgeParas = "None"
        self.Materials:list = []
        self.Boundaries:list = []
        self.Girder = "None"
        self.Pier:list = []

        ops.model("Basic", "-ndm", 3, "-ndf", 6)

        super(BridgeModel, self).__init__("BridgeParts", name)

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


    # def SetBridgeParas(self, paras:BridgeParas):
    #     self.BridgeParas = paras
    #
    # def AddMaterials(self, mat:MaterialParas):
    #     self.Materials.append(mat)
    #
    # def AddBoundaries(self, bou:Boundaries):
    #     self.Boundaries.append(bou)
    #
    # def SetGirderPart(self, gir:GirderPart):
    #     self.Girder = gir
    #
    # def AddPiers(self, pier:PierPart):
    #     self.Pier.append(pier)

    def check(self):
        if self.BridgeParas is not BoxSectParas or\
                len(self.Materials) == 0 or\
                len(self.Boundaries) == 0 or\
                self.Girder is not GirderPart or\
                len(self.Pier) == 0:
            return False
        else:
            return True

# * 截面类, 继承自构件类，派生出 主梁截面、桥墩截面
class CrossSection(Parts):

    def __init__(self, type="CrossSect", name=""):
        self.Paras = 0
        self.Points = 0
        self.Arr = 0
        self.N_axis = 0

        super(CrossSection, self).__init__(type, name)

    def check(self):
        if type(self.Points) != np.ndarray \
                or type(self.Arr) != dict or type(self.N_axis) != tuple:
            return False
        else:
            return True


# 箱梁截面类
class BoxSect(CrossSection):

    def __init__(self, type="BoxSect", name=""):

        super(BoxSect, self).__init__(type, name)

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
def BoxSectBuilder(paras: BoxSectParas, orig_point: tuple, N_axis: tuple):
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

    boxSect.Paras = paras
    boxSect.N_axis = N_axis

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
    x2 = float(N_axis[0])
    y2 = float(N_axis[1])
    z2 = float(N_axis[2])

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

    boxSect.Points = points

    return boxSect


# 桥墩截面类
class PierSect(CrossSection):
    def __init__(self):
        super(PierSect, self).__init__()

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
def PierSectBuilder(paras: PierSectParas, orig_point, N_axis: tuple):
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
    pierSect = PierSect()
    pierSect.Paras = paras
    pierSect.N_axis = N_axis

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

    pierSect.Points = points

    return pierSect


# * 桥梁节点类 表示有限元节点

class BoxGirderElement:
    def __init__(self):
        pass


if __name__ == "__main__":
    x = BridgeModel()
    x.SetBridgeAttribution("is")
