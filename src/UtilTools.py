import numpy as np
from scipy.spatial.transform import Rotation as R

from src import BridgeParas

from .BridgeParas import HRectSectParas, HRoundSectParas, SRoundSectParas
from .Comp import Paras
from .GlobalData import DEFVAL, ReBarArea, ReBarType
from .log import StandardLogger

#%%
    


class SectTools:
    @staticmethod
    def MeanSectParas(sectParas1:Paras, sectParas2:Paras):
        p1 = sectParas1.val
        p2 = sectParas2.val
        p3 = []

        for i,j in zip(p1, p2):
            p3.append((i+j)/2)
        
        return p3

        
    @staticmethod
    def MeanSectAttr(secAttr1:dict, secAttr2:dict):
        
        keys = ['area', 'x', 'y', 'width', 'height',
            'inertia_x', 'inertia_y', 'inertia_j', 'inertia_xy', 
            'inertia_z', 'gyradius_x', 'gyradius_y', 'gyradius_z', 
            'elast_sect_mod_x', 'elast_sect_mod_y', 'elast_sect_mod_z']
        newSect = secAttr1.copy()
        for key in keys:
            newSect[key] = (secAttr1[key] + secAttr2[key]) / 2
        return newSect

class PointsTools:
    @staticmethod
    def PointsDist(p1:tuple[int, ...], p2:tuple[int, ...]) -> float:
        """
        计算2个点之间的距离
        """
        if type(p1) is not np.ndarray:
            p1 = np.array(p1)
        
        if type(p2) is not np.ndarray:
            p2 = np.array(p2)
        return float(np.sqrt(np.sum((p1 - p2) ** 2)))
    
    @staticmethod
    def Point2LineDist(p1, lp1, lp2):
        pl = PointsTools.VectorSub(p1, lp1)
        if Util.isOnlyHas(pl, [0], flatten=True):
            return 0
        l = PointsTools.VectorSub(lp2, lp1)
        ang = PointsTools.VectorAngle(pl, l)
        n_pl = PointsTools.NormOfVector(pl)

        dist = np.sin(ang) * n_pl

        return dist

    @staticmethod
    def LinePointBuilder(p1:tuple[int, ...], p2:tuple[int, ...], intervalList:list[int]):
        """
        根据intervallist长度内插值列表,计算一条直线上对应的点坐标
        """
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dz = p1[2] - p2[2]
        pointsList = [p1]
        Dist = PointsTools.PointsDist(p1, p2)
        if Util.TOL_EQ(Dist, sum(intervalList)):
            rate = [sum(intervalList[:i+1])/sum(intervalList) for i,_ in enumerate(intervalList)]
            temp = 0

            for r in rate:
                temp = (p1[0] - dx * r, p1[1] - dy * r, p1[2] - dz * r)
                pointsList.append(temp)
        else:
            raise Exception("wrong paras, Point {}, Point{} distance {} is not equal sum {} of intervallist".format(p1, p2, Dist, sum(intervalList)))
        return pointsList
    @staticmethod
    def IsLinesParallel(lpi, lpj, lp1, lp2):
        dist_i = PointsTools.Point2LineDist(lpi, lp1, lp2)
        dist_j = PointsTools.Point2LineDist(lpj, lp1, lp2)

        if Util.TOL_EQ(dist_i, dist_j):
            return True
        else:
            return False

    @staticmethod
    def IsPointInLine(p:tuple[float], lp1:tuple[float], lp2:tuple[float], ExtendLine=False):
        v1 = PointsTools.VectorSub(p, lp1)
        v2 = PointsTools.VectorSub(p, lp2)
        n1 = PointsTools.NormOfVector(v1)
        n2 = PointsTools.NormOfVector(v2)
        if Util.TOL_EQ(n1, 0) or Util.TOL_EQ(n2, 0):
            return True

        if PointsTools.IsLinesParallel(p, lp1, p, lp2):
            return True
        else:
            return False

        if Util.TOL_EQ(v1[0]/v2[0], v1[1]/v2[1]) and Util.TOL_EQ(v1[0]/v2[0], v1[2]/v2[2]):
            if not ExtendLine:
                if v1[0]/v2[0] < 0:
                    return True
                else:
                    return False
            else:
                return True



    @staticmethod
    def ndarray2vect(x:np.ndarray):
        return (x[0], x[1], x[2])

    @staticmethod
    def vectAdd(x:tuple[float], y):
        """
        向量加法
        """
        if type(x) is not np.ndarray:
            x = np.array(x)
        return x + y

    @staticmethod
    def VectorSub(x:tuple[float], y):
        """
        向量减法
        """
        if type(x) is not np.ndarray:
            x = np.array(x)
        
        return np.round(x-y, 3)

    @staticmethod
    def vectEnlarge(x:tuple, y):
        """
        拉长向量
        """
        if type(y) == int or type(y) == float:
            return x * y
        else:
            raise Exception("wrong paras")
    
    @staticmethod
    def vectTrueDiv(x, y) -> np.ndarray:
        """
        向量除法
        """
        if type(x) is not np.ndarray:
            x = np.array(x)

        if type(y) == int or type(y) == float:
            return x/y
    
    @staticmethod
    def vectFloorDiv(x, y) -> np.ndarray:
        """
        向量整除
        """

        if type(x) is not np.ndarray:
            x = np.array(x)
        if type(y) == int or type(y) == float:
            return x//y

    @staticmethod
    def NormOfVector(x:tuple[float,...]) -> float:
        """
        向量的模
        """
        if type(x) is not np.ndarray:
            x = np.array(x)
        return PointsTools.PointsDist(x, (0, 0, 0))

    @staticmethod
    def VectorAngle(x:tuple[float], y:tuple[float]) -> float:
        """
        向量的夹角
        """
        if PointsTools.IsVectsLegal(x) and PointsTools.IsVectsLegal(y):
            xnorm = PointsTools.NormOfVector(x)
            ynorm = PointsTools.NormOfVector(y)
            cos = (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]) / ynorm / xnorm

            return np.arccos(cos)
        else:
            raise Exception("Ilegal Vector")

    @staticmethod
    def vectNormalize(x:tuple[float]) -> np.ndarray:
        """
        向量的归一化
        """
        if type(x) is not np.ndarray:
            x = np.array(x)
        norm = PointsTools.NormOfVector(x)

        return x/norm
    
    @staticmethod
    def NormalVectOfPlane(planeVect1:tuple[float, ...], planeVect2:tuple[float, ...]) -> np.ndarray:
        """
        平面的法向量
        """
        if type(planeVect1) is not np.ndarray:
            planeVect1 = np.array(planeVect1)
        if type(planeVect2) is not np.ndarray:
            planeVect2 = np.array(planeVect2)
        
        NormalVect = np.cross(planeVect1, planeVect2)

        NormalVect = PointsTools.vectNormalize(NormalVect)

        return NormalVect

    
    @staticmethod
    def vectPlaneAngle(vect:tuple[float, ...], planeVect1:tuple[float, ...], planeVect2:tuple[float, ...]) -> float:
        """
        向量与截面的夹角
        """
        if type(planeVect1) is not np.ndarray:
            planeVect1 = np.array(planeVect1)
        if type(planeVect2) is not np.ndarray:
            planeVect2 = np.array(planeVect2)

        normalVect = PointsTools.NormalVectOfPlane(planeVect1, planeVect2)

        theta = PointsTools.VectorAngle(vect, normalVect)

        return np.pi/2-theta
    
    @staticmethod
    def IsVectsLegal(vectlist:np.ndarray) -> bool:
        """
        判断向量是否合法
        """

        if vectlist is not np.ndarray:
            vectlist = np.array(vectlist)

        if len(vectlist.shape) == 1:
            vectlist = vectlist.reshape(1, 3)

        if vectlist.shape[-1] != 3:
            return False

        for i in vectlist.tolist():
            if i == [0, 0, 0]:
                return False
            return True

    @staticmethod
    def IsVectorInPlane(vect:np.ndarray, planeVect1:np.ndarray, planeVect2:np.ndarray) -> bool:
        """
        向量是否在平面中
        """
        
        normalVect = PointsTools.NormalVectOfPlane(planeVect1, planeVect2)

        ang = PointsTools.VectorAngle(vect, normalVect)
        if Util.TOL_EQ(np.cos(ang), 0):
            return True
        else:
            return False

    @staticmethod
    def ProjectVectInPlane(vect:np.ndarray, planeVect1:np.ndarray, planeVect2:np.ndarray):
        """
        将向量投影到平面上
        """
        if type(vect) is not np.ndarray:
            vect = np.array(vect)
        if type(planeVect1) is not np.ndarray:
            planeVect1 = np.array(planeVect1)
        if type(planeVect2) is not np.ndarray:
            planeVect2 = np.array(planeVect2)
        n = PointsTools.NormalVectOfPlane(planeVect1, planeVect2)
        norm = PointsTools.NormOfVector(n)
        a = n * vect / norm ** 2
        return vect - (a * n)

    @staticmethod
    def TransXsectPointTo3D(point:np.ndarray):
        """
        将xsect模块生成的2d点转换为3d
        """

        if type(point) is not np.ndarray or point.shape[-1] != 2:
            raise Exception("Wrong Params, point should be produced by Xsect")
        
        zero = np.zeros_like(point[:,0]).reshape(-1, 1)
        p3d = np.hstack([zero, point])
        
        return p3d

    X_AXIS = np.array([1, 0, 0])
    Y_AXIS = np.array([0, 1, 0])
    Z_AXIS = np.array([0, 0, 1])

    XY_PLANE = YX_PLANE = [X_AXIS, Y_AXIS]
    XZ_PLANE = ZX_PLANE = [X_AXIS, Z_AXIS]
    YZ_PLANE = ZY_PLANE = [Y_AXIS, Z_AXIS]

    COORDSYS = np.array([X_AXIS, Y_AXIS, Z_AXIS])

    @staticmethod
    def RotatePointsByPoints(Points:list[float], source:tuple[float], target:tuple[float], coordsys=False) -> tuple[np.ndarray, np.ndarray]:
        """
        根据起始点和终点, 旋转点集, 返回会旋转后的点集和旋转后的坐标轴
        采用4元数描述点的转动
        """
        if type(source) is not np.ndarray:
            source = np.array(source)
        if type(target) is not np.ndarray:
            target = np.array(target)
        if type(Points) is not np.ndarray:
            Points = np.array(Points)
            
        theta = PointsTools.VectorAngle(source , target) / 2      
        NormalAXis = np.cross(source, target)
        NormalAXis = PointsTools.vectNormalize(NormalAXis)

        sinA = np.sin(theta)
        cosA = np.cos(theta)
        quat = (NormalAXis[0] * sinA, NormalAXis[1] * sinA, NormalAXis[2] * sinA, cosA)
        rot = R.from_quat(quat)

        # if coordsys:
        #     return (rot.apply(Points), rot.apply(source), rot.apply(PointsTools.COORDSYS))
        # else:
        #     return (rot.apply(Points), rot.apply(source))
        if coordsys:
            return (rot.apply(Points), rot.apply(PointsTools.COORDSYS))
        else:
            return rot.apply(Points)
    

    @staticmethod
    def RotatePoinsByVects(points:np.ndarray, source:np.ndarray, target:np.ndarray, coordsys:bool=False):
        return PointsTools.RotatePointsByPoints(points, source, target, coordsys)

    @staticmethod
    def ProjectVectsToNewCoordSYS(vect:np.ndarray, new_XAxis:np.ndarray, new_ZAxis:np.ndarray) -> np.ndarray:
        """
        将一个向量映射到
        map a vector in standand coordinate system to a new coordinate system
        """
        if type(vect) is not np.ndarray:
            vect = np.array(vect)
        if type(new_XAxis) is not np.ndarray:
            new_XAxis = np.array(new_XAxis)
        if type(new_ZAxis) is not np.ndarray:
            new_ZAxis = np.array(new_ZAxis)

        new_YAxis = np.cross(new_XAxis, new_ZAxis)

        new_ZAxis = np.cross(new_XAxis, new_YAxis)

        
        theta_x = PointsTools.VectorAngle(vect, new_XAxis)
        vect_x = PointsTools.NormOfVector(vect) * np.cos(theta_x)
        theta_y = PointsTools.VectorAngle(vect, new_YAxis)
        vect_y = PointsTools.NormOfVector(vect) * np.cos(theta_y)
        theta_z = PointsTools.VectorAngle(vect, new_ZAxis)
        vect_z = PointsTools.NormOfVector(vect) * np.cos(theta_z)

        return np.array([vect_x, vect_y, vect_z])

     #  # if 
        
        
        # cos_theta = (x1 * x3 + y1 * y3) / math.sqrt(x3 * x3 + y3 * y3)
        # theta = np.arccos(cos_theta)
        # if y3 < 0:
        #     theta = 2 * np.pi - theta

        # * 绕Y转动，计算x-z平面的角度变化
        # cos_phi = (x1 * x3 + z1 * z3) / math.sqrt(x2 * x3 + z3 * z3)
        # phi = -np.arccos(cos_phi)
        # if z3 < 0:
        #     phi = 2 * np.pi - phi

        # * 绕X转动，计算y-z平面的角度变化
        # cos_psi = 


        # * 绕X轴转动， 计算x-y平面的转角
        # cos_psi = ()

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
        #     points = np.matmul(Points, Trans_Zaxis)
        # except:
        #     pass
        # # * 绕Y转动，计算x-z平面的角度变化
        # try:
        #     cos_theta = (x1 * x2 + z1 * z2) / math.sqrt(x2 * x2 + z2 * z2)
        #     theta = -np.arccos(cos_theta)
        # except:
        #     pass

class Util:
    @staticmethod
    def TOL_EQ(n1, n2):
        try:
            n1 = float(n1)
            n2 = float(n2)
        except:
            raise Exception("Wrong Paras")

        if abs(n1 - n2) <= DEFVAL._TOL_:
            return True
        else:
            return False

    @staticmethod
    def TOLLT(n1, n2):
        try:
            n1 = float(n1)
            n2 = float(n2)
        except:
            raise Exception("Wrong Paras")

        if abs(n1 - n2) > DEFVAL._TOL_ and n1 < n2:
            return True
        else:
            return False

    @staticmethod
    def TOLGT(n1, n2):
        try:
            n1 = float(n1)
            n2 = float(n2)
        except:
            raise Exception("Wrong Paras")
        if abs(n1 - n2) > DEFVAL._TOL_ and n1 > n2:
            return True
        else:
            return False
        
    @staticmethod
    def TOL_LE(n1, n2):
        if not Util.TOLGT(n1, n2):
            return True
        else:
            return False

    @staticmethod
    def TOL_GE(n1, n2):
        if not Util.TOLLT(n1, n2):
            return True
        else:
            return False

    @staticmethod
    def Flatten(obj):
        res = []
        if np.iterable(obj):
            for i in obj:
                res += Util.Flatten(i)
            return res
        
        else:
            return [obj]

    @staticmethod
    def isOnlyHas(obj, item, flatten=False) -> bool:
        if obj is not np.ndarray:
            obj = np.array(obj)
        if item is not np.ndarray:
            item = np.array(item)

        if flatten is True:
            obj = obj.flatten()
            item = item.flatten()

            for i in obj:
                if i not in item:
                    return False
            
            return True
        else:
            if len(item.shape) == 1:
                item = item.reshape(1, -1)

            if len(obj.shape) == 1:
                obj = obj.reshape(1, -1)
                obj = np.vstack([obj, obj])

            if len(obj.shape) != len(item.shape) or obj.shape[-1] != obj.shape[-1]:
                raise Exception("wrong paras")

            for i in obj.tolist():
                if i not in item.tolist():
                    return False
            return True

    @staticmethod
    def iter_isOnlyHas(obj, item):
        new = Util.Flatten(obj)
        return Util.isOnlyHas(new, item)

    @staticmethod
    def iter_Len(obj):
        n = Util.Flatten(obj)
        return len(n)

    def FullPertmuation(barArea, num):
        
        if num == 1:
            for a in barArea:
                yield [a]

        if num > 1:
            num -= 1
            
            for i in barArea:
                f = Util.FullPertmuation(barArea, num)
                for j in f:
                    yield [i] + j

    
class SegmentTools:

    @staticmethod
    def BuildWithSettedLength(totalLen:float, setLen:float):
        num = int(totalLen/setLen)
        rem = totalLen - setLen * num

        if Util.TOL_EQ(rem, 0):
            eleLenList = num * [setLen]
        else:
            eleLenList = num * [setLen] + [rem]
        
        return eleLenList

    def BuildWithSettedNum(totalLen:float, n:int):
        l = totalLen / n
        eleLenList = [l] * n

        return eleLenList

    @staticmethod
    def PowerParasBuilder(paraI:float, paraJ:float, l:float, power:float, intervalList:list[float]):
        try:
            paraI = float(paraI)
            paraJ = float(paraJ)
            l = float(l)
            power = float(power)
        except:
            raise Exception("Wrong paras")
        if Util.TOL_EQ(sum(intervalList), l):
            b = paraI
            a = (paraJ-paraI) / (l**power)
            paraList = [paraI]
            rate = [sum(intervalList[:i+1])/sum(intervalList) for i,_ in enumerate(intervalList)]
            temp = 0
            for r in rate:
                temp = a * (r * l)**power + b
                paraList.append(temp)

            return paraList
        else:
            raise Exception("Wrong Paras")

class BarsTools:
    @staticmethod
    def TargetBarsNum(bar, r: float, area: float) -> int:
        """
        :param bar: 钢筋类型 枚举类BarsType确定
        :param r: 目标配筋率
        :param area: 截面面积
        :return: 需要的钢筋总数
        """
        N = int(area * r / bar)
        return N

    @staticmethod
    def calcBarsArea(Ns: list[list[float]], As: list[list[ReBarType]]) -> float:
        """
        :param Ns: 钢筋数量的列表, [[N11, N12, ...], [N21, N22, ...], ...]
        :param As: 钢筋截面面积元组, [[a1:BarType, a2:BarType], [a1:BarType, a2:BarType], ...]
        :return: 钢筋的总面积
        """
        layers = len(Ns)
        if len(As) != layers:
            raise Exception("Wrong Params")
        area = 0
        for index in range(layers):
            for n, a in zip(Ns[index], As[index]):
                area += n * a.value
        return area

    @staticmethod
    def calcBarsNumOfHRectSect(w, l, t, d1, d2):
        N_x1 = int(w / d1)
        N_x2 = int((w - 2 * t) / d2)
        N_y1 = int(l / d1)
        N_y2 = int((l - 2 * t) / d2)
        sum = (N_x1 + N_x2 + N_y1 + N_y2) * 2
        return (sum, [N_x1, N_x2, N_y1, N_y2])

    @staticmethod
    def calcBarsNumOfHRoundSect(r, t, d1, d2):
        N_x1 = int(r * np.pi * 2 / d1)
        N_x2 = int((r-t) * np.pi *2 / d2)
        sum = N_x1 + N_x2
        return (sum, [N_x1, N_x2])

    @staticmethod
    def calcBarsNum(l_line:list[float], l_D:list[float]):
        if len(l_line) != len(l_D):
            raise Exception("wrong paras")

        Ns = []
        for l, d in zip(l_line, l_D):
            N = int(l/d)
            Ns.append(N)
        
        return Ns

    @staticmethod
    def TryFunc(l_line:list[float], sectArea: float, r: float, decOrder:list[int], decFun):
        if Util.iter_Len(decOrder) != len(l_line):
            raise Exception("Wrong Paras")

        Ns_max = BarsTools.calcBarsNum(l_line, len(l_line)*[DEFVAL._REBAR_D_DEF])
        As_max = ReBarArea.getMaxItem()
        Rat_max = BarsTools.calcBarsArea([Ns_max], [[As_max] * len(Ns_max)]) / sectArea

        if Util.TOLLT(Rat_max, r):
            Ns_ = [Ns_max]
            As_ = [[As_max] * len(Ns_max)]
            Rat_ = [Rat_max]

            l_line = decFun(l_line)

            rat, Ns, As = BarsTools.TryFunc(
                l_line, sectArea, r - Rat_max, decOrder, decFun
            )
            Rat_ += rat
            Ns_ += Ns
            As_ += As

            return (Rat_, Ns_, As_)

        elif Util.TOL_EQ(Rat_max, r):
            Ns_ = [Ns_max]
            As_ = [[As_max] * len(Ns_max)]
            Rat_ = [Rat_max]

            return (Rat_, Ns_, As_)

        elif Util.TOLGT(Rat_max, r):
            As_min = ReBarArea.getMinItem()
            order = decOrder
            count = 0
            Ns_try = Ns_max.copy()

            while Util.TOLGT(BarsTools.calcBarsArea([Ns_try], [[As_min] * len(Ns_try)])/sectArea, r) \
                and not Util.isOnlyHas(Ns_try, [1], flatten=True):

                BarsTools.NsChangeByOrder(Ns_try, order, count, lambda x:x-1)

                count += 1
                count = count % len(order)

            Ns_Lower = []
            Ns_Upper = []

            if Ns_try == Ns_max:
                Ns_Upper = Ns_try
                Ns_Lower = Ns_try
            else:
                Ns_Lower = Ns_try
                Ns_Upper = Ns_try.copy()

                BarsTools.NsChangeByOrder(Ns_Upper, order, count, lambda x: x+1)
            # ---------(Ns_Lower, As_min)----(||Ns_Upper, As_min||)----r--(||Ns_Upper, As_min||)---(Ns_max, As_max)----->
            f = BarsTools.RebarsCombineGenator(order, ReBarArea.listAllItem())

            r_lower = 0
            r_res = BarsTools.calcBarsArea([Ns_Upper], [[As_min]*len(Ns_Upper)]) / sectArea

            Ns_res = Ns_Upper
            As_res = [As_min]*len(Ns_Upper)

            while True:
                try:
                    As_ = next(f)
                    # print(As_)
                    r_lower = BarsTools.calcBarsArea([Ns_Lower], [As_]) / sectArea

                    if (r_lower - r) > 0 and (r_res - r) > 0:
                        r_res = r_lower
                        Ns_res = Ns_Lower
                        As_res = As_
                    elif (r_lower - r) * (r_res - r) < 0:
                        if Util.TOLLT(abs(r_lower - r), abs(r_res - r)):
                            r_res = r_lower
                            Ns_res = Ns_Lower
                            As_res = As_
                        
                    elif (r_lower - r) < 0 and (r_res -r) < 0:
                        break

                except:
                    msg = ("An error occurred while searching for the closest target reinforcement ratio\n {}\nThe result has been set to r:{}\tNs:{}\tAs{},\n a difference of {}".format(r, r_res, Ns_res, As_res, r-r_res))
                    StandardLogger.warning(msg)
                    # raise Exception(msg)
                    break
                # print(As_res)

            return ([r_res], [Ns_res], [As_res])

    @staticmethod
    def RebarsCombineGenator(decOrder, barAreaRange:list):
        num = len(decOrder)
        f = Util.FullPertmuation(barAreaRange, num)
        res = [0] * Util.iter_Len(decOrder)
        for As_ in f:
            if len(As_) != len(decOrder):
                raise Exception("the length of AS and decOrder is not equal")

            for As, ords in zip(As_, decOrder):
                if type(ords) is list or type(ords) is tuple:
                    for ord in ords:
                        res[ord] = As

                elif type(ords) is int:
                    res[ords] = As

                else:
                    raise Exception("Wrong Params")

            yield res
        
            

    # * 计算钢筋分布
    @staticmethod
    def HRectRebarDistr(paras:HRectSectParas, attr:dict, r: float) -> tuple[float, list[int], list[ReBarArea]]:
        """
        :param Paras:桥墩截面截面参数对象
        :param attr: 桥墩截面参数对象
        :param r: 目标配筋率
        :returns r:实际配筋率
        :returns N:各边的钢筋数量
        :returns BarType:钢筋类型及截面积
        """
        #                 ^y
        #            l_l3 | w
        #        1________|________2
        #        |  5_____|_____6  |
        #        |  |     |l_l7 | -|--t
        #        |  |     |     |  |
        #   l_l0-|  |-l_l4|l_l6-|  |--l
        # -------|--|-----|-----|--|---------->x
        #        |  |     |     |  |-l_l2
        #        |  |     |l_l5 |  |
        #        | 8|_____|_|___|7 |
        #       4|________|________|3
        #                 |l_l1
        if r - 0.006 < DEFVAL._TOL_ or r - 0.04 > DEFVAL._TOL_:
            raise Exception("Wrong Re-Bar Ratio")
        w = paras.W - paras.C * 2
        l = paras.L - paras.C * 2
        t = paras.T - paras.C * 2

        l_line = [w, l] * 2 + [w-2*t, l-2*t] * 2
        decOrder = [(5, 7), (4, 6), (1, 3), (0, 2)]

        def decFun(l_line):
            new_line = []
            for l in l_line[:4]:
                l = l-2*DEFVAL._REBAR_D_DEF
                new_line.append(l)
            
            for l in l_line[4:]:
                l = l+2*DEFVAL._REBAR_D_DEF
                new_line.append(l)

            return new_line


        area = attr["area"]
        # * Res:([r]:配筋率, [Ns]:钢筋的个数, [As]:钢筋的截面积)
        Res = BarsTools.TryFunc(l_line, area, r, decOrder, decFun)
        r = sum(Res[0])
        return r, Res[1], Res[2]

    @staticmethod
    def SRoundRebarDistr(paras:SRoundSectParas, attr:dict, r:float):
        if r - 0.006 < DEFVAL._TOL_ or r - 0.04 > DEFVAL._TOL_:
            raise Exception("Wrong Re-Bar Ratio:{}".format(r))
        
        c = paras.C
        R = paras.R
        l_line = [2 * np.pi * (R-c)]
        area = attr['area']
        decOrder = [0]

        def decFunc(l_line):
            return [l_line[0]-2*np.pi*DEFVAL._REBAR_D_DEF]

        Res = BarsTools.TryFunc(l_line, area, r, decOrder, decFunc)
        r = sum(Res[0])
        return r, Res[1], Res[2]
        
    @staticmethod
    def HRoundRebarDistr(paras:BridgeParas.HRoundSectParas, attr:dict, r:float) -> tuple[float, list[int], list[ReBarArea]]:
        """
        :param Paras:桥墩截面截面参数对象
        :param attr: 桥墩截面参数对象
        :param r: 目标配筋率
        :returns r:实际配筋率
        :returns N:各边的钢筋数量
        :returns BarType:钢筋类型及截面积
        """
        if r - 0.006 < DEFVAL._TOL_ or r - 0.04 > DEFVAL._TOL_:
            raise Exception("Wrong Re-Bar Ratio")

        
        c = paras.C
        Rout = paras.Rout - c
        Rin = paras.Rout - paras.T - c
        if c < 0 or Rout < 0 or Rin < 0:
            raise Exception('Wrong Params:{0}'.format([c, Rout, Rin]))
        if paras.T < DEFVAL._REBAR_D_DEF:
            raise Exception('wrong params T too small')
        l_line = [Rout*2*np.pi, Rin*2*np.pi]
        area = attr["area"]
        # * Res:([r]:配筋率, [Ns]:钢筋的个数, [As]:钢筋的截面积)
        decOrder = [1, 0]

        def decFunc(l_line):
            return [l_line[0]-2*np.pi*DEFVAL._REBAR_D_DEF, l_line[1]+2*np.pi*DEFVAL._REBAR_D_DEF]

        Res = BarsTools.TryFunc(l_line, area, r, decOrder, decFunc)
        r = sum(Res[0])
        return r, Res[1], Res[2]

    @staticmethod
    def NsChangeByOrder(Ns:list[int], order:list, count:int, func):
        index = order[count]
        if type(index) is int:
            Ns[index] = func(Ns[index])
            if Ns[index] <= 0:
                Ns[index] = 1
            # if Ns[index] == 1:
            #     return
            # else:
            #     Ns[index] = func(Ns[index])
        elif type(index) is list or type(index) is tuple:
            for i in index:
                if Ns[i] == 1:
                    return
                else:
                    Ns[i] = func(Ns[index])
        else:
            raise Exception("Wrong Paras")
