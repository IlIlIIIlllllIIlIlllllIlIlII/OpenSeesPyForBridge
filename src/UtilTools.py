#%%
import numpy as np
from .Comp import Paras
from .GlobalData import DEFVAL, ReBarArea, ReBarType
from .Paras import HRectSectParas, HRoundSectParas
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
        return float(np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2)))

    @staticmethod
    def LinePointBuilder(p1:tuple[int, ...], p2:tuple[int, ...], intervalList:list[int]):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dz = p1[2] - p2[2]
        pointsList = [p1]
        if Util.TOLEQ(Util.PointsDis(p1, p2), sum(intervalList)):
            rate = [sum(intervalList[:i+1])/sum(intervalList) for i,_ in enumerate(intervalList)]
            temp = 0

            for r in rate:
                temp = (p1[0] - dx * r, p1[1] - dy * r, p2[2] - dz * r)
                pointsList.append(temp)
        else:
            raise Exception("wrong paras")
        return pointsList

    @staticmethod
    def vectAdd(x:tuple, y):
        if type(x) == type(y):
            return (x[0]+y[0], x[1]+y[1], x[2]+y[2])
        elif type(y) == float or type(y) == int:
            return (x[0]+y, x[1]+y, x[2]+y)
        else:
            raise Exception("wrong paras")

    @staticmethod
    def vectSub(x:tuple, y):
        if type(x) == type(y):
            return (x[0]-y[0], x[1]-y[1], x[2]-y[2])
        elif type(y) == float or type(y) == int:
            return (x[0]-y, x[1]-y, x[2]-y)
        else:
            raise Exception("wrong paras")

    @staticmethod
    def vectEnlarge(x, y):
        if type(y) == int or type(y) == float:
            return (x[0]*y, x[1]*y, x[2]*y)
        else:
            raise Exception("wrong paras")
    
    @staticmethod
    def vectTrueDiv(x, y):
        if type(y) == int or type(y) == float:
            return (x[0]/y, x[1]/y, x[2]/y)
    
    @staticmethod
    def vectFloorDiv(x, y):
        if type(y) == int or type(y) == float:
            return (x[0]//y, x[1]//y, x[2]//y)

class Util:

    @staticmethod
    def TOLEQ(n1, n2):
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
    def TOLLE(n1, n2):
        if not Util.TOLGT(n1, n2):
            return True
        else:
            return False

    @staticmethod
    def TOLGE(n1, n2):
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
    def isOnlyHas(obj, item) -> bool:
        if not np.iterable(item):
            item = [item]

        if np.iterable(obj):
            
            flag = True
            for i in obj:
                if i not in item:
                    flag = False

            return flag
        else:
            return (obj in item)

    @staticmethod
    def iter_isOnlyHas(obj, item):
        new = Util.Flatten(obj)
        return Util.isOnlyHas(new, item)

    @staticmethod
    def iter_Len(obj):
        n = Util.Flatten(obj)
        return len(n)

    def FullPertmuation(barArea, num):
        count = 0
        
        if num == 1:
            for a in barArea:
                yield [a]

        if num > 1:
            num -= 1
            
            f = Util.FullPertmuation(barArea, num)
            while True:
                i = [barArea[count]]
                try:
                    j = next(f)
                    yield i+j
                except:
                    count += 1
                    f = Util.FullPertmuation(barArea, num)
    
class SegmentTools:

    @staticmethod
    def BuildWithSettedLength(totalLen, setLen:float):
        num = int(totalLen/setLen)
        rem = totalLen - setLen * num

        if Util.TOLEQ(rem, 0):
            eleLenList =  totalLen._eleLengthList = num * [setLen]
        else:
            eleLenList = num * [setLen] + [rem]
        
        return eleLenList

    def BuildWithSettedNum(totalLen, n:int):
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
        if Util.TOLEQ(sum(intervalList), l):
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

        elif Util.TOLEQ(Rat_max, r):
            Ns_ = [Ns_max]
            As_ = [[As_max] * len(Ns_max)]
            Rat_ = [Rat_max]

            return (Rat_, Ns_, As_)

        elif Util.TOLGT(Rat_max, r):
            As_min = ReBarArea.getMinItem()
            order = decOrder
            count = 0
            Ns_try = Ns_max.copy()
            
            while Util.TOLLT(BarsTools.calcBarsArea([Ns_try], [[As_min] * len(Ns_try)])/sectArea, r) \
                and not Util.isOnlyHas(Ns_try, [1]):

                for i in order[count]:
                    if Ns_try[i] == 1:
                        continue
                    else:
                        Ns_try[i] -= 1

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
                for i in order[count]:
                    Ns_Upper[i] += 1
            # ---------(Ns_Lower, As_min)----(||Ns_Upper, As_min||)----r--(||Ns_Upper, As_min||)---(Ns_max, As_max)----->
            f = BarsTools.RebarsCombineGenator(order, ReBarArea.listAllItem())

            r_lower = 0
            r_res = BarsTools.calcBarsArea([Ns_Upper], [[As_max]*len(Ns_Upper)])
            Ns_res = None
            As_res = None


            while True:
                try:
                    As_ = next(f)
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
                    break

                return (r_res, Ns_res, As_res)

    @staticmethod
    def RebarsCombineGenator(decOrder, barAreaRange:list):
        num = len(decOrder)
        f = Util.FullPertmuation(barAreaRange, num)
        res = [0] * Util.iter_Len(decOrder)
        while True:
            try:
                As_ = next(f)
                if len(As_) != len(decOrder):
                    raise Exception("the length of AS and decOrder is not equal")

                for As, ords in zip(As_, decOrder):
                    for i in ords:
                        res[i] = As

                yield res
            except:
                raise Exception("can not produce more Rebars Area Combine")

    # * 计算钢筋分布
    @staticmethod
    def HRectRebarDistr(Paras:HRectSectParas, attr:dict, r: float) -> tuple[float, list[int], list[ReBarArea]]:
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
        w = Paras.W - Paras.C * 2
        l = Paras.L - Paras.C * 2
        t = Paras.T - Paras.C * 2

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
    def HRoundRebarDistr(Paras:HRoundSectParas, attr:dict, r: float) -> tuple[float, list[int], list[ReBarArea]]:
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

        c = Paras.C
        Rout = Paras.R - c
        Rin = Paras.R - Paras.T - c
        l_line = [Rout*2*np.pi, Rin*2*np.pi]
        area = attr["area"]
        # * Res:([r]:配筋率, [Ns]:钢筋的个数, [As]:钢筋的截面积)
        decOrder = [1, 0]

        def decFunc(l_line):
            return [l_line[0]-2*np.pi*DEFVAL._REBAR_D_DEF, l_line[1]+2*np.pi*DEFVAL._REBAR_D_DEF]

        Res = BarsTools.TryFunc(l_line, area, r, decOrder, decFunc)
        r = sum(Res[0])
        return r, Res[1], Res[2]






