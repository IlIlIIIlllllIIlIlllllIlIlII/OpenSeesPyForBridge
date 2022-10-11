import numpy as np
import time
from abc import abstractmethod
from . import Comp
from . import Part
from . import OpsObject
from . import UtilTools
from . import Paras
from . import GlobalData

# class Boundary(Comp.Parts, metaclass=ABCMeta):
#     @abstractmethod
#     def __init__(self, name=""):
#         super(Comp.Parts, self).__init__(name)
#         self._type += "Boundary"
#         self._activated = False
    
#     @abstractmethod
#     def _activate(self): ...

class BridgeFixedBoundary(Comp.Boundary):
    # __slots__ = ['_type', '_uniqNum', '_name', '_Node', '_OpsFix']
    @Comp.CompMgr()
    def __init__(self, node:Part.BridgeNode, fixDOFs:list, name=""):
        super(BridgeFixedBoundary, self).__init__(name)
        self._type += "->FixedBoundary"
        self._node = node
        self._fixDOFs= fixDOFs
        self._OpsFix = None
     
    def _SectReBuild(self):
        # Comp.CompMgr.removeComp(self._OpsFix)
        # self._OpsFix = OpsObject.OpsFix(self._node.OpsNode, self._fixVal)
        ...
    
    def _activate(self):
        # if self._activated:
        #     return
        self._OpsFix = OpsObject.OpsFix(self._node.OpsNode, self._fixDOFs)
        self._activated = True

    @property
    def FixedNode(self):
        return self._node
    @FixedNode.setter
    def FixedNode(self, newVal):
        if type(newVal) is type(self._point):
            self._point = newVal
            self._SectReBuild()
        else:
            raise Exception("Wrong Paras")
    
    @property
    def fixVal(self):
        return self._fixDOFs
    @fixVal.setter
    def fixVal(self, newVal):
        if type(newVal) is type(self._fixDOFs):
            self._fixDOFs = newVal
        else:
            raise Exception("Wrong Paras")
    
    @property
    def val(self):
        return [self._point, self._fixDOFs]

class BridgeBearingBoundary(Comp.Boundary):
    @Comp.CompMgr()
    def __init__(self, nodeI:Part.BridgeNode, nodeJ:Part.BridgeNode, plasticStrain:float, E:float, DOFs:list[int], name=""):
        super().__init__(name)
        self._type += "Bridge Plastic Bearing Boundary"
        self._EPPMaterial = OpsObject.OpsElasticPPMaterial(E, plasticStrain)
        # self._Rigid = OpsObject.OpsElasticPPMaterial(10000 * E, 10000 * plasticStrain)
        self._NodeI = nodeI
        self._NodeJ = nodeJ
        if UtilTools.Util.isOnlyHas(DOFs, [0, 1], flatten=True):
            self._DOFs = DOFs
        else:
            raise Exception("Wrong Params")
        # self._zMats = []
        # for i, val in enumerate(dirt):
        #     if val == 1:
        #         self._zMats.append(self._EPPMaterial.uniqNum)
        #     else:
        #         self._zMats.append(self._Rigid.uniqNum)

        self._BearingElement = None
    
    def _activate(self):
        # if self._activated:
        #     return
        zMaterials = [self._EPPMaterial] * sum(self._DOFs)
        self._BearingElement = OpsObject.OpsZLElement(self._NodeI.OpsNode, self._NodeJ.OpsNode, zMaterials, self._DOFs)
        self._activated = True
        return self._BearingElement

    @property
    def val(self):
        return [self._EPPMaterial, self._NodeI, self._NodeJ, self._DOFs]

class BridgeEQDOFSBoundary(Comp.Boundary):
    @Comp.CompMgr()
    def __init__(self, nodeI:Part.BridgeNode, nodeJ:Part.BridgeNode, DOFs:list[int], name=""):
        super().__init__(name)
        self._type = 'Bridge Equal Dofs boundary'
        self._nodeI = nodeI
        self._nodeJ = nodeJ
        self._OpsEQDOF = None

        if UtilTools.Util.isOnlyHas(DOFs, [0, 1], flatten=True):
            self._Dofs = DOFs
        else:
            raise Exception('Wrong paras')
    
    def _activate(self):
        # if self._activated:
        #     return

        dofs = []
        for i, val in enumerate(self._Dofs):
            if val == 1:
                dofs.append(i+1)
    
        self._OpsEQDOF = OpsObject.OpsEqualDOF(self._nodeI.OpsNode, self._nodeJ.OpsNode, dofs)
        self._activated = True
        return self._OpsEQDOF

    @property
    def val(self):
        return [self._nodeI, self._nodeJ, self._Dofs]

class BridgeSimplePileSoilBoundary(Comp.Boundary):
    @Comp.CompMgr()
    def __init__(self, nodeI:Part.BridgeNode, nodeJ:Part.BridgeNode, soil:Paras.SoilParas, pileD, Cu, pileEnd=False, h=None, name=""):
        super().__init__(name)
        self._type += 'Bridge Pile-Soil Interaction Boundary'
        self._NodeI = nodeI
        self._NodeJ = nodeJ
        self._pileD = pileD
        self._Cu = Cu
        self._soil = soil
        self._pileEnd = pileEnd
        if h:
            self._z = h
        else:
            self._z = -nodeI[2]
    
    
    def _activate(self):
        # if self._activated:
        #     return
        # self
        paras:Paras.ClayParas = self._soil
        gamma = paras.rho * GlobalData.DEFVAL._G_
        z = self._z
        if paras.clayType == 'SoftClay':
            J = 0.25
            epsu = 0.02
            soilType = 1
        elif paras.clayType == 'MediumClay':
            J = 0.375
            epsu = 0.01
            soilType = 1
        elif paras.clayType == 'StiffClay':
            J = 0.5
            epsu = 0.005
            soilType = 1
        else:
            J = 1.0
            epsu = 0.001
            soilType = 2

        pult = min(9 * self._Cu * self._pileD, [3 + gamma/self._Cu*z + J/self._pileD*z]* self._Cu * self._pileD)
        y50 = 2.5 * epsu * self._pileD
        Cd = 0.3
        py = OpsObject.OpsPySimpleMaterial(pult, y50, Cd, soilType=soilType)

        phi = self._Cu / (gamma * z)
        if phi <= 1:
            alpha = 0.5 * phi ** -0.25
        else:
            alpha = 0.5 * phi ** -0.5
        tult = alpha * self._Cu * self._pileD * np.pi
        z50 = 0.5 * self._pileD / 6
        tz = OpsObject.OpsTzSimpleMaterial(tult, z50, soilType=soilType)

        qult = 9 * self._Cu * np.pi * (self._pileD/2)**2
        q50 = 0.1 * self._pileD
        
        qz = OpsObject.OpsQzSimpleMaterial(qult, q50, soilType=soilType)

        if self._pileEnd:
            zmat = qz
        else:
            zmat = tz
        
        self._activated = True

        return OpsObject.OpsZLElement(self._NodeI.OpsNode, self._NodeJ.OpsNode, [py, py, zmat], [1, 2, 3])



    @abstractmethod    
    def val(self):
        return [self._NodeI, self._NodeJ, self._Cu, self._pileD]

class BridgeFullPileSoilBoundary(Comp.Boundary):
    @Comp.CompMgr()
    def __init__(self, pileSeg_: list[Part.LineSRoundSeg], soilMaterial:Paras.SandParas, soilHalfL_:list[float]=None, soilHalfW_:list[float]=None, soilHalfH_:list[float]=None, name=""):
        super().__init__(name)
        self._type += 'Bridge Full Pile Soil Boundary'
        self._pileSegs = pileSeg_
        self._SoilMaterial = soilMaterial
        self._SoilL = soilHalfL_
        self._SoilW = soilHalfW_
        self._SoilH = soilHalfH_
        self._SoilCuboids = None
        
    @property
    def val(self):
        return [self._pileSegs, self._SoilL, self._SoilW, self._SoilH]
    
    def _activate(self):
        # if self._activated:
        #     return
        if len(self._pileSegs) == 1:
            paras:Paras.SRoundSectParas = self._pileSegs[0]._Secti
            r = paras.R

            soilH = sum(self._SoilH)
            soilL = sum(self._SoilL)
            soilW = sum(self._SoilW)
            segH = sum(self._pileSegs[0].EleLength)

            x0t, y0t, z0t = self._pileSegs[0]._BridgeNodeI.point

            xdt, ydt, zdt = x0t+r, y0t+r, z0t
            xct, yct, zct = x0t+r, y0t-r, z0t
            xbt, ybt, zbt = x0t-r, y0t-r, z0t
            xat, yat, zat = x0t-r, y0t+r, z0t

            xdm, ydm, zdm = x0t+r, y0t+r, z0t-segH
            xcm, ycm, zcm = x0t+r, y0t-r, z0t-segH
            xbm, ybm, zbm = x0t-r, y0t-r, z0t-segH
            xam, yam, zam = x0t-r, y0t+r, z0t-segH

            xdb, ydb, zdb = x0t+r, y0t+r, z0t-segH-soilH
            xcb, ycb, zcb = x0t+r, y0t-r, z0t-segH-soilH
            xbb, ybb, zbb = x0t-r, y0t-r, z0t-segH-soilH
            xab, yab, zab = x0t-r, y0t+r, z0t-segH-soilH

            rev_L = self._SoilL.copy()
            rev_L.reverse()
            
            rev_W =self._SoilW.copy()
            rev_W.reverse()
            """
            |_____y
            |
            |x

            #------------|#----|#------------ 
            |    block3  |  2  | block1     |                       ________________________   ___
            |            |     |            |                                 | |               |
            #-----------%|#b-a%|#-----------%                       ----------|-|-----------    |
            |    block4  |  9  | block8     |  ___                  ----------|-|-----------    segh
            #-----------%|#c-d%|#-----------%   ^                   ----------|-|-----------    |
            |            |     |            |   |                   __________|_|___________   _|_
            |    block5  |  6  | block7     |   soilL                         | |               |
            ----------5-%|----%|------------%  _|_                  __________|_|___________    soilH
                                                                              | |               |
            |<---------->|--d--|<---soilW-->|                       __________|_|___________   _|_
            """
            msgStart = 'create block {}'
            msgEnd = 'block {} finished, spend time {}'

            eleLenH = self._SoilH + self._pileSegs[0].EleLength

            # * block 1 
            print(msgStart.format(1))
            timeStart = time.time()
            p3 = xab-soilL,         yab, zab
            p5 =       xat, yat + soilW, zat
            block1 = Part.SoilCuboid.FromCuboidP3_P5(p3, p5, self._SoilMaterial, rev_L, self._SoilW, eleLenH)
            timeEnd = time.time()
            print(msgEnd.format(1, timeEnd-timeStart))

            # * block 2 
            print(msgStart.format(2))
            timeStart = time.time()
            p3 = xbb-soilL,         ybb, zbb
            p5 =       xbt,     ybt + r*2, zbt

            block2 = Part.SoilCuboid.FromCuboidP3_P5(p3, p5, self._SoilMaterial, rev_L, [r]*2, eleLenH)
            timeEnd = time.time()
            print(msgEnd.format(2, timeEnd-timeStart))

            # * block 3 
            print(msgStart.format(3))
            timeStart = time.time()
            p3 = xbb-soilL, xbb-soilW, zbb
            p5 =       xbt,       xbt, zbt
            block3 = Part.SoilCuboid.FromCuboidP3_P5(p3, p5, self._SoilMaterial, rev_L, rev_W, eleLenH)
            timeEnd = time.time()
            print(msgEnd.format(3, timeEnd-timeStart))

            # * block 4 
            print(msgStart.format(4))
            timeStart = time.time()
            p3 = xcb-r*2, ycb-soilW, zcb
            p5 =   xct,       yct, zct
            block4 = Part.SoilCuboid.FromCuboidP3_P5(p3, p5, self._SoilMaterial, [r]*2, rev_W, eleLenH)
            timeEnd = time.time()
            print(msgEnd.format(4, timeEnd-timeStart))

            # * block 5 
            print(msgStart.format(5))
            timeStart = time.time()
            p3 =       xcb, ycb-soilW, zcb
            p5 = xct+soilL,       yct, zct
            block5 = Part.SoilCuboid.FromCuboidP3_P5(p3, p5, self._SoilMaterial, self._SoilL, rev_W, eleLenH)
            timeEnd = time.time()
            print(msgEnd.format(5, timeEnd-timeStart))

            # * block 6 
            print(msgStart.format(6))
            timeStart = time.time()
            p3 =       xcb,   ycb, zcb
            p5 = xct+soilL, yct+r*2, zct
            block6 = Part.SoilCuboid.FromCuboidP3_P5(p3, p5, self._SoilMaterial, self._SoilL, [r]*2, eleLenH)
            timeEnd = time.time()
            print(msgEnd.format(6, timeEnd-timeStart))

            # * block 7 
            print(msgStart.format(7))
            timeStart = time.time()
            p3 =       xdb,       ydb, zdb
            p5 = xdt+soilL, ydt+soilW, zdt
            block7 = Part.SoilCuboid.FromCuboidP3_P5(p3, p5, self._SoilMaterial, self._SoilL, self._SoilW, eleLenH)
            timeEnd = time.time()
            print(msgEnd.format(7, timeEnd-timeStart))

            # * block 8 
            print(msgStart.format(8))
            timeStart = time.time()
            p3 =   xab,       yab, zab
            p5 = xat+r*2, yat+soilW, zat
            block8 = Part.SoilCuboid.FromCuboidP3_P5(p3, p5, self._SoilMaterial, [r]*2, self._SoilW, eleLenH)
            timeEnd = time.time()
            print(msgEnd.format(8, timeEnd-timeStart))

            # * block 9
            print(msgStart.format(9))
            timeStart = time.time()
            p3 = xbb, ybb, zbb
            p5 = xdm, ydm, zdm
            block9 = Part.SoilCuboid.FromCuboidP3_P5(p3, p5, self._SoilMaterial, [r]*2, [r]*2, self._SoilH)
            timeEnd = time.time()
            print(msgEnd.format(9, timeEnd-timeStart))

            self._SoilCuboids = [block1, block2, block3, block4, block5, block6, block7, block8, block9]

            # * 约束土体底面
            all_fix_node = []
            for b in self._SoilCuboids:
                flag, val = b.FindRangeBridgeNodes(b._Node3.point, b._L, b._W, 0)
                if flag:
                    ns:list[Part.BridgeNode] = val.flatten().tolist()
                    all_fix_node += ns

            all_fix_des = []
            for n in all_fix_node:
                all_fix_des.append(BridgeFixedBoundary(n, [1]*6))
            
            # * equal DOFs
            all_eqDof_des = []
            suface_pair1 = [(0, 2), (7, 3), (6,4)]
            for bI1, bI2 in suface_pair1:
                b1 = self._SoilCuboids[bI1]
                b2 = self._SoilCuboids[bI2]
                p1 = b1._PointsIndex[0, -1, 1]
                p2 = b2._PointsIndex[0, 0, 1]
                flag1, BridgeNs1 = b1.FindRangeBridgeNodes(p1, b1._L, 0, b1._H)
                flag2, BridgeNs2 = b2.FindRangeBridgeNodes(p2, b2._L, 0, b2._H)
                if flag1 and flag2 and BridgeNs1.shape == BridgeNs2.shape:
                    l, w, h = BridgeNs1.shape
                    for x in range(l):
                        for y in range(w):
                            for z in range(h):
                                bn1 = BridgeNs1[x, y, z]
                                bn2 = BridgeNs2[x, y, z]
                                # b = BridgeEQDOFSBoundary(bn1, bn2, [1, 1, 1, 0, 0, 0])
                                # all_eqDof_des.append(b)
                else:
                    raise Exception("wrong paras, can not find points or points num are not equal")
            
            suface_pair2 = [(0, 6), (1, 5), (2,4)]
            for bI1, bI2 in suface_pair2:
                b1 = self._SoilCuboids[bI1]
                b2 = self._SoilCuboids[bI2]
                p1 = b1._PointsIndex[0, 0, 1]
                p2 = b2._PointsIndex[-1, 0, 1]
                flag1, BridgeNs1 = b1.FindRangeBridgeNodes(p1, 0, b1._W, b1._H)
                flag2, BridgeNs2 = b2.FindRangeBridgeNodes(p2, 0, b2._W, b2._H)

                if flag1 and flag2 and BridgeNs1.shape == BridgeNs2.shape:
                    l, w, h = BridgeNs1.shape
                    for x in range(l):
                        for y in range(w):
                            for z in range(h):
                                bn1 = BridgeNs1[x, y, z]
                                bn2 = BridgeNs2[x, y, z]
                                # b = BridgeEQDOFSBoundary(bn1, bn2, [1, 1, 1, 0, 0, 0])
                                # if (-19000, -25000, -40000) in (bn1.point, bn2.point):
                                #     print("boundary {} get".format(b._uniqNum))
                                # all_eqDof_des.append(b)
                else:
                    raise Exception("wrong paras")

            all_connet_des = []
            # * connect soil and pile
            segBottomPoint = self._pileSegs[0]._BridgeNodeJ.point
            x, y, z = segBottomPoint
            flag1, nodes1 = self._SoilCuboids[5].FindRangeBridgeNodes((x+r, y, z), 0, 0, abs(z))
            flag2, nodes2 = self._SoilCuboids[7].FindRangeBridgeNodes((x, y+r, z), 0, 0, abs(z))

            if flag1 and flag2 and nodes1.shape == nodes2.shape:
                nodes1 = nodes1.flatten().tolist()
                nodes2 = nodes2.flatten().tolist()
                segnodes = self._pileSegs[0]._BridgeNodes.copy()
                segnodes = segnodes.reverse()
                if len(nodes1) == len(self._pileSegs[0]._BridgeNodes):
                    for p1, p2, p in zip(nodes1, nodes2, self._pileSegs[0]._BridgeNodes):
                        # all_connet_des.append(BridgeEQDOFSBoundary(p, p1, [1, 1, 0, 0, 0, 0] ))
                        # all_connet_des.append(BridgeEQDOFSBoundary(p, p2, [1, 1, 0, 0, 0, 0] ))
                        ...
                else:
                    raise Exception("Wrong param, nodes in pile are not equal to nodes in soil")
            else:
                raise Exception("Wrong param, canot find nodes or the nodes shape are not equal")


            
            # for bridgeN in self._pileSegs[0].NodeList:
            #     x, y, z = bridgeN.point
            #     p = (x+r, y, z)
            #     # flag , n = self._SoilCuboids[5].FindBridgeNode(p)
            #     flag, n = self._SoilCuboids[5].FindBridgeNode()
            #     if not flag:
            #         raise Exception("Can not find point:{} in block 6".format(p))
            #     all_connet_des.append(BridgeEQDOFSBoundary(bridgeN, n, [1, 1, 0, 0, 0, 0] ))
                
            #     p = (x, y+r, z)
            #     flag , n = self._SoilCuboids[7].FindBridgeNode(p)
            #     if not flag:
            #         raise Exception("Can not find point:{} in block 8".format(p))
            #     all_connet_des.append(BridgeEQDOFSBoundary(bridgeN, n, [1, 1, 0, 0, 0, 0] ))
            
            self._activated = True
            
            return all_fix_des, all_eqDof_des, all_connet_des