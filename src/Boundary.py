import time
from abc import abstractmethod

import numpy as np

from src.log import StandardLogger

from . import BridgeParas, Comp, GlobalData, OpsObject, Part, UtilTools

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
        # if len(fixDOFs) == 6:
        self._fixDOFs= fixDOFs
        # else:
        #     raise Exception("wrong fixDofs:{}, the length of fixDOFs list should be 6".format(fixDOFs))
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
            raise Exception("Wrong Params DOFs{}".format(DOFs))
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

class BridgePyTzQzBoundary(Comp.Boundary):
    @Comp.CompMgr()
    def __init__(self, nodeI:Part.BridgeNode, nodeJ:Part.BridgeNode, soil, pileD, Cu, pileEnd=False, h=None, name=""):
        super().__init__(name)
        self._type += 'Bridge Py, Qz, Tz Boundary'
        self._NodeI = nodeI
        self._NodeJ = nodeJ
        self._pileD = pileD
        self._Cu = Cu
        self._soil = soil
        self._pileEnd = pileEnd
        if h:
            self._z = h
        else:
            self._z = -nodeI.point[2]
    
    
    def _activate(self):
        # if self._activated:
        #     return
        # self
        paras:BridgeParas.ClayParas = self._soil
        gamma = paras.rho * GlobalData.DEFVAL._G_
        z = self._z
        if isinstance(paras, BridgeParas.ClayParas) and paras.soilType == 'SoftClay':
            J = 0.25
            epsu = 0.02
            soilType = 1
        elif isinstance(paras, BridgeParas.ClayParas) and paras.soilType == 'MediumClay':
            J = 0.375
            epsu = 0.01
            soilType = 1
        elif isinstance(paras, BridgeParas.ClayParas) and paras.soilType == 'StiffClay':
            J = 0.5
            epsu = 0.005
            soilType = 1
        else:
            J = 1.0
            epsu = 0.001
            soilType = 2

        pult = min(9 * self._Cu * self._pileD, (3 + gamma/self._Cu*z + J/self._pileD*z)* self._Cu * self._pileD)
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

        return OpsObject.OpsZLElement(self._NodeI.OpsNode, self._NodeJ.OpsNode, [py.uniqNum, py.uniqNum, zmat.uniqNum], [1, 2, 3])

    @property
    def val(self):
        return [self._NodeI, self._NodeJ, self._Cu, self._pileD]

class BridgeSimplyPileSoilBoundary(Comp.Boundary):
    @Comp.CompMgr()
    def __init__(self, segs:list[Part.LineSRoundSeg], soilParas, name: str = ""):
        super().__init__(name)
        self._type += 'Bridge Simple Pile Soil Boundary'
        self._segs = segs
        self._soil = soilParas
        # self._pileEnd = pileEnd
    
    def _activate(self):
        fixedNodes = []
        fixs = []
        pyQzTzs = []
        for seg in self._segs:
            for node in seg.NodeList:
                if node.point[2] == 0:
                    continue

                fixedNode = Part.BridgeNode(node.point[0], node.point[1], node.point[2], dofType=node.OpsNode._dofType)

                fix = BridgeFixedBoundary(fixedNode, [1]*6)
                if isinstance(self._soil, BridgeParas.SandParas):
                    Cu = self._soil.frictionAng*abs(node.point[2])*self._soil.rho
                elif isinstance(self._soil, BridgeParas.ClayParas):
                    Cu = self._soil.cohesi
                if node == seg._BridgeNodeJ:
                    pileEnd = True
                else:
                    pileEnd = False
                D = seg._Secti.R * 2
                PyQzTz = BridgePyTzQzBoundary(node, fixedNode, self._soil, D, Cu, pileEnd)
                fixs.append(fix)
                pyQzTzs.append(PyQzTz)
                fixedNodes.append(fixedNode)

        self._activate = True
        return fixedNodes, pyQzTzs, fixs

class BridgeFullPileSoilBoundary(Comp.Boundary):
    @Comp.CompMgr()
    def __init__(self, pileSeg_: list[Part.LineSRoundSeg], pierEndNode, soilMaterial, soilHalfL_:list[float]=None, soilHalfW_:list[float]=None, soilHalfH_:list[float]=None, name=""):
        super().__init__(name)
        self._type += 'Bridge Full Pile Soil Boundary'
        self._pileSegs = pileSeg_
        self._SoilMaterial = soilMaterial
        self._SoilL = soilHalfL_
        self._SoilW = soilHalfW_
        self._SoilH = soilHalfH_
        self._SoilCuboids = None
        self._pierEdNode = pierEndNode
        
    @property
    def val(self):
        return [self._pileSegs, self._SoilL, self._SoilW, self._SoilH]


    def _activate(self):
        # if self._activated:
        #     return
        # Comp.CompMgr.NdmNdfSwitcher(Comp.DimensionAndNumberEnum.Brick)
            
        if len(self._pileSegs) == 1:
            paras:BridgeParas.SRoundSectParas = self._pileSegs[0]._Secti
            r = paras.R

            x0t, y0t, z0t = self._pileSegs[0]._BridgeNodeI.point
            x0t -= GlobalData.DEFVAL._COOROFFSET_
            y0t -= GlobalData.DEFVAL._COOROFFSET_
            z0t -= GlobalData.DEFVAL._COOROFFSET_

            xbt, ybt, zbt = x0t-r, y0t-r, z0t

            cornerL = self._SoilL
            cornerW = self._SoilW

            coreL = [r]*2
            coreW = [r]*2

            eleLenH = self._SoilH + self._pileSegs[0].EleLength

        elif len(self._pileSegs) > 1:

            cornerL = self._SoilL
            cornerW = self._SoilW

            x_ = []
            y_ = []
            z_ = []

            for seg in self._pileSegs:
                x_.append(seg._BridgeNodeI.point[0])
                y_.append(seg._BridgeNodeI.point[1])
                z_.append(seg._BridgeNodeI.point[2])

            x_.sort()
            y_.sort()
            z_.sort()
            coreL = []
            coreW = []
            coreH = []

            for i in range(len(x_)-1):
                coreL.append(x_[i+1]-x[i])

            for i in range(len(y_)-1):
                coreW.append(y_[i+1]-y[i])

            for i in range(len(z_)-1):
                coreH.append(z_[i+1]-z[i])

            eleLenH = self._SoilH + coreH

            xbt, ybt, zbt = x_[0], y_[0], z_[0]

        self.create_Cuboid((xbt, ybt, zbt), cornerL, cornerW, coreL, coreW, eleLenH)

        # * 约束土体底面
        all_fix_node = []
        for b in self._SoilCuboids:
            flag, val = b.FindRangeBridgeNodes(b._Node3.point, b._L, b._W, 0)
            if flag:
                ns:list[Part.BridgeNode] = val.flatten().tolist()
                all_fix_node += ns
            else:
                StandardLogger.warning("Can not find Bridge node in the plane of cuboid:{}, ignored".format(b._uniqNum))
                StandardLogger.warning("Search nodes range: p3:{}, p5:{}".format(b._Node3.point, (b._Node3.point[0]+b._L, b._Node3.point[1]+b._W, b._Node3.point[2]+0)))

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
                            b = BridgeEQDOFSBoundary(bn1, bn2, [1, 1, 1, 0, 0, 0])
                            all_eqDof_des.append(b)
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
                            b = BridgeEQDOFSBoundary(bn1, bn2, [1, 1, 1, 0, 0, 0])
                            all_eqDof_des.append(b)
            else:
                raise Exception("wrong paras")

        all_connet_des = []
        all_tempNodes = []

        # * connect soil and pilea // pier and pile
        offset = GlobalData.DEFVAL._COOROFFSET_
        for seg in self._pileSegs:
            #*pier with pile
            if len(self._pileSegs) > 1:
                eqDof = BridgeEQDOFSBoundary(seg._BridgeNodeI, self._pierEdNode, [1]*6)
                all_connet_des.append(eqDof)
            x, y, z = seg._BridgeNodeJ.point
            flag, nodes = self._SoilCuboids[8].FindRangeBridgeNodes((x-offset, y-offset, z-offset), 0, 0, abs(z))
            segnodes:list[Part.BridgeNode] = seg._BridgeNodes.copy()
            segnodes.reverse()
            if flag :
                soilnodes = nodes.flatten().tolist()
                for i, (pilep, soilp) in enumerate(zip(segnodes, soilnodes)):
                    if i==len(segnodes)-1:
                        continue

                    if isinstance(self._SoilMaterial, BridgeParas.SandParas):
                        Cu = self._SoilMaterial.frictionAng*abs(pilep.point[2])*self._SoilMaterial.rho
                    elif isinstance(self._SoilMaterial, BridgeParas.ClayParas):
                        Cu = self._SoilMaterial.cohesi

                    if pilep == (x, y, z):
                        pileEnd = True
                    else:
                        pileEnd = False

                    x1, y1, z1 = pilep.point
                    tempNode = Part.BridgeNode(x1, y1, z1, dofType=Comp.DimensionAndNumberEnum.BeamColunm)
                    all_tempNodes.append(tempNode)

                    simpleSSI = BridgePyTzQzBoundary(pilep, tempNode, self._SoilMaterial, seg._Secti.R*2, Cu, pileEnd)
                    all_connet_des.append(simpleSSI)

                    eqDof = BridgeEQDOFSBoundary(tempNode, soilp, [1, 1, 1, 0, 0, 0])
                    all_connet_des.append(eqDof)

            else:
                raise Exception("Wrong param, canot find nodes or the nodes shape are not equal")

            
        self._activated = True
        # Comp.CompMgr.NdmNdfSwitcher(Comp.DimensionAndNumberEnum.Brick)
            
        return all_fix_des, all_eqDof_des, all_connet_des, all_tempNodes


    def create_Cuboid(self, PointB, cornerL:list[float], cornerW:list[float], coreL:list[float], coreW:list[float], eleLenH:list[float] ):
        """
        |_____ y
        |
        |x

        #-------------#-----#------------ 
        |    block3  |  2  | block1     |                       ________________________   ___
        |            |     |            |                                 | |               |
        #-----------%|#b-a%|#-----------%                       ----------|-|-----------    |
        |    block4  |  9  | block8     |  ___                  ----------|-|-----------    segh
        #-----------%|#c-d%|#-----------%   ^                   ----------|-|-----------    |
        |            |     |            |   |                   __________|_|___________   _|_
        |    block5  |  6  | block7     |   soilL                         | |               |
        ----------5-%l----%l------------%  _|_                  __________|_|___________    soilH
                                                                            | |               |
        |<---------->|--d--|<---soilW-->|                       __________|_|___________   _|_
        """
        
        
        cornerSoilL = sum(cornerL)
        cornerSoilW = sum(cornerW)

        coreSoilL = sum(coreL)
        coreSoilW = sum(coreW)
        
        SoilH = sum(eleLenH)
        rvs_cornerL = cornerL.copy()
        rvs_cornerL.reverse()

        rvs_cornerW = cornerW.copy()
        rvs_cornerW.reverse()

        xbt, ybt, zbt = PointB
        xbb, ybb, zbb = xbt, ybt, zbt-SoilH 

        xat, yat, zat = xbt, ybt+coreSoilW, zbt
        xab, yab, zab = xbt, ybt+coreSoilW, zbt-SoilH 

        xct, yct, zct = xbt+coreSoilL, ybt, zbt
        xcb, ycb, zcb = xbt+coreSoilL, ybt, zbt-SoilH 

        xdt, ydt, zdt = xbt+coreSoilL, ybt+coreSoilL, zbt
        xdb, ydb, zdb = xbt+coreSoilL, ybt+coreSoilL, zbt-SoilH 

        msgStart = 'create block {}'
        msgEnd = 'block {} finished, spend time {}'


            # * block 1 
        print(msgStart.format(1))
        timeStart = time.time()
        p3 = xab-cornerSoilL,         yab, zab
        p5 =       xat, yat + cornerSoilW, zat
        block1 = Part.SoilCuboid.FromCuboidP3_P5(p3, p5, self._SoilMaterial, rvs_cornerL, cornerW, eleLenH)
            
        timeEnd = time.time()
        print(msgEnd.format(1, timeEnd-timeStart))

            # * block 2 
        print(msgStart.format(2))
        timeStart = time.time()
        p3 = xbb-cornerSoilL,         ybb, zbb
        p5 =       xbt,     ybt + coreSoilW, zbt

        block2 = Part.SoilCuboid.FromCuboidP3_P5(p3, p5, self._SoilMaterial, rvs_cornerL, coreW, eleLenH)
        timeEnd = time.time()
        print(msgEnd.format(2, timeEnd-timeStart))

            # * block 3 
        print(msgStart.format(3))
        timeStart = time.time()
        p3 = xbb-cornerSoilL, xbb-cornerSoilW, zbb
        p5 =       xbt,       xbt, zbt
        block3 = Part.SoilCuboid.FromCuboidP3_P5(p3, p5, self._SoilMaterial, rvs_cornerL, rvs_cornerW, eleLenH)
        timeEnd = time.time()
        print(msgEnd.format(3, timeEnd-timeStart))

            # * block 4 
        print(msgStart.format(4))
        timeStart = time.time()
        p3 = xcb-coreSoilL, ycb-cornerSoilW, zcb
        p5 =   xct,       yct, zct
        block4 = Part.SoilCuboid.FromCuboidP3_P5(p3, p5, self._SoilMaterial, coreL, rvs_cornerW, eleLenH)
        timeEnd = time.time()
        print(msgEnd.format(4, timeEnd-timeStart))

            # * block 5 
        print(msgStart.format(5))
        timeStart = time.time()
        p3 =       xcb, ycb-cornerSoilW, zcb
        p5 = xct+cornerSoilL,       yct, zct
        block5 = Part.SoilCuboid.FromCuboidP3_P5(p3, p5, self._SoilMaterial, cornerL, rvs_cornerW, eleLenH)
        timeEnd = time.time()
        print(msgEnd.format(5, timeEnd-timeStart))

            # * block 6 
        print(msgStart.format(6))
        timeStart = time.time()
        p3 =       xcb,   ycb, zcb
        p5 = xct+cornerSoilL, yct+coreSoilW, zct
        block6 = Part.SoilCuboid.FromCuboidP3_P5(p3, p5, self._SoilMaterial, cornerL, coreW, eleLenH)
        timeEnd = time.time()
        print(msgEnd.format(6, timeEnd-timeStart))

            # * block 7 
        print(msgStart.format(7))
        timeStart = time.time()
        p3 =       xdb,       ydb, zdb
        p5 = xdt+cornerSoilL, ydt+cornerSoilW, zdt
        block7 = Part.SoilCuboid.FromCuboidP3_P5(p3, p5, self._SoilMaterial, cornerL, cornerW, eleLenH)
        timeEnd = time.time()
        print(msgEnd.format(7, timeEnd-timeStart))

            # * block 8 
        print(msgStart.format(8))
        timeStart = time.time()
        p3 =   xab,       yab, zab
        p5 = xat+coreSoilL, yat+cornerSoilW, zat
        block8 = Part.SoilCuboid.FromCuboidP3_P5(p3, p5, self._SoilMaterial, coreL, cornerW, eleLenH)
        timeEnd = time.time()
        print(msgEnd.format(8, timeEnd-timeStart))

            # * block 9
        print(msgStart.format(9))
        timeStart = time.time()
        p3 = xbb, ybb, zbb
        p5 = xdt, ydt, zdt
        block9 = Part.SoilCuboid.FromCuboidP3_P5(p3, p5, self._SoilMaterial, coreL, coreW, eleLenH)
        timeEnd = time.time()
        print(msgEnd.format(9, timeEnd-timeStart))

        self._SoilCuboids = [block1, block2, block3, block4, block5, block6, block7, block8, block9]