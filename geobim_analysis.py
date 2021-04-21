from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import operator
import functools
import multiprocessing

from collections import defaultdict, Iterable, OrderedDict

# add python-occ library
import OCC.Core.AIS
try:
    from OCC.Display.pyqt5Display import qtViewer3d
except BaseException:
    import OCC.Display
    try:
        import OCC.Display.backend
    except BaseException:
        pass
    try:
        OCC.Display.backend.get_backend("qt-pyqt5")
    except BaseException:
        OCC.Display.backend.load_backend("qt-pyqt5")
    from OCC.Display.qtDisplay import qtViewer3d

# add ifcopenshell library
import ifcopenshell
from ifcopenshell.geom.main import settings, iterator
from ifcopenshell.geom.occ_utils import display_shape,set_shape_transparency
from ifcopenshell import open as open_ifc_file
if ifcopenshell.version < "0.6":
    # not yet ported
    from .. import get_supertype

# add the functions of BIM calculation algorithm
from .functions import *



class analyser():
    
    def __init__(self):
        self.files = {}
        self.storeyElevation_lst = []
        self.floor_elements_lst = []
        self.floor_compound_shapes_lst = []
    
        #---------------------------default variables for BIM calculation algorithm ----------------------------------#
        # DBSCAN clustering
        self.s = 0.2
        self.dbscan =2
        # k-nearest neighbor
        self.k = 16
        # default ground floor number
        self.floornum = int(0)
    
        self.current_ifc_file = None
        self.floor_elements_lst = []
        self.floor_name_lst = []
        self.base_polygon = None
        self.base_overhang_obb_poly = None
        self.base_obb_pt_lst = []
        self.base_overhang_points =None
        self.base_floor_num = 1
        self.overhang_left = True
        self.storeyElevation_lst=[]
        # Georeference parameters
        self.addGeoreference = False
        self.georeference_x = 0.0
        self.georeference_y = 0.0
        self.georeference_z = 0.0
    
    def load(self, path):
        fn = path.split("/")[-1].split(".")[0]
        if fn in self.files:
            return

        f = open_ifc_file(str(path))
        #Run the floor segementation when loading
        self.floor_elements_lst, self.floor_name_lst = GetElementsByStorey(f)
        self.files[fn] = f
        
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_PYTHON_OPENCASCADE, True)
        for i in range(len(self.floor_elements_lst)):
            # -------------- create shape from BIM----------------:
            floor_ifc = self.floor_elements_lst[i]
            shapes = []
            if isinstance(floor_ifc, list):
                for element in floor_ifc:
                    try:
                        if element.Representation:
                            shape = ifcopenshell.geom.create_shape(settings, element).geometry
                            shapes.append(shape)
                    except:
                        print("Create shape failed, ", element.is_a(),',', element.Name)
            # ------------------- create shape from BIM done -----------------------------

            shapes_compound, if_all_compound = list_of_shapes_to_compound(shapes)
            self.floor_compound_shapes_lst.append(shapes_compound)

        storeys = f.by_type("IfcBuildingStorey")
        for st in storeys:
            self.storeyElevation_lst.append(st.Elevation/1000.0) # convert units from mm to meter

        if not os.path.exists('./result'):
            os.makedirs('./result/')
            
    def OverhangOneFloor(self, floornum):

        '''  get the overhang distance calculation of input floor'''

        floornum = int(floornum)
        self.floornum = floornum

        if not self.floor_name_lst:
            return

        if floornum>=0 and floornum < len(self.floor_elements_lst):
            if self.base_overhang_obb_poly:
                current_floor_obb_poly, current_obb_pt_lst, current_all_pt_lst= self.GetFloorOBBPoly_new(floornum)
            else:
                self.base_overhang_obb_poly, self.base_obb_pt_lst, base_all_pt_lst = self.GetFloorOBBPoly_new(self.base_floor_num)
                current_floor_obb_poly, current_obb_pt_lst, current_all_pt_lst= self.GetFloorOBBPoly_new(floornum)
            up_overhang, low_overhang = self.OBBPolyOverhang_new(self.base_obb_pt_lst,current_all_pt_lst,self.overhang_left)
            print("OverhangOneFloor done!")
            print("floor name, ",self.floor_name_lst[floornum], " up_overhang, ",up_overhang, "low_overhang, ", low_overhang)
            return {"floorname": self.floor_name_lst[floornum], "up_overhang": up_overhang, "low_overhang": low_overhang}
        
        else:
            return "error"
        
    def GetFloorOBBPoly_new(self,i):

        ''' calculate the oriented bounding box of one floor'''

        floor_name = self.floor_name_lst[i]
        print("current floor, ", floor_name)
        #shapes = CreateShape(floor_elements)
        compound_shapes = self.floor_compound_shapes_lst[i]

        # get all pt_lst
        all_pt_lst = []
        exp = TopExp_Explorer(compound_shapes, OCC.Core.TopAbs.TopAbs_VERTEX)
        while exp.More():
            vertex = OCC.Core.TopoDS.topods_Vertex(exp.Current())
            pnt = OCC.Core.BRep.BRep_Tool_Pnt(vertex)
            all_pt_lst.append([float("{:.3f}".format(pnt.X())),float("{:.3f}".format(pnt.Y()))])
            exp.Next()
        pts = GetOrientedBoundingBoxShapeCompound(compound_shapes)
        Z_value = []
        for pt in pts:
            Z_value.append(pt.Z())
        z_max = max(Z_value)
        z_min = min(Z_value)
        z_mid = 0.5 * (z_max + z_min)
        pts_low = []
        pts_up = []
        for pt in pts:
            if pt.Z() < z_mid:
                pts_low.append(pt)
            else:
                pts_up.append(pt)
        corners_top = pts_up
        pyocc_corners_list = []
        for pt in corners_top:
            pyocc_corners_list.append(
                [float("{:.3f}".format(pt.X())), float("{:.3f}".format(pt.Y() ))])
        # change the order of pt lst
        pyocc_corners_list = ptsReorder(pyocc_corners_list)
        poly_corners = Polygon(pyocc_corners_list)
        return poly_corners, pyocc_corners_list, all_pt_lst
    
    def OBBPolyOverhang_new(self, base_pt_lst, target_pt_lst, left_side=True ):

        ''' calculate the overhang distance based on oriented bounding boxes'''

        if left_side:
            p_up_0 = base_pt_lst[0]
            p_up_1 = base_pt_lst[3]

            p_low_0 = base_pt_lst[1]
            p_low_1 = base_pt_lst[2]
        else:
            p_up_0 = base_pt_lst[3]
            p_up_1 = base_pt_lst[0]

            p_low_0 = base_pt_lst[2]
            p_low_1 = base_pt_lst[1]

        # up distance:
        dis_up = [0.0]
        dis_low =[0.0]
        for t in target_pt_lst:
            v_up = (p_up_1[0] - p_up_0[0])*(t[1] - p_up_0[1]) - (t[0] - p_up_0[0])*(p_up_1[1]-p_up_0[1])
            if v_up>0:
                d = PT2lineDistance(p_up_0,p_up_1,t)
                dis_up.append(float("{:.3f}".format(d)))
                continue
            v_low = (p_low_1[0] - p_low_0[0])*(t[1] - p_low_0[1]) - (t[0] - p_low_0[0])*(p_low_1[1]-p_low_0[1])
            if v_low <0:
                d2 = PT2lineDistance(p_low_0,p_low_1,t)
                dis_low.append(float("{:.3f}".format(d2)))

        up_overhang = max(dis_up)
        low_overhang = max(dis_low)

        if left_side:
            return up_overhang,low_overhang   #always return up side and low side
        else:
            return low_overhang,up_overhang