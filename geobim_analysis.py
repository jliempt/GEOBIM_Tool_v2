from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import operator
import functools
import multiprocessing
import numpy as np

from collections import defaultdict, Iterable, OrderedDict

# add python-occ library
import OCC.Core.AIS

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
from .functions_nad import *


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
        # Georeference parameters from user
        self.addGeoreference = False
        self.georeference_x = 0.0
        self.georeference_y = 0.0
        self.georeference_z = 0.0
        # Georeference parameters from IFC file
        self.location = (0.0, 0.0, 0.0)
        self.direction = (0.0, 1.0, 0.0)
    
    def load(self, path):
        fn = path.split("/")[-1].split(".")[0]
        if fn in self.files:
            return

        f = open_ifc_file(str(path))
        #Run the floor segementation when loading
        self.floor_elements_lst, self.floor_name_lst = GetElementsByStorey(f)
        self.files[fn] = f

        # Add the georeferencing from the IFC file
        self.location = self.getGeoref()["location"]
        self.direction = self.getGeoref()["direction"]

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

        if floornum >= 0 and floornum < len(self.floor_elements_lst):
            if self.base_overhang_obb_poly:
                current_floor_obb_poly, current_obb_pt_lst, current_all_pt_lst = self.GetFloorOBBPoly_new(floornum)
            else:
                self.base_overhang_obb_poly, self.base_obb_pt_lst, base_all_pt_lst = self.GetFloorOBBPoly_new(
                    self.base_floor_num)
                current_floor_obb_poly, current_obb_pt_lst, current_all_pt_lst = self.GetFloorOBBPoly_new(floornum)
            up_overhang, low_overhang = self.OBBPolyOverhang_new(self.base_obb_pt_lst, current_all_pt_lst,
                                                                 self.overhang_left)
            print("OverhangOneFloor done!")
            print("floor name, ", self.floor_name_lst[floornum], " up_overhang, ", up_overhang, "low_overhang, ",
                  low_overhang)
            return {"floorname": self.floor_name_lst[floornum], "up_overhang": up_overhang,
                    "low_overhang": low_overhang}

        else:
            return "error"

    def overhangRoads(self, guidelines, floor_number=None):
        res = run_overhang_check(guidelines, self.floor_elements_lst, self.floor_name_lst, self.location,
                                 self.direction, floor_number)
        return res

    def overhangRoadsAlphaShape(self, guidelines, floor_number=None):
        res = run_overhang_check_alpha_shape(guidelines, self.floor_elements_lst, self.floor_name_lst, self.location,
                                             self.direction, floor_number)
        return res

    def heightCheck(self, guidelines):
        res = run_height_check(guidelines, self.floor_elements_lst, self.floor_name_lst, self.location, self.direction)
        return res

    def boundaryCheck(self):
        res = run_boundary_check(self.floor_elements_lst, self.floor_name_lst, self.location, self.direction)
        return res

    def OverhangAll_new(self):

        ''' new algorithm for overhang distance calculation of all floors, result save in the folder  ./result/overhang_all.txt'''

        if not self.floor_name_lst:
            return

        result = ""

        self.base_overhang_obb_poly, self.base_obb_pt_lst, base_all_pt_lst = self.GetFloorOBBPoly_new(
            self.base_floor_num)
        up_overhang_lst = []
        low_overhang_lst = []
        for i in range(self.base_floor_num, len(self.floor_elements_lst)):
            current_floor_obb_poly, current_obb_pt_lst, current_all_pt_lst = self.GetFloorOBBPoly_new(i)
            up_overhang, low_overhang = self.OBBPolyOverhang_new(self.base_obb_pt_lst, current_all_pt_lst,
                                                                 self.overhang_left)
            up_overhang_lst.append(up_overhang)
            low_overhang_lst.append(low_overhang)

        up_idx = up_overhang_lst.index(max(up_overhang_lst))
        low_idx = low_overhang_lst.index(max(low_overhang_lst))

        result = {"north": {}, "south": {}}
        result["north"]["floor"] = self.floor_name_lst[up_idx + self.base_floor_num]
        result["north"]["distance"] = max(up_overhang_lst)
        result["south"]["floor"] = self.floor_name_lst[low_idx + self.base_floor_num]
        result["south"]["distance"] = max(low_overhang_lst)

        for i in range(len(up_overhang_lst)):
            floor_num = str(self.floor_name_lst[i + self.base_floor_num])
            result[floor_num] = {}
            result[floor_num]["north"] = up_overhang_lst[i]
            result[floor_num]["south"] = low_overhang_lst[i]

        return result

    def GetFloorOBBPoly_new(self, i):

        ''' calculate the oriented bounding box of one floor'''

        floor_name = self.floor_name_lst[i]
        print("current floor, ", floor_name)
        # shapes = CreateShape(floor_elements)
        compound_shapes = self.floor_compound_shapes_lst[i]

        # get all pt_lst
        all_pt_lst = []
        exp = TopExp_Explorer(compound_shapes, OCC.Core.TopAbs.TopAbs_VERTEX)
        while exp.More():
            vertex = OCC.Core.TopoDS.topods_Vertex(exp.Current())
            pnt = OCC.Core.BRep.BRep_Tool_Pnt(vertex)
            all_pt_lst.append([float("{:.3f}".format(pnt.X())), float("{:.3f}".format(pnt.Y()))])
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
                [float("{:.3f}".format(pt.X())), float("{:.3f}".format(pt.Y()))])
        # change the order of pt lst
        pyocc_corners_list = ptsReorder(pyocc_corners_list)
        poly_corners = Polygon(pyocc_corners_list)
        return poly_corners, pyocc_corners_list, all_pt_lst

    def OBBPolyOverhang_new(self, base_pt_lst, target_pt_lst, left_side=True):

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
        dis_low = [0.0]
        for t in target_pt_lst:
            v_up = (p_up_1[0] - p_up_0[0]) * (t[1] - p_up_0[1]) - (t[0] - p_up_0[0]) * (p_up_1[1] - p_up_0[1])
            if v_up > 0:
                d = PT2lineDistance(p_up_0, p_up_1, t)
                dis_up.append(float("{:.3f}".format(d)))
                continue
            v_low = (p_low_1[0] - p_low_0[0]) * (t[1] - p_low_0[1]) - (t[0] - p_low_0[0]) * (p_low_1[1] - p_low_0[1])
            if v_low < 0:
                d2 = PT2lineDistance(p_low_0, p_low_1, t)
                dis_low.append(float("{:.3f}".format(d2)))

        up_overhang = max(dis_up)
        low_overhang = max(dis_low)

        if left_side:
            return up_overhang, low_overhang  # always return up side and low side
        else:
            return low_overhang, up_overhang

    def footprintWKT(self, floornum):

        floornum = int(floornum)

        '''save the coordinates of footprint polygon into WKT file'''

        if not self.floor_name_lst:
            return

        if floornum >= 0 and floornum < len(self.floor_elements_lst):
            shapes_compound = self.floor_compound_shapes_lst[floornum]
            pts = GetOrientedBoundingBoxShapeCompound(shapes_compound, False)
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
                    [float("{:.3f}".format(pt.X() + self.georeference_x)),
                     float("{:.3f}".format(pt.Y() + self.georeference_y))])
            # convex hull pyocc_corners_list
            from scipy.spatial import ConvexHull
            points = np.array(pyocc_corners_list)
            hull = ConvexHull(points)
            result = []
            for idx in hull.vertices:
                result.append(pyocc_corners_list[idx])
            poly_footprint = Polygon(result)
            str_poly = str(poly_footprint)
            line_str = self.floor_name_lst[floornum] + "|" + str_poly

            return line_str
        else:
            return "error"

    def GetHeight(self):

        '''Get the height value of the input floor '''

        if not self.floor_name_lst:
            return
        top_shape_compound = self.floor_compound_shapes_lst[-1]
        z_lst = []
        exp = TopExp_Explorer(top_shape_compound, OCC.Core.TopAbs.TopAbs_VERTEX)
        while exp.More():
            vertex = OCC.Core.TopoDS.topods_Vertex(exp.Current())
            pnt = OCC.Core.BRep.BRep_Tool_Pnt(vertex)
            if float("{:.3f}".format(pnt.Z())) not in z_lst:
                z_lst.append(float("{:.3f}".format(pnt.Z())))
            exp.Next()
        res = str(max(z_lst))
        print("Max Z value is ", max(z_lst), " meter")
        return res

    def GetBaseHeight(self, floornum):
        floornum = int(floornum)
        if not self.floor_name_lst:
            return

        print("Floor name, ", self.floor_name_lst[floornum])
        top_height = float("{:.3f}".format(self.storeyElevation_lst[floornum + 1]))
        return ({"Floor name": self.floor_name_lst[floornum], "height": top_height})

    def OverlapOneFloor(self, floornum):

        ''' Calculate overlap percentage between input floor and ground floor and save the result in ./result/overlap folder'''

        floornum = int(floornum)
        print(os.getcwd())
        yamlFilepath = "GEOBIM_Tool/Parameters/parameters.yaml"
        result = {}

        if not self.floor_name_lst:
            return

        if floornum or floornum == 0:
            # self.canvas._display.Context.RemoveAll(True)
            if not self.floor_compound_shapes_lst[floornum]:
                return "Current floor has no shapes or geometry," + self.floor_name_lst[floornum]

            floor_name_lst = [self.floor_name_lst[floornum]]
            storey_poly_lst = []

            if self.base_polygon:
                current_floor_poly_lst = self.GetFloorPolygon(floornum, yamlFilepath)
                storey_poly_lst.append(current_floor_poly_lst)
            else:
                base_poly_lst = self.GetFloorPolygon(self.base_floor_num, yamlFilepath)
                self.base_polygon = base_poly_lst[0]
                current_floor_poly_lst = self.GetFloorPolygon(floornum, yamlFilepath)
                storey_poly_lst.append(current_floor_poly_lst)

            result = GetStoreyOverlap(self.base_polygon, storey_poly_lst, floor_name_lst)

        return result

    def OverlapAll(self):

        ''' Calculate overlap percentage between all floors and ground floor and save the result in ./result/overlap folder'''

        if not self.floor_name_lst:
            return

        yamlFilepath = "GEOBIM_Tool/Parameters/parameters.yaml"

        storey_poly_lst = []
        new_floor_name_lst = []
        if self.base_polygon:
            for i in range(self.base_floor_num, len(self.floor_elements_lst)):
                if self.floor_compound_shapes_lst[i]:
                    floor_poly_lst = self.GetFloorPolygon(i, yamlFilepath)
                    storey_poly_lst.append(floor_poly_lst)
                    new_floor_name_lst.append(self.floor_name_lst[i])
        else:
            base_poly_lst = self.GetFloorPolygon(self.base_floor_num, yamlFilepath)
            self.base_polygon = base_poly_lst[0]
            for i in range(self.base_floor_num, len(self.floor_elements_lst)):
                if self.floor_compound_shapes_lst[i]:
                    floor_poly_lst = self.GetFloorPolygon(i, yamlFilepath)
                    storey_poly_lst.append(floor_poly_lst)
                    new_floor_name_lst.append(self.floor_name_lst[i])
        return GetStoreyOverlap(self.base_polygon, storey_poly_lst, new_floor_name_lst)

    def GetFloorPolygon(self, i, yamlFilepath):

        ''' return intersecting surface polygon of floor i, generated from floor cutting '''

        result = ""

        floor_name = self.floor_name_lst[i]
        print("current floor, ", floor_name,
              "******************************************************************************")
        # display shapes
        # v = self.canvas._display

        # set parameters
        s = self.s
        dbscan = self.dbscan
        k = self.k
        calcconvexhull = False
        use_obb = False

        # cutting_height of each building storey
        cutting_height = self.storeyElevation_lst[i] + 1.0
        # loading customize parameters from the .yml file
        yml_file = open(yamlFilepath, 'r')
        yml_data = yaml.load(yml_file, Loader=Loader)
        str1 = "f" + str(i)
        if str1 in yml_data.keys():
            dict2 = yml_data[str1]
            if 'cutting_height' in dict2.keys():
                value = float(dict2['cutting_height'])
                cutting_height = self.storeyElevation_lst[i] + value
            if 'k' in dict2.keys():
                k = float(dict2['k'])
            if 'use_obb' in dict2.keys():
                if dict2['use_obb'] == True:
                    use_obb = True
            if 's' in dict2.keys():
                s = float(dict2['s'])
            if 'dbscan' in dict2.keys():
                dbscan = float(dict2['dbscan'])
            if 'calcconvexhull' in dict2.keys():
                if dict2['calcconvexhull'] == True:
                    calcconvexhull = True
        if use_obb:
            print("use_obb, ", use_obb, "floor name,", floor_name,
                  " ----------------------------------------------------------------------")
            pts = GetOrientedBoundingBoxShapeCompound(self.floor_compound_shapes_lst[i], False)
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
                pyocc_corners_list.append([float("{:.3f}".format(pt.X())), float("{:.3f}".format(pt.Y()))])
            points = np.array(pyocc_corners_list)
            obb_hull = ConvexHull(points)
            result = []
            for idx in obb_hull.vertices:
                result.append(pyocc_corners_list[idx])
            poly_footprint = Polygon(result)
            return [poly_footprint]

        print("cutting height,", cutting_height)
        section_shape = GetSectionShape(cutting_height, self.floor_compound_shapes_lst[i])
        # v.DisplayShape(section_shape, color="RED", update=True)
        # get the section shape edges
        edges = GetShapeEdges(section_shape)
        if s != 0:
            first_xy = GetEdgeSamplePointsPerDistance(edges, s)
        else:
            first_xy = GetEdges2DPT(edges)
        np_points = np.array(first_xy)
        corners = GetNumpyOBB(np_points, calcconvexhull=calcconvexhull, show_plot=False)
        OBB_poly = Polygon(corners.tolist())

        # create result dir
        if not os.path.exists('./result/Overlap/' + floor_name):
            os.makedirs('./result/Overlap/' + floor_name)
        img_filepath = "./result/Overlap/" + floor_name + "/obbAndPoints.png"

        # save result as images in the result folder
        SavePloyAndPoints(OBB_poly, np_points, color='b', filepath=img_filepath)
        cluster_filepath = "./result/Overlap/" + floor_name + "/clusters.png"
        cluster_lst = GetDBSCANClusteringlst(np_points, dbscan, showplot=False, saveplot=cluster_filepath)
        line = str()
        per_floot_poly = []
        poly_count = 0
        for np_member_array in cluster_lst:
            poly_count += 1
            print(len(np_member_array))
            print("starting concave hull")
            hull = concaveHull(np_member_array, k=k, if_optimal=False)
            self.WriteConcave2WKT(hull, floor_name, poly_count)
            poly = Polygon(hull)
            print("polygon validation is: ", poly.is_valid, poly.area)
            poly_filepath = "./result/Overlap/" + floor_name + "/polygon" + str(poly_count) + ".png"
            OBB_points = GetNumpyOBB(np_member_array, show_plot=False)
            OBB_poly = Polygon(OBB_points.tolist())
            print("OBB_poly area,", OBB_poly.area, " name,", floor_name,
                  "---------------------------------------------------------------------")

            if not poly.is_valid:
                print("Try to repair validation:")
                new_poly = poly.buffer(0)
                line = line + "Repaired_" + str(new_poly.is_valid) + "_" + str(float("{:.2f}".format(new_poly.area)))
                print(new_poly.is_valid, new_poly.area)
                un_poly = ops.unary_union(new_poly)
                print(type(un_poly), un_poly.is_valid, "Union area,", un_poly.area)

                # if the poly is wrong, replace with OBB_poly
                if un_poly.area < (0.3 * OBB_poly.area):
                    un_poly = OBB_poly

                per_floot_poly.append(un_poly)

                if un_poly.geom_type == 'MultiPolygon':
                    for geom in un_poly.geoms:
                        xs, ys = geom.exterior.xy
                        plt.plot(xs, ys, color="r")
                    plt.savefig(poly_filepath)
                    plt.close()

                elif un_poly.geom_type == 'Polygon':
                    SavePloyAndPoints(un_poly, np_member_array, filepath=poly_filepath)
                else:
                    print("Error polygon generation from concave hull failed!")
            else:

                line = line + "True, Floor Area" + str(float("{:.2f}".format(poly.area)))
                print("Polygon True, no need repair")
                SavePloyAndPoints(poly, np_member_array, filepath=poly_filepath)
                per_floot_poly.append(poly)
        return per_floot_poly  # [polygon] or [polygons]

    def WriteConcave2WKT(self, hull_lst, floor_name, poly_count):

        ''' Save the concave hull in WKT format in order to load in QGIS, WKT result saved in ./result/WKT folder'''

        if not os.path.exists('./result/WKT'):
            os.makedirs('./result/WKT')
        new_lst = []
        for p in hull_lst:
            new_lst.append([float("{:.3f}".format(p[0] + self.georeference_x)),
                            float("{:.3f}".format(p[1] + self.georeference_y))])
        geo_poly = Polygon(new_lst)
        str_poly = str(geo_poly)
        f = open('./result/WKT/' + floor_name + '_' + str(poly_count) + '.txt', "w+")
        f.write("name|wkt\n")
        line_str = floor_name + '_' + str(poly_count) + "|" + str_poly + '\n'
        f.write(line_str)
        f.close()

    def OverlapOneFloorOBB(self, floornumber):

        ''' Calculate overlap percentage between input floor and ground floor by using their oriented bounding boxes'''

        floor_num = int(floornumber)

        if not self.floor_name_lst:
            return

        self.floornum = floor_num

        if floor_num or floor_num == 0:
            floor_name_lst = [self.floor_name_lst[floor_num]]
            storey_poly_lst = []
            if self.base_polygon:
                if self.floor_compound_shapes_lst[floor_num]:
                    current_poly, current_poly_lst, all_pt_lst = self.GetFloorOBBPoly_new(floor_num)
                    storey_poly_lst.append([current_poly])
                else:
                    #                    msg = QtWidgets.QMessageBox()
                    #                    msg.setIcon(QtWidgets.QMessageBox.Critical)
                    #                    msg.setText("Current floor has no shapes or geometry," + self.floor_name_lst[floor_num])
                    #                    msg.setWindowTitle("Shapes or Geometry Error")
                    #                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
                    #                    msg.show()
                    #                    msg.exec_()
                    return
            else:
                if self.floor_compound_shapes_lst[floor_num]:
                    base_poly, base_poly_lst, base_all_pt_lst = self.GetFloorOBBPoly_new(self.base_floor_num)
                    self.base_polygon = base_poly
                    current_poly, current_poly_lst, all_pt_lst = self.GetFloorOBBPoly_new(floor_num)
                    storey_poly_lst.append([current_poly])
                else:
                    #                    msg = QtWidgets.QMessageBox()
                    #                    msg.setIcon(QtWidgets.QMessageBox.Critical)
                    #                    msg.setText("Current floor has no shapes or geometry," + self.floor_name_lst[floor_num])
                    #                    msg.setWindowTitle("Shapes or Geometry Error")
                    #                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
                    #                    msg.show()
                    #                    msg.exec_()
                    return

            result = GetStoreyOverlap(self.base_polygon, storey_poly_lst, floor_name_lst)
            return result

    def OverlapAllOBB(self):

        ''' Calculate overlap percentage between all floors and ground floor by using their oriented bounding boxes'''

        if not self.floor_name_lst:
            return
        storey_poly_lst = []
        new_floor_name_lst = []
        if self.base_polygon:
            for i in range(self.base_floor_num, len(self.floor_elements_lst)):
                if self.floor_compound_shapes_lst[i]:
                    floor_poly, floor_poly_lst, floor_all_pt_lst = self.GetFloorOBBPoly_new(i)
                    storey_poly_lst.append([floor_poly])
                    new_floor_name_lst.append(self.floor_name_lst[i])
        else:
            base_poly, base_poly_lst, base_all_pt_lst = self.GetFloorOBBPoly_new(self.base_floor_num)
            self.base_polygon = base_poly
            for i in range(self.base_floor_num, len(self.floor_elements_lst)):
                if self.floor_compound_shapes_lst[i]:
                    floor_poly, floor_poly_lst, floor_all_pt_lst = self.GetFloorOBBPoly_new(i)
                    storey_poly_lst.append([floor_poly])
                    new_floor_name_lst.append(self.floor_name_lst[i])

        result = GetStoreyOverlap(self.base_polygon, storey_poly_lst, new_floor_name_lst)
        return result

    def GetFloorOBBPoly_new(self, i):

        ''' calculate the oriented bounding box of one floor'''

        floor_name = self.floor_name_lst[i]
        print("current floor, ", floor_name)
        # shapes = CreateShape(floor_elements)
        compound_shapes = self.floor_compound_shapes_lst[i]

        # get all pt_lst
        all_pt_lst = []
        exp = TopExp_Explorer(compound_shapes, OCC.Core.TopAbs.TopAbs_VERTEX)
        while exp.More():
            vertex = OCC.Core.TopoDS.topods_Vertex(exp.Current())
            pnt = OCC.Core.BRep.BRep_Tool_Pnt(vertex)
            all_pt_lst.append([float("{:.3f}".format(pnt.X())), float("{:.3f}".format(pnt.Y()))])
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
                [float("{:.3f}".format(pt.X())), float("{:.3f}".format(pt.Y()))])
        # change the order of pt lst
        pyocc_corners_list = ptsReorder(pyocc_corners_list)
        poly_corners = Polygon(pyocc_corners_list)
        return poly_corners, pyocc_corners_list, all_pt_lst

    def parkingCalculate(self, ifc_path, zone):

        ''' Calculte the needed parking units number according to the regulation of municipality '''

        print("Parking Units Calculation starts!")

        ifc_file = ifcopenshell.open(ifc_path)
        ifcSpaces = ifc_file.by_type('ifcspace')
        # file_parking = open(outputfile, "w+")

        count_40, count_40_65, count_65_85, count_85_120, count_120_plus = GetIfcSpaceType(ifcSpaces)
        minpp = GetMinParkingUnitNum(count_40, count_40_65, count_65_85, count_85_120, count_120_plus, zone.upper())
        str_apartment = "number of Apartments, \nless 40 square meter: " + str(
            count_40) + "\n40 to 65 square meter: " + str(count_40_65) + "\n65 to 85 square meter: " + str(
            count_65_85) + "\n85 to 120 square meter: " + str(count_85_120) + "\nmore than 120 square meter: " + str(
            count_120_plus) + "\n"
        str_zone = "Zone type: Zone A Metropolitan area\n"
        str_minpp = "min parking units to provide: " + str(minpp)

        return str_apartment + str_zone + str_minpp

    def setBaseFloornum(self, floornum):
        floornum = int(floornum)

        # self.canvas._display.Context.RemoveAll(True)
        self.base_polygon = None

        self.base_floor_num = floornum

    def addGeoreferencePoint(self, x, y, z):

        '''add georefercen point into this tool'''

        self.georeference_x = float(x)
        self.georeference_y = float(y)
        self.georeference_z = float(z)

    def setOverhangdir(self, direction):

        '''set the overhang direction for distance calculation, north-south direction or east-west direction'''

        dialog = CheckInput()
        if dialog.exec():
            self.overhang_left = dialog.getInputs()
        else:
            return
        print(self.overhang_left)

    def setOverlapParameters(self, x, y, z):

        self.s = float(x)
        self.dbscan = float(y)
        self.k = float(z)

    def getGeoref(self):
        f = next(iter(self.files.values()))
        site = f.by_type('IfcSite')[0]
        object_placement = site[5]
        relative_placement = object_placement[1]
        location = relative_placement[0]
        ref_direction = relative_placement[2]
        if location != None and ref_direction != None:
            return {"location": (location[0], location[1], location[2]), "direction": ref_direction[0]}
        else:
            None
