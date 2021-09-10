import fiona
import ifcopenshell
import ifcopenshell.util
import math
import numpy as np

from ifcopenshell.util.selector import Selector
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point
from scipy.interpolate import LinearNDInterpolator
from shapely.strtree import STRtree
from sympy import atan


def is_float_str(str):
    try:
        float(str)
        return True
    except ValueError:
        return False


def read_height_points(shape_file):
    height_points = []
    with fiona.open(shape_file) as records:
        for record in records:
            point_id = record['properties']['PUNTID']
            if point_id == 0:
                pass
            else:
                height_point = (record['properties']['XCOORDINAA'], record['properties']['YCOORDINAA'])
                height = record['properties']['PEILWAARDE']
                height_points.append((height_point[0], height_point[1], height))
    return np.array(height_points)


def get_tin_interpolation(height_points):
    x = height_points[:, 0]
    y = height_points[:, 1]
    z = height_points[:, 2]
    interp = LinearNDInterpolator(list(zip(x, y)), z)
    return interp


def read_asc_file(file):
    with open(file, 'r') as f:
        data = f.read()
    lines = data.splitlines()
    points = {}
    for line in lines:
        item = line.split()
        if is_float_str(item[3]):
            points[item[0]] = (float(item[1]), float(item[2]), float(item[3]))
        else:
            points[item[0]] = (float(item[1]), float(item[2]))
    return points


def read_parcel_gpkg(file):
    parcel_geom = None
    with fiona.open(file) as records:
        for record in records:
            parcel_geom = Polygon(record['geometry']['coordinates'][0][0])
    return parcel_geom


def get_close_geopoints(points, parcel, buffer=100):
    list_points_2d = {}
    for i, point in points.items():
        list_points_2d[i] = Point([point[0], point[1]])
    list_points = list(list_points_2d.values())
    # Create shapely STRtree
    tree_points = STRtree(list_points)
    # map the geometry back to its id
    temp_dict = {}
    for j, pt in list_points_2d.items():
        temp_dict[id(pt)] = j
    query_geom = parcel.buffer(buffer)
    chosen_points = tree_points.query(query_geom)
    dict_chosen_points = {}
    for pt in chosen_points:
        dict_chosen_points[temp_dict[id(pt)]] = points[temp_dict[id(pt)]]
    return dict_chosen_points


def global_to_local(point, origin_point, true_north, scale=1):
    new_x = point[0] - origin_point[0]
    new_y = point[1] - origin_point[1]
    rotation_angle = - np.arctan2(true_north[1], true_north[0])
    a = scale * np.cos(rotation_angle)
    b = scale * np.sin(rotation_angle)
    x = (a * new_x) - (b * new_y)
    y = (b * new_x) + (a * new_y)
    z = point[2] - origin_point[2]
    return x, y, z


def write_txt_file(points, output_file):
    with open(output_file, 'w') as f:
        for i, point in points.items():
            f.write('{0} {1} {2} {3}\n'.format(i, point[0], point[1], point[2]))


def read_txt_file(in_file):
    with open(in_file, 'r') as f:
        data = f.read()
    lines = data.splitlines()
    points = {}
    for line in lines:
        item = line.split()
        points[item[0]] = (float(item[1]), float(item[2]), float(item[3]))
    return points


def create_initial_ifc_file(geo_points, parcel, height_pts, output_ifc_file):
    """
        creates the initial IFC file with local reference points and the parcel limit
        it also writes a text file to link the global reference points to the local ones in the IFC file
    """
    reference_points = get_close_geopoints(geo_points, parcel)
    # Make all points 3D
    interp_tin = get_tin_interpolation(height_pts)
    for idx, reference_point in reference_points.items():
        if len(reference_point) == 2:
            X, Y = reference_point[0], reference_point[1]
            Z = interp_tin(X, Y)
            reference_points[idx] = np.array([X, Y, Z])
    write_txt_file(reference_points, 'reference_3d_points.txt')

    # get side of parcel for global to local reference
    x_parcel_points, y_parcel_points = parcel.exterior.coords.xy
    z_parcel_points = interp_tin(x_parcel_points, y_parcel_points)
    chosen_origin_point = np.array([x_parcel_points[0], y_parcel_points[0], z_parcel_points[0]])
    chosen_second_point = np.array([x_parcel_points[1], y_parcel_points[1], z_parcel_points[1]])
    side_to_straighten = chosen_second_point - chosen_origin_point

    ifc = ifcopenshell.open('empty_2022.ifc')
    # generate IFC reference points
    for k, r_point in reference_points.items():
        local_r_point = global_to_local(r_point, chosen_origin_point, side_to_straighten)
        new_point = ifc.createIfcCartesianPoint(
            (float(local_r_point[0]), float(local_r_point[1]), float(local_r_point[2])))
        new_proxy_point = ifc.createIfcProxy(GlobalId=k, Name='Reference points', Representation=new_point)

    # generate IFC parcel
    for c, coord in enumerate(x_parcel_points[:-1]):
        first_point = np.array([coord, y_parcel_points[c]])
        second_point = np.array([x_parcel_points[c + 1], y_parcel_points[c + 1]])
        segment_vector = second_point - first_point
        local_first_point = global_to_local(np.array([first_point[0], first_point[1], chosen_origin_point[2]]),
                                            chosen_origin_point, side_to_straighten)
        origin = ifc.createIfcCartesianPoint((float(local_first_point[0]), float(local_first_point[1]), 0.))
        vector = ifc.createIfcDirection((float(segment_vector[0]), float(segment_vector[1]), 0.))
        line = ifc.createIfcLine(origin, vector)
        new_proxy_line = ifc.createIfcProxy(GlobalId=ifcopenshell.guid.new(), Name='Parcel', Representation=line)

    ifc.write(output_ifc_file)


def georef_ifc(ifc_file, output_ifc_file):
    """
        Reads the local reference points from the IFC file and outputs a new IFC file with IFC Site correctly
        georeferenced
    """
    ifc = ifcopenshell.open(ifc_file)
    selector = Selector()
    local_reference_points = selector.parse(ifc, '.IfcProxy[Name *= "Reference points"]')
    global_reference_points = read_txt_file('reference_3d_points.txt')

    ref_points = {}
    for point in local_reference_points:
        point_coords = point.Representation.Coordinates
        point_id = point.GlobalId
        ref_points[point_id] = point_coords

    points_ids = list(ref_points.keys())
    local_point_00 = ref_points[points_ids[0]]
    global_point_00 = global_reference_points[points_ids[0]]
    local_point_01 = ref_points[points_ids[1]]
    global_point_01 = global_reference_points[points_ids[1]]
    local_point_02 = ref_points[points_ids[2]]
    global_point_02 = global_reference_points[points_ids[2]]
    A = np.array([[local_point_00[0], -local_point_00[1], 1], [local_point_01[0], -local_point_01[1], 1],
                  [local_point_02[0], -local_point_02[1], 1]])
    B = np.array([global_point_00[0], global_point_01[0], global_point_02[0]])
    X = np.linalg.inv(A).dot(B)
    N = global_point_00[1] - (X[1] * local_point_00[0]) - (X[0] * local_point_00[1])
    H = global_point_00[2] - local_point_00[2]
    R = atan(X[1] / X[0])
    R_cos = math.cos(R)
    R_sin = math.sin(R)
    scale = round((X[0] / R_cos), 2)
    t_north = ifc.createIfcDirection((R_cos, R_sin, 0.))
    translation = ifc.createIfcCartesianPoint((float(X[2]), float(N), float(H)))
    axis = ifc.createIfcDirection((0., 0., 1.))
    placement = ifc.createIfcAxis2Placement3D(translation, axis, t_north)
    local_placement = ifc.createIfcLocalPlacement(None, placement)

    site = ifc.createIfcSite(GlobalId=ifcopenshell.guid.new(), Name='Default', ObjectPlacement=local_placement,
                             CompositionType='ELEMENT', RefLatitude=(42, 24, 53, 508911),
                             RefLongitude=(-71, -15, -29, -58837),
                             RefElevation=0.)
    ifc.write(output_ifc_file)



'''geo_pts = read_asc_file('/www/models-preloaded/Grondslag_DB_export_26-03-20.asc')
parcel_limit = read_parcel_gpkg('/www/models-preloaded/BRON_ID_168822.gpkg')
height_pnts = read_height_points('/www/models-preloaded/UitgiftePeilen/Peil_punten.shp')
create_initial_ifc_file(geo_pts, parcel_limit, height_pnts, 'output_test.ifc')'''

'''input_file = 'output_test_01.ifc'
ifc = ifcopenshell.open(input_file)
selector = Selector()
local_reference_points = selector.parse(ifc, '.IfcProxy[Name *= "Reference points"]')
global_reference_points = read_txt_file('reference_2d_points.txt')

ref_points = {}
for point in local_reference_points:
    point_coords = point.Representation.Coordinates
    point_id = point.GlobalId
    ref_points[point_id] = point_coords

points_ids = list(ref_points.keys())
local_point_00 = ref_points[points_ids[0]]
global_point_00 = global_reference_points[points_ids[0]]
local_point_01 = ref_points[points_ids[1]]
global_point_01 = global_reference_points[points_ids[1]]
local_point_02 = ref_points[points_ids[2]]
global_point_02 = global_reference_points[points_ids[2]]
A = np.array([[local_point_00[0], -local_point_00[1], 1], [local_point_01[0], -local_point_01[1], 1],
              [local_point_02[0], -local_point_02[1], 1]])
B = np.array([global_point_00[0], global_point_01[0], global_point_02[0]])
X = np.linalg.inv(A).dot(B)
N = global_point_00[1] - (X[1] * local_point_00[0]) - (X[0] * local_point_00[1])
H = global_point_00[2] - local_point_00[2]
R = atan(X[1] / X[0])
R_cos = math.cos(R)
R_sin = math.sin(R)
scale = round((X[0] / R_cos), 2)
t_north = ifc.createIfcDirection((R_cos, R_sin, 0.))
translation = ifc.createIfcCartesianPoint((float(X[2]), float(N), float(H)))
axis = ifc.createIfcDirection((0., 0., 1.))
placement = ifc.createIfcAxis2Placement3D(translation, axis, t_north)
local_placement = ifc.createIfcLocalPlacement(None, placement)

site = ifc.createIfcSite(GlobalId=ifcopenshell.guid.new(), Name='Default', ObjectPlacement=local_placement,
                         CompositionType='ELEMENT', RefLatitude=(42, 24, 53, 508911),
                         RefLongitude=(-71, -15, -29, -58837),
                         RefElevation=0.)'''
test = 0
