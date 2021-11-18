import fiona
import numpy as np
import pyvista as pv
import re

from .boundingBox import BoundingBox
from .orientedBoundingBox import OrientedBoundingBox
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay, ConvexHull
from shapely.geometry import shape, Polygon, LineString, Point
from shapely.strtree import STRtree
from .functions import *
from ifcopenshell import open as open_ifc_file


def check_boundary(parcel_limit, bldg_limit):
    """ checks if the building respects the parcel's boundary
    input:
    parcel_limit: POLYGON
    bldg_limit: POLYGON
    output: string Pass or Fail, parcel wkt, ground floor wkt
    """
    check = parcel_limit.contains(bldg_limit)
    if check:
        return { "result": "Pass", "parcel_limit": parcel_limit.wkt, "bldg_limit": bldg_limit.wkt }
    else:
        return { "result": "Fail", "parcel_limit": parcel_limit.wkt,"bldg_limit": bldg_limit.wkt }


def shapefile_to_shapely_roads(shape_file):
    roads_geom = {}
    roads_name = {}
    with fiona.open(shape_file) as records:
        for record in records:
            road_id = record['id']
            road_name = record['properties']['STRAAT']
            poly_geom = Polygon(record['geometry']['coordinates'][0])
            roads_geom[road_id] = poly_geom
            roads_name[road_id] = road_name
    return roads_geom, roads_name


def shapefile_to_shapely_parcels(shape_file):
    parcels_geom = []
    with fiona.open(shape_file) as records:
        for record in records:
            poly_geom = Polygon(record['geometry']['coordinates'][0])
            parcels_geom.append(poly_geom)
    return parcels_geom


def get_close_roads(roads_geom, parcel_limit, buffer=500):
    """
    Put roads in a tree and query roads in 500 m buffer
    """
    list_roads = roads_geom.values()
    # Create shapely STRtree
    tree_roads = STRtree(list_roads)
    # map the geometry back to its id
    temp_dict = {}
    for i, road in roads_geom.items():
        temp_dict[id(road)] = i
    query_geom = parcel_limit.buffer(buffer)  # 500 m buffer around parcel can be other value
    chosen_roads = tree_roads.query(query_geom)
    dict_chosen_roads = {}
    for road in chosen_roads:
        dict_chosen_roads[temp_dict[id(road)]] = road
    return dict_chosen_roads


def side_to_road(roads, bbox):
    """
    for each side of the building get respective road
    input:
    roads: dictionary of Polygons
    bbox: orientedBoundingBox instance
    output:
    dictionary: each side and its respective road
    """
    # for each side get line-normal check road intersection, intersection should be CCW and closest one
    sides = bbox.vertical_sides
    centers, extensions = bbox.get_normal_line()
    centroids = bbox.get_centroid_horizontal()
    # change points from 3D to 2D
    centers = centers[:, [0, 1]]
    extensions = extensions[:, [0, 1]]
    centroid = centroids[:, [0, 1]][0]
    # change to shapely LineString
    shapely_normals = {}
    for i, center in enumerate(centers):
        line = LineString([centroid, extensions[i]])
        shapely_normals[i] = line
    normal_to_road = {}
    for n_id in shapely_normals.keys():
        normal = shapely_normals[n_id]
        distance_to_road = 0
        for road_id in roads.keys():
            road = roads[road_id]
            if normal.intersects(road):
                # if the normal is too short it might not intersect the road find a better a way ensure direction of
                # normal
                line_intersection = normal.intersection(road)
                if line_intersection.type == "MultiLineString":
                    for line in line_intersection:
                        points_intersection = list(line.coords)
                        points_of_normal = list(normal.coords)
                        pt0 = np.array(points_of_normal[0])
                        pt1 = np.array(points_of_normal[1])
                        for point in points_intersection:
                            pt_intersection = np.array(point)
                            # make sure it is pointing away from the bounding box
                            v0 = pt1 - pt0
                            v1 = pt_intersection - pt0
                            check_alignment = np.dot(v0, v1)
                            if check_alignment > 0:
                                # choose closest point
                                dist = np.linalg.norm(v1)
                                if distance_to_road == 0:
                                    normal_to_road[n_id] = road_id
                                    distance_to_road = dist
                                elif dist < distance_to_road:
                                    normal_to_road[n_id] = road_id
                                    distance_to_road = dist
                else:
                    points_intersection = list(line_intersection.coords)
                    points_of_normal = list(normal.coords)
                    pt0 = np.array(points_of_normal[0])
                    pt1 = np.array(points_of_normal[1])
                    for point in points_intersection:
                        pt_intersection = np.array(point)
                        # make sure it is pointing away from the bounding box
                        v0 = pt1 - pt0
                        v1 = pt_intersection - pt0
                        check_alignment = np.dot(v0, v1)
                        if check_alignment > 0:
                            # choose closest point
                            dist = np.linalg.norm(v1)
                            if distance_to_road == 0:
                                normal_to_road[n_id] = road_id
                                distance_to_road = dist
                            elif dist < distance_to_road:
                                normal_to_road[n_id] = road_id
                                distance_to_road = dist
    return normal_to_road


def check_overhang(groundfloor, sides_to_road, sides, roads_name, guideline):
    check = {}
    for side, road_id in sides_to_road.items():
        # get the part of the line that passes the parcel
        side_line_3d = sides[side][0:2]
        temp_line = np.array(side_line_3d)[:, :2]
        side_line_2d = LineString(temp_line.tolist())
        outside_line = side_line_2d.difference(groundfloor)
        # get distance to parcel
        furthest_pt = []
        check_dist = 0
        road_name = roads_name[road_id]
        admissible_overhang = guideline[road_name]
        if outside_line.wkt == 'LINESTRING EMPTY':
            check[road_name] = ("Pass", "Admissible overhang: " + str(admissible_overhang),
                                "Overhang: " + "No overhang", side_line_2d.wkt, road_id)
        else:
            if outside_line.type == 'MultiLineString':
                for line in outside_line:
                    for point in line.coords:
                        pt = Point(point)
                        dist_to_gf = groundfloor.exterior.distance(pt)
                        if dist_to_gf >= check_dist:
                            check_dist = dist_to_gf
                            furthest_pt = pt
                    if admissible_overhang > check_dist:
                        check[road_name] = ("Pass", "Admissible overhang: " + str(admissible_overhang),
                                            "Overhang: " + str(check_dist), side_line_2d.wkt, road_id)
                    else:
                        check[road_name] = ("Fail", "Admissible overhang: " + str(admissible_overhang),
                                            "Overhang: " + str(check_dist), side_line_2d.wkt, road_id)
            else:
                for point in outside_line.coords:
                    pt = Point(point)
                    dist_to_gf = groundfloor.exterior.distance(pt)
                    if dist_to_gf >= check_dist:
                        check_dist = dist_to_gf
                        furthest_pt = pt
                if admissible_overhang > check_dist:
                    check[road_name] = ("Pass", "Admissible overhang: " + str(admissible_overhang),
                                        "Overhang: " + str(check_dist), side_line_2d.wkt, road_id)
                else:
                    check[road_name] = ("Fail", "Admissible overhang: " + str(admissible_overhang),
                                        "Overhang: " + str(check_dist), side_line_2d.wkt, road_id)
    return check


def get_geometry_unchecked_sides(box, sides_facing_roads):
    sides_not_facing_roads = {}
    for i, side in enumerate(box.vertical_sides):
        pt_00 = side[0][:2]
        pt_01 = side[1][:2]
        if i not in sides_facing_roads.keys():
            sides_not_facing_roads[i] = LineString([pt_00, pt_01]).wkt
    return sides_not_facing_roads


def inscribed_r(convex_tr):
    a_00 = convex_tr[:, 1] - convex_tr[:, 0]
    b_00 = convex_tr[:, 2] - convex_tr[:, 0]
    c_00 = convex_tr[:, 2] - convex_tr[:, 1]
    area_2 = np.linalg.norm(np.cross(a_00, b_00), axis=1)
    perim = np.linalg.norm(a_00, axis=1) + np.linalg.norm(b_00, axis=1) + np.linalg.norm(c_00, axis=1)
    r = area_2 / perim
    return r


def alpha_shape_3D(pts, alpha):
    """
    Compute the alpha shape (concave hull) of a set of 3D points.
    Parameters:
        pos - np.array of shape (n,3) points.
        alpha - alpha value.
    return
        outer surface vertex indices, edge indices, and triangle indices
    """
    # get all 3D Delaunay triangles
    tetra = Delaunay(pts)
    tetras_i = tetra.vertices
    tricomb = np.array([(2, 1, 0), (0, 1, 3), (0, 3, 2), (1, 2, 3)])
    all_tr_00 = tetras_i[:, tricomb].reshape(-1, 3)
    all_ = pts[all_tr_00]

    # check which triangles of the convex hull are less than alpha
    r = inscribed_r(all_)

    tr_check_i = all_tr_00[np.where(r < alpha)]
    # tr_sort_i = tr_check_i[np.arange(np.shape(tr_check_i)[0])[:, np.newaxis], np.argsort(tr_check_i)]
    tr_chosen, count = np.unique(tr_check_i, axis=0, return_counts=True)
    hull_tr = tr_chosen[np.where(count == 1)]

    return hull_tr


def read_height_points(shape_file):
    """
    creates dictionary of (x, y) points and their heights
    """
    height_points = []
    heights = []
    with fiona.open(shape_file) as records:
        for record in records:
            point_id = record['properties']['PUNTID']
            if point_id == 0:
                pass
            else:
                height_point = (record['properties']['XCOORDINAA'], record['properties']['YCOORDINAA'])
                height = record['properties']['PEILWAARDE']
                height_points.append(height_point)
                heights.append(height)
    return height_points, heights


def get_height_parcel(parcel_points, points, heights):
    """
    Extracts the height of each point that defines the parcel
    parcel points: points that define the parcel
    points: points from dataset
    heights: height os 'points' from dataset
    """
    point_tree = cKDTree(points)
    # distance and index to nearest point
    dd, ii = point_tree.query(parcel_points, k=1)
    parcel_heights = np.array(heights)[ii]
    return parcel_heights


def check_height(entrance_height, building_height, parcel_heights, maximum_allowed_height):
    """
    checks that the height of the building and the highest parcel point respect the allowed height
    """
    max_parcel_height = np.amax(parcel_heights)
    height_difference = entrance_height - max_parcel_height
    height_to_check = building_height + height_difference
    if height_to_check <= maximum_allowed_height:
        return ("Pass", "Admissible height: " + str(maximum_allowed_height), "Height: " + str(height_to_check)), \
               max_parcel_height
    else:
        return ("Fail", "Admissible height: " + str(maximum_allowed_height), "Height: " + str(height_to_check)), \
               max_parcel_height


def get_gf_floor_idx(storeys_names):
    for i, name in enumerate(storeys_names):
        if name[:2] == '00':
            return i


def get_first_floor_idx(storeys_names):
    for i, name in enumerate(storeys_names):
        if name[:2] == '01':
            return i


def get_georeference(ifc_file):
    origin_point = ()
    true_north = ()
    ifc_dict = {}
    with open(ifc_file, 'r') as f:
        for line in f:
            if line[0] == '#':
                splitting = line.split()
                ifc_dict[splitting[0][:-1]] = splitting[1]
                check_line = splitting[1].split('(')
                if check_line[0] == 'IFCSITE':
                    first_idx = check_line[1].split(',')[5]
                    second_idx = ifc_dict[first_idx].split('(')[1][2:-2]
                    third_idx = ifc_dict[second_idx].split('(')
                    point_idx = third_idx[1].split(',')[0]
                    direction_idx = third_idx[1].split(',')[2][:-2]
                    point = ifc_dict[point_idx].split('(')[2].split(',')
                    origin_point = (float(point[0]), float(point[1]), float(point[2][:-2]))
                    # direction = ifc_dict[direction_idx].split('(')[1].split(',')
                    direction = ifc_dict[direction_idx].split('(')[2].split(',')
                    true_north = (float(direction[0]), float(direction[1]))
                    break
    return np.array(origin_point), np.array(true_north)


def get_georeferenced_point(point, origin_point, true_north):
    rotation_angle = np.arctan2(true_north[1], true_north[0])
    newX = point[0] * np.cos(rotation_angle) - point[1] * np.sin(rotation_angle)
    newY = point[0] * np.sin(rotation_angle) + point[1] * np.cos(rotation_angle)
    rotated_point = np.array([newX, newY])
    translated_point = rotated_point + origin_point[:2]
    return translated_point[0], translated_point[1]


def get_parcel(centroid, parcels):
    chosen_parcels = []
    for parcel in parcels:
        if parcel.contains(centroid):
            chosen_parcels.append(parcel)
    # get smallest polygon
    chosen_parcel = chosen_parcels[0]
    if len(chosen_parcels) > 0:
        for parcel in chosen_parcels[1:]:
            if parcel.within(chosen_parcel):
                chosen_parcel = parcel
    return chosen_parcel


def get_boundary_z_min_z_max(elements, origin_pt, true_n):
    lst_x, lst_y, lst_z = GetAllCoordinates(elements)
    points_2d = np.array(list(zip(lst_x, lst_y)))
    obb_2d = GetNumpyOBB(points_2d, show_plot=False)
    z_min = min(lst_z) + origin_pt[2]
    z_max = max(lst_z) + origin_pt[2]
    bb_2d = [(min(lst_x), min(lst_y)), (max(lst_x), min(lst_y)), (max(lst_x), max(lst_y)), (min(lst_x), max(lst_y))]
    bb_2d_georef = []
    for pt_2d in bb_2d:
        pt_2d_georef = get_georeferenced_point(pt_2d, origin_pt, true_n)
        bb_2d_georef.append(pt_2d_georef)
    obb_2d_georef = []
    for pt_2d in obb_2d:
        pt_2d_georef = get_georeferenced_point(pt_2d, origin_pt, true_n)
        obb_2d_georef.append(pt_2d_georef)
    if Polygon(obb_2d_georef).area > Polygon(bb_2d_georef).area:
        return bb_2d_georef, z_min, z_max
    else:
        return obb_2d_georef, z_min, z_max


def run_overhang_check(guidelines, all_storeys_elements, all_storeys_names, origin_pt, true_n, storey_number):
    # get GF
    gf_floor_idx = get_gf_floor_idx(all_storeys_names)
    gf, z_min, z_max = get_boundary_z_min_z_max(all_storeys_elements[gf_floor_idx], origin_pt, true_n)
    gf = Polygon(gf)
    # Get the parcel of project
    centroid_gf = gf.centroid
    parcels = shapefile_to_shapely_parcels("/www/models-preloaded/BRK_SelectieCentrum.shp")
    parcel = get_parcel(centroid_gf, parcels)
    # Remove the underground floors and GF from the list of floors OR get designated floor number
    first_floor_idx = get_first_floor_idx(all_storeys_names)
    all_storeys_elements = all_storeys_elements[first_floor_idx:]
    all_storeys_names = all_storeys_names[first_floor_idx:]
    if storey_number is not None:
        all_storeys_elements = [all_storeys_elements[storey_number - 1]]
        all_storeys_names = all_storeys_names[storey_number - 1]
    # roads
    geom_roads, name_roads = shapefile_to_shapely_roads("/www/models-preloaded/Wegvakonderdelen.shp")
    close_roads = get_close_roads(geom_roads, gf)
    # get roads to test against
    query_parcel = parcel.buffer(1)  # 1 m buffer around parcel can be other value
    adjacent_roads = {}
    for i, road in close_roads.items():
        if query_parcel.intersects(road):
            adjacent_roads[i] = road
    lst_all_checks = []
    lst_all_rogues = []
    # i=0
    for storey_elements in all_storeys_elements:
        # if i == 4:
        #    break
        # print(i)
        storey, z_min_s, z_max_s = get_boundary_z_min_z_max(storey_elements, origin_pt, true_n)
        corner_00 = storey[0]
        corner_01 = storey[1]
        corner_02 = storey[2]
        corner_03 = storey[3]
        simplified_box = OrientedBoundingBox([corner_00[0], corner_00[1], z_min_s],
                                             [corner_01[0], corner_01[1], z_min_s],
                                             [corner_02[0], corner_02[1], z_min_s],
                                             [corner_03[0], corner_03[1], z_min_s],
                                             [corner_00[0], corner_00[1], z_max_s],
                                             [corner_01[0], corner_01[1], z_max_s],
                                             [corner_02[0], corner_02[1], z_max_s],
                                             [corner_03[0], corner_03[1], z_max_s])
        sides_roads = side_to_road(adjacent_roads, simplified_box)
        check = check_overhang(gf, sides_roads, simplified_box.vertical_sides, name_roads, guidelines)
        rogue_sides = get_geometry_unchecked_sides(simplified_box, sides_roads)
        lst_all_checks.append(check)
        lst_all_rogues.append(rogue_sides)
        # i+=1

    return { "all_checks": lst_all_checks, "rogue_checks": lst_all_rogues }


def run_height_check(guidelines, all_storeys_elements, all_storeys_names, origin_pt, true_n,):
    # get GF
    gf_floor_idx = get_gf_floor_idx(all_storeys_names)
    gf, z_min, z_max = get_boundary_z_min_z_max(all_storeys_elements[gf_floor_idx], origin_pt, true_n)
    gf = Polygon(gf)
    z_min = z_min + origin_pt[2]
    # Get the parcel of project
    centroid_gf = gf.centroid
    parcels = shapefile_to_shapely_parcels("/www/models-preloaded/BRK_SelectieCentrum.shp")
    parcel = get_parcel(centroid_gf, parcels)
    parcel_points = list(zip(*parcel.exterior.coords.xy))
    points, heights = read_height_points("/www/models-preloaded/Peil_punten.shp")
    parcel_heights = get_height_parcel(parcel_points, points, heights)
    # get last floor
    lst_x_last, lst_y_last, lst_z_last = GetAllCoordinates(all_storeys_elements[-1])
    z_max = max(lst_z_last) + origin_pt[2]
    building_height = z_max - z_min
    height_check, highest_road = check_height(z_min, building_height, parcel_heights, guidelines)
    buffer_gf = gf.buffer(20)
    return { "height_check": height_check, "guidelines": guidelines, "highest_road": highest_road, "wkt": buffer_gf.wkt }  # make sure that guidelines is the maximum allowed height


def run_boundary_check(all_storeys_elements, all_storeys_names, origin_pt, true_n,):
    # get GF
    gf_floor_idx = get_gf_floor_idx(all_storeys_names)
    gf, z_min, z_max = get_boundary_z_min_z_max(all_storeys_elements[gf_floor_idx], origin_pt, true_n)
    gf = Polygon(gf)
    # Get the parcel of project
    centroid_gf = gf.centroid
    parcels = shapefile_to_shapely_parcels("/www/models-preloaded/BRK_SelectieCentrum.shp")
    parcel = get_parcel(centroid_gf, parcels)
    boundary_check = check_boundary(parcel, gf)  # test result, parcel polygon wkt, gf polygon wkt
    return boundary_check


def get_tetra_volume(tetrapts):
    pts_num = tetrapts.shape[0]
    c_d = np.reshape((tetrapts[:, [2]] - tetrapts[:, [3]]).flatten(), (pts_num, 3))
    b_d = np.reshape((tetrapts[:, [1]] - tetrapts[:, [3]]).flatten(), (pts_num, 3))
    a_d = np.reshape((tetrapts[:, [0]] - tetrapts[:, [3]]).flatten(), (pts_num, 3))
    cross = np.cross(b_d, c_d)
    dot = np.einsum("ij,ij->i", a_d, cross)
    return np.absolute(np.array(dot)) / 6


def get_tetra_surface(tetrapts):
    pts_num = tetrapts.shape[0]
    c_d = np.reshape((tetrapts[:, [2]] - tetrapts[:, [3]]).flatten(), (pts_num, 3))
    b_d = np.reshape((tetrapts[:, [1]] - tetrapts[:, [3]]).flatten(), (pts_num, 3))
    a_d = np.reshape((tetrapts[:, [0]] - tetrapts[:, [3]]).flatten(), (pts_num, 3))
    a_c = np.reshape((tetrapts[:, [0]] - tetrapts[:, [2]]).flatten(), (pts_num, 3))
    b_c = np.reshape((tetrapts[:, [1]] - tetrapts[:, [2]]).flatten(), (pts_num, 3))
    tr_00 = 0.5 * np.linalg.norm(np.cross(a_c, b_c), axis=-1)
    tr_01 = 0.5 * np.linalg.norm(np.cross(a_d, b_d), axis=-1)
    tr_02 = 0.5 * np.linalg.norm(np.cross(a_d, c_d), axis=-1)
    tr_03 = 0.5 * np.linalg.norm(np.cross(b_d, c_d), axis=-1)
    return tr_00 + tr_01 + tr_02 + tr_03


def get_edges_distance(tetrapts):
    pts_num = tetrapts.shape[0]
    c_d = np.reshape((tetrapts[:, [2]] - tetrapts[:, [3]]).flatten(), (pts_num, 3))
    b_d = np.reshape((tetrapts[:, [1]] - tetrapts[:, [3]]).flatten(), (pts_num, 3))
    a_d = np.reshape((tetrapts[:, [0]] - tetrapts[:, [3]]).flatten(), (pts_num, 3))
    a_c = np.reshape((tetrapts[:, [0]] - tetrapts[:, [2]]).flatten(), (pts_num, 3))
    b_c = np.reshape((tetrapts[:, [1]] - tetrapts[:, [2]]).flatten(), (pts_num, 3))
    a_b = np.reshape((tetrapts[:, [0]] - tetrapts[:, [1]]).flatten(), (pts_num, 3))
    c_d_length = np.linalg.norm(c_d, axis=-1)
    b_d_length = np.linalg.norm(b_d, axis=-1)
    a_d_length = np.linalg.norm(a_d, axis=-1)
    a_c_length = np.linalg.norm(a_c, axis=-1)
    b_c_length = np.linalg.norm(b_c, axis=-1)
    a_b_length = np.linalg.norm(a_b, axis=-1)
    result = np.sqrt((((c_d_length * a_b_length) + (b_d_length * a_c_length) + (a_d_length * b_c_length)) *
                      ((c_d_length * a_b_length) + (b_d_length * a_c_length) - (a_d_length * b_c_length)) *
                      ((c_d_length * a_b_length) - (b_d_length * a_c_length) + (a_d_length * b_c_length)) *
                      (-(c_d_length * a_b_length) + (b_d_length * a_c_length) + (a_d_length * b_c_length))))
    return result


def get_alpha_shape_3d(pts, alpha):
    epsilon = 10 ** (-3)
    # pts = np.unique(pts, axis=0)
    # get all 3D Delaunay triangles
    tetra = Delaunay(pts, incremental=True)
    # get tetrahedrals, volume and edges distance
    tetrapts = np.take(tetra.points, tetra.vertices, axis=0)
    volume = get_tetra_volume(tetrapts)
    # get the circumradii of non-flat tetrahedral by masking where volume < epsilon
    edges = get_edges_distance(tetrapts)
    r = np.ma.masked_where(volume < epsilon, edges) / (24 * np.ma.masked_where(volume < epsilon, volume))

    # tetrahedrals
    chosen_tetras = np.where(r < alpha)
    tetras = tetra.vertices[chosen_tetras]
    # fix tetrahedrals orientation
    chosen_tetrapts = np.take(tetra.points, tetras, axis=0)
    base = chosen_tetrapts[:, [0, 1, 2]]
    apex = chosen_tetrapts[:, 3]
    normal_base = np.cross((base[:, 1] - base[:, 0]), (base[:, 2] - base[:, 0]))
    apex_vertex = apex - base[:, 0]
    dot_product = np.einsum("ij,ij->i", normal_base, apex_vertex)
    correct_tetras = np.where(dot_product < 0)
    flipped_tetras = np.where(dot_product > 0)
    # triangles
    tricomb_correct = np.array([(0, 1, 2), (1, 0, 3), (0, 2, 3), (2, 1, 3)])
    triangles_correct = tetras[correct_tetras][:, tricomb_correct].reshape(-1, 3)
    tricomb_flipped = np.array([(0, 2, 1), (0, 1, 3), (2, 0, 3), (1, 2, 3)])
    triangles_flipped = tetras[flipped_tetras][:, tricomb_flipped].reshape(-1, 3)
    triangles = np.concatenate((triangles_correct, triangles_flipped), axis=0)
    triangles_sorted = np.sort(triangles, axis=1)
    # Remove triangles that occurs twice, because they are within shapes
    trianglesdict = defaultdict(int)
    for tri in triangles_sorted:
        trianglesdict[tuple(tri)] += 1
    final_triangles = []
    for tri in triangles:
        possible_tris = ((tri[0], tri[1], tri[2]), (tri[0], tri[2], tri[1]), (tri[1], tri[0], tri[2]),
                         (tri[1], tri[2], tri[0]), (tri[2], tri[0], tri[1]), (tri[2], tri[1], tri[0]))
        for p_tri in possible_tris:
            if trianglesdict[p_tri] and trianglesdict[p_tri] == 1:
                final_triangles.append(tri)
                break
    final_triangles = np.array(final_triangles)
    return tetra.points, final_triangles


def populate_surface_with_points(face):
    space = 1
    face_x = face[:, 0]
    div_x = (np.max(face_x) - np.min(face_x)) // space
    face_y = face[:, 1]
    div_y = (np.max(face_y) - np.min(face_y)) // space
    face_z = face[:, 2]
    div_z = (np.max(face_z) - np.min(face_z)) // space
    if div_z == 0:
        div_z = 1
    x_ = np.linspace(np.min(face_x), np.max(face_x), int(div_x + 1))
    y_ = np.linspace(np.min(face_y), np.max(face_y), int(div_y + 1))
    z_ = np.linspace(np.min(face_z), np.max(face_z), int(div_z + 1))
    x, y, z = np.meshgrid(x_, y_, z_)
    x = x.reshape((np.prod(x.shape),))
    y = y.reshape((np.prod(y.shape),))
    z = z.reshape((np.prod(z.shape),))
    return list(zip(x, y, z))


def get_elements_surfaces_points(ifc_elements):
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_PYTHON_OPENCASCADE, True)
    settings.set(settings.USE_WORLD_COORDS, True)
    points_on_faces = []
    if isinstance(ifc_elements, list):
        print("load ifc elements list")
        for element in ifc_elements:
            if element.Representation:
                shape = ifcopenshell.geom.create_shape(settings, element)
                exp_face = OCC.Core.TopExp.TopExp_Explorer(shape.geometry, OCC.Core.TopAbs.TopAbs_FACE)
                while exp_face.More():
                    face = OCC.Core.TopoDS.topods_Face(exp_face.Current())
                    face_nurbs_converter = OCC.Core.BRepBuilderAPI.BRepBuilderAPI_NurbsConvert(face)
                    face_nurbs_converter.Perform(face)
                    nurbs_face = face_nurbs_converter.Shape()
                    brep_face = BRep_Tool.Surface(OCC.Core.TopoDS.topods_Face(nurbs_face))
                    bspline_face = geomconvert_SurfaceToBSplineSurface(brep_face)
                    # extract the Control Points of each face
                    n_poles_u = bspline_face.NbUPoles()
                    n_poles_v = bspline_face.NbVPoles()
                    control_polygon_coordinates = np.zeros(
                        shape=(n_poles_u * n_poles_v, 3)
                    )
                    # cycle over the poles to get their coordinates
                    i = 0
                    for pole_u_direction in range(n_poles_u):
                        for pole_v_direction in range(n_poles_v):
                            control_point_coordinates = bspline_face.Pole(
                                pole_u_direction + 1, pole_v_direction + 1
                            )
                            control_polygon_coordinates[i, :] = [control_point_coordinates.X(),
                                                                 control_point_coordinates.Y(),
                                                                 control_point_coordinates.Z()]
                            i += 1
                    populated_face = populate_surface_with_points(control_polygon_coordinates)
                    points_on_faces = points_on_faces + populated_face
                    exp_face.Next()
            else:
                print(str(element.id()) + "No Rrepresentation")
    else:
        if ifc_elements.Representation:
            shape = ifcopenshell.geom.create_shape(settings, ifc_elements)
            exp_face = OCC.Core.TopExp.TopExp_Explorer(shape.geometry, OCC.Core.TopAbs.TopAbs_FACE)
            while exp_face.More():
                face = OCC.Core.TopoDS.topods_Face(exp_face.Current())
                face_nurbs_converter = OCC.Core.BRepBuilderAPI.BRepBuilderAPI_NurbsConvert(face)
                face_nurbs_converter.Perform(face)
                nurbs_face = face_nurbs_converter.Shape()
                brep_face = BRep_Tool.Surface(OCC.Core.TopoDS.topods_Face(nurbs_face))
                bspline_face = geomconvert_SurfaceToBSplineSurface(brep_face)
                # extract the Control Points of each face
                n_poles_u = bspline_face.NbUPoles()
                n_poles_v = bspline_face.NbVPoles()
                control_polygon_coordinates = np.zeros(
                    shape=(n_poles_u * n_poles_v, 3)
                )
                # cycle over the poles to get their coordinates
                i = 0
                for pole_u_direction in range(n_poles_u):
                    for pole_v_direction in range(n_poles_v):
                        control_point_coordinates = bspline_face.Pole(
                            pole_u_direction + 1, pole_v_direction + 1
                        )
                        control_polygon_coordinates[i, :] = [control_point_coordinates.X(),
                                                             control_point_coordinates.Y(),
                                                             control_point_coordinates.Z()]
                        i += 1
                populated_face = populate_surface_with_points(control_polygon_coordinates)
                points_on_faces = points_on_faces + populated_face
                exp_face.Next()
        else:
            print(str(ifc_elements.id()) + "No Rrepresentation")
    return points_on_faces


def get_boundary_z_min_z_max_no_georef(elements):
    lst_x, lst_y, lst_z = GetAllCoordinates(elements)
    points_2d = np.array(list(zip(lst_x, lst_y)))
    obb_2d = GetNumpyOBB(points_2d, show_plot=False)
    z_min = min(lst_z)
    z_max = max(lst_z)
    bb_2d = [(min(lst_x), min(lst_y)), (max(lst_x), min(lst_y)), (max(lst_x), max(lst_y)), (min(lst_x), max(lst_y))]
    if Polygon(obb_2d).area > Polygon(bb_2d).area:
        return bb_2d, z_min, z_max
    else:
        return obb_2d, z_min, z_max


def face_to_road(roads, origin_normal, extension):
    """
    get respective road of a face
    input:
    roads: dictionary of Polygons
    normal_face: np.array(x, y, z)
    origin_normal: np.array(x, y, z)
    output:
    face and its respective road
    """
    # change normal to shapely LineString
    normal = LineString([origin_normal, extension])
    distance_to_road = 0
    normal_to_road = None
    for road_id in roads.keys():
        road = roads[road_id]
        if normal.intersects(road):
            # if the normal is too short it might not intersect the road find a better a way ensure direction of
            # normal
            line_intersection = normal.intersection(road)
            if line_intersection.type == "MultiLineString":
                for line in line_intersection:
                    points_intersection = list(line.coords)
                    points_of_normal = list(normal.coords)
                    pt0 = np.array(points_of_normal[0])
                    pt1 = np.array(points_of_normal[1])
                    for point in points_intersection:
                        pt_intersection = np.array(point)
                        # make sure it is pointing away from the bounding box
                        v0 = pt1 - pt0
                        v1 = pt_intersection - pt0
                        check_alignment = np.dot(v0, v1)
                        if check_alignment > 0:
                            # choose closest point
                            dist = np.linalg.norm(v1)
                            if distance_to_road == 0:
                                normal_to_road = road_id
                                distance_to_road = dist
                            elif dist < distance_to_road:
                                normal_to_road = road_id
                                distance_to_road = dist
            else:
                points_intersection = list(line_intersection.coords)
                points_of_normal = list(normal.coords)
                pt0 = np.array(points_of_normal[0])
                pt1 = np.array(points_of_normal[1])
                for point in points_intersection:
                    pt_intersection = np.array(point)
                    # make sure it is pointing away from the bounding box
                    v0 = pt1 - pt0
                    v1 = pt_intersection - pt0
                    check_alignment = np.dot(v0, v1)
                    if check_alignment > 0:
                        # choose closest point
                        dist = np.linalg.norm(v1)
                        if distance_to_road == 0:
                            normal_to_road = road_id
                            distance_to_road = dist
                        elif dist < distance_to_road:
                            normal_to_road = road_id
                            distance_to_road = dist
    return normal_to_road


def run_overhang_check_alpha_shape(guidelines, alpha, all_storeys_elements, all_storeys_names, origin_pt, true_n, storey_number):
    e = 10**(-8)
    # get roads guidelines
    guide_lines = guidelines
    # get GF 2d to later get the roads
    gf_floor_idx = get_gf_floor_idx(all_storeys_names)
    gf, z_min, z_max = get_boundary_z_min_z_max_no_georef(all_storeys_elements[gf_floor_idx])
    lst_x_last, lst_y_last, lst_z_last = GetAllCoordinates(all_storeys_elements[-1])
    lst_z_max = max(lst_z_last)
    gf = Polygon(gf)
    # get GF 2d for pyvista extrude
    points_2d = np.array(list(zip(*gf.exterior.coords.xy)))[:-1]
    size_points_2d = points_2d.shape[0]
    temp_array = np.full((size_points_2d, 1), z_min)
    points_3d = np.concatenate((points_2d, temp_array), axis=1)
    face = [size_points_2d + 1] + list(range(size_points_2d)) + [0]  # cell connectivity for a single cell
    polygon = pv.PolyData(points_3d, faces=face)

    # extrude along z
    gf_extruded = polygon.extrude((0, 0, (lst_z_max + 10)))
    gf_extruded.compute_normals(cell_normals=True, point_normals=True, flip_normals=True, inplace=True)
    # the definition of faces from a pv mesh is in a certain way, transform it into a dictionary
    point_on_faces = []
    center_faces = []
    idx = 0
    counter = 0
    for i, element in enumerate(gf_extruded.faces):
        if idx == 0:
            idx = element
            counter += element + 1
            continue
        if i - counter + 1 == 0:
            idx = 0
            point_on_faces.append(gf_extruded.points[element])
            face_center = (gf_extruded.points[element] + gf_extruded.points[gf_extruded.faces[i-1]]
                           + gf_extruded.points[gf_extruded.faces[i-2]]) / 3
            center_faces.append(face_center)

    # Get the parcel of project
    georef_gf, z_min, z_max = get_boundary_z_min_z_max(all_storeys_elements[gf_floor_idx], origin_pt, true_n)
    georef_gf = Polygon(georef_gf)
    georef_centroid_gf = georef_gf.centroid
    parcels = shapefile_to_shapely_parcels("/www/models-preloaded/BRK_SelectieCentrum.shp")
    parcel = get_parcel(georef_centroid_gf, parcels)

    # roads
    geom_roads, name_roads = shapefile_to_shapely_roads("/www/models-preloaded/Wegvakonderdelen.shp")
    close_roads = get_close_roads(geom_roads, georef_gf)
    # get roads to test against
    query_parcel = parcel.buffer(1)  # 1 m buffer around parcel can be other value
    adjacent_roads = {}
    for i, road in close_roads.items():
        if query_parcel.intersects(road):
            adjacent_roads[i] = road

    # filter point faces, horizontal faces, and faces with no roads
    sides_roads = {}
    for k, normal in enumerate(gf_extruded.face_normals):
        if -e <= normal[0] <= e and -e <= normal[1] <= e:
            pass
        else:
            origin = center_faces[k]
            extension = origin + normal
            georef_origin = get_georeferenced_point(origin, origin_pt, true_n)
            georef_extension = get_georeferenced_point(extension, origin_pt, true_n)
            face_link_road = face_to_road(adjacent_roads, georef_centroid_gf, georef_extension)
            if face_link_road is None:
                pass
            else:
                sides_roads[k] = name_roads[face_link_road]

    # Remove the underground floors and GF from the list of floors OR get designated floor number
    first_floor_idx = get_first_floor_idx(all_storeys_names)
    all_storeys_elements = all_storeys_elements[first_floor_idx:]
    all_storeys_names = all_storeys_names[first_floor_idx:]
    if storey_number is not None:
        all_storeys_elements = [all_storeys_elements[storey_number - 1]]
        all_storeys_names = all_storeys_names[storey_number - 1]

    lst_all_checks = []
    lst_all_points = []
    lst_all_triangles = []
    i = 0
    for storey_elements in all_storeys_elements:
        storey_points = np.array(get_elements_surfaces_points(storey_elements))
        storey_pts, storey_alpha_tris = get_alpha_shape_3d(storey_points, alpha)
        lst_all_points.append(storey_pts.tolist())
        lst_all_triangles.append(storey_alpha_tris.tolist())
        storey_size = storey_alpha_tris.shape[0]
        storey_temp_array = np.full((storey_size, 1), 3)
        storey_faces = np.hstack(np.concatenate((storey_temp_array, storey_alpha_tris), axis=1))

        storey_mesh = pv.PolyData(storey_pts, storey_faces)
        check = {}
        for j, road_name in sides_roads.items():
            origin = point_on_faces[j]
            normal = gf_extruded.face_normals[j]
            clipped = storey_mesh.clip(normal=normal, origin=origin, invert=False)
            admissible_overhang = guide_lines[road_name]
            if clipped.n_points == 0:
                check[road_name] = ("Pass", "Admissible overhang: " + str(admissible_overhang),
                                    "Overhang: " + "No overhang")
            else:
                points_size = len(clipped.points)
                normal_temp_array = np.full((points_size, 3), normal)
                origin_temp_array = np.full((points_size, 3), origin)
                temp_vector = clipped.points - origin_temp_array
                dist_to_plane = np.einsum("ij,ij->i", temp_vector, normal_temp_array)
                dist_to_check = np.amax(dist_to_plane)
                filename = "mesh{0}{1}.wkt".format(j, i)
                pv.save_meshio(filename, clipped)
                with open(filename, "r") as f:
                    wkt = f.read()
                    
                if admissible_overhang > dist_to_check:
                    check[road_name] = ("Pass", "Admissible overhang: " + str(admissible_overhang),
                                        "Overhang: " + str(dist_to_check), wkt)
                else:
                    check[road_name] = ("Fail", "Admissible overhang: " + str(admissible_overhang),
                                        "Overhang: " + str(dist_to_check), wkt)
        lst_all_checks.append(check)
        i += 1

    return {"all_checks" : lst_all_checks, "all_points": lst_all_points, "all_triangles": lst_all_triangles}
