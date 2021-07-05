import numpy as np


class BoundingBox:
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax):
        self.points = np.array(
            ([xmin, ymin, zmin],
             [xmax, ymin, zmin],
             [xmax, ymax, zmin],
             [xmin, ymax, zmin],
             [xmin, ymin, zmax],
             [xmax, ymin, zmax],
             [xmax, ymax, zmax],
             [xmin, ymax, zmax])
        )

        self.vertical_sides = np.array(
            ([self.points[0], self.points[1], self.points[5], self.points[4]],
             [self.points[1], self.points[2], self.points[6], self.points[5]],
             [self.points[2], self.points[3], self.points[7], self.points[6]],
             [self.points[3], self.points[0], self.points[4], self.points[7]])
        )

        self.horizontal_sides = np.array(
            ([self.points[0], self.points[1], self.points[2], self.points[3]],
             [self.points[4], self.points[5], self.points[6], self.points[7]])
        )

    def get_normal_vector(self):
        """
        get normal of sides with origin O
        output: nparray(x, y, z)
        """
        edge0 = self.vertical_sides[:, 1] - self.vertical_sides[:, 0]
        edge1 = self.vertical_sides[:, 2] - self.vertical_sides[:, 0]
        return np.cross(edge0, edge1)

    def get_centroid(self):
        """
        get centroid of sides
        """
        centroid = (self.vertical_sides[:, 0] + self.vertical_sides[:, 2]) / 2
        return centroid

    def get_normal_line(self):
        """
        get normal of sides as a line defined by centroid and normal vector
        output: nparray(x, y, z)
        """
        centroid = self.get_centroid()
        point = centroid + self.get_normal_vector()
        return centroid, point
