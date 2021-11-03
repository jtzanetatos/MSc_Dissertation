#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from numpy import where, arctan2, delete, zeros, argsort, array


def _minPoint(points):
    '''
    Returns the index of the lowest Y valued point. In case of conflict
    (i.e. more than one Y values), returns value with the lowest X coordinate.
    
    Parameters
    ----------
    points : int/float N x 2 array
        Set of input points.
    
    Returns
    -------
    minPoint : int
        Index where the lowest Y value resides.
    
    '''
    
    # Find min y point
    minPoint = where(points[:, 0].min() == points[:, 0])[0] # Will return tuple otherwise
    
    # Determine if more than one y has min value
    if len(minPoint) > 1:
        # Find min x point
        minPoint = where(points[minPoint, 1].min() == points[:, 1])[0]
        
        # Ensure only one point returns in case of X coord conflict
        if len(minPoint) > 1:
            minPoint = minPoint[0]
    
    return minPoint

def _crossSlopeDir(point_a, point_b, point_c):
    '''
    Counter Clockwise turns geometrical primitive.
    If result is > 0, then points make a clockwise turn,
    if < 0, counter clockwise and if = 0 collinear.
    
    Parameters
    ----------
    point_a : int/float X Y coordinates
        Point a to test if CCW, CW or collinear.
    point_b : int/float X Y coordinates
        Point b to test if CCW, CW or collinear..
    point_c : int/float X Y coordinates
        Point c to test if CCW, CW or collinear..
    
    Returns
    -------
    int/float
        Result of CCW turn.
    
    '''
    
    return (point_b[0] - point_a[0]) * (point_c[1] - point_a[1]) \
          - (point_b[1] - point_a[1]) * (point_c[0] - point_a[0])

def _polarAngle(point, point_0):
    '''
    Evaluates the radiant degrees between anchor point & Nth point of the 
    set of points.
    
    Parameters
    ----------
    point : int/float, 2 element array
        Set of Y X coordinates of Nth point.
    point_0 : int/float, 2 element array
         Set of Y X coordinates of P_0 point (anchor point).
    
    Returns
    -------
    float
        Radiant degrees between anchor point & Nth point.
    
    '''
    
    return arctan2((point[1] - point_0[1]), (point[0] - point_0[0]))

def _sortByDistance(ang_idxs, angles, points, point_0):
    '''
    Utilized to resolve potential conflicts of sorted radiant angles.
    Utilizes the approximate Eucledian distance for efficiency.
    
    Will not return a modified copy of angular indices, instead, it mutates 
    the indices.
    
    Parameters
    ----------
    ang_idxs : int, N x 2 array
        Indices of sorted points based on radiant angle between points & anchor
        point.
    points : int/float
        Set of unsorted points (anchor point excluded).
    point_0 : int/float
        Set of Y X coordinates of anchor point.
    
    Returns
    -------
    Sorts input indices of input array.
    
    '''
    
    # Iterate over indices & locate duplicates
    for i in range(len(ang_idxs)):
        for j in range(i+1, len(ang_idxs)):
            # Duplicate found
            if angles[ang_idxs[i]] == angles[ang_idxs[j]]:
                # Sort by distance - Approximate Eucledian distance
                dist_i = (points[ang_idxs[i], 1] - point_0[1])**2 + (points[ang_idxs[i], 0] - point_0[0])**2
                
                dist_j = (points[ang_idxs[j], 1] - point_0[1])**2 + (points[ang_idxs[j], 0] - point_0[0])**2
                
                # Swap indices
                if dist_i > dist_j:
                    ang_idxs[i] += ang_idxs[j]
                    ang_idxs[j] = ang_idxs[i] - ang_idxs[j]
                    ang_idxs[i] -= ang_idxs[j]

def convex_hull_from_points(points, progress=False):
    '''
    Graham scan algorithm implementation to draw the convex hull of a given
    set of points.
    
    References:
        [1] https://en.wikipedia.org/wiki/Graham_scan
    
    Parameters
    ----------
    points : int/float, N x 2 array
        Set of Y X coordinate points.
    progress : bool, optional
        Flag that plots each step of the algorithm. The default is False.
    
    Returns
    -------
    int/float N x 2 array
        Convex hull of input set of points.
    
    '''
    # Find point with lowest y (& possibly lowest x)
    p_idx = _minPoint(points)
    
    # Min Point
    minPoint = points[p_idx].squeeze()
    
    # Remove minPoint from array of points
    points = delete(points, array((p_idx, p_idx)), axis=0)
    
    # Initialize slope indices
    angles = zeros(len(points))
    
    # Evaluate polar angle of each point
    for i in range(len(points)):
        # Evaluate slope for current point
        angles[i] = _polarAngle(points[i], minPoint)
    # Return indices that would sort angles
    angle_idxs = argsort(angles, kind='heapsort')
    
    # Sort by distance in case of slope conflicts
    _sortByDistance(angle_idxs, angles, points, minPoint)
    
    # Sort points
    points = points[angle_idxs, :]
    
    # Initialize convex hull points
    convex_hull = [minPoint, points[0,:]]
    
    # Iterate over points
    for point in points[1:,]:
        # Backtrack points if clockwise movement
        while _crossSlopeDir(convex_hull[-2], convex_hull[-1], point) <= 0:
            del convex_hull[-1]
            if len(convex_hull) == 1:
                break
        # Show progress
        if progress:
            _scatter_plot(points, convex_hull)
        # Append current point
        convex_hull.append(point)
    
    # Convert convex hull to array obj
    return array(convex_hull)


def point_is_in_polygon(point, convex_hull):
    '''
    Winding Number Algorithm algorithm to determine if input point lies 
    inside / on the convexpolygon or outside the (convex) polygon.
    
    References:
        [1] https://www.eecs.umich.edu/courses/eecs380/HANDOUTS/PROJ2/InsidePoly.html
        [2] https://towardsdatascience.com/is-the-point-inside-the-polygon-574b86472119
    
    Parameters
    ----------
    point : int/float 2 elements array
        Set of Y X coordinates of input point.
    polygon : inf/float N x 2 array
        The convex hull of a set of points.
    
    Returns
    -------
    bool
        True/False based on CCW result.
    
    '''
    
    ccwTest = zeros(convex_hull.shape[0]-1)
    
    for i in range(1, convex_hull.shape[0]):
        ccwTest[i-1] = _crossSlopeDir(convex_hull[i, :], convex_hull[i-1,:], point)
    
    for i in range(1, convex_hull.shape[0]-1):
        if ccwTest[i-1] <= 0 and ccwTest[i] <= 0:
            continue
        elif ccwTest[i-1] >= 0 and ccwTest[i] >= 0:
            continue
        else:
            return False
    
    return True


def do_intersect(polygon1, polygon2):
    '''
    
    
    Parameters
    ----------
    polygon1 : int/float N x 2 array
        Reference polygon.
    polygon2 : int/float N x 2 array
        Polygon that each point is tested whether it intersects reference
        polygon.
    
    Returns
    -------
    bool
        Returns true if polygons intersect, otherwise false.
    
    '''
    
    # Initialize point test results array
    pointRes = zeros(polygon2.shape[0], dtype=bool)
    
    # Iterate over polygon2 points
    for i, point in enumerate(polygon2):
        pointRes[i] = point_is_in_polygon(point, polygon1)
        # Polygons intersect, return true
        if pointRes[i] == True:
            return True
    # Return false
    return False