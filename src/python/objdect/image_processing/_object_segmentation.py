from ._image_processing import convex_hull_from_points
from ._image_processing import point_is_in_polygon
from ._image_processing import do_intersect
from ._image_processing import overlapping_points
from numpy import (float32, uint8, reshape, zeros, array, where)
from cv2 import (TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, KMEANS_PP_CENTERS,
                 kmeans, bitwise_and, dilate, MORPH_CLOSE, findContours,
                 RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)


def _kmeans_cv(frame, n_clusters):
    """

    K-Means function that utilizes the already built-in function found in the
    opencv library. The k-means clustering algorithm is utilized in order to
    separate potential present objects in the current frame. The algorithm
    utilizes the Kmeans++ initialization. The criteria for the K-Means are
    defined as, max number of iterations set to 300, and the acceptable error
    rate is set to 1e-4.

    Parameters
    ----------
    frame : uint8 array
        Input (background subtracted) frame.
    n_clusters : uint
        Number of clusters to segment input frame.

    Returns
    -------
    res_frame : uint8 array
        Clustered frame.

    """
    # Define criteria
    criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 300, 1e-4)

    # Flatten input frame
    inpt_cv = float32(frame.reshape((-1, 3)))

    # Fit current frame to the k-means algorithm
    ret, label, center = kmeans(inpt_cv, n_clusters, None, criteria,
                                10, KMEANS_PP_CENTERS)

    # Obtain labels
    center = uint8(center)

    return center, label


def _frame_clusters(frame, n_clusters):

    # Segment input frame
    center, label = _kmeans_cv(frame, n_clusters)

    # Initialize output clusters
    clusters = zeros((frame.shape, n_clusters), dtype=uint8)

    # Construct clusters into frames
    for i in range(n_clusters):
        clusters[:, :, :, i] = center[where(
            label.flatten() == i)].reshape((frame.shape))

    return clusters


def _points_from_clusters(clusters):

    # Initialize points
    points = zeros(clusters.shape[3], dtype=object)

    # Find points of each cluster
    for i in range(clusters.shape[3]):
        points[i], hierarchy = findContours(clusters[:, :, :, i],
                                            RETR_EXTERNAL,
                                            CHAIN_APPROX_SIMPLE)
    return points


def _convex_hull_from_clusters(points):

    # Initialize convex hulls
    conv_hulls = zeros(len(points), dtype=object)

    # Iterate over clusters & construct convex hull
    for i in range(len(points)):
        conv_hulls[i] = convex_hull_from_points(points)

    return conv_hulls


def _overlapping_hulls(conv_hulls):

    # Initialize indices of overlapping hulls
    idxs = []

    # Iterate over hulls
    for i in range(len(conv_hulls)):
        for k in range(i+1, len(conv_hulls)):
            if do_intersect(conv_hulls[i], conv_hulls[k]):
                idxs.append([i, k])
    # Return None if no overlappings occur
    if len(idxs) > 0:
        return idxs
    else:
        return None


def _cluster_segmentation(frame, n_clusters):

    # Get clusters of input frame
    clusters = _frame_clusters(frame, n_clusters)

    # Get points
    points = _points_from_clusters(clusters)

    # Get convex hulls
    conv_hulls = _convex_hull_from_clusters(points)

    # Overlapping convexes
    overlap = _overlapping_hulls(conv_hulls)

    # Return convex hulls & overlapping clusters
    return overlap, conv_hulls

# TODO: Need a model that can determine relationship of overlapping objects
#     * Can a GMM arrive on the conclusion of whether to stop bg subtraction?
#     * Stop the bg subtraction & feed image to ICP & match with a known
#       object? If no match, create class of current object's position?
#     * Casuality?
#     * Need to give it memory (casuallity) i.e. determine object behind/inside
#       other object.
#     * Can send convex hull of an object to ICP & convert to semi-supervised
#       learning, then semi-unsupervised.


def _getBBox(conv_hull):

    return array([[conv_hull[:, 0].min(), conv_hull[:, 1].min()],
                  [conv_hull[:, 0].max(), conv_hull[:, 1].max()]])


def _drawBBox(frame, color, conv_hull):
    """
    Parameters.
    ----------
    frame : TYPE
        DESCRIPTION.
    color : TYPE
        DESCRIPTION.
    conv_hull : TYPE
        DESCRIPTION.

    Returns
    -------
    frame : TYPE
        DESCRIPTION.

    """
    # Get BBox leftmost & righmost coords
    maxY, minY = _getBBox(conv_hull)

    # Draw horizontal lines
    for i in range(maxY[1], minY[1]):
        frame[maxY[0], i] = color
        frame[minY[0], i] = color

    # Draw vertical lines
    for i in range(maxY[0], minY[1]):
        frame[i, maxY[1]] = color
        frame[i, minY[1]] = color

    return frame


overlapping_points


def _reconstruct_frame(frame, conv_hull):

    # Initialize convex hull mask
    mask = zeros(frame.shape, dtype=uint8)

    # Get bounding box min & max corner points of convex hull
    maxY, minY = _getBBox(conv_hull)

    # Iterate over bbox region
    for i in range(maxY[0], minY[0]):
        for k in range(maxY[1], minY[1]):
            if point_is_in_polygon(frame[i, k, 0], conv_hull):
                mask[i, k] = 255
    # Return reconstructed cluster from convex hull
    return bitwise_and(frame, frame, mask)

# def _segment_objects(idxs):


#     while True:
#         if idxs is None:
#             break
#         else:
#             # Store & compaire with previous results; if same, break
