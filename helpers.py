import numpy as np

relative = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]))
relativeT = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]), 0)

def distance(p1, p2):
    """
    Args:
        p1: p2: points in the image plane
    Returns:
        The distance between the points
    """
    dist = np.asarray(p1) - np.asarray(p2)
    return np.sqrt(np.dot(dist, dist))

def eyes_close(points):
    """
    input: face landmarks and raw frame
    output: mean between the two eyes of the following ratio: 1/2 as:
    1: vertical length (in pixels) of the distance between the eyes edges
    2: distance between the bottom side of the eyelid to the bottom part of the eye
    """
    upper_left = points.landmark[386].y
    lower_left = points.landmark[374].y
    vertical_left1 = points.landmark[362].x
    vertical_left2 = points.landmark[263].x

    upper_right = points.landmark[159].y
    lower_right = points.landmark[145].y
    vertical_right1 = points.landmark[33].x
    vertical_right2 = points.landmark[133].x

    # Ratio average, every eye distance was multiplied by 100 for easier represntation 
    return (distance(upper_left, lower_left) / distance(vertical_left1, vertical_left2) * 100 \
            + distance(upper_right, lower_right) / distance(vertical_right1, vertical_right2, ) * 100) / 2