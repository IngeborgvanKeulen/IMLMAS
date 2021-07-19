import math


def add_edge(start, end, edge_x, edge_y, length_frac=0.8, arrow_angle=20, dot_size=20):
    # Get start and end cartesian coordinates
    x0, y0 = start
    x1, y1 = end

    # Incorporate the fraction of this segment covered by a dot into total reduction
    length = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    arrow_length = length * 0.1
    arrow_length = 0.003
    dot_size_conversion = .005 / dot_size  # length units per dot size
    converted_dot_diameter = dot_size * dot_size_conversion
    length_frac_reduction = converted_dot_diameter / length
    length_frac = length_frac - length_frac_reduction

    # If the line segment should not cover the entire distance, get actual start and end coords
    skipX = (x1 - x0) * (1 - length_frac)
    skipY = (y1 - y0) * (1 - length_frac)
    x0 = x0 + skipX / 2
    x1 = x1 - skipX / 2
    y0 = y0 + skipY / 2
    y1 = y1 - skipY / 2

    # Draw arrow
    # Find the point of the arrow; assume is at end unless told middle
    pointx = x1
    pointy = y1
    eta = math.degrees(math.atan((x1 - x0) / (y1 - y0)))

    # Find the directions the arrows are pointing
    signx = (x1 - x0) / abs(x1 - x0)
    signy = (y1 - y0) / abs(y1 - y0)

    # Append first arrowhead
    dx = arrow_length * math.sin(math.radians(eta + arrow_angle))
    dy = arrow_length * math.cos(math.radians(eta + arrow_angle))
    edge_x.append(pointx)
    edge_x.append(pointx - signx ** 2 * signy * dx)
    edge_x.append(None)
    edge_y.append(pointy)
    edge_y.append(pointy - signx ** 2 * signy * dy)
    edge_y.append(None)

    # And second arrowhead
    dx = arrow_length * math.sin(math.radians(eta - arrow_angle))
    dy = arrow_length * math.cos(math.radians(eta - arrow_angle))
    edge_x.append(pointx)
    edge_x.append(pointx - signx ** 2 * signy * dx)
    edge_x.append(None)
    edge_y.append(pointy)
    edge_y.append(pointy - signx ** 2 * signy * dy)
    edge_y.append(None)

    return edge_x, edge_y
