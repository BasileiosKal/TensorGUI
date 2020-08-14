from math import sin, cos, radians
import numpy as np
from tkinter import *


def sigmoid(x):
    return 1/(1+np.exp(-x))


def scaling(x, max_size):
    print("SCALING ERROR ---->> ", x)
    if int(x) < max_size*(4/5):
        return int(x)*2
    else:
        return max_size*(4/5) + max_size*(1/5)*sigmoid(int(x))


def paint_rectangle_layer(working_canvas, name, activation,
                          size, scaled_size, x_coordinate, config_window, canvas_colour_config):
    """paint in the canvas a rectangle that will
    correspond to a dense or flatten layer. Each
    rectangle will have text above it with the
    name and dimension of the layer.
    """

    y_middle = 350
    y_start = y_middle - (scaled_size / 2)
    y_end = y_middle + (scaled_size / 2)
    Layer_rectangle = working_canvas.create_rectangle(x_coordinate, y_start, x_coordinate + 40, y_end,
                                                      **canvas_colour_config["rectangle_layer"])

    working_canvas.tag_bind(Layer_rectangle, "<Button-1>", config_window)
    working_canvas.move(Layer_rectangle, 0, 0)

    # Labels
    working_canvas.create_text(x_coordinate + 20, y_start - 25, **canvas_colour_config["text"],
                               text="{" + name)
    working_canvas.create_text(x_coordinate + 20, y_start - 10, **canvas_colour_config["text"],
                               text=" size: " + str(size) + "}")
    # Arrows
    working_canvas.create_text(x_coordinate+72, y_middle-10, text=activation,
                               **canvas_colour_config["text"])
    working_canvas.create_line(x_coordinate+50, y_middle, x_coordinate+100, y_middle, arrow=LAST,
                               **canvas_colour_config["text"])

    return Layer_rectangle


def draw_convolution_layer(working_canvas, layer_config, middle, r, theta, alpha, beta, config_window, canvas_colour):
    rads = radians(theta)
    rads_2 = radians(90-theta)
    d = r*sin(rads)
    d_2 = r*sin(rads_2)
    t = r*cos(rads)
    t_2 = r*cos(rads_2)

    # Points
    x = middle[0] - t - alpha + d_2 - (r/2)*cos(rads)
    y = middle[1] - (beta/2) + (r/2)*sin(rads)
    A = (x+t, y-d)
    B = (A[0]+alpha, A[1])
    G = (B[0], B[1]+beta)
    D = (G[0]-d_2, G[1]+t_2)
    F = (D[0], D[1]-beta)
    Conv2D_colours = canvas_colour["Conv2D"]

    # the sides of the cube
    polygon = working_canvas.create_polygon(x, y, *A, *B, *G, *D, *F,
                                            fill=Conv2D_colours["side"], outline=Conv2D_colours["lines"])
    working_canvas.create_line(*F, *B, fill=Conv2D_colours["lines"])

    # the rectangle of the front
    rectangle = working_canvas.create_rectangle(x, y, *D, fill=Conv2D_colours["front"], outline=Conv2D_colours["lines"])

    # Bind the rectangle shape to open the configure window
    working_canvas.tag_bind(polygon, "<Button-1>", config_window)
    working_canvas.tag_bind(rectangle, "<Button-1>", config_window)

    #  the arrow in front of the cube
    P1 = middle
    arrow_length = 50
    dist_from_layer = 40
    working_canvas.create_line(P1[0] + dist_from_layer, P1[1],
                               P1[0] + dist_from_layer + arrow_length, P1[1],
                               fill=Conv2D_colours["arrow"], arrow=LAST)

    shape = layer_config["shape"]

    # Labels
    working_canvas.create_text(x + t + 20, y - d - 40, **canvas_colour["text"],
                               text="{" + layer_config["name"])
    working_canvas.create_text(x + t + 20, y - d - 20, **canvas_colour["text"],
                               text=" size: " + str(shape) + "}")