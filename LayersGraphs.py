from tkinter import *
from tkinter import ttk
import numpy as np
import json
from math import ceil
from GraphsUtils import paint_rectangle_layer, draw_convolution_layer, scaling
from LayersGUI import DenseLayerWindow, ConvConfigureWindow, FlattenLayerWindow


ActivationFunctions = ["linear", "sigmoid", "tanh", "relu", "softmax"]
with open("Colours.json") as f:
    colourConfig = json.load(f)


def paint_layers(Root, In_canvas, layers_list, summaryFrame):
    In_canvas.delete("all")
    y_coordinate = 350

    GraphObjects = {"Input": InputGraph,
                    "Flatten": FlattenLayerGraph,
                    "Dense": DenseLayerGraph,
                    "Conv2D": Conv2DGraph}

    for index, layer in enumerate(layers_list.LayersList):
        # size = layer["layer"].units
        x_coordinate = 100 + 150 * index

        graph = GraphObjects[layer["type"]]
        graphObj = graph(Root, layers_list, layer, In_canvas, summaryFrame)
        graphObj.recalculate_size()
        graphObj.calculate_parameters()
        # graphObj.UpdateModelSummary()

        graphObj.draw(x_coordinate, y_coordinate)
        graphObj.UpdateModelSummary()


# <<<<<<<<<<<<<<<<<<<<<<< BASE Graph Obj >>>>>>>>>>>>>>>>>>>>>>>>>>> #
class LayerGraphWindow:
    def __init__(self, parent, layers, current_layer, working_canvas, modelSummaryFrame):
        self.parent = parent
        self.canvas = working_canvas
        self.Layers = layers
        self.current_layer = current_layer
        self.summaryFrame = modelSummaryFrame

    def UpdateModelSummary(self):
        for widget in self.summaryFrame.winfo_children():
            widget.destroy()


        parameters = [layer["parameters"] for layer in self.Layers.LayersList]
        totalParameters = sum(parameters)
        self.summaryFrame.grid_columnconfigure(1, weight=1)

        Label(self.summaryFrame, text="Model Summary",
              **colourConfig["ModelSummary"]["Labels"]).grid(row=0, column=0, sticky=W, padx=(5, 5))

        # Separator
        ttk.Separator(self.summaryFrame, orient=HORIZONTAL).grid(row=1, column=0,
                                                                columnspan=2, sticky=W+E, pady=(5, 5), padx=(5, 5))
        # Labels
        Label(self.summaryFrame, text="Total parameters: ",
              **colourConfig["ModelSummary"]["Labels"]).grid(row=2, column=0, sticky=W, padx=(4, 0))
        Label(self.summaryFrame, text=str(totalParameters),
              **colourConfig["ModelSummary"]["Labels"]).grid(row=2, column=1, sticky=E, padx=(0, 0))

        # Separator
        ttk.Separator(self.summaryFrame, orient=HORIZONTAL).grid(row=3, column=0,
                                                                 columnspan=2, sticky=W+E, pady=(5, 5), padx=(5, 5))
        # Labels
        Label(self.summaryFrame, text="Layer",
              **colourConfig["ModelSummary"]["Labels"]).grid(row=4, column=0, sticky=W, padx=(4, 0))
        Label(self.summaryFrame, text="Param #",
              **colourConfig["ModelSummary"]["Labels"]).grid(row=4, column=1, sticky=E, padx=(4, 0))

        row = 5
        for layer in self.Layers.LayersList[1:]:
            Label(self.summaryFrame, text=layer["name"],
                  **colourConfig["ModelSummary"]["Labels"]).grid(row=row, column=0, sticky=W, padx=(4, 0))
            Label(self.summaryFrame, text=str(layer["parameters"]),
                  **colourConfig["ModelSummary"]["Labels"]).grid(row=row, column=1, sticky=E, padx=(4, 0))
            row += 1

    def calculate_parameters(self):
        self.current_layer["parameters"] = 0

    def set_ConfWindow(self, confWindow):
        self.confWindow = confWindow

    def SetPosition(self):
        try:
            layer_index = self.Layers.LayersList.index(self.current_layer)
            if layer_index == len(self.Layers.LayersList) - 1:  # If the layer is at the end
                self.confWindow.PosRadiobuttonVar.set(0)
            else:  # if it is a hidden layer
                self.confWindow.PosRadiobuttonVar.set(1)
                self.confWindow.Pos_Combobox.current(layer_index - 1)
                self.confWindow.Pos_Combobox_active()
        except AttributeError as err:
            print(err)

    def reconfigureButtons(self):
        try:
            self.confWindow.AddButton.config(text="Save", command=self.Save)
            self.confWindow.QuitButton.config(text="Cancel")
        except AttributeError as err:
            print(err)

    def Save(self):
        self.Layers.Remove_layer(self.current_layer)
        self.confWindow.Add()
        self.confWindow.Window.destroy()

    def delete_layer(self):
        self.Layers.Remove_layer(self.current_layer)
        self.canvas.delete("all")
        paint_layers(self.parent, self.canvas, self.Layers, self.summaryFrame)
        self.confWindow.Window.destroy()


# <<<<<<<<<<<<<<<<<<<<<<< Input Graph Obj >>>>>>>>>>>>>>>>>>>>>>>>>>> #
class InputGraph(LayerGraphWindow):

    def draw(self, x_coordinate, y_coordinate):
        if len(self.current_layer["keras_layer"].shape) == 1:
            paint_rectangle_layer(self.canvas, self.current_layer, "", 400, 400, x_coordinate, self.config_window,
                                  colourConfig["Canvas"])
        else:
            start = (x_coordinate, y_coordinate)
            draw_convolution_layer(self.canvas, self.current_layer, start, 40, 70, 64, 64, self.config_window,
                                   colourConfig["Canvas"])

    def recalculate_size(self):
        pass

    def config_window(self, event):
        pass


# <<<<<<<<<<<<<<<<<<<<<<< Flatten Graph Obj >>>>>>>>>>>>>>>>>>>>>>>>>>> #
class FlattenLayerGraph(LayerGraphWindow):

    def draw(self, x_coordinate, *args, **kwards):
        size = self.current_layer["shape"]
        name = self.current_layer["name"]
        activation = ""
        scaled_size = 400
        paint_rectangle_layer(self.canvas, name, activation, size, scaled_size, x_coordinate, self.config_window,
                              colourConfig["Canvas"])

    def recalculate_size(self):
        layer_index = self.Layers.LayersList.index(self.current_layer)
        prev_layer = self.Layers.LayersList[layer_index-1]
        self.current_layer["shape"] = np.prod(prev_layer["shape"])
        print("Flatten shape: ", self.current_layer["shape"])

    def config_window(self, event):
        self.ConfigWindow = FlattenLayerWindow(self.parent, self.canvas, self.Layers, self.summaryFrame)
        self.ConfigWindow.Create()
        self.set_ConfWindow(self.ConfigWindow)

        self.ConfigWindow.NameEntry.insert(END, self.current_layer["name"])
        self.SetPosition()
        self.reconfigureButtons()

        Button(self.ConfigWindow.Window, text="Delete", **colourConfig["Button"],
               command=self.delete_layer).grid(row=5, column=1, sticky=E, pady=(10, 5))


# <<<<<<<<<<<<<<<<<<<<<<< Dense Graph Obj >>>>>>>>>>>>>>>>>>>>>>>>>>> #
class DenseLayerGraph(LayerGraphWindow):
    """Object for the visual representation of a dense layer
    in the canvas. By clicking on a layer you can open the
    configuration window for that layer in witch you can save
    changes or delete the layer. The configuration window is
    created using the FCLayerWindow class changing the command
    of the add button to save the changes to the layer and adding
    a new button to delete the layer.

    """

    def draw(self, x_coordinate, *args, **kwards):
        max_size = 700
        size = self.current_layer["configuration"]["units"]
        name = self.current_layer["name"]
        activation = self.current_layer["configuration"]["activation"]
        scaled_size = scaling(size, max_size)
        paint_rectangle_layer(self.canvas, name, activation, size, scaled_size, x_coordinate, self.config_window,
                              colourConfig["Canvas"])

    def recalculate_size(self):
        size = self.current_layer["configuration"]["units"]
        self.current_layer["shape"] = size

    def calculate_parameters(self):
        layer = self.current_layer
        layers_shape = layer["shape"]
        print("layers_shape: ", layers_shape)
        layer_index = self.Layers.LayersList.index(layer)
        prevLayer = self.Layers.LayersList[layer_index - 1]
        prevLayer_shape = prevLayer["shape"]
        print("prevLayer_shape: ", prevLayer_shape)
        parameters_N = layers_shape * (prevLayer_shape + 1)
        self.current_layer["parameters"] = parameters_N

        print(" Dense Parameters -------->", parameters_N)


    def config_window(self, event):
        self.ConfigWindow = DenseLayerWindow(self.parent, self.canvas, self.Layers, self.summaryFrame)
        self.ConfigWindow.Create()
        self.set_ConfWindow(self.ConfigWindow)
        # making the preselected tags
        self.ConfigWindow.NameEntry.insert(END, self.current_layer["name"])
        self.ConfigWindow.SizeEntry.insert(END, self.current_layer["shape"])
        activation = self.current_layer["configuration"]["activation"]
        self.ConfigWindow.ActivationCompo.current(ActivationFunctions.index(activation))

        reg_val_list = list(self.ConfigWindow.regularizers.values())

        RegMethod = self.current_layer["configuration"]["activity_regularizer"]


        self.ConfigWindow.RegCompo.current(reg_val_list.index(RegMethod))

        self.SetPosition()
        self.reconfigureButtons()

        Button(self.ConfigWindow.Window, text="Delete", **colourConfig["Button"],
               command=self.delete_layer).grid(row=10, column=1, sticky=W)


# <<<<<<<<<<<<<<<<<<<<<<< Conv2D Graph Obj >>>>>>>>>>>>>>>>>>>>>>>>>>> #
class Conv2DGraph(LayerGraphWindow):

    def draw(self, x_coordinate, y_coordinate):
        theta = 71
        alpha, beta, r = self.current_layer["shape"]
        middle = (x_coordinate+alpha, y_coordinate)

        draw_convolution_layer(self.canvas, self.current_layer,
                               middle, r, theta, alpha, beta, self.config_window, colourConfig["Canvas"])

    def recalculate_size(self):
        layer = self.current_layer
        layer_index = self.Layers.LayersList.index(layer)
        prevLayer = self.Layers.LayersList[layer_index-1]
        prevLayer_shape = prevLayer["shape"]
        assert len(prevLayer_shape) > 2

        W, H, _ = prevLayer_shape[0:3]
        F_w, F_h = layer["configuration"]["kernel_size"]
        S_w, S_h = layer["configuration"]["strides"]
        padding = layer["configuration"]["padding"]
        if padding == "same":
            P_w = ceil((W * (S_w - 1) - S_w + F_w)/2)
            P_h = ceil((H * (S_h - 1) - S_h + F_h) / 2)
        else:
            P_w = 0
            P_h = 0

        output_W = ((W - F_w + 2 * P_w) / S_w) + 1
        output_H = ((H - F_h + 2 * P_h) / S_h) + 1

        layer["shape"] = (int(output_W), int(output_H), layer["configuration"]["filters"])

    def calculate_parameters(self):
        layer = self.current_layer
        layer_index = self.Layers.LayersList.index(layer)
        prevLayer = self.Layers.LayersList[layer_index - 1]
        prevLayer_shape = prevLayer["shape"]

        filters = self.current_layer["configuration"]["filters"]
        kernel_size = self.current_layer["configuration"]["kernel_size"]

        prev_filters = prevLayer_shape[-1]

        parameters_N = (int(kernel_size[0])*int(kernel_size[1])*int(prev_filters) + 1)*int(filters)

        print("Conv2D Parameters -----> ", parameters_N)

        self.current_layer["parameters"] = parameters_N

    def config_window(self, event):
        confWindow = ConvConfigureWindow(self.parent, self.canvas, self.Layers, self.summaryFrame)
        self.confWindow = confWindow
        self.confWindow.Create()

        # get the information of the current layer
        name = self.current_layer["name"]
        filters = self.current_layer["configuration"]["filters"]
        kernel_size = self.current_layer["configuration"]["kernel_size"]
        strides = self.current_layer["configuration"]["strides"]
        padding = self.current_layer["configuration"]["padding"]
        activation = self.current_layer["configuration"]["activation"]
        kernel_initializer = self.current_layer["configuration"]["kernel_initializer"]
        bias_initializer = self.current_layer["configuration"]["bias_initializer"]
        kernel_regularizer = self.current_layer["configuration"]["kernel_regularizer"]
        bias_regularizer = self.current_layer["configuration"]["bias_regularizer"]
        activity_regularizer = self.current_layer["configuration"]["activity_regularizer"]

        # set the confWindow options to the configuration of the current layer
        confWindow.NameEntry.insert(0, name)
        confWindow.FilterEntry.insert(0, filters)
        confWindow.KernelSizeEntry1.insert(0, kernel_size[0])
        confWindow.KernelSizeEntry2.insert(0, kernel_size[1])
        confWindow.StridesSizeEntry1.insert(0, strides[0])
        confWindow.StridesSizeEntry2.insert(0, strides[1])
        confWindow.PadingCombobox.current(confWindow.pading_options.index(padding))
        confWindow.ActivationCombobox.current(confWindow.activation_options.index(activation))
        confWindow.KernelInitCombobox.current(confWindow.kernelInit_options.index(kernel_initializer['class_name']))
        confWindow.BiasInitCombobox.current(confWindow.biasInit_options.index(bias_initializer['class_name']))

        regularizers_values_list = list(confWindow.regularizers.values())
        confWindow.Kernel_Reg.current(regularizers_values_list.index(kernel_regularizer))
        confWindow.Bias_Reg.current(regularizers_values_list.index(bias_regularizer))
        confWindow.Activation_Reg.current(regularizers_values_list.index(activity_regularizer))

        # set the position of the layer
        self.SetPosition()

        # reconfiguring the configure window to save the changes made on the layer
        self.reconfigureButtons()

        # button fir deleting the layer
        Button(confWindow.Bottom_Frame, text="Delete", command=self.delete_layer, **colourConfig["Button"],
               width=4).pack(side=RIGHT, padx=(5, 2))

