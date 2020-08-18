from tkinter import *
from tkinter import ttk
import numpy as np
import tensorflow as tf
from Utils import Label_and_Entry, Label_and_2Entries, Label_and_Combobox, get_tuple_from_str, \
    layer_position
import json


ActivationFunctions = ["linear", "sigmoid", "tanh", "relu", "softmax"]
RegularizationMethods = ["None", "Dropout", "L2"]
data_libraries = {"boston_housing": tf.keras.datasets.boston_housing,
                  "cifar10": tf.keras.datasets.cifar10,
                  "cifar100": tf.keras.datasets.cifar100,
                  "fashion_mnist": tf.keras.datasets.fashion_mnist,
                  "imdb": tf.keras.datasets.imdb,
                  "mnist": tf.keras.datasets.mnist,
                  "reuters": tf.keras.datasets.reuters}

with open("Colours.json") as f:
    colourConfig = json.load(f)


# <<<<<<<<<<<<<<<<<<<<<<< BASE Config Window >>>>>>>>>>>>>>>>>>>>>>>>>>> #
class LayerConfigWindow:
    def __init__(self, parent, working_canvas, layers_list, summaryFrame):
        self.parent = parent
        self.canvas = working_canvas
        self.Layers = layers_list
        self.Layers_names = layers_list.get_LayersNames()
        self.summaryFrame = summaryFrame

    def Create(self):
        pass

    def Add(self):
        pass


# ==================================================================================== #
#                                    FLATTEN                                           #
# ==================================================================================== #

# <<<<<<<<<<<<<<<<<<<<<<< Flatten Config Window >>>>>>>>>>>>>>>>>>>>>>>>>>> #
class FlattenLayerWindow(LayerConfigWindow):
    """The configuration window for a flatten
    layer.
    Contains:
    - Entry for the input size.
    - Buttons for adding the layer and quiting.
    """

    def Create(self):
        window_colours = colourConfig["TopLevel"]

        self.Window = Toplevel(self.parent, bg=window_colours["bg"])
        self.Window.geometry("279x159")
        # Name
        Label(self.Window, text="Name:", **window_colours["Labels"]).grid(row=0, column=0, sticky=W,
                                                                          pady=(5, 5), padx=(5, 0))
        self.NameEntry = Entry(self.Window, **window_colours["Entry"])
        self.NameEntry.grid(row=0, column=1, columnspan=2, sticky=W, pady=(5, 5))
        # Position
        # Radiobutton and labels for the position of the layer
        self.PosRadiobuttonVar = IntVar()
        self.LayerPosLabel = Label(self.Window, text="Choose the position of the layer: ", **window_colours["Labels"])
        self.PutInEnd_RadButton = Radiobutton(self.Window, text="In the End", variable=self.PosRadiobuttonVar, value=0,
                                              command=self.Pos_Combobox_disabled, **window_colours["Radiobutton"])
        self.PutAfter_RadButton = Radiobutton(self.Window, text="After Layer", variable=self.PosRadiobuttonVar, value=1,
                                              command=self.Pos_Combobox_active, **window_colours["Radiobutton"])
        self.LayerPosLabel.grid(row=2, column=0, columnspan=3)
        self.PutInEnd_RadButton.grid(row=3, column=0)
        self.PutAfter_RadButton.grid(row=3, column=1)
        # Combobox for the position of the layer
        self.Pos_Combobox = ttk.Combobox(self.Window, value=self.Layers_names, state="disabled")
        self.Pos_Combobox.grid(row=4, column=1, columnspan=2, pady=(5, 10))
        # Buttons
        self.QuitButton = Button(self.Window, text="Quit", command=self.Window.destroy,
                                 **colourConfig["Button"])
        self.QuitButton.grid(row=5, column=0, sticky=W, pady=(10, 5), padx=(5, 0))
        self.AddButton = Button(self.Window, text="Create", command=self.Add,
                                **colourConfig["Button"])
        self.AddButton.grid(row=5, column=2, sticky=E, pady=(10, 5))

    def Add(self):
        # Get the Layers info
        name = self.NameEntry.get()
        # Position of the layer
        if self.PosRadiobuttonVar.get():
            choice = self.Pos_Combobox.get()
            PosIndex = self.Layers_names.index(choice) + 1
        else:
            PosIndex = len(self.Layers_names)
        # layer size
        prev_layer_shape = list(self.Layers.LayersList[PosIndex-1]["shape"])
        flatLayer_shape = np.prod(prev_layer_shape)

        Layer = {"name": name,
                 "type": "Flatten",
                 "shape": flatLayer_shape,
                 "parameters": 0,
                 "configuration": {},
                 "keras_layer": tf.keras.layers.Flatten(**{})}

        # self.Layers_names.insert(PosIndex, name)
        self.Layers.Add_layer(Layer, PosIndex)

        self.Pos_Combobox.configure(value=self.Layers_names)
        self.canvas.delete("all")
        import LayersGraphs
        LayersGraphs.paint_layers(self.parent, self.canvas, self.Layers, self.summaryFrame)

        return self.Layers

    def Pos_Combobox_active(self):
        self.Pos_Combobox.configure(state="readonly")

    def Pos_Combobox_disabled(self):
        self.Pos_Combobox.configure(state="disabled")


# ==================================================================================== #
#                                     DENSE                                            #
# ==================================================================================== #

# <<<<<<<<<<<<<<<<<<<<<<< Dense Config Window >>>>>>>>>>>>>>>>>>>>>>>>>>> #
class DenseLayerWindow(LayerConfigWindow):
    """Window for configuring the FC (dense) layers
    Contains:
    - Entry's for the name and size of the layer.
    - Combobox for the activation function
    - Combobox for the regularization of the layer
    - Radio box for choosing between adding the layer
      in the end of the network or after another
      layer.
    - Combobox for choosing after what layer the
      current layer will go if the user chooses this
      option

    """

    def __init__(self, parent, working_canvas, layers_list, summaryFrame):
        super().__init__(parent, working_canvas, layers_list, summaryFrame)
        self.regularizers = {
            "None": None,
            "l1": {'class_name': 'L1L2', 'config': {'l1': 0.009999999776482582, 'l2': 0.0}},
            "l2": {'class_name': 'L1L2', 'config': {'l1': 0.0, 'l2': 0.009999999776482582}},
        }

    def Create(self):
        window_colours = colourConfig["TopLevel"]
        self.Window = Toplevel(self.parent, bg=window_colours["bg"])
        self.Window.geometry("315x250")
        # Name Label and Entry
        self.LabelName = Label(self.Window, text="Name: ", **window_colours["Labels"])
        self.NameEntry = Entry(self.Window, **window_colours["Entry"])
        self.LabelName.grid(row=1, column=0, sticky=E, pady=(10, 5))
        self.NameEntry.grid(row=1, column=1, sticky=W, pady=(10, 5))
        # Size Label and Entry
        self.LabelSize = Label(self.Window, text="Size: ", **window_colours["Labels"])
        self.SizeEntry = Entry(self.Window, **window_colours["Entry"])
        self.LabelSize.grid(row=2, column=0, sticky=E, pady=(0, 5))
        self.SizeEntry.grid(row=2, column=1, sticky=W, pady=(0, 5))
        # Regularization Combobox and Label
        l1 = 0.01
        l2 = 0.01

        self.RegLabel = Label(self.Window, text="Regularization: ", **window_colours["Labels"])
        self.RegCompo = ttk.Combobox(self.Window, value=list(self.regularizers.keys()))
        self.RegCompo["state"] = "readonly"
        self.RegCompo.current(0)
        self.RegLabel.grid(row=3, column=0, sticky=E, pady=(0, 5))
        self.RegCompo.grid(row=3, column=1, sticky=W, pady=(0, 5))
        # Activation Combobox and Label
        self.ActivationLabel = Label(self.Window, text="Activation function: ", **window_colours["Labels"])
        self.ActivationCompo = ttk.Combobox(self.Window, value=ActivationFunctions)
        self.ActivationCompo["state"] = "readonly"
        self.ActivationCompo.current(0)
        self.ActivationLabel.grid(row=4, column=0, sticky=E, pady=(0, 10))
        self.ActivationCompo.grid(row=4, column=1, sticky=W, pady=(0, 10))
        # Separator
        ttk.Separator(self.Window, orient=HORIZONTAL).grid(row=5, columnspan=3, sticky=W + E)

        LayerPosFrame = Frame(self.Window,  bg=window_colours["bg"])
        LayerPosFrame.grid(row=6, column=0, columnspan=4, sticky="we")
        self.PosRadiobuttonVar, self.Pos_Combobox = layer_position(LayerPosFrame,
                                                                   self.Layers_names,
                                                                   self.Pos_Combobox_active,
                                                                   self.Pos_Combobox_disabled,
                                                                   window_colours)

        # Seperator
        ttk.Separator(self.Window, orient=HORIZONTAL).grid(row=9, columnspan=3, sticky=E + W)
        # Buttons
        self.QuitButton = Button(self.Window, text="Quit", command=self.Window.destroy, **colourConfig["Button"])
        self.AddButton = Button(self.Window, text="Create", command=self.Add, **colourConfig["Button"])
        self.QuitButton.grid(row=10, column=0, sticky=W, padx=(20, 0))
        self.AddButton.grid(row=10, column=1, sticky=E, padx=(0, 20))

    def get_configuration(self):
        size = int(self.SizeEntry.get())
        LayerConfig = {
            "name": self.NameEntry.get(),
            "configuration": {
                "units": size,
                "activation": self.ActivationCompo.get(),
                "activity_regularizer": self.regularizers[self.RegCompo.get()]
            }
        }
        return LayerConfig

    def Add(self):
        # Get the Layers info


        layer_conf = self.get_configuration()

        if self.PosRadiobuttonVar.get():
            choice = self.Pos_Combobox.get()
            PosIndex = self.Layers_names.index(choice) + 1
        else:
            PosIndex = len(self.Layers_names)

        Layer = {"name": layer_conf["name"],
                 "shape": layer_conf["configuration"]["units"],
                 "parameters": 0,
                 "type": "Dense",
                 "configuration": layer_conf["configuration"],
                 "keras_layer": tf.keras.layers.Dense(**layer_conf["configuration"])}

        self.Layers_names.insert(PosIndex, Layer["name"])
        self.Layers.Add_layer(Layer, PosIndex)

        self.Pos_Combobox.configure(value=self.Layers_names)
        self.canvas.delete("all")
        import LayersGraphs
        LayersGraphs.paint_layers(self.parent, self.canvas, self.Layers, self.summaryFrame)

        return self.Layers

    def Pos_Combobox_active(self):
        self.Pos_Combobox.configure(state="readonly")

    def Pos_Combobox_disabled(self):
        self.Pos_Combobox.configure(state="disabled")

    def get_layers(self):
        pass


# ==================================================================================== #
#                                     CONV2D                                           #
# ==================================================================================== #

# <<<<<<<<<<<<<<<<<<<<<<< Conv2D Config Window >>>>>>>>>>>>>>>>>>>>>>>>>>> #
class ConvConfigureWindow(LayerConfigWindow):
    def __init__(self, parent, working_canvas, layers_list, summaryFrame):
        super().__init__(parent, working_canvas, layers_list, summaryFrame)
        self.regularizers = {
            "None": None,
            "l1": {'class_name': 'L1L2', 'config': {'l1': 0.009999999776482582, 'l2': 0.0}},
            "l2": {'class_name': 'L1L2', 'config': {'l1': 0.0, 'l2': 0.009999999776482582}},
        }
        regularization_options = {"None": None,
                                  "L1": "l1",
                                  "L2": "l2"}
        self.regularizers_options = regularization_options

    def Create(self):
        window_colours = colourConfig["TopLevel"]
        self.Window = Toplevel(self.parent, bg=window_colours["bg"])
        # --- Left Frame --- #
        Left_Frame = Frame(self.Window, bg=window_colours["bg"], width=150, heigh=200)
        Left_Frame.pack_propagate(0)
        Left_Frame.grid(row=0, column=0, rowspan=2, sticky="nw")
        # --- Middle Tom Frame --- #
        Middle_Frame_Top = Frame(self.Window, bg=window_colours["bg"], width=500, heigh=100)
        Middle_Frame_Top.pack_propagate(0)
        Middle_Frame_Top.grid(row=0, column=1)
        # --- Middle Bottom Frame --- #
        Middle_Frame_Bottom = Frame(self.Window, bg=window_colours["bg"], width=300, heigh=100)
        Middle_Frame_Bottom.pack_propagate(0)

        Middle_Frame_Bottom.grid(row=1, column=1, sticky="w")

        # --- Separate --- #
        ttk.Separator(self.Window, orient=HORIZONTAL).grid(row=0, column=2, rowspan=2, sticky="ns", padx=(3, 3))

        # --- Right Frame --- #
        Right_Frame = Frame(self.Window, bg=window_colours["bg"], width=210, heigh=130)
        Right_Frame.grid_propagate(0)
        Right_Frame.grid(row=0, column=3, rowspan=2, sticky="n", pady=(3, 0))
        # --- Bottom Frame --- #
        Bottom_Frame = Frame(self.Window, bg=window_colours["bg"], width=210, heigh=40)
        Bottom_Frame.pack_propagate(0)
        Bottom_Frame.grid(row=1, column=3, sticky="s")

        # ---------------------------------------------------- #
        #                      Left Frame                      #
        # ---------------------------------------------------- #
        # Name
        self.NameEntry = Label_and_Entry(Left_Frame, "Name: ", window_colours, row=0, width=18,
                                         column_span=3, pady=(5, 5))
        # Filter
        self.FilterEntry = Label_and_Entry(Left_Frame, "Filters: ", window_colours, row=1, width=18,
                                           column_span=3, pady=(5, 5))
        # Kernel Size
        self.KernelSizeEntry1, self.KernelSizeEntry2 = Label_and_2Entries(Left_Frame, "Kernel Size: ", window_colours,
                                                                          row=2, width=7, pady=(5, 5))
        # Strides
        self.StridesSizeEntry1, self.StridesSizeEntry2 = Label_and_2Entries(Left_Frame, "Strides: ", window_colours,
                                                                           row=3, width=7, pady=(5, 5))
        # Pading
        self.pading_options = ["same", "valid"]
        self.PadingCombobox = Label_and_Combobox(Left_Frame, "Pading: ", self.pading_options, window_colours,
                                                 row=4, width=17, pady=(5, 5), column_span=3)
        # Activation functions
        self.activation_options = ActivationFunctions
        self.ActivationCombobox = Label_and_Combobox(Left_Frame, "Activations: ", self.activation_options, window_colours,
                                                     row=5, width=17, pady=(5, 5), column_span=3)
        # Separator
        ttk.Separator(Left_Frame, orient=HORIZONTAL).grid(row=0, column=4, rowspan=7, sticky='ns', padx=(3, 5))

        # ---------------------------------------------------- #
        #                   Middle Top Frame                   #
        # ---------------------------------------------------- #
        Label(Middle_Frame_Top, text="Initializing", **window_colours["Labels"]).grid(row=0, column=0, columnspan=2,
                                                                                            pady=(5, 5))
        # Kernel initializing
        self.kernelInit_options = ["GlorotUniform"]
        self.KernelInitCombobox = Label_and_Combobox(Middle_Frame_Top, "Kernels: ", self.kernelInit_options, window_colours,
                                                     row=1, pady=(5, 5), width=17)
        self.KernelInitCombobox.current(0)
        # Bias initializing
        self.biasInit_options = ["Zeros"]
        self.BiasInitCombobox = Label_and_Combobox(Middle_Frame_Top, "Bias: ", self.biasInit_options, window_colours,
                                                   row=2, pady=(5, 5), width=17)
        self.BiasInitCombobox.current(0)

        # ---------------------------------------------------- #
        #                  Middle bottom Frame                 #
        # ---------------------------------------------------- #
        # Separator
        ttk.Separator(Middle_Frame_Bottom, orient=VERTICAL).grid(row=0, column=0, columnspan=3, sticky='we', padx=(0, 5),
                                                                pady=(0, 5))
        # Position of the layer
        self.PosRadiobuttonVar, self.Pos_Combobox = layer_position(Middle_Frame_Bottom, self.Layers_names,
                                                                   self.Pos_Combobox_active, self.Pos_Combobox_disabled,
                                                                   window_colours, row=1)
        self.Pos_Combobox.configure(width=15)

        # ---------------------------------------------------- #
        #                  Middle bottom Frame                 #
        # ---------------------------------------------------- #
        Label(Right_Frame, text="Regularization", **window_colours["Labels"]).grid(row=0, column=0, columnspan=2,
                                                                                   pady=(5, 5))

        self.Kernel_Reg = Label_and_Combobox(Right_Frame, "Kernels: ", list(self.regularizers.keys()),
                                             window_colours, row=1, pady=(5, 5))
        self.Kernel_Reg.current(0)
        self.Bias_Reg = Label_and_Combobox(Right_Frame, "Bias: ", list(self.regularizers.keys()),
                                           window_colours, row=2, pady=(5, 5))
        self.Bias_Reg.current(0)
        self.Activation_Reg = Label_and_Combobox(Right_Frame, "Activation: ", list(self.regularizers.keys()),
                                                 window_colours, row=3, pady=(5, 5))
        self.Activation_Reg.current(0)

        # ---------------------------------------------------- #
        #                    Bottom Frame                      #
        # ---------------------------------------------------- #
        # Buttons
        self.QuitButton = Button(Bottom_Frame, text="Cancel", **colourConfig["Button"], width=4,
                                 command=self.Window.destroy)
        self.QuitButton.pack(side=LEFT, padx=(5, 5))

        self.AddButton = Button(Bottom_Frame, text="Add", **colourConfig["Button"], command=self.Add)
        self.AddButton.pack(side=RIGHT, padx=(0, 5))

        self.Bottom_Frame = Bottom_Frame  # to be used in the layers graph config window

    def Pos_Combobox_active(self):
        self.Pos_Combobox.configure(state="readonly")

    def Pos_Combobox_disabled(self):
        self.Pos_Combobox.configure(state="disabled")

    def get_configuration(self):
        """
         Returns the configuration of the Layer
        in JSON format
        """
        kernel_x = int(self.KernelSizeEntry1.get())
        kernel_y = int(self.KernelSizeEntry2.get())
        strides_x = int(self.StridesSizeEntry1.get())
        strides_y = int(self.StridesSizeEntry2.get())

        kernel_reg_str = self.Kernel_Reg.get()
        bias_reg_str = self.Bias_Reg.get()
        activ_reg_str = self.Activation_Reg.get()

        # kernel_regularizer = self.regularizers_options[kernel_reg_str]
        # bias_regularizer = self.regularizers_options[bias_reg_str]
        # activity_regularizer = self.regularizers_options[activ_reg_str]

        kernel_regularizer = kernel_reg_str
        bias_regularizer = bias_reg_str
        activity_regularizer = activ_reg_str


        LayerConfig = {
            "name": self.NameEntry.get(),
            "configuration": {
                "filters": int(self.FilterEntry.get()),
                "kernel_size": (kernel_x, kernel_y),
                "strides": (strides_x, strides_y),
                "padding": self.PadingCombobox.get(),
                "activation": self.ActivationCombobox.get(),
                "kernel_initializer": {'class_name': self.KernelInitCombobox.get(), 'config': {'seed': None}},
                "bias_initializer": {'class_name': self.BiasInitCombobox.get(), 'config': {}},
                "kernel_regularizer": self.regularizers[kernel_regularizer],
                "bias_regularizer": self.regularizers[bias_regularizer],
                "activity_regularizer": self.regularizers[activity_regularizer],
            }
        }
        return LayerConfig

    def get_pos_index(self):
        if self.PosRadiobuttonVar.get():
            choice = self.Pos_Combobox.get()
            PosIndex = self.Layers_names.index(choice) + 1
        else:
            PosIndex = len(self.Layers_names)
        return PosIndex

    def Add(self):
        # create the keras layer
        Layer_config = self.get_configuration()
        keras_layer = tf.keras.layers.Conv2D(**Layer_config["configuration"])
        # get the position of the layer
        PosIndex = self.get_pos_index()

        Layer = {"name": Layer_config["name"],
                 "type": "Conv2D",
                 "shape": (),
                 "parameters": 0,
                 "configuration": Layer_config["configuration"],
                 "keras_layer": keras_layer}


        self.Layers.Add_layer(Layer, PosIndex)
        self.Layers_names.insert(PosIndex, Layer_config["name"])

        self.Pos_Combobox.configure(value=self.Layers_names)
        self.canvas.delete("all")

        import LayersGraphs
        LayersGraphs.paint_layers(self.parent, self.canvas, self.Layers, self.summaryFrame)
