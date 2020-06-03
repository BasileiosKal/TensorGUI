from tkinter import *
from tkinter import ttk
import numpy as np
import tensorflow as tf
from tkinter import filedialog
import os

ActivationFunctions = ["linear", "sigmoid", "tanh", "relu"]
RegularizationMethods = ["None", "Dropout", "L2"]
data_libraries = {"boston_housing": tf.keras.datasets.boston_housing,
                  "cifar10": tf.keras.datasets.cifar10,
                  "cifar100": tf.keras.datasets.cifar100,
                  "fashion_mnist": tf.keras.datasets.fashion_mnist,
                  "imdb": tf.keras.datasets.imdb,
                  "mnist": tf.keras.datasets.mnist,
                  "reuters": tf.keras.datasets.reuters}

def sigmoid(x):
    return 1/(1+np.exp(-x))


def scaling(x, max_size):
    if x < max_size*(4/5):
        return x*2
    else:
        return max_size*(4/5) + max_size*(1/5)*sigmoid(x)


def paint_layers(Root, In_canvas, layers_list, max_size=700):
    x_coordinate = 0
    print("Paint layers list: ", layers_list)
    for index, layer in enumerate(layers_list):
        # size = layer["layer"].units
        x_coordinate = 50 + 110 * index
        keras_layer = layer["layer"]
        print("kera layer name", keras_layer.name)
        if keras_layer.name == "flatten":
            InputLayerGraph(Root, layers_list, layer, In_canvas, x_coordinate)
        elif keras_layer.name[0:5] == "dense":

            Obj = FCLayerGraph(Root, layers_list, layer, In_canvas, x_coordinate)

        if index == 0:
            In_canvas.create_line(x_coordinate + 50, 400, x_coordinate + 100, 400, arrow=LAST)
        else:
            activation = keras_layer.get_config()["activation"]
            In_canvas.create_text(x_coordinate+72, 390, text=activation)
            In_canvas.create_line(x_coordinate+50, 400, x_coordinate+100, 400, arrow=LAST)


def paint_rectangle_layer(canvas, name, size, scaled_size, x_coordinate, config_window):
    """paint in the canvas a rectangle"""
    # scaled_size = scaling(size, max_size)
    y_middle = 400
    y_start = y_middle - (scaled_size / 2)
    y_end = y_middle + (scaled_size / 2)
    Layer_rectangle = canvas.create_rectangle(x_coordinate, y_start, x_coordinate + 40, y_end, fill="black",
                                              activeoutline="red")

    canvas.tag_bind(Layer_rectangle, "<Button-1>", config_window)
    canvas.move(Layer_rectangle, 0, 0)

    # Labels
    canvas.create_text(x_coordinate + 20, y_start - 25, text="{" + name)
    canvas.create_text(x_coordinate + 20, y_start - 10, text=" size: " + str(size) + "}")

    return Layer_rectangle


class InputLayer:
    def __init__(self):
        self.loadingRoot = Tk()
        self.loadingRoot.geometry("420x300")

        # ==================== tf data ======================== #
        # Radio button for choosing a tf data library
        self.RadioVar = IntVar()
        self.tfDataRadio = Radiobutton(self.loadingRoot, text="load data from a TensorFlow library:",
                                       variable=self.RadioVar, value=0, command=self.tfDataCombo_activate)
        self.tfDataRadio.grid(row=0, column=0, columnspan=2, sticky=W, pady=(10, 0))
        # Drop down menu for the tf data library
        libraries_names = list(data_libraries.keys())
        self.tfDataCombo = ttk.Combobox(self.loadingRoot, value=libraries_names, state="normal")
        self.tfDataCombo.grid(row=1, column=1)

        # -------- separator --------- #
        ttk.Separator(self.loadingRoot, orient=HORIZONTAL).grid(row=2, columnspan=3, sticky=W + E,
                                                                padx=(10, 10), pady=(10, 10))

        # =================== npz file path ====================== #
        # Radio button for loading data form a .npz file
        self.npRadio = Radiobutton(self.loadingRoot, text="load from a .npz file", variable=self.RadioVar, value=1,
                                   command=self.npzCombo_activate)
        self.npRadio.grid(row=3, column=0, sticky=W)
        # Entry for the .npz file path
        self.npzPathEntry = Entry(self.loadingRoot, state="disabled")
        self.npzPathEntry.grid(row=4, column=1, sticky=W)
        Label(self.loadingRoot, text="File Path:").grid(row=4, column=0, sticky=E)
        # Browse button for .npz file path
        self.browseButton = Button(self.loadingRoot, text="Browse", command=self.browseFile, state="disable")
        self.browseButton.grid(row=4, column=2, sticky=W)

        # -------- separator --------- #
        ttk.Separator(self.loadingRoot, orient=HORIZONTAL).grid(row=5, columnspan=3, sticky=W + E,
                                                                padx=(10, 10), pady=(10, 10))

        # ================= add flatten layer ==================== #
        self.flattVar = IntVar()
        flattCheck = Checkbutton(self.loadingRoot, text="Flatten the input", variable=self.flattVar)
        flattCheck.grid(row=6, column=0, sticky=W)
        # ================= normalize the data =================== #
        self.normVar = IntVar()
        normCheck = Checkbutton(self.loadingRoot, text="Normalize", variable=self.normVar,
                                command=self.normCheck_activate)
        normCheck.grid(row=7, column=0, sticky=W, pady=(10, 0))
        self.normLabel = Label(self.loadingRoot, text="div by:", state="disable")
        self.normLabel.grid(row=7, column=0, sticky=E, pady=(10, 0))
        self.normEntry = Entry(self.loadingRoot, state="disable")
        self.normEntry.grid(row=7, column=1, pady=(10, 0), sticky=W)


        self.CreateButton = Button(self.loadingRoot, text="Next", command=self.load_the_data).grid(row=8, column=2)


        self.loadingRoot.mainloop()

    def tfDataCombo_activate(self):
        self.npRadio.deselect()
        self.tfDataCombo.configure(state="readonly")
        self.npzPathEntry.configure(state="disable")
        self.browseButton.configure(state="disable")

    def npzCombo_activate(self):
        self.tfDataRadio.deselect()
        self.tfDataCombo.configure(state="disable")
        self.npzPathEntry.configure(state="normal")
        self.browseButton.configure(state="normal")

    def browseFile(self):
        if os.path.exists("/"):
            initial_dir = "/"
        elif os.path.exists("C:"):
            initial_dir = "C:"
        else:
            raise ValueError("Unknown file system")

        self.loadingRoot.filePath = filedialog.askopenfilename(initialdir=initial_dir, title="Select file",
                                                               filetypes=(("numpy files", "*.npz"),
                                                                          ("all files", "*.*")))
        self.npzPathEntry.delete(0, END)
        self.npzPathEntry.insert(END, self.loadingRoot.filePath)

    def normCheck_activate(self):
        if self.normVar.get:
            self.normLabel.configure(state="normal")
            self.normEntry.configure(state="normal")
        else:
            self.normLabel.configure(state="disable")
            self.normEntry.configure(state="disable")

    def load_the_data(self):
        if not self.RadioVar.get():
            choice = self.tfDataCombo.get()
            data = data_libraries[choice]
            (train_data, train_labels), (test_data, test_labels) = data.load_data()
            print(train_data.shape)
        else:
            pass
        self.loadingRoot.destroy()
        self.data = [(train_data, train_labels), (test_data, test_labels)]


class InputLayerGraph:
    def __init__(self, parent, Layers, current_layer, working_canvas, x_coordinate, max_size=700):
        self.canvas = working_canvas
        self.parent = parent
        self.layer = current_layer
        self.Layers = Layers
        config = current_layer["layer"].get_config()
        size = config['batch_input_shape'][1:]
        name = current_layer["name"]
        scaled_size = 400
        paint_rectangle_layer(self.canvas, name, size, scaled_size, x_coordinate, self.config_window)

    def config_window(self, event):
        pass




class FCLayerWindow:
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
    def __init__(self, parent, canvas, Layers):
        self.canvas = canvas
        self.parent = parent
        self.Window = Toplevel(parent)
        self.Window.geometry("315x250")
        self.Layers_names = [layer["name"] for layer in Layers]
        self.Layers = Layers

        # Name Label and Entry
        self.LabelName = Label(self.Window, text="Name: ")
        self.NameEntry = Entry(self.Window)
        self.LabelName.grid(row=1, column=0, sticky=E, pady=(10, 5))
        self.NameEntry.grid(row=1, column=1, sticky=W, pady=(10, 5))
        # Size Label and Entry
        self.LabelSize = Label(self.Window, text="Size: ")
        self.SizeEntry = Entry(self.Window)
        self.LabelSize.grid(row=2, column=0, sticky=E, pady=(0, 5))
        self.SizeEntry.grid(row=2, column=1, sticky=W, pady=(0, 5))
        # Regularization Combobox and Label
        self.RegLabel = Label(self.Window, text="Regularization: ")
        self.RegCompo = ttk.Combobox(self.Window, value=RegularizationMethods)
        self.RegCompo.current(0)
        self.RegLabel.grid(row=3, column=0, sticky=E, pady=(0, 5))
        self.RegCompo.grid(row=3, column=1, sticky=W, pady=(0, 5))
        # Activation Combobox and Label
        self.ActivationLabel = Label(self.Window, text="Activation function: ")
        self.ActivationCompo = ttk.Combobox(self.Window, value=ActivationFunctions)
        self.ActivationCompo.current(0)
        self.ActivationLabel.grid(row=4, column=0, sticky=E, pady=(0, 10))
        self.ActivationCompo.grid(row=4, column=1, sticky=W, pady=(0, 10))
        # Separator
        ttk.Separator(self.Window, orient=HORIZONTAL).grid(row=5, columnspan=3, sticky=W + E)
        # Radiobutton and labels for the position of the layer
        self.PosRadiobuttonVar = IntVar()
        self.LayerPosLabel = Label(self.Window, text="Choose the position of the layer: ")
        self.PutInEnd_RadButton = Radiobutton(self.Window, text="In the End", variable=self.PosRadiobuttonVar, value=0,
                                              command=self.Pos_Combobox_disabled)
        self.PutAfter_RadButton = Radiobutton(self.Window, text="After Layer", variable=self.PosRadiobuttonVar, value=1,
                                              command=self.Pos_Combobox_active)
        self.LayerPosLabel.grid(row=6, column=0, columnspan=3)
        self.PutInEnd_RadButton.grid(row=7, column=0)
        self.PutAfter_RadButton.grid(row=7, column=1)
        # Combobox for the position of the layer
        self.Pos_Combobox = ttk.Combobox(self.Window, value=self.Layers_names, state="disabled")
        self.Pos_Combobox.grid(row=8, column=1, pady=(5, 10))
        # Seperator
        ttk.Separator(self.Window, orient=HORIZONTAL).grid(row=9, columnspan=3, sticky=E + W)
        # Buttons
        self.exitButton = Button(self.Window, text="Quit", command=self.Window.destroy)
        self.createButton = Button(self.Window, text="Create", command=self.createFCLayer)
        self.exitButton.grid(row=10, column=0, sticky=W, padx=(20, 0))
        self.createButton.grid(row=10, column=1, sticky=E, padx=(0, 20))

    def createFCLayer(self):
        # Get the Layers info
        size = int(self.SizeEntry.get())
        name = self.NameEntry.get()
        activation = self.ActivationCompo.get()
        Regularization = self.RegCompo.get()
        if self.PosRadiobuttonVar.get():
            choice = self.Pos_Combobox.get()
            PosIndex = self.Layers_names.index(choice) + 1
        else:
            PosIndex = len(self.Layers_names)
        print(name, size, activation, Regularization, PosIndex)
        Layer = {"name": name, "regularization": Regularization,
                 "layer": tf.keras.layers.Dense(size, activation=activation)}

        self.Layers_names.insert(PosIndex, name)
        self.Layers.insert(PosIndex, Layer)

        self.Pos_Combobox.configure(value=self.Layers_names)
        print("--->", self.Layers_names)
        self.canvas.delete("all")
        paint_layers(self.parent, self.canvas, self.Layers)

        return self.Layers

    def Pos_Combobox_active(self):
        self.Pos_Combobox.configure(state="readonly")

    def Pos_Combobox_disabled(self):
        self.Pos_Combobox.configure(state="disabled")

    def get_layers(self):
        pass


class FCLayerGraph:
    """Object for the visual representation of a dense layer
    in the canvas. By clicking on a layer you can open the
    configuration window for that layer in witch you can save
    changes or delete the layer. The configuration window is
    created using the FCLayerWindow class changing the command
    of the add button to save the changes to the layer and adding
    a new button to delete the layer.

    """
    def __init__(self, parent, Layers, current_layer, working_canvas, x_coordinate, max_size=700):
        self.canvas = working_canvas
        self.parent = parent
        self.layer = current_layer
        self.Layers = Layers
        size = current_layer["layer"].units
        name = current_layer["name"]
        scaled_size = scaling(size, max_size)
        paint_rectangle_layer(self.canvas, name, size, scaled_size, x_coordinate, self.config_window)

    def config_window(self, event):
        self.ConfigWindow = FCLayerWindow(self.parent, self.canvas, self.Layers)
        # making the preselected tags
        self.ConfigWindow.NameEntry.insert(END, self.layer["name"])
        self.ConfigWindow.SizeEntry.insert(END, self.layer["layer"].units)
        activation = self.layer["layer"].get_config()["activation"]
        self.ConfigWindow.ActivationCompo.current(ActivationFunctions.index(activation))
        RegMethod = self.layer["regularization"]
        self.ConfigWindow.RegCompo.current(RegularizationMethods.index(RegMethod))

        layer_index = self.Layers.index(self.layer)
        if layer_index == len(self.Layers)-1:  # If the layer is at the end
            self.ConfigWindow.PosRadiobuttonVar.set(0)
        else:  # if it is a hidden layer
            self.ConfigWindow.PosRadiobuttonVar.set(1)
            self.ConfigWindow.Pos_Combobox.current(layer_index-1)
            self.ConfigWindow.Pos_Combobox_active()

        self.ConfigWindow.createButton.configure(text="Shave", command=self.reconfigure)
        # Button to delete the layer
        Button(self.ConfigWindow.Window, text="Delete", command=self.delete_layer).grid(row=10, column=1, sticky=W)

    def reconfigure(self):
        self.Layers.remove(self.layer)
        self.ConfigWindow.createFCLayer()
        self.ConfigWindow.Window.destroy()

    def delete_layer(self):
        self.Layers.remove(self.layer)
        self.canvas.delete("all")
        paint_layers(self.parent, self.canvas, self.Layers)
        self.ConfigWindow.Window.destroy()
