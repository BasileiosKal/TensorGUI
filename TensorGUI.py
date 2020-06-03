from tkinter import *
from tkinter import ttk
import numpy as np
from LayersGUI import FCLayerWindow, paint_layers, InputLayer
import tensorflow as tf
from tensorflow import keras


class AddLayerWindow:
    """Window in which the user will choose what type of layer
    to add in the network.
    Contains:
    - Combobox with the choices for the layers.
    - Button for going to the next window where the user will
      choose the configuration of the layer.
    - Button for quiting the window.

     """
    def __init__(self, parent, canvas, Layers):
        self.Layers = Layers
        self.parent = parent
        self.canvas = canvas
        self.Window = Toplevel(parent)
        self.label = Label(self.Window, text="Choose a layer type : ")
        self.label.grid(row=0, column=0, sticky=W)

        self.combo = ttk.Combobox(self.Window, value=["...", "Foully Connected", "Convoloutional", "Padding",
                                                      "Pooling"])
        self.combo.current(0)
        self.combo.grid(row=0, column=1)

        self.CancelButton = Button(self.Window, text="Cancel", command=self.Window.destroy)
        self.NextButton = Button(self.Window, text="Next", command=self.nextWindow)
        self.CancelButton.grid(row=1, column=0)
        self.NextButton.grid(row=1, column=1)

    def nextWindow(self):
        if self.combo.get() == "Foully Connected":
            FCLayerWindow(self.parent, self.canvas, self.Layers)
            self.Window.destroy()


class App:
    """The main window of the app.
    Contains:
    - Toolbar with a button to add new layers and a button to train the network.
    - A Compiler frame with the options for the compiling and training of
      the model: optimization algorithms, accuracy metrics, loss function and
      number of epochs.
    - The Canvas with the visual representation of the layers.

    Inputs:
    - Layers: A list of layers in the form: {"name": the layers name, "layer": the keras layer}

    The App will draw in the Canvas the layers the input when it is constructed.

    """
    def __init__(self, Layers):
        self.Layers = Layers
        self.root = Tk()
        self.root.geometry("1000x700")
        # ===== Toolbar ===== #
        self.top_frame = Frame(self.root, heigh=50, bg="red")
        self.addButton = Button(self.top_frame, text="Add", fg="red", bg="black",
                                command=self.Add_layer).pack(side=LEFT)
        self.trainButton = Button(self.top_frame, text="Train", fg="red", bg="black", command=self.train).pack(side=LEFT)
        self.top_frame.pack(side=TOP, fill='x', expand=0)

        # ===== Compiler ===== #
        self.CompileFrame = Frame(self.root, heigh=100)
        # Optimization Algorithms
        Label(self.CompileFrame, text="Optimization Algorithm:").grid(row=0, column=0, sticky=W, pady=(5, 0))
        OptAlgorithms = ['adam']
        self.OptCombobox = ttk.Combobox(self.CompileFrame, value=OptAlgorithms, state="normal")
        self.OptCombobox.current(0)
        self.OptCombobox.grid(row=1, column=0, sticky=W, pady=(2, 10), padx=(2, 2))

        # Separator
        ttk.Separator(self.CompileFrame, orient=VERTICAL).grid(row=0, column=1, rowspan=2, sticky='ns',
                                                               padx=(10, 10))
        # Loss Functions
        Label(self.CompileFrame, text="Losses:").grid(row=0, column=3, sticky=W, pady=(5, 0))
        losses = ['SparseCategoricalCrossentropy']
        self.LossCombobox = ttk.Combobox(self.CompileFrame, value=losses, state="normal", width=27)
        self.LossCombobox.current(0)
        self.LossCombobox.grid(row=1, column=3, sticky=W, pady=(2, 10), padx=(2, 2))

        # Separator
        ttk.Separator(self.CompileFrame, orient=VERTICAL).grid(row=0, column=4, rowspan=2, sticky='ns',
                                                               padx=(10, 10))
        # metrics
        Label(self.CompileFrame, text="Metrics:").grid(row=0, column=5, sticky=W, pady=(5, 0))
        metrics = ['accuracy']
        self.MetricsCombobox = ttk.Combobox(self.CompileFrame, value=metrics, state="normal")
        self.MetricsCombobox.current(0)
        self.MetricsCombobox.grid(row=1, column=5, sticky=W, pady=(2, 10), padx=(2, 2))

        # Separator
        ttk.Separator(self.CompileFrame, orient=VERTICAL).grid(row=0, column=6, rowspan=2, sticky='ns',
                                                               padx=(10, 10))
        # Epochs
        Label(self.CompileFrame, text="Epochs:").grid(row=0, column=7, sticky=W, pady=(5, 0))
        self.epochsEntry = Entry(self.CompileFrame)
        self.epochsEntry.grid(row=1, column=7, sticky=W, pady=(2, 10), padx=(2, 2))


        self.CompileFrame.pack(side=TOP, fill='x', expand=0)

        # ===== Canvas ===== #
        self.canvas_frame = Frame(self.root, width=300, height=300)  # .pack(expand=True, fill=BOTH)
        self.canvas = Canvas(self.canvas_frame, width=300, height=300, scrollregion=(0, 0, 20000, 20000))
        self.canvas_frame.pack(expand=True, fill=BOTH)
        # Scrollbar for the canvas
        s = Scrollbar(self.canvas_frame, orient=HORIZONTAL)
        s.pack(side=BOTTOM, fill=X)
        s.config(command=self.canvas.xview)
        # Canvas configuration
        self.canvas.config(width=300, height=300)
        self.canvas.config(xscrollcommand=s.set)
        self.canvas.pack(side=LEFT, expand=True, fill=BOTH)

        paint_layers(self.root, self.canvas, Layers)

        self.root.update()
        height = self.canvas.winfo_height()
        print(height)
        self.root.mainloop()

    def Add_layer(self):
        AddLayerWindow(self.root, self.canvas, self.Layers)

    def train(self):
        optimization = self.OptCombobox.get()
        loss = self.OptCombobox.get()
        metric = self.MetricsCombobox.get()
        epochs = int(self.epochsEntry.get())
        # Compile
        Keras_Layers = [layer["layer"] for layer in self.Layers]
        model = keras.Sequential(Keras_Layers)
        model.compile(optimizer=optimization,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[metric])
        model.fit(train_X, train_Y, epochs=epochs)



if __name__ == "__main__":
    Input = InputLayer()
    (train_X, train_Y), (test_X, test_Y) = Input.data
    size = train_X.shape[1:]
    Layers = []
    if Input.flattVar.get():
        size = np.prod(np.array(train_X.shape[1:]))
        Layers.append({"name": "Flatten Input", "layer": tf.keras.layers.Flatten(input_shape=(28, 28))})
    App(Layers)
    print("END ", Layers)
