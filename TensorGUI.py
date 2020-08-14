import os
from tkinter import *
from tkinter import messagebox,filedialog
from tkinter import ttk
from LayersGUI import DenseLayerWindow, InputLayerWindow, FlattenLayerWindow, ConvConfigureWindow
import tensorflow as tf
from tensorflow import keras
import json
from Utils import Combobox_colour_config
from matplotlib import pyplot as plt
from matplotlib.backend_bases import key_press_handler
import copy

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

# Loading the colour configurations
with open("Colours.json") as f:
    colourConfig = json.load(f)

Input_colour_config = colourConfig["Input"]
Widget_ColourConfig = colourConfig["Widget"]
Label_ColourConfig = colourConfig["Label"]
Button_ColourConfig = colourConfig["Button"]
Entry_ColourConfig = colourConfig["Entry"]
Combobox_ColourConfig = colourConfig["Combobox"]
canvas_colour_config = colourConfig["Canvas"]


class AdamConfigObj:
    def __init__(self):
        self.epsilon = 1e-07
        self.beta_1 = 0.9
        self.beta_2 = 0.999

    def set_values(self, epsilon=1e-07, beta_1=0.9, beta_2=0.999):
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def get_optimizer(self):
        return tf.keras.optimizers.Adam(beta_1=self.beta_1,
                                        beta_2=self.beta_2,
                                        epsilon=self.epsilon)


class AdamConfigWindow:
    def __init__(self, par, ConfigObj):
        self.Config = ConfigObj
        epsilon = ConfigObj.epsilon
        beta_1 = ConfigObj.beta_1
        beta_2 = ConfigObj.beta_2

        self.root = Toplevel(par)
        self.root.title("Adam Parameters")

        pad_x = (5, 5)
        pad_y = (5, 5)

        # epsilon
        label = Label(self.root, text='epsilon: ')
        label.grid(row=0, column=0, padx=pad_x, pady=pad_y)
        self.eps_entry = Entry(self.root)
        self.eps_entry.insert(0, str(epsilon))
        self.eps_entry.grid(row=0, column=1, padx=pad_x, pady=pad_y)

        # beta 1
        label = Label(self.root, text='beta 1: ')
        label.grid(row=1, column=0, padx=pad_x, pady=pad_y)
        self.beta1_entry = Entry(self.root)
        # self.beta1_entry.delete(0, END)
        self.beta1_entry.insert(0, str(beta_1))
        self.beta1_entry.grid(row=1, column=1, padx=pad_x, pady=pad_y)

        # beta 2
        label = Label(self.root, text='beta 2: ')
        label.grid(row=2, column=0, padx=pad_x, pady=pad_y)
        self.beta2_entry = Entry(self.root)
        self.beta2_entry.insert(0, str(beta_2))
        self.beta2_entry.grid(row=2, column=1, padx=pad_x, pady=pad_y)

        # cancel button
        cancel_button = Button(self.root, text="Cancel", command=self.root.destroy)
        cancel_button.grid(row=3, column=0, padx=pad_x, pady=pad_y)

        # save button
        save_button = Button(self.root, text="Save", command=self.save)
        save_button.grid(row=3, column=1, sticky=E, padx=pad_x, pady=pad_y)

    def save(self):
        epsilon = float(self.eps_entry.get())
        beta_1 = float(self.beta1_entry.get())
        beta_2 = float(self.beta2_entry.get())
        self.Config.set_values(epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)
        self.root.destroy()




class AddLayerWindow:
    """Window in which the user will choose what type of layer
    to add in the network.
    Contains:
    - Combobox with the choices for the layers.
    - Button for going to the next window where the user will
      choose the configuration of the layer.
    - Button for quiting the window.

     """
    def __init__(self, parent, canvas, layers, SummaryFrame):
        self.Layers = layers
        self.parent = parent
        self.canvas = canvas
        self.summaryFrame = SummaryFrame
        self.Window = Toplevel(parent)
        self.label = Label(self.Window, text="Choose a layer type : ")
        self.label.grid(row=0, column=0, sticky=W)

        self.combo = ttk.Combobox(self.Window, value=["...", "Foully Connected", "Flatten", "Convolution"])
        self.combo.current(0)
        self.combo.grid(row=0, column=1)

        self.CancelButton = Button(self.Window, text="Cancel", command=self.Window.destroy)
        self.NextButton = Button(self.Window, text="Next", command=self.nextWindow)
        self.CancelButton.grid(row=1, column=0)
        self.NextButton.grid(row=1, column=1)

    def nextWindow(self):
        if self.combo.get() == "Foully Connected":
            self.Window.destroy()
            DenseLayerWindow(self.parent, self.canvas, self.Layers, self.summaryFrame).Create()
        elif self.combo.get() == "Flatten":
            self.Window.destroy()
            FlattenLayerWindow(self.parent, self.canvas, self.Layers, self.summaryFrame).Create()
        elif self.combo.get() == "Convolution":
            self.Window.destroy()
            ConvConfigureWindow(self.parent, self.canvas, self.Layers, self.summaryFrame).Create()


class LoadingBarWindow:
    def __init__(self, parent):
        self.parent = parent

    def show(self, total_epochs, total_batches, train_size):
        # loading bar
        self.loadingBar_window = Toplevel(self.parent, bg=colourConfig["TopLevel"]["bg"])
        self.loadingBar_window.geometry("550x130")
        # Loading bar for each epoch
        self.epoch_loading_bar = ttk.Progressbar(self.loadingBar_window, orient=HORIZONTAL, length=400,
                                                 mode='determinate')

        self.epoch_label = Label(self.loadingBar_window, text=f'Epoch: 0 /{total_epochs}',
                                 **colourConfig["TopLevel"]["Labels"])

        # Loading bar for each bach
        self.batch_loading_bar = ttk.Progressbar(self.loadingBar_window, orient=HORIZONTAL, length=400,
                                                 mode='determinate')
        self.batch_label = Label(self.loadingBar_window, text=f'Batch 0/{train_size/ total_batches}',
                                 **colourConfig["TopLevel"]["Labels"])

        self.epoch_end_loss_label = Label(self.loadingBar_window, text='Loss: 0.0',
                                          **colourConfig["TopLevel"]["Labels"])
        self.epoch_end_loss_label.grid(row=3, column=0, sticky=W)

        self.batch_end_loss_label = Label(self.loadingBar_window, text='Accuracy: 0.0',
                                          **colourConfig["TopLevel"]["Labels"])
        self.batch_end_loss_label.grid(row=4, column=0, sticky=W)

        return self

    def load_bar_on_epoch_end(self, epoch, logs, total_epochs):
        self.epoch_loading_bar['value'] = ((epoch + 1) / total_epochs) * 100
        self.epoch_label.config(text=f'Epoch: {epoch + 1} /{total_epochs}')
        self.epoch_end_loss_label.config(text=f'Loss: {logs["loss"]}')
        self.batch_end_loss_label.config(text=f'Accuracy: {logs["accuracy"]}')
        self.loadingBar_window.update_idletasks()

    def pack_loading_bar(self):
        self.epoch_loading_bar.grid(row=0, column=0, columnspan=2, pady=(5, 10), padx=(10, 10))
        self.epoch_label.grid(row=0, column=3, columnspan=2, sticky=W, pady=(5, 10))
        self.batch_loading_bar.grid(row=1, column=0, columnspan=2, pady=(5, 10), padx=(10, 10))
        self.batch_label.grid(row=1, column=3, columnspan=2, sticky=W, pady=(5, 10))
        self.loadingBar_window.update_idletasks()

    def load_bar_on_bach_end(self, logs, batch, train_size, total_batches):
        batches = train_size / total_batches
        self.batch_loading_bar["value"] = ((batch + 1) / int(batches)) * 100

        if (batch + 1) % ((train_size / total_batches) / 30) == 0:  # print the loss every ~50 batches
            self.batch_label.config(text=f'Batch: {batch + 1} /{train_size / total_batches}')
            self.loadingBar_window.update_idletasks()



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
    def __init__(self, input_layer, data):

        self.Layers = Layers(self)
        self.Layers.Append(input_layer)
        self.data = data

        App_W = 1200
        App_H = 700

        self.root = Tk()
        self.root.geometry(str(App_W)+"x"+str(App_H))
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)

        # ===== Toolbar ===== #
        # top frame gray5
        # Compiler gray20

        # ------- Top Frame ------- #
        self.TopFrame = Frame(self.root, heigh=50, bg="gray5", width=App_W, height=App_H/16)
        self.TopFrame.grid_propagate(0)
        self.TopFrame.grid(row=0, columnspan=2, column=0, sticky="we")

        # ------ Middle Frame ------ #
        self.CompileFrame = Frame(self.root, heigh=100, bg="gray20", width=App_W, height=App_H/11)
        self.CompileFrame.config(highlightbackground="white", highlightthickness=1)
        # self.CompileFrame.pack(side=TOP, fill='x', expand=0)
        self.CompileFrame.grid_propagate(0)
        self.CompileFrame.grid(row=1, column=0, columnspan=2, sticky="we")

        # ------- Bottom frame --------- #
        BottomFrame_H = App_H*(1 - (1/16) - (1/11))
        self.BottomFrame = Frame(self.root, bg="black", width=App_W, height=BottomFrame_H)
        self.BottomFrame.columnconfigure(0, weight=1)
        self.BottomFrame.rowconfigure(0, weight=1)
        self.BottomFrame.grid(row=2, column=0, columnspan=2, sticky=W+E+N+S)

        #     ------> Canvas Frame
        CanvasFrame_W = App_W-250
        CanvasFrame_H = BottomFrame_H
        self.CanvasFrame = Frame(self.BottomFrame, bg="green", width=CanvasFrame_W, height=CanvasFrame_H)
        self.CanvasFrame.grid(row=0, column=0, sticky=W+E+N+S)

        #     ------> Model summary Frame
        SummaryFrame_W = 200
        SummaryFrame_H = CanvasFrame_H
        self.SummaryFrame = Frame(self.BottomFrame, width=SummaryFrame_W, height=SummaryFrame_H,
                                  **colourConfig["ModelSummary"]["Frame"])
        self.SummaryFrame.grid_propagate(0)
        self.SummaryFrame.grid(row=0, column=1, sticky=E+N+S)

        # Canvas and scrollbar
        self.canvas = Canvas(self.CanvasFrame, width=App_W, height=App_H*(149/176), scrollregion=(0, 0, 20000, 20000),
                             bg=canvas_colour_config["canvas_bg"])

        # Scrollbar for the canvas
        s = Scrollbar(self.CanvasFrame, orient=HORIZONTAL, bg="red")
        s.pack(side=BOTTOM, fill='x')
        self.canvas.config(xscrollcommand=s.set)
        s.config(command=self.canvas.xview)
        # Canvas configuration
        # self.canvas.config(width=300, height=300)
        self.canvas.pack(side=LEFT, expand=True, fill=BOTH)

        # ================================================================== #

        self.TopFrame.config(highlightbackground="white", highlightthickness=1)

        SavePhotoImage = PhotoImage(file="Icons/Save/Save1_x22.png")
        self.addButton = Button(self.TopFrame, text="Save", image=SavePhotoImage, command=self.Add_layer,
                                bg="gray5", highlightbackground="gray5",
                                activebackground="gray15", bd=0,
                                width=30, height=30).grid(row=0, column=0, padx=(2, 5), pady=(5, 2))

        # self.saveButton = Button(self.TopFrame, text="Load", command=self.save,
        #                          **Button_ColourConfig).grid(row=0, column=1, padx=(2, 2), pady=(5, 2))


        RunPhotoImage = PhotoImage(file="Icons/Run/play1_x22.png")
        # RunPhotoImage = RunPhotoImage.subsample(2, 2)

        self.trainButton = Button(self.TopFrame, text="Train", image=RunPhotoImage, command=self.train,
                                  bg="gray5", highlightbackground="gray5", activebackground="gray15", bd=0,
                                  width=30, height=30)
        self.trainButton.grid_propagate(0)
        self.trainButton.grid(row=0, column=2, padx=(2, 5), pady=(5, 2))

        ttk.Separator(self.TopFrame, orient=VERTICAL).grid(row=0, column=3, sticky=N+S)

        Label(self.TopFrame, text="Saved ", bg="gray5", fg="white").grid(row=0, column=4, padx=(2, 2), pady=(5, 2))

        self.saveFrame = Frame(self.TopFrame, width=10, heigh=10, bg="red")
        self.saveFrame.grid_propagate(0)

        self.saveFrame.grid(row=0, column=5, padx=(0, 2), pady=(5, 2))

        Label(self.TopFrame, text="Compiled ", bg="gray5", fg="white").grid(row=0, column=6, padx=(2, 2), pady=(5, 2))

        self.compileFrame = Frame(self.TopFrame, width=10, heigh=10, bg="red")
        self.compileFrame.grid_propagate(0)

        self.compileFrame.grid(row=0, column=7, padx=(0, 2), pady=(5, 2))



        # ================================================================== #
        self.Menubar = Menu(self.root)
        self.root.config(menu=self.Menubar)

        # ----> File drop down menu
        def _save_model():
            if os.path.exists("/"):
                initial_dir = "/"
            elif os.path.exists("C:"):
                initial_dir = "C:"
            else:
                raise ValueError("Unknown file system")

            filePath = filedialog.asksaveasfilename(initialdir=initial_dir, title="Select file",
                                                    filetypes=(("HDF5", "*.h5"),
                                                               ("SavedModel", "*"),
                                                               ("all files", "*")))
            self.model.save(str(filePath))

        def _load_model():
            if os.path.exists("/"):
                initial_dir = "/"
            elif os.path.exists("C:"):
                initial_dir = "C:"
            else:
                raise ValueError("Unknown file system")

            filePath = filedialog.askopenfilename(initialdir=initial_dir, title="Select file",
                                                    filetypes=(("HDF5", "*.h5"),
                                                               ("SavedModel", "*"),
                                                               ("all files", "*")))

            model = keras.models.load_model(str(filePath))

            self.Layers.LayersFromModelConfig(model)
            self.model = model

            from LayersGraphs import paint_layers
            self.canvas.delete("all")
            paint_layers(self.root, self.canvas, self.Layers, self.SummaryFrame)




        FileMenu = Menu(self.Menubar, tearoff=0)
        FileMenu.add_command(label="Save model", command=_save_model)
        FileMenu.add_command(label="Load Model", command=_load_model)
        self.Menubar.add_cascade(label="File", menu=FileMenu)

        # ----> Run drop down menu
        RunMenu = Menu(self.Menubar, tearoff=0)
        RunMenu.add_command(label="Save", command=self.save)
        RunMenu.add_command(label="Compile", command=self.compile)
        RunMenu.add_separator()
        RunMenu.add_command(label="Re-initialize", command=self.re_initialize)
        RunMenu.add_separator()
        RunMenu.add_command(label="Train", command=self.train)
        self.Menubar.add_cascade(label="Run", menu=RunMenu)

        # ----> Add a layer drop down menu
        def _add_Dense():
            DenseLayerWindow(self.root, self.canvas, self.Layers, self.SummaryFrame).Create()

        def _add_Flatten():
            FlattenLayerWindow(self.root, self.canvas, self.Layers, self.SummaryFrame).Create()

        def _add_Conv2D():
            ConvConfigureWindow(self.root, self.canvas, self.Layers, self.SummaryFrame).Create()

        AddMenu = Menu(self.Menubar, tearoff=0)
        AddMenu.add_command(label="Dense", command=_add_Dense)
        AddMenu.add_command(label="Flatten", command=_add_Flatten)
        AddMenu.add_command(label="Conv2D", command=_add_Conv2D)
        self.Menubar.add_cascade(label="Add layer", menu=AddMenu)

        # ----> Weights drop down menu
        def _save_weights():
            if os.path.exists("/"):
                initial_dir = "/"
            elif os.path.exists("C:"):
                initial_dir = "C:"
            else:
                raise ValueError("Unknown file system")

            filePath = filedialog.asksaveasfilename(initialdir=initial_dir, title="Select file",
                                                    filetypes=(("HDF5", "*.h5"),
                                                               ("TensorFlow Checkpoint", "*.tf"),
                                                               ("all files", "*")))

            print("------------->>>>> File Path: ", filePath[-2:])
            try:
                self.model.save_weights(str(filePath), save_format="h5")
            except ValueError as error:
                print("Save Error: ", error)

        def _load_weights():
            if os.path.exists("/"):
                initial_dir = "/"
            elif os.path.exists("C:"):
                initial_dir = "C:"
            else:
                raise ValueError("Unknown file system")

            filePath = filedialog.askopenfilename(initialdir=initial_dir, title="Select file",
                                                  filetypes=(("HDF5", "*.h5"),
                                                               ("TensorFlow Checkpoint", "*.tf"),
                                                               ("all files", "*")))

            try:
                self.model.load_weights(str(filePath))
            except ValueError as error:
                print("Load Error: ", error)

        WeightsMenu = Menu(self.Menubar, tearoff=0)
        WeightsMenu.add_command(label="Save weights", command=_save_weights)
        WeightsMenu.add_command(label="Load weights", command=_load_weights)
        self.Menubar.add_cascade(label="Weights", menu=WeightsMenu)

        # ================================================================== #

        # ===== Compiler ===== #
        labels_bg = "gray20"
        labels_fg = "white"
        button_bg = "dark green"
        button_fg = "white"

        # Optimization Algorithms
        OptAlgorithms = ['adam']
        self.OptConfigs = {"adam": AdamConfigObj()}

        Label(self.CompileFrame, text="Optimization Algorithm:",
              bg=labels_bg, fg=labels_fg).grid(row=0, column=0, sticky=W, pady=(5, 0))

        style = ttk.Style()
        Combobox_colour_config(self.root, Combobox_ColourConfig, style)

        self.OptCombobox = ttk.Combobox(self.CompileFrame, value=OptAlgorithms, state="normal", style='TCombobox')
        self.OptCombobox["state"] = "readonly"
        self.OptCombobox.current(0)
        self.OptCombobox.grid(row=1, column=0, sticky=W, pady=(2, 10), padx=(2, 2))

        opt_config = Button(self.CompileFrame, text="config.", command=self.config_optimizer, bg=button_bg,
                            fg=button_fg, highlightbackground="black")
        opt_config.config(height=1, width=3)
        opt_config.grid(row=1, column=1, pady=(0, 10))

        # Separator
        ttk.Separator(self.CompileFrame, orient=VERTICAL).grid(row=0, column=2, rowspan=2, sticky='ns',
                                                               padx=(10, 10))
        # Loss Functions
        Label(self.CompileFrame, text="Losses:", bg=labels_bg,
              fg=labels_fg).grid(row=0, column=3, sticky=W, pady=(5, 0))

        self.losses = {'SparseCategoricalCrossentropy': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)}
        self.LossCombobox = ttk.Combobox(self.CompileFrame, value=list(self.losses.keys()), state="normal", width=27,
                                         style='TCombobox')
        self.LossCombobox["state"] = "readonly"
        self.LossCombobox.current(0)
        self.LossCombobox.grid(row=1, column=3, sticky=W, pady=(2, 10), padx=(2, 2))

        # Separator
        ttk.Separator(self.CompileFrame, orient=VERTICAL).grid(row=0, column=4, rowspan=2, sticky='ns',
                                                               padx=(10, 10))
        # Metrics
        Label(self.CompileFrame, text="Metrics:", bg=labels_bg,
              fg=labels_fg).grid(row=0, column=5, sticky=W, pady=(5, 0))
        metrics = ['accuracy']
        self.MetricsCombobox = ttk.Combobox(self.CompileFrame, value=metrics, state="normal", style='TCombobox')
        self.MetricsCombobox["state"] = "readonly"
        self.MetricsCombobox.current(0)
        self.MetricsCombobox.grid(row=1, column=5, sticky=W, pady=(2, 10), padx=(2, 2))

        # Separator
        ttk.Separator(self.CompileFrame, orient=VERTICAL).grid(row=0, column=6, rowspan=2, sticky='ns',
                                                               padx=(10, 10))
        # Epochs
        Label(self.CompileFrame, text="Epochs:", bg=labels_bg,
              fg=labels_fg).grid(row=0, column=7, sticky=W, pady=(5, 0))
        self.epochsEntry = Entry(self.CompileFrame)
        self.epochsEntry.grid(row=1, column=7, sticky=W, pady=(2, 10), padx=(2, 2))

        import LayersGraphs
        LayersGraphs.paint_layers(self.root, self.canvas, self.Layers, self.SummaryFrame)

        def on_closing():
            if messagebox.askokcancel("Quit", "Do you want to quit?"):
                self.root.quit()
                self.root.destroy()

        self.root.protocol("WM_DELETE_WINDOW", on_closing)

        self.root.update()
        # height = self.canvas.winfo_height()
        self.root.mainloop()

    def Add_layer(self):
        AddLayerWindow(self.root, self.canvas, self.Layers, self.SummaryFrame)

    def config_optimizer(self):
        ConfigClasses = {"adam": AdamConfigWindow}
        optimizer_str = self.OptCombobox.get()
        configClass = ConfigClasses[optimizer_str]
        configClass(self.root, self.OptConfigs[optimizer_str])

    def train(self):
        (train_X, train_Y) = self.data["train"]
        train_size = train_X.shape[0]

        loadingBar = LoadingBarWindow(self.root)

        # custom callback for saving logs in a json file
        json_logs = open("loss_log.json", "wt")
        logDict = {"logs": []}
        json_logging_callback = keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: logDict["logs"].append({"epoch": epoch,
                                                                    "loss": logs["loss"],
                                                                     "accuracy": logs["accuracy"]}),
            on_train_end=lambda logs: json_logs.write(json.dumps(logDict, indent=2))
        )

        # callback for updating the loading bar
        logging_callback = keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: loadingBar.load_bar_on_epoch_end(epoch, logs, epochs),
            on_train_begin=lambda logs: loadingBar.pack_loading_bar(),
            on_batch_end=lambda batch, logs: loadingBar.load_bar_on_bach_end(logs, batch, train_size, batch_N)
        )

        try:
            epochs = int(self.epochsEntry.get())
            batch_N = 32

            loadingBar.show(epochs, batch_N, train_size)

            print("START COPY W = ", self.Keras_Layers_copy[-1].get_weights()[1])
            print("START W = ", self.Keras_Layers[-1].get_weights()[1])

            results = self.model.fit(train_X, train_Y, epochs=epochs,
                                     batch_size=batch_N,
                                     callbacks=[logging_callback,
                                                json_logging_callback])

            print("END COPY W = ", self.Keras_Layers_copy[-1].get_weights()[1])
            print("END W = ", self.Keras_Layers[-1].get_weights()[1])

            ResultsWindow().create(self.root, results.history)

        except ValueError as error:
            # loadingBar.loadingBar_window.destroy()
            try:
                loadingBar.loadingBar_window.destroy()
            except AttributeError:
                pass
            errorWindow = Toplevel(self.root)
            Label(errorWindow, text=str(error)).pack()
        except RuntimeError as error:
            try:
                loadingBar.loadingBar_window.destroy()
            except AttributeError:
                pass
            loadingBar.loadingBar_window.destroy()
            errorWindow = Toplevel(self.root)
            Label(errorWindow, text=str(error)).pack()
        except AttributeError as error:
            # loadingBar.loadingBar_window.destroy()
            try:
                loadingBar.loadingBar_window.destroy()
            except AttributeError:
                pass
            errorWindow = Toplevel(self.root)
            Label(errorWindow, text=str(error)).pack()

        json_logs.close()

    def save(self):
        self.Keras_Layers = self.Layers.get_KerasLayers()
        self.Keras_Layers_copy = []
        for layer in self.Keras_Layers:
            self.Keras_Layers_copy.append(copy.copy(layer))

        try:
            self.model = keras.Sequential(self.Keras_Layers)
            self.saveFrame.config(bg="green")
        except ValueError as error:
            errorWindow = Toplevel(self.root)
            Label(errorWindow, text=str(error)).pack()

    def re_initialize(self):
        self.Layers.Re_initialize()

    def compile(self):
        # get the optimizer
        optimizer_str = self.OptCombobox.get()
        optimizer_config = self.OptConfigs[optimizer_str]
        optimization = optimizer_config.get_optimizer()

        loss = self.LossCombobox.get()
        metric = self.MetricsCombobox.get()

        try:
            self.model.compile(optimizer=optimization,
                               loss=self.losses[loss],
                               metrics=[metric])
            self.compileFrame.config(bg="green")
        except AttributeError as error:
            errorWindow = Toplevel(self.root)
            Label(errorWindow, text=str(error)).pack()
        except ValueError as error:
            errorWindow = Toplevel(self.root)
            Label(errorWindow, text=str(error)).pack()


class ResultsWindow:
    def create(self, root, logDict):
        # self.Window = Toplevel(root, bg=colourConfig["TopLevel"]["bg"])
        # losses = [logs["loss"] for logs in logDict["logs"]]
        # accuracy = [logs["accuracy"] for logs in logDict["logs"]]

        losses = logDict["loss"]
        accuracy = logDict["accuracy"]

        lossesFig = plt.figure(0, figsize=(4, 3), dpi=100)
        lossesAx = lossesFig.subplots()
        lossesAx.plot(losses)

        accuracyFig = plt.figure(1, figsize=(4, 3), dpi=100)
        accuracyAx = accuracyFig.subplots()
        accuracyAx.plot(accuracy)

        ResultWindow = Toplevel(root, bg=colourConfig["TopLevel"]["bg"])
        ResultWindow.wm_title("Results")

        fig = plt.Figure(figsize=(4, 3), dpi=100)
        ax = fig.add_subplot()
        ax.plot(accuracy)

        canvas1Frame = Frame(ResultWindow)
        canvas1 = FigureCanvasTkAgg(fig, master=canvas1Frame)  # A tk.DrawingArea.
        canvas1.draw()
        canvas1.get_tk_widget().pack(side=TOP, expand=True)

        toolbar = NavigationToolbar2Tk(canvas1, canvas1Frame)
        toolbar.update()
        canvas1.get_tk_widget().pack(side=TOP, expand=True)

        canvas1Frame.grid(row=0, column=0)

        canvas2Frame = Frame(ResultWindow)
        canvas2 = FigureCanvasTkAgg(lossesFig, master=canvas2Frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side=TOP, expand=True)

        toolbar = NavigationToolbar2Tk(canvas2, canvas2Frame)
        toolbar.update()
        canvas1.get_tk_widget().pack(side=TOP, expand=True)

        canvas2Frame.grid(row=0, column=1)


        def on_key_press(event):
            print("you pressed {}".format(event.key))
            key_press_handler(event, canvas1, toolbar)

        canvas1.mpl_connect("key_press_event", on_key_press)

        def _quit():
            # Window.quit()  # stops mainloop
            ResultWindow.destroy()  # this is necessary on Windows to prevent
            # Fatal Python Error: PyEval_RestoreThread: NULL

        button = Button(master=ResultWindow, text="Quit", command=_quit)
        button.grid(row=0, column=2)

        # plt.show()


class Layers:
    def __init__(self, Application):
        self.LayersList = []
        self._layer_types = {"Flatten": tf.keras.layers.Flatten,
                             "Dense": tf.keras.layers.Dense,
                             "Conv2D": tf.keras.layers.Conv2D}
        self.App = Application

    def Add_layer(self, layer, position):
        """
        Add a layer in the list in a certain position.
        Each layer must be of the form:

         Layer = {"name": layers name,
                 "type": layers type,
                 "shape": layers shape,
                 "parameters": number of trainable parameters,
                 "configuration": {the configuration of the layer},
                 "keras_layer": the keras layer}

        Each layer can go in certain positions only. For example
        a Conv2D layer can't be placed after a Dense layer.

        :param layer: the layer to be added in the above form
        :param position: the position of the layer in the network
        :return: Nothing
        """
        assert isinstance(position, int)
        assert position >= 0
        assert "name" in layer.keys()
        assert "type" in layer.keys()
        assert "shape" in layer.keys()
        assert "parameters" in layer.keys()
        assert "configuration" in layer.keys()
        assert "keras_layer" in layer.keys()
        assert layer["type"] in self._layer_types.keys()

        self.App.saveFrame.config(bg="red")
        self.App.compileFrame.config(bg="red")
        self.LayersList.insert(position, layer)

    def Remove_layer(self, layer):
        self.App.saveFrame.config(bg="red")
        self.App.compileFrame.config(bg="red")
        self.LayersList.remove(layer)

    def Append(self, layer):
        if hasattr(App, "saveFrame") and hasattr(App, "compileFrame"):
            self.App.saveFrame.config(bg="red")
            self.App.compileFrame.config(bg="red")
        self.LayersList.append(layer)

    def Re_initialize(self):
        for layer in self.LayersList[1:]:
            try:
                del layer["keras_layer"]
                layer["keras_layer"] = self._layer_types[layer["type"]](**layer["configuration"])
            except KeyError:
                pass

    def get_KerasLayers(self):
        return [layer["keras_layer"] for layer in self.LayersList]

    def get_LayersNames(self):
        return [layer["name"] for layer in self.LayersList]

    def LayersFromModelConfig(self, model):
        self.LayersList = [self.LayersList[0]]
        keras_layers = model.layers
        modelConfig = model.get_config()
        for layer_config, keras_layer in zip(modelConfig["layers"], keras_layers):

            AppLayer = {"name": layer_config["config"]["name"],
                        "type": layer_config["class_name"],
                        "parameters": 0,
                        "shape": (),
                        "configuration": layer_config["config"],
                        "keras_layer": keras_layer
                        }

            self.Append(AppLayer)






if __name__ == "__main__":
    Input = InputLayerWindow()

    (train_X, train_Y) = Input.data["train"]
    (test_X, test_Y) = Input.data["test"]

    keras_Input = tf.keras.Input(shape=train_X.shape[1:])

    Input_Layer = {"name": "Input",
                   "type": "Input",
                   "shape": keras_Input.shape[1:],
                   "parameters": 0,
                   "configuration":
                       {
                           "shape": keras_Input.shape
                       },
                   "keras_layer": keras_Input}


    App(Input_Layer, Input.data)
    # print("END ", Layers)
