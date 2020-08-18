from tkinter import *
from tkinter import ttk
import numpy as np
import tensorflow as tf
from tkinter import filedialog, messagebox
from Utils import Combobox_colour_config, Label_and_Entry, get_tuple_from_str
import os
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


def load_the_data(InputConf):

    if InputConf["Use_Keras_Data"] and InputConf["Keras_data"]["Data_library"] != '':
        data_lib = data_libraries[InputConf["Keras_data"]["Data_library"]]
        (train_data, train_labels), (test_data, test_labels) = data_lib.load_data()
    else:
        train_data = None
        train_labels = None
        test_data = None
        test_labels = None

    if InputConf["Reshape"]["Do_reshape"]:
        reshapeEntry = list(get_tuple_from_str(InputConf["Reshape"]["Shape"]))
        reshapeList_train = [train_data.shape[0]]
        reshapeList_test = [test_data.shape[0]]
        for dim in reshapeEntry:
            reshapeList_train.append(dim)
            reshapeList_test.append(dim)
        train_data = train_data.reshape(tuple(reshapeList_train))
        test_data = test_data.reshape(tuple(reshapeList_test))

    data = {"train": (train_data, train_labels), "test": (test_data, test_labels)}
    return data


class NewProjectWindow:
    def __init__(self, parent, MiddleFrame, Width, Height):
        self.parent = parent
        self.data = None
        self.Configuration = None

        Colours = colourConfig["Input"]
        Widget_ColourConfig = colourConfig["Widget"]
        Label_ColourConfig = colourConfig["Label"]
        Button_ColourConfig = colourConfig["Button"]
        Entry_ColourConfig = colourConfig["Entry"]
        Combobox_ColourConfig = colourConfig["Combobox"]

        # -------> Project Name and Dir Frame
        NameDirFrame = Frame(MiddleFrame, width=Width, height=40, **Colours["Frame"])
        NameDirFrame.grid_propagate(0)
        NameDirFrame.grid(row=0, column=0, columnspan=2, sticky=W + E + N, pady=(10, 0))


        # -------> Load Data Frame
        DataFrame_H = 150
        DataFrame_W = 385

        LoadDataFrameTitle = Frame(MiddleFrame, width=DataFrame_W, **Colours["Frame"])
        LoadDataFrameTitle.grid(row=1, column=0, sticky=W + E, pady=(10, 0))
        # ttk.Separator(LoadDataFrameTitle, orient=HORIZONTAL).pack(side=TOP, fill="x", pady=(3, 3), padx=(10, 10))
        Label(LoadDataFrameTitle, text="Load the data", **Colours["Labels"]).pack()

        LoadDataFrame = Frame(MiddleFrame, height=DataFrame_H, width=DataFrame_W, **Colours["Frame"])
        LoadDataFrame.grid_propagate(0)
        LoadDataFrame.grid(row=2, column=0, sticky=W + N, pady=(10, 0))

        # -------> Manipulate Data Frame
        ManipulateDataFrameTitle = Frame(MiddleFrame, width=DataFrame_W, **Colours["Frame"])
        ManipulateDataFrameTitle.grid(row=3, column=0, sticky=W + E, pady=(10, 0))
        #ttk.Separator(ManipulateDataFrameTitle, orient=HORIZONTAL).pack(side=TOP, fill="x", pady=(3, 3), padx=(10, 10))
        Label(ManipulateDataFrameTitle, text="Manipulate the data", **Colours["Labels"]).pack()

        ManipulateDataFrame = Frame(MiddleFrame, height=150, width=DataFrame_W, **Colours["Frame"])
        ManipulateDataFrame.grid_propagate(0)
        ManipulateDataFrame.grid(row=4, column=0, sticky=W, pady=(10, 0))

        # -------> Data Summary Frame
        DataSummaryFrame = Frame(MiddleFrame, bg="black", height=Height-50, width=Width - DataFrame_W)
        DataSummaryFrame.grid_propagate(0)
        DataSummaryFrame.pack_propagate(0)
        DataSummaryFrame.grid(row=1, rowspan=4, column=1, sticky=W + E + N, pady=(10, 0))
        self.DataSummaryFrame = DataSummaryFrame

        # ------------------------------------------------------------------------------- #
        # ------------------------------------------------------------------------------- #
        # Name and Directory of the project
        # Label(NameDirFrame, text="Name:", **Colours["Labels"]).grid(row=0, column=0, padx=(10, 5))
        # self.ProjectNameEntry = Entry(NameDirFrame, width=10, **Colours["Entry"])
        # Label(NameDirFrame, text="Directory:", **Colours["Labels"]).grid(row=0, column=2, padx=(10, 5))
        # self.ProjectDirEntry = Entry(NameDirFrame, **Colours["Entry"])
        #
        # self._browseProjectDirButton = Button(NameDirFrame, text="...", **Colours["Button"], height=1)
        # self._browseProjectDirButton.grid_propagate(0)
        # self.ProjectNameEntry.grid(row=0, column=1)
        # self.ProjectDirEntry.grid(row=0, column=3)
        # self._browseProjectDirButton.grid(row=0, column=4, padx=(5, 0))
        # ttk.Separator(NameDirFrame, orient=HORIZONTAL).pack(side=BOTTOM, fill="x", pady=(3, 3), padx=(5, 5))
        Label(NameDirFrame, text="Name:", **Colours["Labels"]).pack(side=LEFT, padx=(10, 5))
        self.ProjectNameEntry = Entry(NameDirFrame, width=10, **Colours["Entry"])
        self.ProjectNameEntry.pack(side=LEFT, padx=(10, 5))
        Label(NameDirFrame, text="Directory:", **Colours["Labels"]).pack(side=LEFT, padx=(10, 5))
        self.ProjectDirEntry = Entry(NameDirFrame, **Colours["Entry"])
        self.ProjectDirEntry.pack(side=LEFT, padx=(10, 5))

        self._browseProjectDirButton = Button(NameDirFrame, text="...", **Colours["Button"], height=1)
        self._browseProjectDirButton.pack_propagate(0)
        self._browseProjectDirButton.pack(side=LEFT, padx=(10, 5))


        # Loading data for a new project
        # -----> Keras Data
        # separator
        # ttk.Separator(LoadDataFrame, orient=HORIZONTAL).grid(row=0, columnspan=3, sticky=W + E,
        #                                                      padx=(10, 10), pady=(0, 0))
        self.KerasData_RadioVar = IntVar()
        self.tfDataRadio = Radiobutton(LoadDataFrame, text="load data from a TensorFlow library:",
                                       variable=self.KerasData_RadioVar, value=0, command=self._tfDataCombo_activate,
                                       **Widget_ColourConfig)

        style = ttk.Style()
        Combobox_colour_config(LoadDataFrame, Combobox_ColourConfig, style)
        libraries_names = list(data_libraries.keys())

        self._tfDataCombo = ttk.Combobox(LoadDataFrame, value=libraries_names, style='TCombobox')
        self._tfDataCombo["state"] = "readonly"
        self._tfDataCombo.bind("<<ComboboxSelected>>", self.updateDataSummary)


        # -----> .npz Data
        # Radio button for loading data form a .npz file
        self._npzRadio = Radiobutton(LoadDataFrame, text="load from a .npz file",
                                     variable=self.KerasData_RadioVar, value=1, command=self._npzDataCombo_activate,
                                     **Widget_ColourConfig)
        # Entry, Label and Button for the .npz file path
        self._npzPathEntry = Entry(LoadDataFrame, state="disable", **Colours["Entry"])
        self._npzLabel = Label(LoadDataFrame, text="File Path:", state="disable", **Colours["Labels"])
        self._npzBrowseButton = Button(LoadDataFrame, text="...", command=self._npzBrowseFile, state="disable",
                                       **Button_ColourConfig)
        # -----> .csv Data
        # Radio button for loading data form a .csv file
        self._csvRadio = Radiobutton(LoadDataFrame, text="load from a .csv file",
                                     variable=self.KerasData_RadioVar, value=2, command=self._csvDataCombo_activate,
                                     **Widget_ColourConfig)
        # Entry for the .csv file path
        self._csvPathEntry = Entry(LoadDataFrame, state="disable", **Colours["Entry"])
        self._csvLabel = Label(LoadDataFrame, text="File Path:", state="disabled", **Colours["Labels"])

        # Browse button for .csv file path
        self._csvBrowseButton = Button(LoadDataFrame, text="...", command=self._csvBrowseFile, state="disable",
                                       **Button_ColourConfig)

        # ------- grid the widgets of the LoadDataFrame frame ------- #
        self.tfDataRadio.grid(row=1, column=0, columnspan=2, sticky=W, pady=(0, 0))
        self._tfDataCombo.grid(row=2, column=1)

        self._npzRadio.grid(row=3, column=0, sticky=W)
        self._npzPathEntry.grid(row=4, column=1, sticky=W)
        self._npzLabel.grid(row=4, column=0, sticky=E)
        self._npzBrowseButton.grid(row=4, column=2, sticky=W)

        self._csvRadio.grid(row=6, column=0, sticky=W)
        self._csvLabel.grid(row=7, column=0, sticky=E)
        self._csvPathEntry.grid(row=7, column=1, sticky=W)
        self._csvBrowseButton.grid(row=7, column=2, sticky=W)

        # ------------------------------------------------------------------------------- #
        # ------------------------------------------------------------------------------- #
        # Manipulate the Data
        # -----> Reshape the input
        self.reshape_RadioVar = IntVar()
        self._reshapeCheckbutton = Checkbutton(ManipulateDataFrame, text="Reshape the input",
                                               variable=self.reshape_RadioVar, command=self._reshapeCheck_activate,
                                               **Widget_ColourConfig)
        self._reshapeLabel = Label(ManipulateDataFrame, text="Shape:", state="disable", **Label_ColourConfig)
        v = StringVar(parent, value='Example: (28, 28, 1)')
        self._reshapeEntry = Entry(ManipulateDataFrame, textvariable=v, state="disable", **Entry_ColourConfig)
        self._reshapeEntry.bind("<Button-1>", self.reshapeEntry_on_click)
        self._reshapeEntry.bind("<Return>", self._reshapeData_on_enter)

        # -----> Normalize the data
        self.normalize_RadioVar = IntVar()
        self._normalizeCheckbutton = Checkbutton(ManipulateDataFrame, text="Normalize",
                                                 variable=self.normalize_RadioVar, command=self._normCheck_activate,
                                                 **Widget_ColourConfig)
        self._normalizeLabel_div = Label(ManipulateDataFrame, text="div by:", state="disable", **Label_ColourConfig)
        self._normalizeEntry_div = Entry(ManipulateDataFrame, state="disable", width=5, **Entry_ColourConfig)

        self._normalizeLabel_minus = Label(ManipulateDataFrame, text="minus:", state="disable", **Label_ColourConfig)
        self._normalizeEntry_minus = Entry(ManipulateDataFrame, state="disable", width=5, **Entry_ColourConfig)
        # -----> One hot encoding
        self.toCategorical_RadioVar = IntVar()
        self._toCategoricalCheckbutton = Checkbutton(ManipulateDataFrame, text="One hot encode the data",
                                                     variable=self.toCategorical_RadioVar,
                                                     command=self._normCheck_activate, **Widget_ColourConfig)

        # ------- grid the widgets of the ManipulateDataFrame frame ------- #
        self._reshapeCheckbutton.grid(row=0, column=0, sticky=W)
        self._reshapeLabel.grid(row=1, column=0, sticky=E)
        self._reshapeEntry.grid(row=1, column=1, columnspan=2, sticky=W)

        self._normalizeCheckbutton.grid(row=2, column=0, sticky=W)
        self._normalizeLabel_div.grid(row=3, column=0, sticky=E)
        self._normalizeEntry_div.grid(row=3, column=1, sticky=W)
        self._normalizeLabel_minus.grid(row=3, column=1, sticky=E, padx=(25, 0))
        self._normalizeEntry_minus.grid(row=3, column=2, sticky=W)

        self._toCategoricalCheckbutton.grid(row=4, column=0, sticky=E)

        # ------------------------------------------------------------------------------- #
        # ------------------------------------------------------------------------------- #
        self._title = Label(DataSummaryFrame, text="Data Summary", bg="black", fg="white")
        self._trainShape = Label(DataSummaryFrame, text="Train Data Shape:", bg="black", fg="white")
        self._trainShape_val = Label(DataSummaryFrame, text="", bg="black", fg="white")
        self._testShape = Label(DataSummaryFrame, text="Test Data Shape:", bg="black", fg="white")
        self._testShape_val = Label(DataSummaryFrame, text="", bg="black", fg="white")
        self._trainMean = Label(DataSummaryFrame, text="Train Data Mean:", bg="black", fg="white")
        self._trainMean_val = Label(DataSummaryFrame, text="", bg="black", fg="white")

        # ------- grid the widgets of the ManipulateDataFrame frame ------- #
        self._title.grid(row=0, column=0, columnspan=3)
        self._trainShape.grid(row=1, column=0, sticky=W)
        self._trainShape_val.grid(row=1, column=1, sticky=E)
        self._testShape.grid(row=2, column=0, sticky=W)
        self._testShape_val.grid(row=2, column=1, sticky=E)
        self._trainMean.grid(row=3, column=0, sticky=W)
        self._trainMean_val.grid(row=3, column=1, sticky=E)

        # ------------------------------------------------------------------------------- #

    def _tfDataState(self, state="normal"):
        """
          Helper function for activating and disabling the choices
          for loading data from a keras library.

          """
        if state == "normal":
            self._tfDataCombo.configure(state="readonly")
        elif state == "disable":
            self._tfDataCombo.configure(state="disable")
        else:
            raise ValueError("_tfDataState: Unknown state. It must be -normal or -disable")

    def _npzDataState(self, state="normal"):
        """
          Helper function for activating and disabling the choices
          for loading data from a .npz file.

          """
        if state == "normal":
            self._npzRadio.configure(state="normal")
            self._npzPathEntry.configure(state="normal")
            self._npzBrowseButton.configure(state="normal")
            self._npzLabel.configure(state="normal")
        elif state == "disable":
            self._npzRadio.configure(state="normal")
            self._npzPathEntry.configure(state="disable")
            self._npzBrowseButton.configure(state="disable")
            self._npzLabel.configure(state="disable")
        else:
            raise ValueError("_npzDataState: Unknown state. It must be -normal or -disable")

    def _csvDataState(self, state="normal"):
        """
         Helper function for activating and disabling the choices
         for loading data from a .csv file.

           """
        if state == "normal":
            self._csvRadio.configure(state="normal")
            self._csvPathEntry.configure(state="normal")
            self._csvBrowseButton.configure(state="normal")
            self._csvLabel.configure(state="normal")
        elif state == "disable":
            self._csvRadio.configure(state="normal")
            self._csvPathEntry.configure(state="disabled")
            self._csvBrowseButton.configure(state="disable")
            self._csvLabel.configure(state="disabled")
        else:
            raise ValueError("_csvDataState: Unknown state. It must be -normal or -disable")

    def _tfDataCombo_activate(self):
        self._npzRadio.deselect()
        self._tfDataCombo.configure(state="readonly")
        self._npzDataState(state="disable")
        self._csvDataState(state="disable")

    def _npzDataCombo_activate(self):
        self.tfDataRadio.deselect()
        self._csvDataState(state="disable")
        self._tfDataState(state="disable")
        self._npzDataState(state="normal")

    def _csvDataCombo_activate(self):
        self._tfDataState(state="disable")
        self._npzDataState(state="disable")
        self._csvDataState(state="normal")

    def get_InputConf(self):
        Conf = {
            "Use_Keras_Data": self.KerasData_RadioVar.get() == 0,
            "Keras_data": {
                "Data_library": self._tfDataCombo.get()
            },
            "Use_npz_Data": self.KerasData_RadioVar.get() == 1,
            "Reshape": {
                "Do_reshape": self.reshape_RadioVar.get() == 1,
                "Shape": self._reshapeEntry.get()
            }
        }
        return Conf

    def load_the_data(self, InputConf=None):
        if InputConf is None:
            InputConf = self.get_InputConf()

        try:
            self.data = load_the_data(InputConf)
            self.Configuration = InputConf
            return self.data
        except KeyError:
            messagebox.showerror("Data Load", "KeyError")

    def updateDataSummary(self, event):
        # Update the data summary
        # shape
        self._trainShape_val.config(text="Loading...")
        self._testShape_val.config(text="Loading...")
        self._trainMean_val.config(text="Loading...")

        self.DataSummaryFrame.update_idletasks()
        self.DataSummaryFrame.update()
        self.parent.update_idletasks()
        self.parent.update()

        self.data = self.load_the_data()

        train_data_X, train_data_Y = self.data["train"]
        train_data_shape = train_data_X.shape
        self._trainShape_val.config(text=str(train_data_shape))
        self._testShape_val.config(text=str(train_data_Y.shape))
        # mean
        mean = np.mean(train_data_X)
        self._trainMean_val.config(text=str(mean))
        # self.LoadingLabel.config(text="...")

    def _npzBrowseFile(self):
        if os.path.exists("/"):
            initial_dir = "/"
        elif os.path.exists("C:"):
            initial_dir = "C:"
        else:
            raise ValueError("Unknown file system")

        filePath = filedialog.askopenfilename(initialdir=initial_dir, title="Select file",
                                              filetypes=(("numpy files", "*.npz"),
                                                         ("all files", "*.*")))
        self._npzPathEntry.delete(0, END)
        self._npzPathEntry.insert(END, filePath)

    def _csvBrowseFile(self):
        if os.path.exists("/"):
            initial_dir = "/"
        elif os.path.exists("C:"):
            initial_dir = "C:"
        else:
            raise ValueError("Unknown file system")

        filePath = filedialog.askopenfilename(initialdir=initial_dir, title="Select file",
                                              filetypes=(("CSV", "*.csv"),
                                                         ("all files", "*.*")))
        self._csvPathEntry.delete(0, END)
        self._csvPathEntry.insert(END, filePath)

    def _normCheck_activate(self):
        if self.normalize_RadioVar.get():
            self._normalizeLabel_div.configure(state="normal")
            self._normalizeEntry_div.configure(state="normal")
            self._normalizeLabel_minus.configure(state="normal")
            self._normalizeEntry_minus.configure(state="normal")
        else:
            self._normalizeLabel_div.configure(state="disabled")
            self._normalizeEntry_div.configure(state="disabled")
            self._normalizeLabel_minus.configure(state="disabled")
            self._normalizeEntry_minus.configure(state="disabled")

    def _reshapeCheck_activate(self):
        if self.reshape_RadioVar.get():
            self._reshapeLabel.configure(state="normal")
            self._reshapeEntry.configure(state="normal")
        else:
            self.load_the_data()
            # self.updateDataSummary(None)
            self._reshapeLabel.configure(state="disabled")
            self._reshapeEntry.configure(state="disabled")

    def reshapeEntry_on_click(self, event):
        if self._reshapeEntry.get() == "Example: (28, 28, 1)":
            self._reshapeEntry.delete(0, END)

    def _reshapeData_on_enter(self, event):
        if self.data is not None:
            self.updateDataSummary(None)


# <<<<<<<<<<<<<<<<<<<<<<< Input Config Window >>>>>>>>>>>>>>>>>>>>>>>>>>> #
class InputLayerWindow:
    """ The window for choosing and prepossessing the input data.
    The app can load data from the tensorflow libraries or from a
    numpy file (*.npz). Through this window the user can also choose
    to flatten the input or to normalize it. When the window is clossed
    by the "Next" button, the appropriate load_data function is called
    and the data are alocated to the "data" class variable so that they
    could be achesed after the window is closed.

    Example:
    >>> Input = InputLayerWindow()
    >>> (train_X, train_Y), (test_X, test_Y) = Input.data

    Contains:
    - Radiobutton and combobox to choose a tensorflow data librarie.
    - Radiobutton, Entry and Browse button to choose a .npz data file.
    - Checkbuttons for flattening and normilizing the data.
    - Entry for the number with witch to normilize the data if the
      user wishes so.
    - "Quit" and "Next" Buttons

    """
    def __init__(self):
        Width = 700
        Height = 520
        self.Width = Width

        self.root = Tk()
        self.root.title("Open")
        self.root.geometry(f"{Width}x{Height}")
        self.root.resizable(True, True)
        self.data = None
        self.NextPressed = False

        Colours = colourConfig["Input"]

        # =============================================================================== #
        #                                    Frames                                       #
        # =============================================================================== #
        # -------> Top Frame
        TopFrame = Frame(self.root, height=90, width=Width, **Colours["Frame"])
        TopFrame.grid_propagate(0)
        TopFrame.grid(row=0, column=0, sticky=W+E)

        # -------> Middle Frame Base
        MiddleFrame_H = 397
        self.MiddleFrame_H = MiddleFrame_H
        self.MiddleFrameBase = Frame(self.root, height=MiddleFrame_H, width=Width, **Colours["Frame"])
        self.MiddleFrameBase.grid_propagate(0)
        self.MiddleFrameBase.grid(row=1, column=0, sticky=W+E)

        # -------> Bottom Frame
        BottomFrame = Frame(self.root, height=40, width=Width, **Colours["Frame"])
        BottomFrame.grid_propagate(0)
        BottomFrame.pack_propagate(0)
        BottomFrame.grid(row=2, column=0, sticky=W+E)
        # =============================================================================== #

        # ------------------------------------------------------------------------------- #
        # =======> Opening a existing or a new project
        self.OpenProject_RadioVar = IntVar()
        # Choose an existing project
        self.OpenProject_Radiobutton = Radiobutton(TopFrame, text="Open an existing project: ",
                                                   variable=self.OpenProject_RadioVar, value=0,
                                                   command=self._openProject, **Colours["Radiobutton"])
        self._ProjectDirEntry = Label_and_Entry(TopFrame, "Project Directory: ", Colours,
                                                row=1, starting_column=1, width=30)


        self._OpenProjectButton = Button(TopFrame, text="...", command=self._openProject_dialog, width=1, height=1,
                                         **Colours["Button"])
        self._ModelPreviewButton = Button(TopFrame, text="Preview", **Colours["Button"])
        # Choose to create a new project
        self.NewProject = Radiobutton(TopFrame, text="New project", variable=self.OpenProject_RadioVar, value=1,
                                      command=self._newProject,
                                      **Colours["Radiobutton"])

        # separator
        ttk.Separator(TopFrame, orient=HORIZONTAL).grid(row=3, columnspan=5, sticky=W + E,
                                                        padx=(0, 0), pady=(2, 2))

        # ------- grid the widgets of the top frame ------- #
        self.OpenProject_Radiobutton.grid(row=0, column=0, sticky=W)
        self._OpenProjectButton.grid_propagate(0)
        self._OpenProjectButton.grid(row=1, column=3, padx=(5, 0))
        self._ModelPreviewButton.grid(row=1, column=4, padx=(5, 0))
        self.NewProject.grid(row=2, column=0, sticky=W, pady=(2, 0))

        # ------------------------------------------------------------------------------- #
        # =======> Buttons
        # separator
        ttk.Separator(BottomFrame, orient=HORIZONTAL).pack(fill="x")

        Button(BottomFrame, text="Next", command=self.Next, **Colours["Button"]).pack(side=RIGHT, padx=(5, 5))
        Button(BottomFrame, text="Cancel", command=self.root.destroy, **Colours["Button"]).pack(side=RIGHT, padx=(5, 5))

        self.root.mainloop()

    def _openProject_dialog(self):
        if os.path.exists("/"):
            initial_dir = "/"
        elif os.path.exists("C:"):
            initial_dir = "C:"
        else:
            raise ValueError("Unknown file system")

        self.ProjectFilePath = filedialog.askdirectory(initialdir=initial_dir, title="Select file")

        self._ProjectDirEntry.delete(0, END)
        self._ProjectDirEntry.insert(END, self.ProjectFilePath)

    def _newProject(self):
        # deactivate the new project widgets
        self._ProjectDirEntry.configure(state="disable")
        self._OpenProjectButton.configure(state="disable")
        self._ModelPreviewButton.configure(state="disable")

        Colours = colourConfig["Input"]
        MiddleFrame = Frame(self.MiddleFrameBase, height=self.MiddleFrame_H, width=self.Width, **Colours["Frame"])
        self.MiddleFrame = MiddleFrame
        MiddleFrame.grid_propagate(0)
        MiddleFrame.grid(row=1, column=0, sticky=W + E)
        self.ProjectWindow = NewProjectWindow(self.MiddleFrameBase, self.MiddleFrame, self.Width, self.MiddleFrame_H)

    def _openProject(self):
        # deactivate the new project widgets
        self._ProjectDirEntry.configure(state="normal")
        self._OpenProjectButton.configure(state="normal")
        self._ModelPreviewButton.configure(state="normal")

        self.MiddleFrame.destroy()

    def load_the_data(self, InputConf):
        self.data = load_the_data(InputConf)

    def Next(self):
        self.NextPressed = True
        self.root.destroy()
