from tkinter import *
from tkinter import ttk


def Combobox_colour_config(root, Combobox_ColourConfig, style):
    """Helper function to configure the
    colours of a combobox"""
    style.map('TCombobox', fieldbackground=[('readonly', Combobox_ColourConfig["drop_down"]),
                                            ('disabled', Combobox_ColourConfig["disabled"])])
    style.map('TCombobox', selectbackground=[('readonly', Combobox_ColourConfig["drop_down"])])
    style.map('TCombobox', selectforeground=[('readonly', Combobox_ColourConfig["foreground"])])
    style.map('TCombobox', foreground=[('readonly', Combobox_ColourConfig["foreground"])])

    root.option_add("*TCombobox*Listbox.background", Combobox_ColourConfig["drop_down"])
    root.option_add("*TCombobox*Listbox.foreground", Combobox_ColourConfig["foreground"])
    root.option_add("*TCombobox*Listbox.selectBackground", Combobox_ColourConfig["selectBackground"])


def Label_and_Entry(parent, Label_text, colour_config, row=0, starting_column=0, width=10,
                    column_span=None, pady=(0, 0), padx=(0, 0)):
    """Helper function for creating a Label and an Entry obj in the same row.
    """
    Label(parent, text=Label_text, **colour_config["Labels"]).grid(row=row, column=starting_column,
                                                                    columnspan=column_span,
                                                                    pady=pady, padx=padx,
                                                                    sticky="w")
    EntryObj = Entry(parent, **colour_config["Entry"], width=width)
    EntryObj.grid(row=row, column=starting_column+1, columnspan=column_span,
                  pady=pady, padx=padx, sticky="e")

    return EntryObj


def Label_and_2Entries(parent, Label_text, colour_config, row=0, starting_column=0, width=7,
                       column_span=1, pady=(0, 0), padx=(0, 0)):
    """
    Helper function for creating a Label with two entries in the same row in the form:
    Label: Entry x Entry

    :param parent: Where the widgets will be placed
    :param Label_text: The text of the label
    :param colour_config: Json format file with the colours for the labels and the entries
    :param row: The row of the widjets
    :param starting_column: The column where the Label will be
    :param width: Width of the entry objects
    :param column_span: Columnspan of the entry objects
    :param pady: pady for the entries
    :param padx: padx for the entries
    :return: The two entry objects
    """
    Label(parent, text=Label_text, **colour_config["Labels"]).grid(row=row, column=starting_column, sticky="w")
    Entry1 = Entry(parent, width=width, **colour_config["Entry"])
    Entry1.grid(row=row, column=starting_column+1, pady=pady, padx=padx,
                          columnspan=column_span, sticky="w")
    Label(parent, text="x", **colour_config["Labels"], width=2).grid(row=row, column=starting_column+2)
    Entry2 = Entry(parent, width=width, **colour_config["Entry"])
    Entry2.grid(row=row, column=starting_column+3, pady=pady, padx=padx, columnspan=column_span, sticky="e")

    return Entry1, Entry2


def Label_and_Combobox(parent, Label_text, dropdown_list, colour_config, row=0, starting_column=0, width=14,
                       column_span=1, pady=(0, 0), padx=(0, 0)):
    Label(parent, text=Label_text, **colour_config["Labels"]).grid(row=row, column=starting_column, sticky="w")
    ComboboxObj = ttk.Combobox(parent, value=dropdown_list, width=width, style='TCombobox')
    ComboboxObj["state"] = "readonly"
    ComboboxObj.grid(row=row, column=starting_column+1, columnspan=column_span,
                     pady=pady, padx=padx, sticky="e")

    return ComboboxObj


def get_tuple_from_str(string):
    """
    helper function to get a tuple from a str.
    For example:
    get_tuple_from_str("(1, 2, 3)") = (1, 2, 3)
    """
    charList = string[1:-1]
    charList = charList.split()
    intList = []
    for character in charList:
        if ',' in character:
            character = character.replace(',', '')
        intList.append(int(character))
    intTuple = tuple(intList)
    return intTuple


def layer_position(Frame_to_use, Layers_names, Radiobutton_active, Radiobutton_disabled,
                   colour_config, row=0):
    """
    Helper function for giving the options of the position of the layer to the user in the
    configuration window of each layer.

    :param row: The row of the label
    :param Frame_to_use: The frame in witch the widgets will be placed.
    :param Layers_names: The names of the layers that will be displayed in the Pos_Combobox
                         drop down menu.
    :param Radiobutton_active: function for activating the Pos_Combobox when the option to
                               put the layer after another layer is chosen.
    :param Radiobutton_disabled: function for de-activating the Pos_Combobox when the option to
                               put the layer in the end is chosen.
    :return: PosRadiobuttonVar - the choice between putting the layer at the end or after another layer.
             Pos_Combobox - the combobox that indicates after which layer the current layer will be placed
                            ig chosen to not place it in the end of the network.
    """
    # Radiobutton and labels for the position of the layer
    Label(Frame_to_use, text="Position",
          **colour_config["Labels"]).grid(row=row, column=0, columnspan=3)

    PosRadiobuttonVar = IntVar()
    PutInEnd_RadButton = Radiobutton(Frame_to_use, text="In the End", variable=PosRadiobuttonVar, value=0,
                                     command=Radiobutton_disabled, **colour_config["Radiobutton"])
    PutAfter_RadButton = Radiobutton(Frame_to_use, text="After Layer", variable=PosRadiobuttonVar, value=1,
                                     command=Radiobutton_active, **colour_config["Radiobutton"])
    PutInEnd_RadButton.grid(row=row+1, column=0, sticky="w")
    PutAfter_RadButton.grid(row=row+1, column=2, sticky="e")

    # Combobox for the position of the layer
    Pos_Combobox = ttk.Combobox(Frame_to_use, value=Layers_names, state="disabled")
    Pos_Combobox.grid(row=row+2, column=2, pady=(5, 10), padx=(20, 0), sticky="e")

    return PosRadiobuttonVar, Pos_Combobox
