import matplotlib as mpl

# Change the default style of matplotlib here
def set_style():
    mpl.rcParams['font.size'] = 15
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['figure.subplot.left'] = 0.17
    mpl.rcParams['figure.subplot.right'] = 0.945
    mpl.rcParams['figure.subplot.bottom'] = 0.16
    mpl.rcParams['figure.subplot.top'] = 0.93