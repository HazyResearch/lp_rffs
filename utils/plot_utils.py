import matplotlib
import matplotlib.pyplot as plt

#def get_colors():
#  prop_cycle = plt.rcParams['axes.prop_cycle']
#  colors = prop_cycle.by_key()['color']
#  colors_dict = {}
#  colors_dict["exact"] = colors[0]
#  colors_dict["fp"] = colors[1]
#  for idx, nbit in enumerate([1,2,4,8,16,32] ):
#    colors_dict[str(nbit)] = colors[idx + 2]
#  colors_dict["pca"] = colors[len(colors_dict.keys() ) ]
#  colors_dict["pca_2"] = colors[len(colors_dict.keys() ) ]
#  # colors_dict["pca_3"] = colors[len(colors_dict.keys() ) ]
#  # colors_dict["pca_4"] = colors[len(colors_dict.keys() ) ]
#  #print colors_dict
#  return colors_dict

def get_colors():
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors_dict = {}
    colors_dict["Exact"] = colors[0]
    colors_dict["FP-RFF"] = colors[1]
    colors_dict["FP-Nystrom"] = colors[2]
    colors_dict["Cir. FP-RFF"] = colors[3]
    for idx, nbit in enumerate([1,2,4,8,16] ):
        colors_dict["LP-RFF " + str(nbit) ] = colors[idx + 4]
#    for idx, nbit in enumerate([1,2,4,8,16] ):
#        colors_dict["Cir. LP-RFF " + str(nbit) ] = colors[idx + 4]
#     colors_dict["pca"] = colors[len(colors_dict.keys() ) ]
#     colors_dict["pca_2"] = colors[len(colors_dict.keys() ) ]
    # colors_dict["pca_3"] = colors[len(colors_dict.keys() ) ]
    # colors_dict["pca_4"] = colors[len(colors_dict.keys() ) ]
    #print colors_dict
    return colors_dict

#def get_colors():
#    prop_cycle = plt.rcParams['axes.prop_cycle']
#    colors = prop_cycle.by_key()['color']
#    colors_dict = {}
#    colors_dict["exact"] = colors[0]
#    colors_dict["fp RFF"] = colors[1]
#    colors_dict["fp Nystrom"] = colors[2]
#    colors_dict["fp cir. RFF"] = colors[3]
#    for idx, nbit in enumerate([1,2,4,8,16] ):
#        colors_dict["lp cir. RFF " + str(nbit) + " bits"] = colors[idx + 4]
##     colors_dict["pca"] = colors[len(colors_dict.keys() ) ]
##     colors_dict["pca_2"] = colors[len(colors_dict.keys() ) ]
#    # colors_dict["pca_3"] = colors[len(colors_dict.keys() ) ]
#    # colors_dict["pca_4"] = colors[len(colors_dict.keys() ) ]
#    #print colors_dict
#    return colors_dict

