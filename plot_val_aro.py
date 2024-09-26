import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


x_axis = 'VAL'
y_axis = 'ARO'
title = x_axis +' - '+ y_axis
def visualize_single(data_set):
    df = pd.DataFrame(data_set[1], columns=['VAL', 'ARO', 'STATE'])

    color_map = {'NEUTRAL': 'yellow', 'PAIN': 'red', 'STRAIN': 'blue'}
    colors = df['STATE'].map(color_map)

    plt.figure(figsize=(10, 6))
    dot_size = 100
    plt.scatter(df['VAL'], df['ARO'], c=colors, edgecolors='k', s=dot_size)

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(data_set[0])

    size = 10
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Neutral', markersize=size, markerfacecolor='yellow', markeredgecolor='k'),
                       Line2D([0], [0], marker='o', color='w', label='Happy', markersize=size, markerfacecolor='green',markeredgecolor='k'),
                       Line2D([0], [0], marker='o', color='w', label='Pain', markersize=size, markerfacecolor='red', markeredgecolor='k'),
                       Line2D([0], [0], marker='o', color='w', label='Strain', markersize=size, markerfacecolor='blue', markeredgecolor='k')]
    plt.legend(handles=legend_elements, loc='best')

    plt.grid(True)
    plt.show()

def visualize_va(main_dic, drawJoined=False):
    folderPath = main_dic + "\\VAL_ARO"
    os.chdir(folderPath)

    all_files = glob.glob('**/*.csv', recursive=True)
    datas = []
    for file in all_files:

        csv = pd.read_csv(file)
        rows = []
        for row in csv.itertuples():
            my_list = [row.VAL, row.ARO, file.split('_')[2].split('.')[0]]

            rows.append(my_list)
        datas.append([file.split('_')[0],rows])
    print('4: Successfully extracted VAL and ARO from CSV files')

    #print(datas[0])

    data_joined = [title,[]]
    for data in datas:
        if not drawJoined:
            visualize_single(data)
        data_joined[1] += data[1]

    if drawJoined:
        visualize_single(data_joined)


    print('5: Successfully plotted VAL and ARO values')