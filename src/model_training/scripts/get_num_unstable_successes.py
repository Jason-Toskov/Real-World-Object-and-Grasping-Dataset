from src.scripts.data_loading_tools import make_grasp_data_generator
import json
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from labellines import labelLines

if __name__ == '__main__':
    # load config
    with open('../../../config.json', 'r') as config_file:
        config = json.load(config_file)

    grasp_generator = make_grasp_data_generator(
        config, all_data=True, use_processed=True)

    i = 0
    unstable_by_object = {}
    for grasp in tqdm(grasp_generator):
        datapoint, json_dict, rgb_img, depth_img, pcl_obj = grasp

        if unstable_by_object.get(json_dict['object_id']) is None:
            unstable_by_object[json_dict['object_id']] = {}
            unstable_by_object[json_dict['object_id']]['unstable_success'] = 0
            unstable_by_object[json_dict['object_id']]['success'] = 0
            unstable_by_object[json_dict['object_id']]['stable_success'] = 0

        # get the number of unstable successes
        if json_dict['success'] == 1 and json_dict['stable_success'] == 0:
            unstable_by_object[json_dict['object_id']]['unstable_success'] += 1
            i += 1

        if json_dict['success'] == 1:
            unstable_by_object[json_dict['object_id']]['success'] += 1
            # i += 1

        if json_dict['stable_success'] == 1:
            unstable_by_object[json_dict['object_id']]['stable_success'] += 1
            # i += 1

    for obj_id in unstable_by_object.keys():
        assert unstable_by_object[obj_id]['success'] == unstable_by_object[obj_id]['unstable_success'] \
            + unstable_by_object[obj_id]['stable_success']

    print(unstable_by_object)
    print(i)

    with open('/home/jason/Real-World-Object-and-Grasping-Dataset/fig_data/success_stability_data.json', 'w') as f:
        json.dump(unstable_by_object, f)

    # Plot in scatter
    points_x = []
    points_y = []
    for obj_id in unstable_by_object.keys():
        points_x.append(unstable_by_object[obj_id]["success"]/75*100)
        points_y.append(unstable_by_object[obj_id]["stable_success"]/75*100)
        
    fig = plt.figure("Success rate", (16,12))
    ax = fig.add_subplot(111)
    
    used_cmap = mpl.cm.tab20
    plt.scatter(points_x, points_y, s=80, c=list(range(1,21)), cmap=used_cmap)
    plt.xlim([0,100])
    plt.ylim([0,100])
    plt.plot(list(range(101)),list(range(101)), 'k--', label="# Successes = # Stable Successes")

    labelLines(ax.get_lines(), fontsize=16, xvals=[30],backgroundcolor="white")
    
    label_font = {'size':18}
    
    ax.spines['right'].set_position(('data', 50))
    ax.spines['left'].set_position(('data', 50))
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_position(('data', 50))
    ax.spines['bottom'].set_position(('data', 50))
    ax.spines['bottom'].set_color('none')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('right')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.set_xlabel("Success rate (%)", loc='left', fontdict=label_font)
    ax.set_ylabel("Stable success rate (%)", loc='top', fontdict=label_font)
    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=used_cmap), location="right")
    
    tick_locs = (np.arange(20) + 0.5)/20
    
    labels = ["Fish Flakes",
            "Sunscreen Roll On",
            "Bathroom Cleaner",
            "Bubbles",
            "Burger Sauce",
            "Nail Polish Remover",
            "Diced Tomatoes Can",
            "Hazelnut Spread",
            "Intimate Wash",
            "Dishwashing Liquid",
            "Glow Sticks",
            "Dishwasher Powder",
            "Chest Rub Ointment",
            "Gel Nail Polish Remover",
            "Toilet Cleaner",
            "Soap Box",
            "BBQ Sauce",
            "Water Crackers",
            "Salt",
            "Sunscreen Tube"]
    cbar.set_ticks(tick_locs, labels=labels)
    cbar.ax.tick_params(labelsize=14)
    
    ax.set_aspect('equal')

    plt.savefig('/home/jason/Real-World-Object-and-Grasping-Dataset/fig_data/success_v_stablesuccess.png')

    plt.show()