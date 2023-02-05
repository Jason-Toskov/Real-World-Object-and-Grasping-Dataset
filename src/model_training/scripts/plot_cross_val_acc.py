from src.model_training.arg_parsing import parse_args
import os
import json

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    args = parse_args()
    output_path = args.output_path

    tmp_acc_dict = {}
    tmp_acc_obj_dict = {}
    for type in os.listdir(output_path):
        for stability in os.listdir(output_path + '/' + type):
            if stability not in tmp_acc_dict.keys():
                tmp_acc_dict[stability] = {}
            if stability not in tmp_acc_obj_dict.keys():
                tmp_acc_obj_dict[stability] = {}
            accs = []
            obj_accs = []
            for cross_val in os.listdir(output_path + '/' + type + '/' + stability):
                # load metrics
                with open(output_path + '/' + type + '/' + stability+'/'+cross_val+'/metrics.json', 'r') as f:
                    metrics = json.load(f)
                accs.append(metrics['metrics']['test_acc'])

                with open(output_path + '/' + type + '/' + stability+'/'+cross_val+'/per_object_acc_table.txt', 'r') as f:
                    obj_accs_txt = f.readlines()
                obj_accs.append([float(x) for x in obj_accs_txt[2].split('\t')[1:-1]])

            tmp_acc_obj_dict[stability][type] = np.mean(np.array(obj_accs),axis=0)
            tmp_acc_obj_dict[stability][type+'std']=np.std(np.array(obj_accs),axis=0)
            tmp_acc_dict[stability][type] = np.array(accs)
            # breakpoint()
    
    for stability in tmp_acc_dict.keys():
        for type in tmp_acc_dict[stability].keys():
            acc_means = np.mean(tmp_acc_dict[stability][type], axis=0)
            acc_stds = np.std(tmp_acc_dict[stability][type], axis=0)
            # breakpoint()
            plt.plot(range(len(acc_means)), acc_means, label=type)
            plt.fill_between(range(len(acc_means)), acc_means-acc_stds, acc_means+acc_stds, alpha=0.5)
        plt.xlim(0,200)
        plt.ylim(50,80)
        plt.legend(loc='lower right')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.savefig(output_path + '/' + stability + '_acc.png')
        plt.clf()
    
    for stability in tmp_acc_dict.keys():
        for type in tmp_acc_dict[stability].keys():
            max_accs = np.max(tmp_acc_dict[stability][type], axis=1)
            max_acc_means = np.mean(max_accs)
            max_acc_stds = np.std(max_accs)
            print(stability, type, np.round(max_accs,2), round(max_acc_means,2), round(max_acc_stds,2))

    print('\n')

    for stability in tmp_acc_dict.keys():
        print(stability)
        print("Fish_Flakes	Sunscreen_Roll_On	Bathroom_Cleaner	Bubbles	Burger_Sauce	Nail_Polish_Remover	Diced_Tomatoes_Can	Hazelnut_Spread	Intimate_Wash	Dishwashing_Liquid	Glow_Sticks	Dishwasher_Powder	Chest_Rub_Ointment	Gel_Nail_Polish_Remover	Toilet_Cleaner	Soap_Box	BBQ_Sauce	Water_Crackers	Salt	Sunscreen_Tube	")
        for type in tmp_acc_dict[stability].keys():
            print(type, np.round(np.mean(tmp_acc_obj_dict[stability][type]),2), np.round(tmp_acc_obj_dict[stability][type], 2))
            
    print("\n\n\n Standard devs:\n")
    for stability in tmp_acc_dict.keys():
        print(stability)
        print("Fish_Flakes	Sunscreen_Roll_On	Bathroom_Cleaner	Bubbles	Burger_Sauce	Nail_Polish_Remover	Diced_Tomatoes_Can	Hazelnut_Spread	Intimate_Wash	Dishwashing_Liquid	Glow_Sticks	Dishwasher_Powder	Chest_Rub_Ointment	Gel_Nail_Polish_Remover	Toilet_Cleaner	Soap_Box	BBQ_Sauce	Water_Crackers	Salt	Sunscreen_Tube	")
        for type in tmp_acc_dict[stability].keys():
            if "Gripper" in type:
                print(type, np.round(np.mean(tmp_acc_obj_dict[stability][type]),2), np.round(tmp_acc_obj_dict[stability][type+'std'], 2))