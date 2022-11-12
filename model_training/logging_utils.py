import json

import numpy as np 
import matplotlib.pyplot as plt

class MetricLogger:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.object_map = config['object_ds_data']['obj_grasp_id_map']

        self.metrics = {}
        self.metrics['train_loss'] = []
        self.metrics['test_loss'] = []
        self.metrics['train_acc'] = []
        self.metrics['test_acc'] = []
        # per object final accuracy
        self.metrics['acc_per_obj'] = {mode:{id:0 for id, obj in self.object_map.items()} for mode in ['train', 'test']}

    def update(self, train_loss, test_loss, train_acc, test_acc):
        self.metrics['train_loss'].append(train_loss)
        self.metrics['test_loss'].append(test_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['test_acc'].append(test_acc)
    
    def update_per_object(self, acc, id, mode):
        self.metrics['acc_per_obj'][mode][id] = acc

    def save(self, out_path):
        with open(out_path+'metrics.json', 'w') as f:
            json.dump(self.metrics, f)
    
    def plot_epoch(self, out_path):
        # plot accuracy and loss separately
        plt.plot(self.metrics['train_loss'], label='Train')
        plt.plot(self.metrics['test_loss'], label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss plots\nHighest test loss: %.2f, Epoch: %i' % (np.max(self.metrics['test_loss']),np.argmax(self.metrics['test_loss'])))
        plt.xlim(0, len(self.metrics['train_loss']))
        plt.grid(True)
        plt.legend()
        plt.savefig(out_path+'loss.png')
        plt.clf()

        plt.plot(self.metrics['train_acc'], label='Train')
        plt.plot(self.metrics['test_acc'], label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy plots\nHighest test acc: %.2f %%, Epoch: %i' % (np.max(self.metrics['test_acc']),np.argmax(self.metrics['test_acc'])))
        plt.xlim(0, len(self.metrics['train_loss']))
        plt.ylim(0,100)
        plt.grid(True)
        plt.legend()
        plt.savefig(out_path+'accuracy.png')
    
    def per_object_acc_table(self, out_path):
        # save per object accuracy table
        with open(out_path+'per_object_acc_table.txt', 'w') as f:
            for row in ['Object', 'Train', 'Test']:
                str_to_write = '%s\t' % row
                for id, obj in self.object_map.items():
                    if row == 'Object':
                        str_to_write += '%s\t' % obj
                    else:
                        str_to_write += '%.2f\t' % self.metrics['acc_per_obj'][row.lower()][id]
                # replace last tab with newline
                str_to_write += '\n'
                f.write(str_to_write)