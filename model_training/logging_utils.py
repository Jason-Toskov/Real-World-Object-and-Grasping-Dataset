import json

import numpy as np 
import matplotlib.pyplot as plt

class MetricLogger:
    def __init__(self):
        self.metrics = {}
        self.metrics['train_loss'] = []
        self.metrics['test_loss'] = []
        self.metrics['train_acc'] = []
        self.metrics['test_acc'] = []

    def update(self, train_loss, test_loss, train_acc, test_acc):
        self.metrics['train_loss'].append(train_loss)
        self.metrics['test_loss'].append(test_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['test_acc'].append(test_acc)

    def save(self, out_path):
        with open(out_path+'metrics.json', 'w') as f:
            json.dump(self.metrics, f)
    
    def plot(self, out_path):
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