import os
import json

import matplotlib.pyplot as plt


def main():
    path = f'var/log/resnet_cifar10_history.json'
    with open(path, 'r', encoding='UTF-8') as f:
        history = json.load(f)

    x = range(len(history['acc']))
    plt.figure(figsize=(6, 4))
    plt.plot(x, history['acc'], 'b--', label='train_acc', linewidth=.5)
    plt.legend(loc='lower right')
    plt.plot(x, history['val_acc'], 'r-', label='val_acc', linewidth=.5)
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.0])
    plt.xlim([0, 200])
    plt.grid(which='major', color='black', linestyle='--')
    plt.subplots_adjust(left=.1, right=.95, bottom=.1, top=.95)
    path = f'figure/resnet_cifar10_history.png'
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    plt.savefig(path)
    plt.close()


if __name__ == '__main__':
    main()
