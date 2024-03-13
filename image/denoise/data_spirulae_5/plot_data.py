import numpy as np
import matplotlib.pyplot as plt
import json


def plot_images(images, names=None):
    n = len(images)
    num_rows = int(np.ceil(np.sqrt(n)))
    num_cols = int(np.ceil(n / num_rows))
    
    # Create a figure and subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    
    for i in range(n):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col] if n > 1 else axs
        
        # Plot the image
        #print(np.mean(images[i]), np.mean(images[i]==0.0))
        img = np.einsum('kij->ijk', images[i])
        #img = img ** (1/2.2)
        img = 1.019*img/(img+0.155)
        ax.imshow(img, cmap='gray', origin='lower', vmin=0, vmax=1, interpolation='none')
        if names:
            ax.set_title(names[i])
        ax.axis('off')
    
    # Hide any unused subplots
    for i in range(n, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axs[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()


def load_data(filename):
    with open(filename, 'rb') as fp:
        flat_data = np.fromfile(fp, dtype=np.byte)
    header_length = np.frombuffer(flat_data[:4], dtype=np.uint32)[0]
    #print(header_length)
    header = flat_data[4:4+header_length].tobytes().decode('utf-8')
    #print(header)
    buffer = flat_data[4+header_length:]
    bufferview = json.loads(header)['buffers']
    for info in bufferview:
        b = buffer[info['byte_offset']:info['byte_offset']+info['byte_length']]
        dtype = np.__dict__[info['type']]
        img = np.frombuffer(b, dtype=dtype).reshape(info['shape'])
        info['buffer'] = img
    return bufferview


def plot_data(filename):
    data = load_data(filename)
    plot_images([_['buffer'] for _ in data],
                [_['name'] for _ in data])

def plot_all_data(dirname):
    import os
    images = []
    names = []
    total_w = 0.0
    for root, dirs, files in os.walk(dirname):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                bufferview = load_data(filepath)
                img_sum = 0.0 * bufferview[0]['buffer']
                w = 0.0
                for image in bufferview:
                    if not image['name'] >= 1:
                        continue
                    w += image['name']
                    img_sum += image['name'] * image['buffer']
                images.append(img_sum/w)
                names.append(filename)
                total_w += w+1.0
            except:
                pass
    print(len(images), "images,", total_w/len(images), "spp average")
    plot_images(images)


if __name__ == "__main__":
    
    #filename = "data/implicit3-rt_0adeb0f1623f_256.bin"
    #plot_data(filename)

    dirname = "data/"
    plot_all_data(dirname)

