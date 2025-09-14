import numpy as np
import matplotlib.pyplot as plt

def debug_show_img(batch):
    first_img = batch[0].detach().cpu().numpy()
    first_img = np.transpose(first_img, (1, 2, 0))
    plt.figure(figsize=(4, 4))
    plt.imshow(first_img)
    plt.title("First Image in Batch")
    plt.axis("off")
    plt.show()