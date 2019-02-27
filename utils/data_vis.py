import matplotlib.pyplot as plt
import torch

def plot_img_and_mask(img, mask, ground_true):

    cmap = plt.cm.gray
    alpha = 0.3
    fig = plt.figure()
    a = fig.add_subplot(1, 3, 1)
    a.set_title('Input image')
    plt.axis("off")
    img = img.cpu()
    plt.imshow(img, cmap=cmap)

    b = fig.add_subplot(1, 3, 2)
    plt.axis("off")
    b.set_title('Output mask')
    mask = mask.cpu().detach().numpy()
    plt.imshow(img, cmap=cmap)
    plt.imshow(mask, cmap=cmap, alpha=alpha)

    c = fig.add_subplot(1, 3, 3)
    plt.axis("off")
    c.set_title('manual mask')
    plt.imshow(img, cmap=cmap)
    plt.imshow(ground_true, cmap=cmap, alpha=alpha)
    plt.show()



