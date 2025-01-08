"""plot and visualization functions for cs131 hw7"""
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import downscale_local_mean, rescale, resize


def plot_part1(avg_face, face_hog):
    """plot average face and hog representatitons of face."""
    plt.subplot(1, 2, 1)
    plt.imshow(avg_face)
    plt.axis('off')
    plt.title('average face image')

    plt.subplot(1, 2, 2)
    plt.imshow(face_hog)
    plt.title('hog representation of face')
    plt.axis('off')

    plt.show()


def plot_part2(image, r, c, response_map_resized, response_map, winW, winH):
    """plot window with highest hog score and heatmap."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10,15))

    im = ax1.imshow(image)
    rect = patches.Rectangle((c - winW // 2, r - winH // 2),
                             winW,
                             winH,
                             linewidth=1,
                             edgecolor='r',
                             facecolor='none')
    ax1.add_patch(rect)
    fig.colorbar(im, ax=ax1)

    ax2.set_title('Sliding Window Response Map')
    im = ax2.imshow(response_map_resized, cmap='viridis', interpolation='nearest')
    fig.colorbar(im, ax=ax2)

    ax3.set_title('Unresized Sliding Window Response Map')
    im = ax3.imshow(response_map, cmap='viridis', interpolation='nearest')
    fig.colorbar(im, ax=ax3)
    plt.tight_layout()
    plt.show()


def plot_part3_1(images):
    """plot image pyramid."""
    sum_r = 0
    sum_c = 0
    for i, result in enumerate(images):
        (scale, image) = result
        if i == 0:
            sum_c = image.shape[1]
        sum_r += image.shape[0]

    composite_image = np.zeros((sum_r, sum_c))

    pointer = 0
    for i, result in enumerate(images):
        (scale, image) = result
        composite_image[pointer:pointer +
                        image.shape[0], :image.shape[1]] = image
        pointer += image.shape[0]

    plt.imshow(composite_image)
    plt.axis('off')
    plt.title('Image Pyramid')
    plt.show()


def plot_part3_2(image, max_scale, winW, winH, maxc, maxr, max_response_map):
    """plot window with highest hog score and heatmap."""
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10,10))
    im = ax1.imshow(rescale(image, max_scale))
    rect = patches.Rectangle((maxc - winW // 2, maxr - winH // 2),
                             winW,
                             winH,
                             linewidth=1,
                             edgecolor='r',
                             facecolor='none')
    ax1.add_patch(rect)
    fig.colorbar(im, ax=ax1)

    ax2.set_title('Pyramid Score Response Map')
    im = ax2.imshow(max_response_map, cmap='viridis', interpolation='nearest')
    fig.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.show()


def plot_part4(avg, hog, part_name):
    """plot average and hog representatitons of deformable parts."""
    plt.subplot(1, 3, 1)
    plt.imshow(avg)
    plt.axis('off')
    plt.title('average ' + part_name + ' image')

    plt.subplot(1, 3, 2)
    plt.imshow(hog)
    plt.axis('off')
    plt.title('average hog image')
    plt.show()


def plot_part5_1(response_map):
    """plot heatmaps."""
    fig, ax = plt.subplots(1, figsize=(10,5))
    im = ax.imshow(response_map, cmap='viridis', interpolation='nearest')
    fig.colorbar(im, ax=ax)
    plt.show()


def plot_part5_2_face(face_heatmap_shifted):
    """plot heatmaps."""
    fig, ax = plt.subplots(1, figsize=(10,5))
    im = ax.imshow(face_heatmap_shifted, cmap='viridis', interpolation='nearest')
    fig.colorbar(im, ax=ax)
    plt.show()


def plot_part5_2_parts(lefteye_heatmap_shifted, righteye_heatmap_shifted,
                 nose_heatmap_shifted, mouth_heatmap_shifted):
    """plot heatmaps."""
    f, axarr = plt.subplots(2, 2, figsize=(14,7))

    im = axarr[0, 0].imshow(
        lefteye_heatmap_shifted, cmap='viridis', interpolation='nearest')
    f.colorbar(im, ax=axarr[0,0])

    im = axarr[0, 1].imshow(
        righteye_heatmap_shifted, cmap='viridis', interpolation='nearest')
    f.colorbar(im, ax=axarr[0,1])

    im = axarr[1, 0].imshow(
        nose_heatmap_shifted, cmap='viridis', interpolation='nearest')
    f.colorbar(im, ax=axarr[1,0])

    im = axarr[1, 1].imshow(
        mouth_heatmap_shifted, cmap='viridis', interpolation='nearest')
    f.colorbar(im, ax=axarr[1,1])

    plt.show()


def plot_part6_1(winH, winW, heatmap, image, i, j):
    """plot heatmaps and optimal window."""
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10,10))
    im = ax1.imshow(resize(image, heatmap.shape))
    rect = patches.Rectangle((j - winW // 2, i - winH // 2),
                             winW,
                             winH,
                             linewidth=1,
                             edgecolor='r',
                             facecolor='none')
    ax1.add_patch(rect)
    fig.colorbar(im, ax=ax1)

    ax2.set_title('Gaussian Filter Heatmap')
    im = ax2.imshow(heatmap, cmap='viridis', interpolation='nearest')
    rect = patches.Rectangle((j - winW // 2, i - winH // 2),
                             winW,
                             winH,
                             linewidth=1,
                             edgecolor='r',
                             facecolor='none')
    ax2.add_patch(rect)
    fig.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.show()
