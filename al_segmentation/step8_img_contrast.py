import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from skimage import data, img_as_float
from skimage import exposure
from config import *
from glob import glob
from tqdm import tqdm

"""
matplotlib.rcParams['font.size'] = 8
def plot_img_and_hist(img, axes, bins=256):
    img = img_as_float(img)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(img, cmap=plt.cm.gray)
    ax_img.set_axis_off()
    ax_img.set_adjustable('box-forced')

    # Display histogram
    ax_hist.hist(img.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


# Load an example image
img = cv2.imread(RSNA_SEG_ALL+"/14879_seg.png", cv2.IMREAD_GRAYSCALE)

# Contrast stretching
p2, p98 = np.percentile(img, (2, 98))
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

# Equalization
img_eq = exposure.equalize_hist(img)

# Adaptive Equalization
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

# Display results
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2, 4), dtype=np.object)
axes[0, 0] = fig.add_subplot(2, 4, 1)
for i in range(1, 4):
    axes[0, i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])
for i in range(0, 4):
    axes[1, i] = fig.add_subplot(2, 4, 5+i)

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
ax_img.set_title('Low contrast image')

y_min, y_max = ax_hist.get_ylim()
ax_hist.set_ylabel('Number of pixels')
ax_hist.set_yticks(np.linspace(0, y_max, 5))

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
ax_img.set_title('Contrast stretching')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
ax_img.set_title('Histogram equalization')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
ax_img.set_title('Adaptive equalization')

ax_cdf.set_ylabel('Fraction of total intensity')
ax_cdf.set_yticks(np.linspace(0, 1, 5))

# prevent overlap of y-axis labels
fig.tight_layout()
plt.show()
"""


def enhance_image(img):
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_adapteq


def enhance_all():
    output_dir = RSNA_SEG_ENHANCE
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    img_files = glob(RSNA_SEG_ALL + "/*.*")
    for img_file in tqdm(img_files):
        img_name = img_file[img_file.rfind("/")+1: img_file.rfind("_seg")]
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        img_enh = enhance_image(img)
        plt.imsave(fname=RSNA_SEG_ENHANCE + "/{}_seg.png".format(img_name), arr=img_enh, cmap="gray")
        #cv2.imwrite(RSNA_SEG_ENHANCE + "/{}_seg.png".format(img_name), img_enh)

if __name__ == "__main__":
    enhance_all()