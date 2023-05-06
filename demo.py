import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from PIL import Image
import numpy as np

pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo


def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    pil_image = Image.open(url).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


def imshow(img, caption):
    plt.imshow(img[:, :, [2, 1, 0]])
    # plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)
    plt.show()


def imsave(img, path):
    plt.imsave(path, img[:, :, [2, 1, 0]])


# config_file = "configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
# weight_file = "glip_tiny_model_o365_goldg_cc_sbu.pth"
# config_file = "configs/pretrain/glip_A_Swin_T_O365.yaml"
# weight_file = "glip_a_tiny_o365.pth"
config_file = "configs/pretrain/glip_Swin_T_O365.yaml"
weight_file = "glip_tiny_model_o365.pth"

cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

glip_demo = GLIPDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
    show_mask_heatmaps=False
)
print(glip_demo.model)
# image = load('cat_remote.jpg')
# caption = 'There is two cat and a remote in the picture'

image = load('demo.jpg')
caption = 'There are a lot of cars here.'
result, _ = glip_demo.run_on_web_image(image, caption, 0.5)
imsave(result, 'cat_remote_pred.jpg')

# caption = 'cat'
# result, _ = glip_demo.run_on_web_image(image, caption, 0.5)
# imsave(result, 'cat_remote_pred1.jpg')
#
# caption = 'cat . remote . '
# result, _ = glip_demo.run_on_web_image(image, caption, 0.5)
# imsave(result, 'cat_remote_pred2.jpg')

# caption = ['cat', 'remote']
# result, _ = glip_demo.run_on_web_image(image, caption, 0.5)
# imsave(result, 'cat_remote_pred3.jpg')
