import numpy as np
from PIL import ImageFilter
from PIL import Image
import random
from detectron2.data import detection_utils as utils
import torchvision.transforms as transforms
import os
import cv2


class PairTransform:
    """Apply the same transformation to the image pair."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, im0, im1):
        im0 = self.base_transform(im0)
        im1 = self.base_transform(im1)
        return [im0, im1]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def convert_PIL_to_numpy(image, format):
    """
    Convert PIL image to numpy array of target format.

    Args:
        image (PIL.Image): a PIL image
        format (str): the format of output image

    Returns:
        (np.ndarray): also see `read_image`
    """
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)

    # handle formats not supported by PIL
    elif format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    elif format == "YUV-BT.601":
        image = image / 255.0
        image = np.dot(image, np.array(_M_RGB2YUV).T)

    return image


if __name__ == "__main__":
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    augmentation = [
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.ToTensor(),
    ]

    output_dir = "./debug"
    pair = "tools/demo"
    # Load cfg
    img_file1 = os.path.join(pair, "view_0.png")
    img_file2 = os.path.join(pair, "view_1.png")

    # im0 = utils.read_image(img_file1, format='BGR')
    # im1 = utils.read_image(img_file2, format='BGR')

    # PIL Format
    im0 = utils.read_image(img_file1, format="BGR")
    im1 = utils.read_image(img_file2, format="BGR")
    im0 = Image.fromarray(im0)
    im1 = Image.fromarray(im1)
    tf = PairTransform(transforms.Compose(augmentation))

    for i in range(10):
        im0_aug, im1_aug = tf(im0, im1)
        import pdb; pdb.set_trace()
        cv2.imwrite(
            f"tools/demo/augmentation/all/{i}_view_0.png",
            im0_aug.numpy().transpose(1, 2, 0) * 255,
        )
        cv2.imwrite(
            f"tools/demo/augmentation/all/{i}_view_1.png",
            im1_aug.numpy().transpose(1, 2, 0) * 255,
        )
