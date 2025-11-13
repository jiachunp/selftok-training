from torchvision import transforms
import torch
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image, ImageFile


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class ConditionalResize(transforms.Resize):
    """
    Resize transform but only if the input is smaller than the resize dims
    """

    @property
    def resize_height(self):
        return self.size if isinstance(self.size, int) else self.size[0]

    @property
    def resize_width(self):
        return self.size if isinstance(self.size, int) else self.size[1]

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        r_h, r_w = self.resize_height, self.resize_width
        if isinstance(img, torch.Tensor):
            h, w = img.shape[1:]
        else:  # PIL Image
            w, h = img.size
        if w < r_w or h < r_h:
            return super().forward(img)
        else:
            return img

class RandomResize(transforms.Resize):
    """
    Resize transform but only if the input is smaller than the resize dims
    """
    def __init__(self, size, interpolation=F.InterpolationMode.BILINEAR, max_size=None, antialias="warn", ratio=0.5):
        super().__init__(size, interpolation, max_size, antialias)
        self.ratio = ratio

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        if isinstance(img, torch.Tensor):
            h, w = img.shape[1:]
        else:  # PIL Image
            w, h = img.size
        r = torch.rand(1)
        if r > self.ratio:
            # print('fix 256')
            return F.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
        else:
            r = (r*1./self.ratio)[0].item()
            res = r * self.size + (1-r) * min(w,h)
            res = int(max(res, self.size))
            # print(f'{res}')
            return F.resize(img, res, self.interpolation, self.max_size, self.antialias)


class NormalizeToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, reshape=True):
        self.reshape = reshape

    def __call__(self, image):
        image = np.array(image).astype(np.float32)
        image = (image / 127.5 - 1.0).astype(np.float32)
        if self.reshape:
            image = np.reshape(image, (image.shape[0], image.shape[1], -1))
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)

val_transform_256 = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(256),
        NormalizeToTensor(),
    ]
)

train_transform_rand1_0_256 = transforms.Compose(
    [
        RandomResize(256, ratio=1.0, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(256),
        NormalizeToTensor(),
    ]
)


val_transform_512 = {'512':transforms.Compose(
    [
        transforms.Resize(512),
        transforms.CenterCrop(512),
        NormalizeToTensor(),
    ]
),'256':transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(256),
        NormalizeToTensor(),
    ]
)}

val_transform_256_from512 = transforms.Compose(
    [
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.Resize(256),
        NormalizeToTensor(),
    ]
    
    

)

extract_tokens =  transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ]
                                           )


transforms_dict = {
    "256": val_transform_256,
    "512_old": val_transform_512,
    "512": val_transform_512,
    "512enc-256dec": [val_transform_256_from512, val_transform_512],        # first decoder, then encoder
    "train-rand1.0-256": train_transform_rand1_0_256,
    "extract_tokens":extract_tokens
}