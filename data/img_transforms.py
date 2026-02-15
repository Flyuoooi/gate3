# from torchvision.transforms import *
# from PIL import Image
# import random
# import math


# class ResizeWithEqualScale(object):
#     """
#     Resize an image with equal scale as the original image.

#     Args:
#         height (int): resized height.
#         width (int): resized width.
#         interpolation: interpolation manner.
#         fill_color (tuple): color for padding.
#     """
#     def __init__(self, height, width, interpolation=Image.BILINEAR, fill_color=(0,0,0)):
#         self.height = height
#         self.width = width
#         self.interpolation = interpolation
#         self.fill_color = fill_color

#     def __call__(self, img):
#         width, height = img.size
#         if self.height / self.width >= height / width:
#             height = int(self.width * (height / width))
#             width = self.width
#         else:
#             width = int(self.height * (width / height))
#             height = self.height 

#         resized_img = img.resize((width, height), self.interpolation)
#         new_img = Image.new('RGB', (self.width, self.height), self.fill_color)
#         new_img.paste(resized_img, (int((self.width - width) / 2), int((self.height - height) / 2)))

#         return new_img


# class RandomCroping(object):
#     """
#     With a probability, first increase image size to (1 + 1/8), and then perform random crop.

#     Args:
#         p (float): probability of performing this transformation. Default: 0.5.
#     """
#     def __init__(self, p=0.5, interpolation=Image.BILINEAR):
#         self.p = p
#         self.interpolation = interpolation

#     def __call__(self, img):
#         """
#         Args:
#             img (PIL Image): Image to be cropped.

#         Returns:
#             PIL Image: Cropped image.
#         """
#         width, height = img.size
#         if random.uniform(0, 1) >= self.p:
#             return img
        
#         new_width, new_height = int(round(width * 1.125)), int(round(height * 1.125))
#         resized_img = img.resize((new_width, new_height), self.interpolation)
#         x_maxrange = new_width - width
#         y_maxrange = new_height - height
#         x1 = int(round(random.uniform(0, x_maxrange)))
#         y1 = int(round(random.uniform(0, y_maxrange)))
#         croped_img = resized_img.crop((x1, y1, x1 + width, y1 + height))

#         return croped_img


# class RandomErasing(object):
#     """ 
#     Randomly selects a rectangle region in an image and erases its pixels.

#     Reference:
#         Zhong et al. Random Erasing Data Augmentation. arxiv: 1708.04896, 2017.

#     Args:
#         probability: The probability that the Random Erasing operation will be performed.
#         sl: Minimum proportion of erased area against input image.
#         sh: Maximum proportion of erased area against input image.
#         r1: Minimum aspect ratio of erased area.
#         mean: Erasing value. 
#     """
    
#     def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
#         self.probability = probability
#         self.mean = mean
#         self.sl = sl
#         self.sh = sh
#         self.r1 = r1
       
#     def __call__(self, img):

#         if random.uniform(0, 1) >= self.probability:
#             return img

#         for attempt in range(100):
#             area = img.size()[1] * img.size()[2]
       
#             target_area = random.uniform(self.sl, self.sh) * area
#             aspect_ratio = random.uniform(self.r1, 1/self.r1)

#             h = int(round(math.sqrt(target_area * aspect_ratio)))
#             w = int(round(math.sqrt(target_area / aspect_ratio)))

#             if w < img.size()[2] and h < img.size()[1]:
#                 x1 = random.randint(0, img.size()[1] - h)
#                 y1 = random.randint(0, img.size()[2] - w)
#                 if img.size()[0] == 3:
#                     img[0, x1:x1+h, y1:y1+w] = self.mean[0]
#                     img[1, x1:x1+h, y1:y1+w] = self.mean[1]
#                     img[2, x1:x1+h, y1:y1+w] = self.mean[2]
#                 else:
#                     img[0, x1:x1+h, y1:y1+w] = self.mean[0]
#                 return img
#         return img


from torchvision.transforms import *
from PIL import Image
import random
import math


class ResizeWithEqualScale(object):
    """
    Resize an image with equal scale as the original image.
    (保持原样，未修改)
    
    Args:
        height (int): resized height.
        width (int): resized width.
        interpolation: interpolation manner.
        fill_color (tuple): color for padding.
    """
    def __init__(self, height, width, interpolation=Image.BILINEAR, fill_color=(0,0,0)):
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.fill_color = fill_color

    def __call__(self, img):
        width, height = img.size
        if self.height / self.width >= height / width:
            height = int(self.width * (height / width))
            width = self.width
        else:
            width = int(self.height * (width / height))
            height = self.height 

        resized_img = img.resize((width, height), self.interpolation)
        new_img = Image.new('RGB', (self.width, self.height), self.fill_color)
        new_img.paste(resized_img, (int((self.width - width) / 2), int((self.height - height) / 2)))

        return new_img


class RandomCroping(object):
    """
    [Logic Fixed for CoCoOp-ReID]
    Implementation changed from "Zoom-in -> Crop" to "Pad -> Crop".
    This prevents cutting off heads/feet which causes MetaNet drift, 
    while keeping the exact same class interface.

    Args:
        p (float): probability of performing this transformation. Default: 0.5.
        interpolation: Kept for interface compatibility (unused in Pad logic).
    """
    def __init__(self, p=0.5, interpolation=Image.BILINEAR):
        self.p = p
        # interpolation 参数虽然在 Padding 逻辑中用不到，
        # 但为了不报错（防止你的外部代码传入了这个参数），必须保留在 __init__ 中。
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        width, height = img.size
        if random.uniform(0, 1) >= self.p:
            return img
        
        # --- 核心修改开始 ---
        # 你的原始逻辑是放大 1.125 倍 (1 + 1/8)
        # 这里我们计算等效的 Padding 大小，来实现同样的位移空间，但不放大图片
        
        # 计算 Padding (保持 0.125 的比例)
        pad_w = int(round(width * 0.125 / 2))
        pad_h = int(round(height * 0.125 / 2))

        # 1. 创建黑色背景的大图 (Padding)
        padded_width = width + 2 * pad_w
        padded_height = height + 2 * pad_h
        padded_img = Image.new('RGB', (padded_width, padded_height), (0, 0, 0))
        
        # 2. 将原图贴在中间 (保留完整人体语义，这是解决 MetaNet 偏移的关键)
        padded_img.paste(img, (pad_w, pad_h))

        # 3. 随机裁剪回原尺寸
        x_maxrange = padded_width - width
        y_maxrange = padded_height - height
        
        # 使用 randint 确保整数坐标
        x1 = random.randint(0, x_maxrange)
        y1 = random.randint(0, y_maxrange)
        
        croped_img = padded_img.crop((x1, y1, x1 + width, y1 + height))
        # --- 核心修改结束 ---

        return croped_img
# class RandomCroping(object):
#     """
#     [Updated] Pad -> Crop strategy for CoCoOp-ReID.
#     Preserves head/feet information to stabilize Meta-Net prompts.
#     """
#     def __init__(self, p=0.5, interpolation=Image.BILINEAR):
#         self.p = p
#         self.interpolation = interpolation

#     def __call__(self, img):
#         width, height = img.size
#         if random.uniform(0, 1) >= self.p:
#             return img
        
#         # 计算 Padding 大小 (模拟原先 1.125 倍放大的位移空间)
#         pad_w = int(round(width * 0.125 / 2))
#         pad_h = int(round(height * 0.125 / 2))

#         # 1. 创建黑边背景
#         padded_width = width + 2 * pad_w
#         padded_height = height + 2 * pad_h
#         padded_img = Image.new('RGB', (padded_width, padded_height), (0, 0, 0))
        
#         # 2. 原图居中粘贴 (核心：保留了完整人体)
#         padded_img.paste(img, (pad_w, pad_h))

#         # 3. 随机裁剪
#         x_maxrange = padded_width - width
#         y_maxrange = padded_height - height
#         x1 = random.randint(0, x_maxrange)
#         y1 = random.randint(0, y_maxrange)
        
#         croped_img = padded_img.crop((x1, y1, x1 + width, y1 + height))

#         return croped_img
    

class RandomErasing(object):
    """ 
    Randomly selects a rectangle region in an image and erases its pixels.
    (保持原样，确认逻辑对 Tensor 有效)

    Args:
        probability: The probability that the Random Erasing operation will be performed.
        sl: Minimum proportion of erased area against input image.
        sh: Maximum proportion of erased area against input image.
        r1: Minimum aspect ratio of erased area.
        mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            # 兼容 Tensor 输入 (C, H, W)
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img