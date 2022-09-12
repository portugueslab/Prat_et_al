from skimage import color
import numpy as np

def overimpose_shade(img, mask, alpha=0.6, col=(1, 0, 0)):

    # Construct RGB version of grey-level image
    img_color = np.dstack((img, )*3)
    img_mask = (np.dstack((mask, )*3)*col).astype(img_color.dtype)

    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(img_mask)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked


def merge_anatomy_and_mask(anatomy_stack, mask_stack, color, gamma=0.6):
    """
    Function to beautifully merge anatomy and mask stacks.
    :param anatomy_stack: Anatomy stack
    :param mask_stack: Mask stack
    :param color: Desired color on which the ROI will be plotted (RGB or Hex formats accepted)
    :param gamma: Gamma to be applied to the anatomy stack
    :return:
    """

    # Get RGB color
    if type(color) == str:
        rgb_color = tuple(int(color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
    elif type(color) == tuple or list and len(color) == 3:
        rgb_color = color
    else:
        print('Define color in RGB or Hex format')

    # Create RGB anatomy stack in grayscale
    anatomy_stack = np.sum(anatomy_stack, 0)
    anatomy_stack = ((anatomy_stack - anatomy_stack.min()) / (anatomy_stack.max() - anatomy_stack.min()))
    anatomy_stack = np.power((anatomy_stack), gamma) * 255
    anatomy_stack = np.dstack((anatomy_stack,) * 3).astype(np.uint8)

    # Create RGB mask stack with desired color
    mask_stack = np.sum(mask_stack, 0)
    mask_stack = ((mask_stack - mask_stack.min()) / (mask_stack.max() - mask_stack.min()))
    mask_stack = np.dstack((mask_stack,) * 3)
    for chanel in range(3):
        mask_stack[:, :, chanel] = mask_stack[:, :, chanel] * rgb_color[chanel]
    mask_stack = mask_stack.astype(np.uint8)

    # Get pixels belonging to mask, and set their value to 0 in anatomy stack
    mask_pixels = np.argwhere(mask_stack[:, :, 0] != 0)

    for chanel in range(3):
        for pixel in mask_pixels:
            anatomy_stack[pixel[0], pixel[1], chanel] = 0

    merged_anatomy = anatomy_stack + mask_stack

    return (merged_anatomy)