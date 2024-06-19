from PIL import Image
import numpy as np
import torch
from rembg import new_session, remove 

providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': 8 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'HEURISTIC',
    })
] 
session = new_session(providers=providers) 

def rotate_normalmap_by_angle_torch(normal_map, angle):
    """
    rotate along y-axis
    normal_map: torch.Tensor, shape=(H, W, 3) in [-1, 1], device='cuda'
    angle: float, in degree
    """
    angle = torch.tensor(angle / 180 * np.pi).to(normal_map)
    R = torch.tensor([[torch.cos(angle), 0, torch.sin(angle)], 
                      [0, 1, 0], 
                      [-torch.sin(angle), 0, torch.cos(angle)]]).to(normal_map)
    return torch.matmul(normal_map.view(-1, 3), R.T).view(normal_map.shape)


def do_rotate(rgba_normal, angle):
    rgba_normal = torch.from_numpy(rgba_normal).float().cuda() / 255
    rotated_normal_tensor = rotate_normalmap_by_angle_torch(rgba_normal[..., :3] * 2 - 1, angle)
    rotated_normal_tensor = (rotated_normal_tensor + 1) / 2
    rotated_normal_tensor = rotated_normal_tensor * rgba_normal[:, :, [3]]    # make bg black
    rgba_normal_np = torch.cat([rotated_normal_tensor * 255, rgba_normal[:, :, [3]] * 255], dim=-1).cpu().numpy()
    return rgba_normal_np



def rotate_normals_torch(normal_pils, return_types='np', rotate_direction=1):
    n_views = len(normal_pils)
    ret = []
    for idx, rgba_normal in enumerate(normal_pils):
        # rotate normal
        angle = rotate_direction * idx * (360 / n_views)
        rgba_normal_np = do_rotate(np.array(rgba_normal), angle)
        if return_types == 'np':
            ret.append(rgba_normal_np)
        elif return_types == 'pil':
            ret.append(Image.fromarray(rgba_normal_np.astype(np.uint8)))
        else:
            raise ValueError(f"return_types should be 'np' or 'pil', but got {return_types}")
    return ret

def remove_color(arr):
    # todo: change this, use same alpha for both image and normal 
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    # calc diffs
    base = arr[0, 0]
    diffs = np.abs(arr.astype(np.int32) - base.astype(np.int32)).sum(axis=-1)
    alpha = (diffs <= 80)
    
    arr[alpha] = 255
    alpha = ~alpha
    arr = np.concatenate([arr, alpha[..., None].astype(np.int32) * 255], axis=-1)
    return arr


def rgba_to_rgb(rgba: Image.Image, bkgd="WHITE"):
    new_image = Image.new("RGBA", rgba.size, bkgd)
    new_image.paste(rgba, (0, 0), rgba)
    new_image = new_image.convert('RGB')
    return new_image

def change_rgba_bg(rgba: Image.Image, bkgd="WHITE"):
    rgb_white = rgba_to_rgb(rgba, bkgd)
    new_rgba = Image.fromarray(np.concatenate([np.array(rgb_white), np.array(rgba)[:, :, 3:4]], axis=-1))
    return new_rgba


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def simple_preprocess(input_image, rembg_session=session, background_color=255):
    RES = 2048
    input_image.thumbnail([RES, RES], Image.Resampling.LANCZOS)
    if input_image.mode != 'RGBA':
        image_rem = input_image.convert('RGBA')
        input_image = remove(image_rem, alpha_matting=False, session=rembg_session)

    arr = np.asarray(input_image)
    alpha = np.asarray(input_image)[:, :, -1]
    x_nonzero = np.nonzero((alpha > 60).sum(axis=1))
    y_nonzero = np.nonzero((alpha > 60).sum(axis=0))
    x_min = int(x_nonzero[0].min())
    y_min = int(y_nonzero[0].min())
    x_max = int(x_nonzero[0].max())
    y_max = int(y_nonzero[0].max())
    arr = arr[x_min: x_max, y_min: y_max]
    input_image = Image.fromarray(arr)
    input_image = expand2square(input_image, (background_color, background_color, background_color, 0))
    return input_image
