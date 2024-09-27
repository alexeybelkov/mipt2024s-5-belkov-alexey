import os
import barcode
from barcode.writer import ImageWriter
import qrcode
import segno
import treepoem
from aztec_code_generator import AztecCode


def generate_code(code_type, data, directory='generated_codes'):

    if not os.path.exists(directory):
        os.makedirs(directory)

    try:
        if code_type in ['EAN13', 'UPCA', 'EAN8', 'CODE128', 'CODE39']:
            code = barcode.get(code_type, data, writer=ImageWriter())
            # filename = f"{code_type}_{data}"
            # if filename[-3:] not in ('jpg', 'png'):
            filename = os.path.join(directory, f"{code_type}_{data}")
            code.save(filename)
            return filename + '.png'

        elif code_type == 'CODE93':
            image = treepoem.generate_barcode(code_type.lower(), data=data)
            filename = f"CODE93_{data}"
            if filename[-3:] not in ('jpg', 'png'):
              filename = os.path.join(directory, f"CODE93_{data}.png")
            image.save(filename)
            return filename

        elif code_type == 'AZTECCODE':
            # raise NotImplementedError("Aztec Code generation requires a different library.")
            aztec_code = AztecCode(data)

            filename = f"AZTECODE_{data}"
            if filename[-3:] not in ('jpg', 'png'):
              filename = os.path.join(directory, f"AZTECODE_{data}.png")
            aztec_code.save(filename, module_size=4, border=1)

        elif code_type == 'PDF417':
            # Generate PDF417 using treepoem
            image = treepoem.generate_barcode(code_type.lower(), data=data)
            filename = f"PDF417_{data}"
            if filename[-3:] not in ('jpg', 'png'):
              filename = os.path.join(directory, f"PDF417_{data}.png")
            image.save(filename)
            return filename

        elif code_type == 'DATAMATRIX':
            # Generate DataMatrix using treepoem
            image = treepoem.generate_barcode(code_type.lower(), data=data)
            filename = f"DATAMATRIX_{data}"
            if filename[-3:] not in ('jpg', 'png'):
              filename = os.path.join(directory, f"DATAMATRIX_{data}.png")
            image.save(filename)
            return filename

        elif code_type == 'QRCODE':
            # Generate QR Code using qrcode library
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(data)
            qr.make(fit=True)

            img = qr.make_image(fill_color="black", back_color="white")
            filename = os.path.join(directory, f"QRCODE_{data}.png")
            img.save(filename)
            return filename

        elif code_type == 'AZTECRUNE':
            # Generate Aztec Rune using segno
            aztec = segno.make(data)
            filename = os.path.join(directory, f"AZTECRUNE_{data}.png")
            aztec.save(filename)
            return filename

    except Exception as e:
        return -1

# Example usage:
code_types = ['EAN13', 'UPCA', 'CODE93', 'EAN8', 'CODE128', 'CODE39', 'PDF417', 'DATAMATRIX', 'QRCODE', 'AZTECRUNE', 'AZTECCODE']
data_samples = ['123456789012', '012345678901', 'CODE93DATA', '1234567', 'CODE128DATA', 'CODE39DATA']

from random import randint
import random


def calc_check_digit(upc):
    check_sum = sum(int(x) * (1 + (1 - i % 2) * 2) for i, x in enumerate(upc))
    check_digit = (10 - (check_sum % 10)) % 10
    return str(check_digit)


def generate_upca():
    upc_prefix = "0"
    upc_middle = str(randint(10000, 99999))
    upc_check_digit = calc_check_digit(upc_prefix + upc_middle)
    upc = upc_prefix + upc_middle + upc_check_digit
    return upc


def generate_code39():
    # Define the characters allowed in a Code 39 barcode
    code39_chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%'

    code39_data = ''.join(random.choices(code39_chars, k=10))
    # Generate random data of length 10

    # Calculate the checksum character for Code 39
    checksum = sum([code39_chars.index(char) for char in code39_data]) % 43
    code39_encoded = code39_data + code39_chars[checksum]

    return code39_encoded


def generate_code93(length=None):
    if length is None:
        length = randint(1, 50)
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%'
    input_data = [random.choice(characters) for _ in range(length)]
    return ''.join(input_data)


def random_ascii(sz=256):
    return ''.join([chr(randint(0, 127)) for i in range(randint(1, sz))])


gens = {
    'ean13': lambda: str(randint(10**11, 10**12-1)),
    'upca': generate_upca,
    'plessey': lambda: hex(randint(0, 16**20)).upper()[2:],
    'code39': generate_code39,
    'code93': generate_code93,
    'datamatrix': random_ascii,
    'qrcode': random_ascii,
    'azteccode': random_ascii,
    'aztecrune': lambda: str(randint(0, 255)),
    'microqrcode': lambda: random_ascii(35),
}


dims = {
    'ean8': '1d',
    'ean13': '1d',
    'upca': '1d',
    'plessey': '1d',
    'code39': '1d',
    'code93': '1d',
    'datamatrix': '2d',
    'qrcode': '2d',
    'azteccode': '2d',
    'aztecrune': '2d',
    'microqrcode': '2d',
}

import augraphy

augs = {
    "BadPhotoCopy": augraphy.augmentations.BadPhotoCopy(),
    "BrightnessTexturize": augraphy.augmentations.BrightnessTexturize(),
    "ColorPaper": augraphy.augmentations.ColorPaper(),
    "Folding": augraphy.augmentations.Folding(),
    "LightingGradient": augraphy.augmentations.LightingGradient(),
    "NoisyLines": augraphy.augmentations.NoisyLines(),
    "ShadowCast": augraphy.augmentations.ShadowCast(),
}

from PIL import Image
import cv2
import json
import argparse
import treepoem
import numpy as np
from matplotlib import pyplot as plt

# from data_generator import gens, dims
# from augmentations import augs


def load_json(fname, *args, **kwargs):
    with open(fname) as f:
        return json.load(f, *args, **kwargs)


def save_json(jd, fname, *args, indent=4, **kwargs):
    with open(fname, 'w') as f:
        json.dump(jd, f, *args, indent=indent, **kwargs)


def aligned_affine(bar, distortion_matrix, fix_position=True):
    xps = [0, 0, 1, 1]
    yps = [1, 0, 0, 1]

    height, width, _ = bar.shape
    corners = np.array([[x*width, y*height, 1] for x, y in zip(xps, yps)]).T

    if fix_position:
        distortion_matrix = distortion_matrix.copy()
        distortion_matrix[:, -1] *= 0
        distortion_matrix[:, -1] = -np.min(distortion_matrix@corners, axis=-1)

    new_sz = np.ceil(np.max(distortion_matrix@corners, axis=-1)).astype(np.int32)
    img = cv2.warpAffine(bar, distortion_matrix, new_sz)
    return img, distortion_matrix@corners


def generate_perspective_distort(img, alpha=0.1, beta=0.01):
    xps = [0, 0, 1, 1]
    yps = [1, 0, 0, 1]
    height, width, _ = img.shape
    corners = np.array([[x*width, y*height, 1] for x, y in zip(xps, yps)]).T

    distortion_matrix = np.zeros((3, 3))
    distortion_matrix[:-1, :-1] = np.random.randn(2, 2)*alpha / 2 + np.eye(2)*(1.-alpha) / 2
    distortion_matrix[-1, :-1] = beta*np.abs(np.random.randn(1, 2)) / 2
    distortion_matrix[:-1, -1] *= 0
    distortion_matrix[-1, -1] = 1
    coords = (distortion_matrix@corners)[:-1]
    distortion_matrix[:-1, -1] = -np.min(coords, axis=-1)
    return distortion_matrix


def generate_aligned_perspective_distort(img, scale=0.1):
    xps = [0, 0, 1, 1]
    yps = [1, 0, 0, 1]
    height, width, _ = img.shape
    corners = np.array([[x*width, y*height] for x, y in zip(xps, yps)])
    corners_old = corners.copy()

    dx, dy = np.random.exponential(scale=width*scale / 2, size=corners.shape).T

    corners[0, 0] -= dx[0]
    corners[0, 1] += dy[0]

    corners[1, 0] -= dx[1]
    corners[1, 1] -= dy[1]

    corners[2, 0] += dx[2]
    corners[2, 1] -= dy[2]

    corners[3, 0] += dx[3]
    corners[3, 1] += dy[3]

    theta = np.pi*np.random.random()*2 / 5
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array(((c, -s), (s, c)))

    corners = corners@rotation_matrix

    distortion_matrix = cv2.getPerspectiveTransform(corners_old.astype(np.float32), corners.astype(np.float32))

    corners = np.array([[x*width, y*height, 1] for x, y in zip(xps, yps)]).T
    distortion_matrix[:-1, -1] *= 0
    distortion_matrix[-1, -1] = 1
    coords = (distortion_matrix@corners)[:-1]
    distortion_matrix[:-1, -1] = -np.min(coords, axis=-1)
    coords = distortion_matrix@corners
    coords = coords[:-1]/coords[-1]

    return distortion_matrix


def aligned_perspective(img, distortion_matrix):
    xps = [0, 0, 1, 1]
    yps = [1, 0, 0, 1]
    height, width, _ = img.shape
    corners = np.array([[x*width, y*height, 1] for x, y in zip(xps, yps)]).T
    coords = distortion_matrix@corners
    coords = coords[:-1]/coords[-1]

    new_sz = np.ceil(np.max(coords, axis=-1)).astype(np.int32)

    img = cv2.warpPerspective(img, distortion_matrix, new_sz)
    return img, coords


def coords_to_regions(coords, dimensions):
    res = []
    for i in range(len(coords)):
        ptsx, ptsy = coords[i]

        res.append({\
            'shape_attributes': {
                'name': 'polygon',
                'all_points_x': list(ptsx),
                'all_points_y': list(ptsy)
            },
            'region_attributes': {'barcode': dimensions[i]}
        })
    return res


def export(img, name, coords, dimensions):
    # np.clip(img, 0, 1)
    plt.imsave(f'{name}.jpg', img)
    res = {
        f'{name}.jpg813086': {
            'filename': f'../code/{name}.jpg',
            'size': 813086,
            'regions': coords_to_regions(coords, dimensions),
            'file_attributes': {}
        }
    }
    save_json(res, f'{name}.json')


def generate_distorted(barcode_types, content_barcodes, source_img=None,
                       augms=[], distortions=None, codes_to_augm=None):
    '''
    Generates image with barcodes and returns list of coords of barcodes on img
    Parameters:

    barcode_types: Iterable[str] -- list of barcode types to generate
                   see treepoem docs for available options

    content_barcodes: Iterable -- list of content to encode in barcodes

    source_img: np.array -- np.array of an image

    augms: Iterable[str] -- list of augmentation names,
                            see `augmentations.py` for options

    distortions: Iterable[np.array] -- list of parameters for spatial distort
    '''

    filenames = []
    if codes_to_augm is None:
        for code_type in barcode_types:
            for content in content_barcodes:
                result = generate_code(code_type, content)
                if isinstance(result, str):
                    filenames.append(result)

    # print(filenames)
        barimgs = [cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB) for fname in filenames]
    else:
        barimgs = codes_to_augm

    # barimgs = [np.array(treepoem.generate_barcode(typ, content)) for typ, content in zip(barcode_types, content_barcodes)]

    for aug_name in augms:
        barimgs = [augs[aug_name](img) for img in barimgs]
    if distortions is None:
        distortions = [generate_aligned_perspective_distort(np.array(img)) for img in barimgs]
    # imgs, coords = zip(*[aligned_affine(np.array(img), dis) for img, dis in zip(barimgs, distortions)])
    imgs, coords = zip(*[aligned_perspective(np.array(img), dis) for img, dis in zip(barimgs, distortions)])
    # masks, _ = zip(*[aligned_affine(np.ones_like(img), dis) for img, dis in zip(barimgs, distortions)])
    masks, _ = zip(*[aligned_perspective(np.ones_like(img), dis) for img, dis in zip(barimgs, distortions)])

    if source_img is None:
        width, height, _ = np.max([img.shape for img in imgs], axis=0)*len(imgs)//3
        combined = np.zeros((width, height, 3), dtype=imgs[0].dtype)
    else:
        combined = plt.imread(source_img)[:, :, :3]
        width, height, _ = combined.shape

    for i in range(len(imgs)):
        w, h, _ = imgs[i].shape
        # if width - w < 0 or height - h < 0:
        #     continue
        dw = np.random.randint(0, max(width - w, 10))
        dh = np.random.randint(0, max(height - h, 10))
        coords[i][0] += dh
        coords[i][1] += dw
        expanded_img = np.zeros_like(combined)
        expanded_img[dw:w+dw, dh:h+dh] = imgs[i][:width-dw, :height-dh]
        expanded_mask = np.zeros_like(combined)
        expanded_mask[dw:w+dw, dh:h+dh] = masks[i][:width-dw, :height-dh]
        combined = combined*(1-expanded_mask) + expanded_img
    return combined, coords, imgs
