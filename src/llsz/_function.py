import numpy as np
import pyclesperanto_prototype as cle
from napari_plugin_engine import napari_hook_implementation
from napari_tools_menu import register_function
from napari_time_slicer import time_slicer

@napari_hook_implementation
def napari_experimental_provide_function():
    return [deskew_y]


def _affine_transform(source, at: cle.AffineTransform3D = None):
    """
    Applies an AffineTransform3D to a given source image. What's different to the affine_transform in clesperanto:
    Assuming the transform rotates the image out of the field of view (negative coordinates), it will be moved back
    so that all pixels are visible. Thus, the applied transform actually contains a translation, that was not passed
    as parameter.
    """

    # define coordinates of all corners of the current stack
    from itertools import product
    nx, ny, nz = source.shape
    original_bounding_box = [list(x) + [1] for x in product((0, nz), (0, ny), (0, nx))]
    # transform the corners using the given affine transform
    transformed_bounding_box = np.asarray(list(map(lambda x: at._matrix @ x, original_bounding_box)))

    # the min and max coordinates tell us from where to where the image ranges (bounding box)
    min_coordinate = transformed_bounding_box.min(axis=0)
    max_coordinate = transformed_bounding_box.max(axis=0)
    # determin the size of the transformed bounding box
    new_size = (max_coordinate - min_coordinate)[0:3].astype(int).tolist()[::-1]

    # create a new stack on GPU
    destination = cle.create(new_size)

    # we make a copy to not modify the original transform
    transform_copy = cle.AffineTransform3D()
    transform_copy._concatenate(at._matrix)

    # if the new minimum-coordinate is `-x`, we need to
    # translate the stack by `x` so that the new origin is (0,0,0)
    translation = -min_coordinate
    transform_copy.translate(
        translate_x=translation[0],
        translate_y=translation[1],
        translate_z=translation[2]
    )

    # apply transform and return result
    return cle.affine_transform(source, destination, transform=transform_copy)

@register_function(menu="Transform > Deskew in Y (llsz)")
@time_slicer
def deskew_y(raw_image:"napari.types.ImageData", rotation_angle: float = 30, keep_orientation:bool = False, viewer:"napari.Viewer"=None) -> "napari.types.ImageData":
    """
    Deskew an image stack.
    """

    # from https://github.com/SpimCat/unsweep/blob/6592b2667bda304336360e099ac015654a87787a/src/main/java/net/haesleinhuepf/spimcat/unsweep/Unsweep.java#L45
    import math
    deskew_factor = 1.0 / math.tan(rotation_angle * math.pi / 180)

    deskew_transform = cle.AffineTransform3D()
    # shearing
    shear_mat = np.array([
        [1.0, 0, 0, 0],
        [0, 1.0, deskew_factor, 0],
        [0, 0, 1.0, 0],
        [0, 0.0, 0.0, 1.0]
    ])
    deskew_transform._concatenate(shear_mat)

    # rotation
    delta = 0
    if keep_orientation:
        delta = 90
    deskew_transform.rotate(angle_in_degrees=delta-rotation_angle, axis=0)

    # apply transform
    import time
    start_time = time.time()
    result = _affine_transform(raw_image, at=deskew_transform)
    print("Deskew took", time.time() - start_time, "s on", cle.get_device())
    return result
