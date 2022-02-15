
import numpy as np
import pyclesperanto_prototype as cle
import dask.array as da
from typing import Union
from napari.layers.shapes import shapes

from llsz.utils import calculate_crop_bbox


#pass shapes data from single ROI to crop the volume from original data
def crop_volume_deskew(original_volume:Union[da.core.Array,np.array], 
                        deskewed_volume:Union[da.core.Array,np.array], 
                        roi_shape:Union[shapes.Shapes,list,np.array], 
                        angle_in_degrees:float, 
                        voxel_size_x:float, 
                        voxel_size_y:float, 
                        voxel_size_z:float, 
                        z_start:int, 
                        z_end:int):

    """
        Uses coordinates from deskewed space to find corresponding coordinates in original volume 
        and deskew only the specific volume
    Args:
        original_volume (np.array): Volume to deskew (xyz)
        deskewed_volume (np.array): Deskewed volume (xyz)
        roi_shape ((Union[shapes.Shapes,list,np.array])): if shapes layer or data passed, use onl
        angle_in_degrees ([float): deskewing angle in degrees
        voxel_size_x (float): [description]
        voxel_size_y (float): [description]
        voxel_size_z (float): [description]
        z_start (int): [description]
        z_end (int): [description]
    """

    assert len(original_volume.shape) == 3, print("Shape of original volume must be 3")
    assert len(deskewed_volume.shape) == 3, print("Shape of deskewed volume must be 3")
    #assert len(shape) == 4, print("Shape must be an array of shape 4 ")
    shape = None

    if type(roi_shape) is shapes.Shapes:
        shape = roi_shape.data[0]
    elif type(roi_shape) is list:#if list of rois, use only first one
        shape = roi_shape[0]
        #len(roi_shape) >= 1:  
    elif len(roi_shape) == 4 and type(roi_shape) is np.ndarray:
        shape = roi_shape
    
    assert len(shape) == 4, print("Shape must be an array of shape 4") 


    crop_bounding_box, crop_vol_shape = calculate_crop_bbox(shape,z_start,z_end)
    
    #get reverse transform by rotating around original volume
    reverse_aff, excess_bounds = get_inverse_affine_transform(original_volume,angle_in_degrees,voxel_size_x,voxel_size_y,voxel_size_z)

    #apply the transform to get corresponding bounding boxes in original volume
    crop_transform_bbox = np.asarray(list(map(lambda x: reverse_aff._matrix @ x,crop_bounding_box)))

    #get shape of original volume in xyz
    orig_shape = original_volume.shape[::-1]

    #Default taking entire frame of y. 
    #TODO: allow cropping in all axes
    y_start = 0
    y_end = orig_shape[1]

    min_coordinate = crop_transform_bbox.min(axis=0)
    max_coordinate = crop_transform_bbox.max(axis=0)

    x_start = min_coordinate[0].astype(int)
    x_end = max_coordinate[0].astype(int)

    #The crop_transform bbox has 4 values of z
    #First and last z value is out of bounds of the original volume
    #We only take the 2nd and 3rd value
    z_end_vol = np.abs(crop_transform_bbox[2][2]).astype(int)
    z_end_vol = np.clip(z_end_vol, 0,orig_shape[2]) #clip to z bounds of original volume

    z_start_vol = np.abs(crop_transform_bbox[1][2]).astype(int)
    z_start_vol = np.clip(z_start_vol, 0,orig_shape[2]) #clip to z bounds of original volume

    #make sure z_start > z_end
    if z_start > z_end:
        #tuple swap  #https://docs.python.org/3/reference/expressions.html#evaluation-order
        z_start,z_end = z_end,z_start
        
    if type(original_volume) is da.core.Array:
        #If using dask, use .map_blocks(np.copy) to copy subset (faster)
        crop_volume = original_volume[z_start_vol:z_end_vol,y_start:y_end,x_start:x_end].map_blocks(np.copy).squeeze()
    else:
        crop_volume = original_volume[z_start_vol:z_end_vol,y_start:y_end,x_start:x_end]

    deskewed_prelim = cle.deskew_y(crop_volume, 
                        angle_in_degrees=angle_in_degrees, 
                        voxel_size_x=voxel_size_x, 
                        voxel_size_y=voxel_size_y, 
                        voxel_size_z=voxel_size_z)
    #The height of deskewed_prelim will be larger than specified shape
    # as the coordinates of the ROI are skewed in the original volume

    deskewed_height = deskewed_prelim.shape[1]
    crop_height = crop_vol_shape[1]
    #Find excess volume on both sides
    crop_excess = int((deskewed_height  - crop_height)/2)
    #Crop in Y
    deskewed_crop = deskewed_prelim[:,crop_excess:crop_height+crop_excess,:]

    return deskewed_crop

#Get reverse affine transform by rotating around a user-specified volume
def get_inverse_affine_transform(original_volume,angle_in_degrees,voxel_x,voxel_y,voxel_z):
    """
    Calculate the inverse deskew transform and the excess z_bounds 
    Difference from using inverse on deskew_y transform is the rotation here is fixed around a
    specified volume and final affine matrix will be based on the ref volume used
    Args:
        original_volume (_type_): _description_
        angle_in_degrees (_type_): _description_
        voxel_x (_type_): _description_
        voxel_y (_type_): _description_
        voxel_z (_type_): _description_

    Returns:
        cle.AffineTransform3D, int: Affine transform and Excess z slices on either side of the undeskewed volume
    """    
    #calculate the deskew transform for specified volume
    deskew_transform = _deskew_y_vol_transform(original_volume,angle_in_degrees,voxel_x,voxel_y,voxel_z)

    #Get the deskew transform after bringing the volume into bounds
    deskewed_shape, new_deskew_transform, _ = cle._tier8._affine_transform._determine_translation_and_bounding_box(
                                        original_volume, deskew_transform)
    
    #Get the inverse of adjusted desnew transform
    deskew_inverse = new_deskew_transform.inverse()

    #We use the shape of deskewed volume to get the new vertices of deskewed volume in x,y and z
    from itertools import product
    nz, ny, nx = deskewed_shape
    deskewed_bounding_box = [list(x) + [1] for x in product((0, nx), (0, ny), (0, nz))]

    # transform the corners of deskewed volume using the reverse affine transform
    undeskew_bounding_box = np.asarray(list(map(lambda x: deskew_inverse._matrix @ x, deskewed_bounding_box)))

    #Get the maximum z value and subtract it from shape of original volume to get excess bounds of bounding box
    max_bounds = undeskew_bounding_box.max(axis=0).astype(int)
    rev_deskew_z = max_bounds[2]
    extra_bounds = int((rev_deskew_z - original_volume.shape[0]))

    return deskew_inverse, extra_bounds

#Get deskew transform where rotation is around centre of "original_volume"
def _deskew_y_vol_transform(original_volume, angle_in_degrees:float = 30, voxel_size_x: float = 1,
              voxel_size_y: float = 1, voxel_size_z: float = 1, scale_factor: float = 1):
    """Return deskew transform for specified volume
       Rotation is performed around centre of "original_volume"
    Args:
        crop ([type]): Volume to deskew (zyx)
        original_volume ([type]): Reference volume around with to perform rotation (zyx)
        angle_in_degrees (float): Deskewing angle
        voxel_size_x (float, optional): [description]. Defaults to 1.
        voxel_size_y (float, optional): [description]. Defaults to 1.
        voxel_size_z (float, optional): [description]. Defaults to 1.
        scale_factor (float, optional): [description]. Defaults to 1.

    Returns:
        cle.AffineTransform3D
    """    
    import math
    
    transform = cle.AffineTransform3D()
    transform.shear_in_x_plane(angle_y_in_degrees = 90 - angle_in_degrees)
    
    #As deskewing is performed by rotation around ref_vol, we need to define a rotation matrix
    #that uses centre of ref_vol
    transform._concatenate(rotate_around_vol_mat(original_volume, -(90-angle_in_degrees)))
    
    # make voxels isotropic, calculate the new scaling factor for Z after shearing
    # https://github.com/tlambert03/napari-ndtiffs/blob/092acbd92bfdbf3ecb1eb9c7fc146411ad9e6aae/napari_ndtiffs/affine.py#L57
    new_dz = math.sin(angle_in_degrees * math.pi / 180.0) * voxel_size_z
    scale_factor_z = (new_dz / voxel_size_y) * scale_factor
    transform.scale(scale_x=scale_factor, scale_y=scale_factor, scale_z=scale_factor_z)
    # correct orientation so that the new Z-plane goes proximal-distal from the objective.
    transform.rotate(angle_in_degrees=90, axis=0)

    return transform

#Calculate rotation transform around a volume
def rotate_around_vol_mat(ref_vol,angle_in_degrees:float=30.0):
    """Return the rotation matrix , so its rotated around centre of ref_vol

    Args:
        ref_vol (tuple): Shape of the ref volume (zyx)
        angle_in_degrees (float, optional): [description]. Defaults to 30.0.

    Returns:
        Rotation matrix: Will be returned in the form xyz for clesperanto affine transforms
    """    
    angle_in_rad = angle_in_degrees * np.pi / 180.0
    #rotate_transform = cle.AffineTransform3D()
    #rotate_transform._matrix
    # first translate the middle of the image to the origin
    nz,ny,nx = ref_vol.shape
    T1 = np.array([
            [1, 0, 0, nx / 2],
            [0, 1, 0, ny / 2],
            [0, 0, 1, nz / 2],
            [0, 0, 0, 1]
            ])

    R = np.array([
                [1, 0, 0, 0],
                [0, np.cos(angle_in_rad), np.sin(angle_in_rad), 0],
                [0, -np.sin(angle_in_rad), np.cos(angle_in_rad),0],
                [0, 0, 0, 1]
                ])
                
    T2 = np.array([
            [1, 0, 0, -nx / 2],
            [0, 1, 0, -ny / 2],
            [0, 0, 1, -nz / 2],
            [0, 0, 0, 1]
            ])
    T = np.eye(4)
    rotate_mat = np.dot(np.dot(np.dot(T, T1), R), T2)
    #print(rotate_mat)
    return rotate_mat

