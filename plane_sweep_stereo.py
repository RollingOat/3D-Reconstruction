import numpy as np
import cv2


EPS = 1e-8


def backproject_corners(K, width, height, depth, Rt):
    """
    Backproject 4 corner points in image plane to the imaginary depth plane

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        width -- width of the image
        heigh -- height of the image
        depth -- depth value of the imaginary plane: the projection center of ref to the imaginary plane?
    Output:
        points -- 2 x 2 x 3 array of 3D coordinates of backprojected points
    """

    P_pixel = np.array([
        [0, 0, 1],
        [width, 0, 1],
        [0, height, 1],
        [width, height, 1]
    ]) # (4,3)

    Pc = depth * np.linalg.inv(K) @ P_pixel.T # (3, 4)
    Rwc = Rt[:,0:3]
    twc = Rt[:,3]
    Pw = Rwc.T @ (Pc - twc.reshape(-1,1)) # (3,4)
    points = np.zeros((2,2,3))
    points[0,0,:] = Pw[:,0]
    points[0,1,:] = Pw[:,1]
    points[1,0,:] = Pw[:,2]
    points[1,1,:] = Pw[:,3]
    return points

def project_points(K, Rt, points):
    """
    what's the shape of points?
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- points_height x points_width x 3 array of 3D points
    Output:
        projections -- points_height x points_width x 2 array of 2D projections
    """
    h = points.shape[0]
    w = points.shape[1]
    points_2dim = points.transpose(2,0,1).reshape(3,-1) # (3, h*w)
    points_homo = np.vstack((
        points_2dim, 
        np.ones(points_2dim.shape[1])
    )) # (4, h*w)
    Pc = K @ Rt @ points_homo # (3, h*w)
    Pc_homo = Pc/ Pc[2]
    # pixel coordinate after projecting images from imaginary plane into reference plane
    Pc_3dim = Pc_homo[0:2].reshape(2,h,w).transpose(1,2,0) 
    Pc_3dim = Pc_3dim.astype(np.float32)
    return Pc_3dim

def warp_neighbor_to_ref(backproject_fn, project_fn, depth, neighbor_rgb, K_ref, Rt_ref, K_neighbor, Rt_neighbor):
    """ 
    Warp the neighbor view into the reference view 
    via the homography induced by the imaginary depth plane at the specified depth

    Make use of the functions you've implemented in the same file (which are passed in as arguments):
    - backproject_corners
    - project_points

    Also make use of the cv2 functions:
    - cv2.findHomography
    - cv2.warpPerspective

    Input:
        backproject_fn -- backproject_corners function
        project_fn -- project_points function
        depth -- scalar value of the depth at the imaginary depth plane
        neighbor_rgb -- height x width x 3 array of neighbor rgb image
        K_ref -- 3 x 3 camera intrinsics calibration matrix of reference view
        Rt_ref -- 3 x 4 camera extrinsics calibration matrix of reference view
        K_neighbor -- 3 x 3 camera intrinsics calibration matrix of neighbor view
        Rt_neighbor -- 3 x 4 camera extrinsics calibration matrix of neighbor view
    Output:
        warped_neighbor -- height x width x 3 array of the warped neighbor RGB image
    """

    height, width = neighbor_rgb.shape[:2]
    # neighboring plane coord
    pixelCoord_neighbor = np.array((
        (0, 0),
        (width, 0),
        (0, height),
        (width, height),
    ), dtype=np.float32)
    # 2 x 2 x 3 array of 3D coordinates of backprojected points, world coordinate on depth plane
    depth_plane_coord = backproject_fn(K_ref, width, height, depth, Rt_ref)
    #  points_height x points_width x 2 array of 2D projections on ref image plane
    pixelCoord_ref = project_fn(K_neighbor, Rt_neighbor, depth_plane_coord)
    # 2dim array
    pixelCoord_ref_ = np.zeros((4,2))
    pixelCoord_ref_[0] = pixelCoord_ref[0,0,:]
    pixelCoord_ref_[1] = pixelCoord_ref[0,1,:]
    pixelCoord_ref_[2] = pixelCoord_ref[1,0,:]
    pixelCoord_ref_[3] = pixelCoord_ref[1,1,:]
    # compute homography ref = H @ neighbor, but use inverse warping here neighbor = H @ ref
    H, _ = cv2.findHomography(pixelCoord_ref_, pixelCoord_neighbor)
    # warp neighboring image to get its images warped into ref image plane
    warped_neighbor = cv2.warpPerspective(neighbor_rgb, H, (width,height))
    return warped_neighbor


def zncc_kernel_2D(src, dst):
    """ 
    Compute the cost map between src and dst patchified images via the ZNCC metric
    
    IMPORTANT NOTES:
    - Treat each RGB channel separately but sum the 3 different zncc scores at each pixel

    - When normalizing by the standard deviation, add the provided small epsilon value, 
    EPS which is included in this file, to both sigma_src and sigma_dst to avoid divide-by-zero issues
    m = zncc(i1, i2) is the zero-mean normalized cross-correlation between the two equally sized image patches i1 and i2. 
    The result m is a scalar in the interval -1 to 1 that indicates similarity. 
    A value of 1 indicates identical pixel patterns.
    The ZNCC similarity measure is invariant to affine changes in image intensity (brightness offset and scale).
    Input:
        src -- height x width x K**2 x 3, (h,w,k**2,3) patches
        dst -- height x width x K**2 x 3, patches
    Output:
        zncc -- height x width array of zncc metric computed at each pixel
    """
    assert src.ndim == 4 and dst.ndim == 4
    assert src.shape[:] == dst.shape[:]
    k = src.shape[2]
    h = src.shape[0]
    w = src.shape[1]
    zncc = np.zeros((h,w,3))
    for c in range(3):
        src_patch = src[:,:,:,c]
        src_patch = src_patch.transpose(2,0,1).reshape(k,-1) # (k*k, h*w)
        dst_patch = dst[:,:,:,c]
        dst_patch = dst_patch.transpose(2,0,1).reshape(k,-1) # (k*k, h*w)
        src_mu = np.mean(src_patch, axis =0) # (h,w)
        src_std = np.std(src_patch, axis = 0)
        dst_mu = np.mean(dst_patch, axis =0)
        dst_std = np.std(dst_patch, axis = 0)
        numerator = np.sum((src_patch - src_mu)*(dst_patch - dst_mu), axis = 0) #(h,w)
        zncc[:,:,c] = (numerator/((src_std+EPS)*(dst_std+EPS))).reshape(h,w)
    zncc = np.sum(zncc,axis =2)
    return zncc  # height x width


def backproject(dep_map, K):
    """ 
    Backproject image points to 3D coordinates wrt the camera frame according to the depth map

    Input:
        K -- camera intrinsics calibration matrix
        dep_map -- height x width array of depth values
    Output:
        points -- height x width x 3 array of 3D coordinates of backprojected points
    """
    _u, _v = np.meshgrid(np.arange(dep_map.shape[1]), np.arange(dep_map.shape[0]))
    h, w = dep_map.shape
    # image coordinate (3, h*w)
    pixelCoord = np.zeros((3, h*w))
    pixelCoord[0] = _u.flatten()
    pixelCoord[1] = _v.flatten()
    pixelCoord[2] = np.ones(h*w)
    
    # calibrated coord (3, h*w)
    calibCoord = np.linalg.inv(K) @ pixelCoord

    # camera coordinate (3,h*w)
    cameraCoord = dep_map.flatten() * calibCoord

    xyz_cam = cameraCoord.reshape(3,h,w).transpose(1,2,0) 
    return xyz_cam

