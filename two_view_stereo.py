import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import imageio
from tqdm import tqdm
from transforms3d.euler import mat2euler, euler2mat
import pyrender
import trimesh
import cv2
import open3d as o3d


from dataloader import load_middlebury_data
from utils import viz_camera_poses

EPS = 1e-8


def homo_corners(h, w, H):
    corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_aft = cv2.perspectiveTransform(corners_bef, H).squeeze(1)
    u_min, v_min = corners_aft.min(axis=0)
    u_max, v_max = corners_aft.max(axis=0)
    return u_min, u_max, v_min, v_max


def rectify_2view(rgb_i, rgb_j, R_irect, R_jrect, K_i, K_j, u_padding=20, v_padding=20):
    """Given the rectify rotation, compute the rectified view and corrected projection matrix

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    R_irect,R_jrect : [3,3]
        p_rect_left = R_irect @ p_i
        p_rect_right = R_jrect @ p_j
    K_i,K_j : [3,3]
        original camera matrix
    u_padding,v_padding : int, optional
        padding the border to remove the blank space, by default 20

    Returns
    -------
    [H,W,3],[H,W,3],[3,3],[3,3]
        the rectified images
        the corrected camera projection matrix. WE HELP YOU TO COMPUTE K, YOU DON'T NEED TO CHANGE THIS
    """
    # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
    assert rgb_i.shape == rgb_j.shape, "This hw assumes the input images are in same size"
    h, w = rgb_i.shape[:2]

    ui_min, ui_max, vi_min, vi_max = homo_corners(h, w, K_i @ R_irect @ np.linalg.inv(K_i))
    uj_min, uj_max, vj_min, vj_max = homo_corners(h, w, K_j @ R_jrect @ np.linalg.inv(K_j))

    # The distortion on u direction (the world vertical direction) is minor, ignore this
    w_max = int(np.floor(max(ui_max, uj_max))) - u_padding * 2
    h_max = int(np.floor(min(vi_max - vi_min, vj_max - vj_min))) - v_padding * 2

    assert K_i[0, 2] == K_j[0, 2], "This hw assumes original K has same cx"
    K_i_corr, K_j_corr = K_i.copy(), K_j.copy()
    K_i_corr[0, 2] -= u_padding
    K_i_corr[1, 2] -= vi_min + v_padding
    K_j_corr[0, 2] -= u_padding
    K_j_corr[1, 2] -= vj_min + v_padding

    """Student Code Starts"""
    Hi = K_i_corr @ R_irect @ np.linalg.inv(K_i)
    Hj = K_j_corr @ R_jrect @ np.linalg.inv(K_j)
    rgb_i_rect = cv2.warpPerspective(rgb_i, Hi, dsize = (w_max, h_max))
    rgb_j_rect = cv2.warpPerspective(rgb_j, Hj, dsize = (w_max, h_max))
    """Student Code Ends"""

    return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr


def compute_right2left_transformation(R_wi, T_wi, R_wj, T_wj):
    """Compute the transformation that transform the coordinate from j coordinate to i

    Parameters
    ----------
    R_wi, R_wj : [3,3]
    T_wi, T_wj : [3,1]
        p_i = R_wi @ p_w + T_wi
        p_j = R_wj @ p_w + T_wj
    Returns
    -------
    [3,3], [3,1], float
        p_i = R_ji @ p_j + T_ji, B is the baseline
    """

    """Student Code Starts"""
    R_ji = R_wi @ R_wj.T 
    T_ji = - R_wi @ R_wj.T @ T_wj + T_wi
    B = np.linalg.norm(T_ji) # distance between the projection center of two cameras in stereo
    """Student Code Ends"""

    return R_ji, T_ji, B


def compute_rectification_R(T_ji):
    """Compute the rectification Rotation

    Parameters
    ----------
    T_ji : [3,1]

    Returns
    -------
    [3,3]
        p_rect = R_irect @ p_i
    """
    # check the direction of epipole, should point to the positive direction of y axis
    e_i = T_ji.squeeze(-1) / (T_ji.squeeze(-1)[1] + EPS)

    """Student Code Starts"""
    r2 = T_ji/np.linalg.norm(T_ji) # (3,1)
    r1 = np.cross(e_i, np.array([0, 0, 1])).reshape(-1,1)
    r3 = np.cross(r1.flatten(), r2.flatten()).reshape(-1,1) # (3,1)
    R_irect = np.array([
        r1.T.flatten(), r2.T.flatten(), r3.T.flatten()
    ])
    """Student Code Ends"""

    return R_irect


def ssd_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    M = src.shape[0]
    N = dst.shape[0]
    ssd = np.zeros((M,N,3))
    for c in range(3):
        for i in range(M):
            diff = src[i,:,c] - dst[:,:,c]
            ssd[i,:,c] = np.linalg.norm(diff, axis = 1)**2 # (N,)
    ssd = np.sum(ssd,axis = 2) # (M,N)
    """Student Code Ends"""

    return ssd  # M,N


def sad_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    M = src.shape[0]
    N = dst.shape[0]
    sad = np.zeros((M,N,3))
    for c in range(3):
        for i in range(M):
            sad[i,:,c] = np.sum(np.abs(src[i,:,c] - dst[:,:,c]), axis = 1)
    sad = np.sum(sad, axis = 2) # (M,N)
    """Student Code Ends"""
    """Student Code Ends"""

    return sad  # M,N


def zncc_kernel(src, dst):
    """Compute negative zncc similarity, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    M = src.shape[0]
    N = dst.shape[0]
    
    src_mu = np.mean(src, axis = 1) # (M,3)
    src_std = np.std(src, axis =1) # (M,3)
    dst_mu = np.mean(dst, axis = 1) # (N,3)
    dst_std = np.std(dst, axis =1) # (N,3)
    zncc = np.zeros((M,N,3))
    for c in range(3):
        for i in range(M):
            # for j in range(N):
            #     numerator = (src[i,:,c]-src_mu[i,c])*(dst[j,:,c]-dst_mu[j,c])
            #     zncc[i,j,c] = np.mean(numerator) / (src_std[i,c]*dst_std[j,c] + EPS)
            parts_to_sum = (src[i, : ,c] - src_mu[i,c]) * (dst[:, : ,c] - dst_mu[:,c].reshape(-1,1)) # (k*k, ) * (M,k*k)
            zncc[i,:,c] = np.sum(parts_to_sum, axis = 1) / (src_std[i,c]*dst_std[:,c]+EPS)
    zncc = np.sum(zncc, axis =2)
    """Student Code Ends"""

    return zncc * (-1.0)  # M,N


def image2patch(image, k_size):
    """get patch buffer for each pixel location from an input image; For boundary locations, use zero padding

    Parameters
    ----------
    image : [H,W,3]
    k_size : int, must be odd number; your function should work when k_size = 1

    Returns
    -------
    [H,W,k_size**2,3]
        The patch buffer for each pixel
    """

    
    H = image.shape[0]
    W = image.shape[1]
    patch_buffer = np.zeros((H,W,k_size*k_size,3))
    # pad the image
    pad_w = k_size // 2 # pad width
    padded_image= np.pad(image, pad_w, mode='constant', constant_values=0)
    padded_image = padded_image[:,:,pad_w:padded_image.shape[2]-pad_w]
    patch_buffer = np.zeros((H,W,k_size**2,3))
    # store patch indices of every pixel in an H,W,K*K array
    for i in range(H):
        for j in range(W):
            center_row=i+pad_w
            center_col=j+pad_w
            patch_col, patch_row = np.meshgrid(np.arange(j,center_col + pad_w + 1), np.arange(i,center_row + pad_w + 1))
            for c in range(3):
                patch_buffer[i,j,:,c] = padded_image[patch_row, patch_col, c].flatten()
    

    return patch_buffer  # H,W,K**2,3


def compute_disparity_map(rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel):
    """Compute the disparity map from two rectified view

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3], rectified images
    d0 : see the hand out, the bias term of the disparty caused by different K matrix
    k_size : int, optional
        The patch size, by default 3
    kernel_func : function, optional
        the kernel used to compute the patch similarity, by default ssd_kernel

    Returns
    -------
    disp_map: [H,W], dtype=np.float64
        The disparity map, the disparity is defined in the handout as d0 + vL - vR
        vL is the pixel location of the left correspondence
        vR is the pixel location of the right correspondence

    lr_consistency_mask: [H,W], dtype=np.float64
        For each pixel, 1.0 if LR consistent, otherwise 0.0
    """
    h, w = rgb_i.shape[:2]

    patches_i = image2patch(rgb_i.astype(float) / 255.0, k_size)  # [h,w,k*k,3]
    patches_j = image2patch(rgb_j.astype(float) / 255.0, k_size)  # [h,w,k*k,3]
    vi_idx, vj_idx = np.arange(h), np.arange(h)
    disp_candidates = vi_idx[:, None] - vj_idx[None, :] + d0
    valid_disp_mask = disp_candidates > 0.0

    positive_disp_mask = np.zeros((h,w))
    disp_map = np.zeros((h,w))
    lr_consistency_mask = np.zeros((h,w))

    for u in tqdm(range(w)):
        # take each column patch to find correspondence
        buf_i, buf_j = patches_i[:, u], patches_j[:, u]
        # value  = (M,N) with the (i,j) storing the difference betwenn ith col patch from the left and the jth col patch from the right
        value = kernel_func(buf_i, buf_j)  
        # Students, why we compute this `_upper` ?
        _upper = value.max() + 1.0
        value[~valid_disp_mask] = _upper

        # left, right disparity consistency
        # the right patches of the smallest disparity with the left patches should correspond to the same left patch that gives the smallest disparity
        best_matched_right_pixel = np.argmin(value, axis = 1).flatten() #(H, )
        best_matched_left_pixel = np.argmin(value[:,best_matched_right_pixel], axis = 0).flatten() #(H, )
        consistent_flag = best_matched_left_pixel == np.arange(h) # (H, ) the lr consistency for the uth col of images
        lr_consistency_mask[:,u] = consistent_flag 
        # compute disparity map
        vL = np.arange(h)
        vR = best_matched_right_pixel
        d = d0 + vL - vR
        positive_disp = d > 0
        positive_disp_mask [:,u] = positive_disp
        disp_map[:,u] = d

    # valid_disp_mask = np.logical_or(positive_disp_mask, lr_consistency_mask) # valid disp should be larger than 0 and satisfies lr_consistency
    # valid_disp_map = np.zeros((h,w))
    # valid_disp_map[valid_disp_mask] = disp_map[valid_disp_mask] # set the disparity to zeros for invalid pixels

    return disp_map, lr_consistency_mask


def compute_dep_and_pcl(disp_map, B, K):
    """Given disparity map d = d0 + vL - vR, the baseline and the camera matrix K
    compute the depth map and backprojected point cloud

    Parameters
    ----------
    disp_map : [H,W]
        disparity map
    B : float
        baseline
    K : [3,3]
        camera matrix

    Returns
    -------
    [H,W]
        dep_map
    [H,W,3]
        each pixel is the xyz coordinate of the back projected point cloud in camera frame
    """

    
    fy = K[1,1]
    dep_map = fy*B * 1/disp_map

    h, w = disp_map.shape
    xyz_cam = np.zeros((h,w,3))
    # (h,w,3) stores pixel coordinate
    x_pixel, y_pixel = np.meshgrid(np.arange(w), np.arange(h))
    xyz_pixel = np.zeros((3,h*w))
    xyz_pixel[0] = x_pixel.flatten()
    xyz_pixel[1] = y_pixel.flatten()
    xyz_pixel[2] = 1

    # compute homogenous camera coordinate (3,h*w)
    xyz_cam_2d = dep_map.flatten() * (np.linalg.inv(K) @ xyz_pixel)

    # reshape xyz_cam into shape (h,w,3)
    for i in range(h):
        for j in range(w):
            n = i*w + j
            xyz_cam[i,j,:] = xyz_cam_2d[:,n]
    

    return dep_map, xyz_cam


def postprocess(
    dep_map,
    rgb,
    xyz_cam,
    R_wc,
    T_wc,
    consistency_mask=None,
    hsv_th=45,
    hsv_close_ksize=11,
    z_near=0.45,
    z_far=0.65,
):
    """
    Your goal in this function is: 
    given pcl_cam [N,3], R_wc [3,3] and T_wc [3,1]
    compute the pcl_world with shape[N,3] in the world coordinate
    """

    # extract mask from rgb to remove background
    mask_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., -1]
    mask_hsv = (mask_hsv > hsv_th).astype(np.uint8) * 255
    # imageio.imsave("./debug_hsv_mask.png", mask_hsv)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hsv_close_ksize, hsv_close_ksize))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, morph_kernel).astype(float)
    # imageio.imsave("./debug_hsv_mask_closed.png", mask_hsv)

    # constraint z-near, z-far
    mask_dep = ((dep_map > z_near) * (dep_map < z_far)).astype(float)
    # imageio.imsave("./debug_dep_mask.png", mask_dep)

    mask = np.minimum(mask_dep, mask_hsv)
    if consistency_mask is not None:
        mask = np.minimum(mask, consistency_mask)
    # imageio.imsave("./debug_before_xyz_mask.png", mask)

    # filter xyz point cloud
    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcl_cam.reshape(-1, 3).copy())
    cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    _pcl_mask = np.zeros(pcl_cam.shape[0])
    _pcl_mask[ind] = 1.0
    pcl_mask = np.zeros(xyz_cam.shape[0] * xyz_cam.shape[1])
    pcl_mask[mask.reshape(-1) > 0] = _pcl_mask
    mask_pcl = pcl_mask.reshape(xyz_cam.shape[0], xyz_cam.shape[1])
    # imageio.imsave("./debug_pcl_mask.png", mask_pcl)
    mask = np.minimum(mask, mask_pcl)
    # imageio.imsave("./debug_final_mask.png", mask)

    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    pcl_color = rgb.reshape(-1, 3)[mask.reshape(-1) > 0]

    """Student Code Starts"""
    pcl_world = (R_wc.T @ (pcl_cam.T - T_wc) ).T
    """Student Code Ends"""

    # np.savetxt("./debug_pcl_world.txt", np.concatenate([pcl_world, pcl_color], -1))
    # np.savetxt("./debug_pcl_rect.txt", np.concatenate([pcl_cam, pcl_color], -1))

    return mask, pcl_world, pcl_cam, pcl_color


def two_view(view_i, view_j, k_size=5, kernel_func=ssd_kernel):
    # Full pipeline

    # * 1. rectify the views
    R_wi, T_wi = view_i["R"], view_i["T"][:, None]  # p_i = R_wi @ p_w + T_wi
    R_wj, T_wj = view_j["R"], view_j["T"][:, None]  # p_j = R_wj @ p_w + T_wj

    R_ji, T_ji, B = compute_right2left_transformation(R_wi, T_wi, R_wj, T_wj)
    assert T_ji[1, 0] > 0, "here we assume view i should be on the left, not on the right"

    R_irect = compute_rectification_R(T_ji)

    rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr = rectify_2view(
        view_i["rgb"],
        view_j["rgb"],
        R_irect,
        R_irect @ R_ji,
        view_i["K"],
        view_j["K"],
        u_padding=20,
        v_padding=20,
    )

    # * 2. compute disparity
    assert K_i_corr[1, 1] == K_j_corr[1, 1], "This hw assumes the same focal Y length"
    assert (K_i_corr[0] == K_j_corr[0]).all(), "This hw assumes the same K on X dim"
    assert (
        rgb_i_rect.shape == rgb_j_rect.shape
    ), "This hw makes rectified two views to have the same shape"
    disp_map, consistency_mask = compute_disparity_map(
        rgb_i_rect,
        rgb_j_rect,
        d0=K_j_corr[1, 2] - K_i_corr[1, 2],
        k_size=k_size,
        kernel_func=kernel_func,
    )

    # * 3. compute depth map and filter them
    dep_map, xyz_cam = compute_dep_and_pcl(disp_map, B, K_i_corr)

    mask, pcl_world, pcl_cam, pcl_color = postprocess(
        dep_map,
        rgb_i_rect,
        xyz_cam,
        R_wc=R_irect @ R_wi,
        T_wc=R_irect @ T_wi,
        consistency_mask=consistency_mask,
        z_near=0.5,
        z_far=0.6,
    )

    return pcl_world, pcl_color, disp_map, dep_map


def main():
    DATA = load_middlebury_data("data/templeRing")
    # viz_camera_poses(DATA)
    two_view(DATA[0], DATA[3], 5, zncc_kernel)

    return


if __name__ == "__main__":
    main()
