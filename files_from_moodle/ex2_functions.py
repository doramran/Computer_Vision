import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import math
from random import randint
def compute_homography_naive(src, dst):
    """Input: Src and destination matching 2*n points
       Output: Homography matrix found by solving P' = HP according to least squares"""
    stack = np.vstack((src, dst))

    for i, c in enumerate(stack.T):

        if i == 0:
            #M = np.array([[-c[0], -c[1], -1, 0, 0, 0, c[0]*c[2], c[1]*c[2], c[2]],
              #[0, 0, 0, -c[0], -c[1], -1, c[0]*c[3], c[1]*c[3], c[3]]]) version a
            M = np.array([[c[0], c[1], 1, 0, 0, 0, -c[0]*c[2], -c[1]*c[2]],
               [0, 0, 0, c[0], c[1], 1, -c[0]*c[3], -c[1]*c[3]]])

            b = np.array(c[2:])

        else:
            #pi = np.array([[-c[0], -c[1], -1, 0, 0, 0, c[0] * c[2], c[1] * c[2], c[2]],
                  #[0, 0, 0, -c[0], -c[1], -1, c[0] * c[3], c[1] * c[3], c[3]]]) version a
            pi = np.array([[c[0], c[1], 1, 0, 0, 0, -c[0] * c[2], -c[1] * c[2]],
                            [0, 0, 0, c[0], c[1], 1, -c[0] * c[3], -c[1] * c[3]]])
            M = np.vstack((M, pi))

            b = np.hstack((b, c[2:]))
    b = np.reshape(b, (b.shape[0], 1))
    M1 = M
    a = np.linalg.inv(np.dot(M1.T, M1))
    d = np.dot(M1.T, b)
    H = np.dot(a, d)
    h_tag = np.zeros((9,1))
    h_tag[:8,0] = H[:,0]
    h_tag[8] = 1
    H = np.reshape(h_tag, (3,3))
    '''FOR COMPARISION '''
    H2 = (np.linalg.lstsq(M, b, rcond=-1)[0])
    #H2 = np.reshape(H, (3, 3))

    return H

def forward_mapping(H, src):
    """ Input : A homography matrix 3*3, src points to transfer as a 3*n array
        output: dst points as a 3*n array"""
    a = np.ones((1, src.shape[1]))
    D3_vecs = np.vstack((src, a))
    D3_vecs = D3_vecs.T
    dst_vecs = D3_vecs.dot(H.T)
    dst_vecs = dst_vecs.T

    f_map = dst_vecs[:2, :]/dst_vecs[2,:]

    return f_map
def get_all_image_indices(HOLOG,src_img):
    img = np.asarray(src_img[:,:,1])
    ys , xs = np.where(img!=None)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    xs = np.reshape(xs,(xs.shape[0],1)).T
    ys = np.reshape(ys,(ys.shape[0],1)).T
    orig_ind = np.vstack((xs,ys))
    target_ind = forward_mapping(HOLOG, orig_ind)
    target_ind = target_ind.round()
    target_ind = target_ind.astype(int)
    return orig_ind,target_ind

def forward_image_mapping(HOLOG, src_img):
    """ Input: A homography matrix 3*3, src img  to transfer
        Output: dst image (after transformation) scaled according to destination max coordinates"""
    orig_ind , target_ind = get_all_image_indices(HOLOG,src_img)
    src_img_np = np.asarray(src_img)
    rows,cols,D = src_img_np.shape
    corners = np.zeros((2,4))
    cor_src = np.zeros((3,4))
    cor_src[:, 0] = np.array([0,0,1]).T
    cor_src[:, 1] = np.array([0,rows,1]).T
    cor_src[:, 2] = np.array([cols,0,1]).T
    cor_src[:, 3] = np.array([cols,rows,1]).T
    cor_dst = np.dot(HOLOG,cor_src)
    corners[0,:] = cor_dst[0,:]/cor_dst[2,:]
    corners[1, :] = cor_dst[1, :] / cor_dst[2, :]
    x1 = math.floor(min(corners[0,:]))
    x2 = math.ceil(max(corners[0, :]))
    y1 = math.floor(min(corners[1, :]))
    y2 = math.ceil(max(corners[1, :]))
    width = x2-x1
    hight = y2-y1
    target_img = np.zeros((hight+1,width+1,D))
    """limit target pixel values in case of outliers in section A"""
    row1 = target_ind[0,:]
    row1 = row1-x1
    row1 = np.array([min(max(row1[i], 1), width) for i in range(row1.shape[0])])
    row2 = target_ind[1, :]
    row2 = row2 - y1
    row2 = np.array([min(max(row2[i], 1), hight) for i in range(row2.shape[0])])
    target_ind = np.vstack((row1,row2))
    target_ind = target_ind.astype(int)
    mapping = np.vstack((orig_ind,target_ind))

    for indexes in mapping.T:
        #if (indexes[2]>=0 and indexes[2]<max_j_new) and (indexes[3]>=0 and indexes[3]<max_i_new):
        print('indexes are xs,ys,xd,ys {}'.format(indexes))
        target_img[indexes[3],indexes[2],:] = src_img_np[indexes[1],indexes[0],:]
    target_img = target_img.astype(int)
    plt.imshow(target_img)
    plt.show()
    return mapping


def test_homography(H, mp_src, mp_dst, max_err):
    """Input: Src and dest index arrays before and after homogrpahy as a 2d ndarrays (i,j) in each coloumn
       Output: percentage of inliers, average distance error of the inlieres and inliers indices"""

    mp_dst_2 = forward_mapping(H, mp_src)
    temp = (mp_dst_2-mp_dst)*(mp_dst_2-mp_dst)
    error_vec = np.sqrt(np.sum(temp,axis =0))
    dist_mse = sum([err for err in error_vec if err <= max_err])/mp_dst_2.shape[1]
    inliers_idx = [i for i in range(error_vec.shape[0]) if error_vec[i] <= max_err]
    fit_percent = len(inliers_idx)/mp_dst_2.shape[1]
    return fit_percent,dist_mse

def compute_inliers(H, mp_src, mp_dst, max_err):
    """Input: Src and dest index arrays before and after homogrpahy as a 2d ndarrays (i,j) in each coloumn
       Output: inliers idx in arrays given"""
    mp_dst_2 = forward_mapping(H, mp_src)
    temp = (mp_dst_2-mp_dst)*(mp_dst_2-mp_dst)
    error_vec = np.sqrt(np.sum(temp,axis =0))
    inliers_idx = [i for i in range(error_vec.shape[0]) if error_vec[i] <= max_err]
    return inliers_idx

def compute_homography(mp_src, mp_dst, inliers_percent, max_err,):
    """Input:Src and dest index arrays before and after homogrpahy as a 2d ndarrays (i,j) in each coloumn
                 inliers_percent: value describing the inliers percent in match points given, max_err- max dist in pixel
                  for which we consider a transformation to be valid """
    p = 0.99
    n = 6
    w = inliers_percent
    k = np.ceil(np.log(1-p)/np.log(1-w**n)).astype(int)
    best_fit = 0
    for i in range(k):
        idx = [randint(0, mp_src.shape[1]-1) for ind in range(4)]
        source = np.take(mp_src,idx,axis = 1)
        dest = np.take(mp_dst,idx,axis = 1)
        H = compute_homography_naive(source, dest)
        fit_percent, dist_mse = test_homography(H, mp_src, mp_dst, max_err)
        inliers_idx = compute_inliers(H, mp_src, mp_dst, max_err)
        num_inl = len(inliers_idx)
        if fit_percent >= best_fit:
            best_fit = fit_percent
            best_H = H
            best_inliers = inliers_idx
    return best_H


def Backward_Mapping(H,src_img):
    """forward mapping to src image"""
    src_img_np = np.asarray(src_img)
    rows,cols,D = src_img_np.shape
    corners = np.zeros((2,4))
    cor_src = np.zeros((2,4))
    cor_src[:, 0] = np.array([0,0]).T
    cor_src[:, 1] = np.array([0,rows]).T
    cor_src[:, 2] = np.array([cols,0]).T
    cor_src[:, 3] = np.array([cols,rows]).T
    corners = forward_mapping(H, cor_src)
    x1 = math.floor(min(corners[0,:]))
    x2 = math.ceil(max(corners[0, :]))
    y1 = math.floor(min(corners[1, :]))
    y2 = math.ceil(max(corners[1, :]))
    width = x2-x1
    hight = y2-y1
    back_map_img = np.uint8(np.zeros([hight, width, 3]))
    """backward mapping mat"""
    H_inv = np.linalg.inv(H)
    """backmap corners"""


    for i in range(0, width):
        for j in range(0 , hight):
            print('currently in width: {}/{} and height: {}/{} '.format(i, width, j, hight))
            dest_p = np.array([i+x1, j+y1,1]).reshape((3, 1))
            back_map_p = np.dot(H_inv,dest_p)
            back_map_p = back_map_p[:2, :] / back_map_p[2, :]
            x = back_map_p[0]
            y = back_map_p[1]

            if x < 0 or x > cols - 1:
                continue
            if y < 0 or y > rows - 1:
                continue
            else:
                Ind_1 = [math.floor(x),math.ceil(y)]
                Ind_2 = [math.floor(x),math.floor(y)]
                Ind_3 = [math.ceil(x),math.ceil(y)]
                Ind_4 = [math.ceil(x),math.floor(y)]

                a = x[0] - math.floor(x)
                b = math.ceil(y) - y

                w11 = (1 - a) * (1 - b) * src_img[Ind_2[1], Ind_2[0], :]
                w22 = a * (1 - b) * src_img[Ind_4[1], Ind_4[0], :]
                w33 = a * b * src_img[Ind_3[1], Ind_3[0], :]
                w44 = (1 - a) * b * src_img[Ind_1[1], Ind_1[0], :]

                back_map_img[j, i, :] = w11 + w22 + w33 + w44
    plt.figure()
    plt.imshow(back_map_img)
    plt.show()
    return back_map_img

def panorama(img_src, img_dst, mp_src, mp_dst, inliers_percent, max_err):
    H_ransac = compute_homography(mp_src, mp_dst, inliers_percent, max_err)

    """forward mapping to src image"""
    src_img_np = np.asarray(img_src)
    dst_img_np = np.asarray(img_dst)

    src_shp_y = src_img_np.shape[0]
    src_shp_x = src_img_np.shape[1]

    dst_shp_y = dst_img_np.shape[0]
    dst_shp_x = dst_img_np.shape[1]

    back_map_img = Backward_Mapping(H_ransac, img_src)
    y_back_shp = back_map_img.shape[0]
    x_back_shp = back_map_img.shape[1]

    # calculate offsets

    orig_ind, target_ind = get_all_image_indices(H_ransac, img_src)

    x_offset = np.abs(np.min([np.min(target_ind[0]),0])).astype(int)  # calculate the offset on x axis
    y_offset = np.abs(np.min([np.min(target_ind[1]), 0])).astype(int)  # calculate the offset on y axis

    # set panorama image size
    y_pan = np.max([dst_shp_y + y_offset, y_back_shp])
    x_pan = np.max([dst_shp_x + x_offset, x_back_shp])
    img_pan = np.zeros((y_pan, x_pan, 3))

    # insert the source image after homographic transform
    img_pan[:y_back_shp, :x_back_shp] = back_map_img

    # insert the destination image
    img_pan[y_offset:dst_shp_y + y_offset, x_offset:dst_shp_x + x_offset] = img_dst
    img_pan = img_pan.astype(np.uint8)

    return img_pan







