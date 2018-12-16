import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

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
    #H2 = (np.linalg.lstsq(M, b, rcond=-1)[0])
    #H2 = np.reshape(H, (3, 3))

    return H

"""wrong - to delete"""
def forward_mapping2(H, src):

    a = np.ones((1, src.shape[1]))
    D3_vecs = np.vstack((src, a))

    for i, c in enumerate(D3_vecs.T):
        print(i)
        dst_vec = np.dot(H, c)
        dst_vec = dst_vec.reshape((3,1))
        if i == 0:
            dst_vecs = dst_vec
        else:
            dst_vecs = np.hstack((dst_vecs, dst_vec))

    f_map = dst_vecs[:2, :]/dst_vecs[2,:]

    return f_map

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

def forward_image_mapping(H, src_img):
    """ Input: A homography matrix 3*3, src img  to transfer
        Output: dst image (after transformation) scaled according to destination max coordinates"""
    src_img_np = np.asarray(src_img)
    img = np.asarray(src_img[:,:,1])
    ys , xs = np.where(img!=None)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    xs = np.reshape(xs,(xs.shape[0],1)).T
    ys = np.reshape(ys,(ys.shape[0],1)).T

    orig_ind = np.vstack((xs,ys))
    target_ind = forward_mapping(H, orig_ind)
    target_ind = target_ind.round()
    target_ind = target_ind.astype(int)

    """Shift all pixel locations by a constant to avoid negative pixel location values """
    min_i_new = min(target_ind[0,:])
    min_j_new = min(target_ind[1,:])
    if min_i_new < 0:
        target_ind[0,:] += -1 * min(target_ind[0,:])
    if min_j_new < 0:
        target_ind[1, :] += -1 * min(target_ind[1, :])
    max_i_new = max(target_ind[0, :])
    max_j_new = max(target_ind[1, :])
    #H,W,D = src_img_np.shape
    mapping = np.vstack((orig_ind,target_ind))
    """Create a large enough target image to avoid loosing any original image pixels"""
    target_img = np.zeros((max_j_new+1,max_i_new+1,src_img_np.shape[2]))
    for indexes in mapping.T:
        #if (indexes[2]>=0 and indexes[2]<W) and (indexes[3]>=0 and indexes[3]<H):
        print('indexes are xs,ys,xd,ys {}'.format(indexes))
        target_img[indexes[3],indexes[2],:] = src_img_np[indexes[1],indexes[0],:]
    target_img = target_img.astype(int)
    plt.imshow(target_img)
    plt.show()

