import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io
import time
import os
import cv2
from ex2_functions import *
def tic():
    return time.time()
def toc(t):
    return float(tic()) - float(t)

##########################################################
# Don't forget to fill in your IDs!!!
# students' IDs:
ID1 = 302236054
ID2 = 302824099
##########################################################


# Parameters
max_err = 25
inliers_percent = 0.8

# Read the data:
path = 'C:/Users/dorim/Documents/GitHub/Computer_Vision/files_from_moodle'
img_src = mpimg.imread(os.path.join(path, 'src.jpg'))
img_dst = mpimg.imread(os.path.join(path, 'dst.jpg'))

matches = scipy.io.loadmat(os.path.join(path, 'matches_perfect')) #loading perfect matches
match_p_dst = matches['match_p_dst'].astype(float)
match_p_src = matches['match_p_src'].astype(float)

matches2 = scipy.io.loadmat(os.path.join(path, 'matches')) #matching points and some outliers
match_p_dst2 = matches2['match_p_dst'].astype(float)
match_p_src2 = matches2['match_p_src'].astype(float)

# Compute naive homography

tt = time.time()
H_naive = compute_homography_naive(match_p_src, match_p_dst)
H_2 = compute_homography_naive(match_p_src2, match_p_dst2)
mp_src , mp_dst_naive = get_all_image_indices(H_naive,img_src)
print('Naive Homography {:5.4f} sec'.format(toc(tt)))
#forward_image_mapping(H_naive, img_src)


# Test naive homography
tt = time.time()
fit_percent, dist_mse = test_homography(H_naive, match_p_src, match_p_dst, max_err)
print('Naive Homography Test {:5.4f} sec'.format(toc(tt)))
print([fit_percent, dist_mse])


# Compute RANSAC homography
tt = tic()
#q8
H_ransac_1 = compute_homography(match_p_src, match_p_dst, inliers_percent, max_err)
#forward_image_mapping(H_ransac_1, img_src)
#q9
H_ransac_2 = compute_homography(match_p_src2, match_p_dst2, inliers_percent, max_err)
#forward_image_mapping(H_ransac_2, img_src)
print('RANSAC Homography {:5.4f} sec'.format(toc(tt)))
print(H_ransac_2)


# Test RANSAC homography
tt = tic()
fit_percent, dist_mse = test_homography(H_ransac_2, match_p_src, match_p_dst, max_err)
print('RANSAC Homography Test {:5.4f} sec'.format(toc(tt)))
print([fit_percent, dist_mse])

#present backward mapping
#back_map_img = Backward_Mapping(H_ransac_2,img_src)


# Build panorama
tt = tic()
#img_pan = panorama(img_src, img_dst, match_p_src, match_p_dst, inliers_percent, max_err)
#print('Panorama {:5.4f} sec'.format(toc(tt)))

#plt.figure()
#panplot = plt.imshow(img_pan)
#plt.title('Great Panorama')
#plt.show()



## Student Files
#first run "create_matching_points.py" with your own images to create a mat file with the matching coordinates.
max_err = 25 # <<<<< YOU MAY CHANGE THIS
inliers_percent = 0.8 # <<<<< YOU MAY CHANGE THIS

img_src_test = mpimg.imread('G1971625.jpg')
img_dst_test = mpimg.imread('G1971624.jpg')

matches_test = scipy.io.loadmat('matches_test')

match_p_dst = matches_test['match_p_dst']
match_p_src = matches_test['match_p_src']

# Build student panorama

tt = tic()
img_pan = panorama(img_src_test, img_dst_test, match_p_src, match_p_dst, inliers_percent, max_err)
print('Student Panorama {:5.4f} sec'.format(toc(tt)))

plt.figure()
panplot = plt.imshow(img_pan)
plt.title('Awesome Panorama')
plt.show()


