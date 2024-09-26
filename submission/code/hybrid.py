from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr
import math
import cv2

def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)

def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int) (np.abs(2*r+1 - R))
    cpad = (int) (np.abs(2*c+1 - C))
    return np.pad(
        im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
             (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad),
             (0, 0)], 'constant')

def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape
    
    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale, channel_axis=2)
    else:
        im2 = sktr.rescale(im2, 1./dscale, channel_axis=2)
    return im1, im2

def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta*180/np.pi)
    return im1, dtheta

def match_img_size(im1, im2):
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    if h1 < h2:
        im2 = im2[int(np.floor((h2-h1)/2.)) : -int(np.ceil((h2-h1)/2.)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1-h2)/2.)) : -int(np.ceil((h1-h2)/2.)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2-w1)/2.)) : -int(np.ceil((w2-w1)/2.)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1-w2)/2.)) : -int(np.ceil((w1-w2)/2.)), :]
    assert im1.shape == im2.shape, f"Image shapes do not match: {im1.shape} vs {im2.shape}"
    return im1, im2

def align_images(im1, im2):
    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2

def low_pass_filter(image, sigma=1):
    gauss = cv2.getGaussianKernel(ksize=20, sigma=sigma)
    gauss_2d = np.outer(gauss, np.transpose(gauss))
    blurred_channels = []
    for i in range(3):
        channel = image[:, :, i]
        blurred = signal.convolve2d(channel, gauss_2d, mode='same', boundary='symm')
        blurred_channels.append(blurred)

    return np.stack(blurred_channels, axis=-1)

def high_pass_filter(image, sigma):
    low_pass = low_pass_filter(image, sigma)
    return image - low_pass

def hybrid_image(im1, im2, sigma1, sigma2):
    low_frequencies = low_pass_filter(im1, sigma1)
    high_frequencies = high_pass_filter(im2, sigma2)
    return low_frequencies + high_frequencies

def compute_and_display_fourier(image, title):
    if image.ndim == 3: gray_image = np.mean(image, axis=2)
    else: gray_image = image    
    plt.figure(figsize=(8, 6))
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray_image)))))
    plt.title(f'Fourier Transform: {title}')
    plt.colorbar()
    plt.show()

im1 = plt.imread('images/hybrid/happy.jpg')
im2 = plt.imread('images/hybrid/angry.jpg')

im1_aligned, im2_aligned = align_images(im1, im2)

sigma1 = 10
sigma2 = 15

hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

compute_and_display_fourier(im1_aligned, 'Input Image 1')
compute_and_display_fourier(im2_aligned, 'Input Image 2')
compute_and_display_fourier(low_pass_filter(im1_aligned, sigma1), 'Low-pass Filtered Image 1')
compute_and_display_fourier(high_pass_filter(im2_aligned, sigma2), 'High-pass Filtered Image 2')
compute_and_display_fourier(hybrid, 'Hybrid Image')

plt.imshow(hybrid)
plt.axis('off')
plt.show()
