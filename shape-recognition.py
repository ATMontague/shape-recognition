import cv2 as cv
import numpy as np

def compactness(img):
    '''
        Input
        -----
        img : np.array

        Output
        ------
        c : float
    '''

    contours, hierarchy = cv.findContours(img, 1, 2)
    cnt = contours[0]

def main():
    '''
    Quick implementation of:
        Object Shape Recognition in Image for Machine Vision Application
    '''

    # get image in HSL (hue, saturation. light )colorspace
    img = cv.imread('input.jpg', cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    img = img[:,:,1]

    # binarize image with Otsu's method
    val, binary_img = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    cv.imwrite('binary.jpg', binary_img)

    # 'Image Fills' ?

    # median filter for noise reduction
    img = cv.medianBlur(binary_img,3)

    # Sobel operator to get image gradient
    k = 5
    sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=k)
    sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=k)
    abs_grad_x = cv.convertScaleAbs(sobel_x)
    abs_grad_y = cv.convertScaleAbs(sobel_y)
    gradient = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    cv.imwrite('gradient.jpg', gradient)

    # thinning the image
    trash, gradient  = cv.threshold(gradient, 0, 255, cv.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    img = cv.erode(gradient, kernel, iterations=1)
    cv.imwrite('thinned.jpg', img)

    # calculate compactness (paper doesn't say how to find perimeter or area)
    c = compactness(img)

if __name__ == '__main__':
    main()
