import cv2
import numpy as np



def gen_triang(canvas=[256, 256], color1=(0,255,0), color2=(0,0,255)):
    """Generate a triangle centered at the center of canvas"""
    w, h = canvas[0], canvas[1]
    image1 = np.zeros((w, h, 3), np.uint8) * 255
    image2 = np.zeros((w, h, 3), np.uint8) * 255
    pt1 = ((np.random.rand()-1)*w/4+w/2, (np.random.rand())*h/4+h/2)
    pt2 = ((np.random.rand()-0.5)*w/4+w/2, (np.random.rand()-1)*h/4+h/2)
    pt3 = ((np.random.rand())*w/4+w/2, (np.random.rand())*h/4+h/2)
    triangle_cnt = np.array([pt1, pt2, pt3]).reshape((-1,1,2)).astype(np.int32)
    center = (np.sum(triangle_cnt, 0)/3)
    shift = np.array([w/2, h/2]) - center
    triangle_cnt += shift.astype(np.int32)
    cv2.drawContours(image1, [triangle_cnt], 0, color1, -1)
    cv2.drawContours(image2, [triangle_cnt], 0, color2, -1)
    return image1, image2


def gen_circle(canvas=[256, 256], color1=(0,255,0), color2=(0,0,255)):
    """Generate a circle centered at the center of canvas"""
    w, h = canvas[0], canvas[1]
    image1 = np.zeros((w, h, 3), np.uint8) * 255
    image2 = np.zeros((w, h, 3), np.uint8) * 255
    radius = np.random.rand()*min(w,h)/3
    cv2.circle(image1, (int(w/2),int(h/2)), int(radius), color1, -1)
    cv2.circle(image2, (int(w/2),int(h/2)), int(radius), color2, -1)
    return image1, image2

def add_gaussian_noise(ima, sigma=120):
    """Add gaussian noise with a given sigma"""
    rows, columns, channels = ima.shape
    noise = np.zeros((rows, columns, channels))
    mean = (0,0,0) 
    s = (sigma, sigma, sigma)
    cv2.randn(noise, mean, s)
    noisy_image = np.maximum(np.minimum((0.9*ima + 0.1*noise),255), 0)
    return noisy_image.astype(np.uint8)

def main():

    # image1 = np.zeros((1, 1, 3), np.uint8) * 255
    # image2 = np.zeros((1, 1, 3), np.uint8) * 255
    # pt =np.array((0,0)).reshape((-1,1,2)).astype(np.int32)
    # cv2.drawContours(image1, [pt], 0, (0,0,255), -1)
    # image1=cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # for x in range(255):
    #     image2 = np.zeros((1, 1, 3), np.uint8) * 255
    #     cv2.drawContours(image2, [pt], 0, (0,x,0), -1)
    #     image2=cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    #     pix1 = image1[0,0]
    #     pix2 = image2[0,0]
        
    #     if pix1==pix2:
    #         print(x)
    # return

    output_dir = "Datasets/Colorization"
    # output_dir = "Datasets/Circles"
    amount = 120
    for i in range(amount):
        if i%100 == 0:
            print("Generated %d instances" % i)
        # im1,im2 = gen_triang()
        tr,_ = gen_triang(color1=(0,130,0))
        _,circ = gen_circle()
        # cv2.imshow("Circle 1", im1)
        # cv2.imshow("Circle 2", im2)
        # cv2.imshow("Circle 1-gray", cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY))
        # cv2.imshow("Circle 2-gray", cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY))
        # cv2.waitKey()
        cv2.imwrite(output_dir+"/A/t%d.jpg" % i, tr)
        cv2.imwrite(output_dir+"/A/c%d.jpg" % i, circ)
        cv2.imwrite(output_dir+"/B/t%d.jpg" % i, cv2.cvtColor(tr, cv2.COLOR_BGR2GRAY))
        cv2.imwrite(output_dir+"/B/c%d.jpg" % i, cv2.cvtColor(circ, cv2.COLOR_BGR2GRAY))

if __name__ == '__main__':
    main()
