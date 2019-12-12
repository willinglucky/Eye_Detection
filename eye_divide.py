# usage :python eye_divide.py predictImage  testList.txt

import os
import sys
import time

import numpy as np
import cv2

os.environ['GLOG_minloglevel'] = '2'
import caffe
import dlib
import matplotlib.pyplot as plt

system_height = 650
system_width = 1280
channels = 1
test_num = 1
pointNum = 68

S0_width = 60
S0_height = 60
vgg_height = 224
vgg_width = 224
M_left = -0.15
M_right = +1.15
M_top = -0.10
M_bottom = +1.25
pose_name = ['Pitch', 'Yaw', 'Roll']  # respect to  ['head down','out of plane left','in plane right']

OUT_PUT_PATH = './output'


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


def recover_coordinate(largetBBox, facepoint, width, height):
    point = np.zeros(np.shape(facepoint))
    cut_width = largetBBox[1] - largetBBox[0]
    cut_height = largetBBox[3] - largetBBox[2]
    scale_x = cut_width * 1.0 / width;
    scale_y = cut_height * 1.0 / height;
    point[0::2] = [float(j * scale_x + largetBBox[0]) for j in facepoint[0::2]]
    point[1::2] = [float(j * scale_y + largetBBox[2]) for j in facepoint[1::2]]
    return point


def show_image(img, facepoint, bboxs, headpose, file_path):
    # plt.figure(figsize=(20,10))
    # for faceNum in range(0,facepoint.shape[0]):
    for faceNum in range(0, 1):
        # cv2.rectangle(img, (int(bboxs[faceNum,0]), int(bboxs[faceNum,2])), (int(bboxs[faceNum,1]), int(bboxs[faceNum,3])), (0,0,255), 2)
        # for p in range(0,3):
        #     plt.text(int(bboxs[faceNum,0]), int(bboxs[faceNum,2])-p*30,
        #         '{:s} {:.2f}'.format(pose_name[p], headpose[faceNum,p]),
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=12, color='white')
        # for i in range(0,facepoint.shape[1]//2):
        #     if i == 36:
        #         cv2.circle(img,(int(round(facepoint[faceNum,i*2])),int(round(facepoint[faceNum,i*2+1]))),1,(0,255,0),2)
        if round(facepoint[faceNum, 19 * 2 + 1]) > round(facepoint[faceNum, 24 * 2 + 1]):
            down_min = round(facepoint[faceNum, 19 * 2 + 1]) - 3
        else:
            down_min = round(facepoint[faceNum, 24 * 2 + 1]) - 3

        if round(facepoint[faceNum, 29 * 2 + 1]) < round(facepoint[faceNum, 28 * 2 + 1]):
            top_max = round(facepoint[faceNum, 29 * 2 + 1]) + 3
        else:
            top_max = round(facepoint[faceNum, 28 * 2 + 1]) + 3
        # print(img.shape)
        img = img[int(down_min):int(top_max),
              int(round(facepoint[faceNum, 36 * 2]) - 5):int(round(facepoint[faceNum, 45 * 2]) + 5)]
        # print(int(round(facepoint[faceNum,36*2])-3),int(round(facepoint[faceNum,45*2])+3),int(top_max),int(down_min))
        # print(img.shape)
        height = img.shape[0]
        width = img.shape[1]
        if height > system_height or width > system_width:
            height_radius = system_height * 1.0 / height
            width_radius = system_width * 1.0 / width
            radius = min(height_radius, width_radius)
            img = cv2.resize(img, (0, 0), fx=radius, fy=radius)

        # pic_name = './result/' + str(id_num) + '-no-' + str(faceNum) + '.jpg'
        top_max = 0
        cv2.imwrite(file_path, img)
        # img = img[:,:,[2,1,0]]
    # pic_name = './result/'+str(id_num)+'.jpg'
    # cv2.imwrite(pic_name, img)
    #
    # plt.clf()
    # plt.close()
    # plt.imshow(img)
    # plt.show()


def recoverPart(point, bbox, left, right, top, bottom, img_height, img_width, height, width):
    largeBBox = getCutSize(bbox, left, right, top, bottom)
    retiBBox = retifyBBoxSize(img_height, img_width, largeBBox)
    recover = recover_coordinate(retiBBox, point, height, width)
    recover = recover.astype('float32')
    return recover


def getRGBTestPart(bbox, left, right, top, bottom, img, height, width):
    largeBBox = getCutSize(bbox, left, right, top, bottom)
    retiBBox = retifyBBox(img, largeBBox)
    # cv2.rectangle(img, (int(retiBBox[0]), int(retiBBox[2])), (int(retiBBox[1]), int(retiBBox[3])), (0,0,255), 2)
    # cv2.imshow('f',img)
    # cv2.waitKey(0)
    face = img[int(retiBBox[2]):int(retiBBox[3]), int(retiBBox[0]):int(retiBBox[1]), :]
    face = cv2.resize(face, (height, width), interpolation=cv2.INTER_AREA)
    face = face.astype('float32')
    return face


def batchRecoverPart(predictPoint, totalBBox, totalSize, left, right, top, bottom, height, width):
    recoverPoint = np.zeros(predictPoint.shape)
    for i in range(0, predictPoint.shape[0]):
        recoverPoint[i] = recoverPart(predictPoint[i], totalBBox[i], left, right, top, bottom, totalSize[i, 0],
                                      totalSize[i, 1], height, width)
    return recoverPoint


def retifyBBox(img, bbox):
    img_height = np.shape(img)[0] - 1
    img_width = np.shape(img)[1] - 1
    if bbox[0] < 0:
        bbox[0] = 0
    if bbox[1] < 0:
        bbox[1] = 0
    if bbox[2] < 0:
        bbox[2] = 0
    if bbox[3] < 0:
        bbox[3] = 0
    if bbox[0] > img_width:
        bbox[0] = img_width
    if bbox[1] > img_width:
        bbox[1] = img_width
    if bbox[2] > img_height:
        bbox[2] = img_height
    if bbox[3] > img_height:
        bbox[3] = img_height
    return bbox


def retifyBBoxSize(img_height, img_width, bbox):
    if bbox[0] < 0:
        bbox[0] = 0
    if bbox[1] < 0:
        bbox[1] = 0
    if bbox[2] < 0:
        bbox[2] = 0
    if bbox[3] < 0:
        bbox[3] = 0
    if bbox[0] > img_width:
        bbox[0] = img_width
    if bbox[1] > img_width:
        bbox[1] = img_width
    if bbox[2] > img_height:
        bbox[2] = img_height
    if bbox[3] > img_height:
        bbox[3] = img_height
    return bbox


def getCutSize(bbox, left, right, top, bottom):  # left, right, top, and bottom

    box_width = bbox[1] - bbox[0]
    box_height = bbox[3] - bbox[2]
    cut_size = np.zeros((4))
    cut_size[0] = bbox[0] + left * box_width
    cut_size[1] = bbox[1] + (right - 1) * box_width
    cut_size[2] = bbox[2] + top * box_height
    cut_size[3] = bbox[3] + (bottom - 1) * box_height
    return cut_size


def detectFace(img):
    detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
    dets = detector(img, 1)
    bboxs = np.zeros((len(dets), 4))
    for i, d in enumerate(dets):
        face = d.rect
        bboxs[i, 0] = face.left();
        bboxs[i, 1] = face.right();
        bboxs[i, 2] = face.top();
        bboxs[i, 3] = face.bottom();
    return bboxs;


def predictImage(dir_path):
    vgg_point_MODEL_FILE = '/home/willing/intership/model/face_model/deploy.prototxt'
    vgg_point_PRETRAINED = '/home/willing/intership/model/face_model/68point_dlib_with_pose.caffemodel'
    mean_filename='/home/willing/intership//model/face_model/VGG_mean.binaryproto'
    vgg_point_net = caffe.Net(vgg_point_MODEL_FILE, vgg_point_PRETRAINED, caffe.TEST)
    # caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    caffe.set_device(0)

    print(dlib.DLIB_USE_CUDA)
    index = 0
    proto_data = open(mean_filename, "rb").read()
    a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
    mean = caffe.io.blobproto_to_array(a)[0]

    start = time.time()
    makedir(OUT_PUT_PATH)
    for root, dirs, files in os.walk(dir_path):
        if len(files) == 0:
            continue

        for file_name in files:
            imgPath = root + '/' + file_name
            print(imgPath)
            colorImage = cv2.imread(imgPath)
            bboxs = detectFace(colorImage)
            faceNum = bboxs.shape[0]
            # print('face_num:',faceNum)
            faces = np.zeros((1, 3, vgg_height, vgg_width))
            predictpoints = np.zeros((faceNum, pointNum * 2))
            predictpose = np.zeros((faceNum, 3))
            imgsize = np.zeros((2))
            imgsize[0] = colorImage.shape[0] - 1
            imgsize[1] = colorImage.shape[1] - 1
            TotalSize = np.zeros((faceNum, 2))
            for i in range(0, faceNum):
                TotalSize[i] = imgsize
            for i in range(0, faceNum):
                bbox = bboxs[i]
                colorface = getRGBTestPart(bbox, M_left, M_right, M_top, M_bottom, colorImage, vgg_height, vgg_width)
                normalface = np.zeros(mean.shape)
                normalface[0] = colorface[:, :, 0]
                normalface[1] = colorface[:, :, 1]
                normalface[2] = colorface[:, :, 2]
                normalface = normalface - mean
                faces[0] = normalface

                blobName = '68point'
                data4DL = np.zeros([faces.shape[0], 1, 1, 1])
                vgg_point_net.set_input_arrays(faces.astype(np.float32), data4DL.astype(np.float32))
                vgg_point_net.forward()
                predictpoints[i] = vgg_point_net.blobs[blobName].data[0]

                blobName = 'poselayer'
                pose_prediction = vgg_point_net.blobs[blobName].data
                predictpose[i] = pose_prediction * 50

            if faceNum > 0:
                predictpoints = predictpoints * vgg_height / 2 + vgg_width / 2
                level1Point = batchRecoverPart(predictpoints, bboxs, TotalSize, M_left, M_right, M_top, M_bottom,
                                               vgg_height, vgg_width)

                file_path = OUT_PUT_PATH + '/' + os.path.split(root)[-1]
                makedir(file_path)
                file_path = file_path + '/' + file_name
                print(file_path)
                show_image(colorImage, level1Point, bboxs, predictpose, file_path)
    end = time.time()
    print(end - start)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        # func = globals()[sys.argv[1]]
        # func(*sys.argv[2:])
        predictImage(sys.argv[1])
