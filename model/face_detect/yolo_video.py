import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import os
import cv2
import numpy as np

def detect_img(yolo):
#     imgAddress = "/data/user/larry5/2021_summer/face/keras-yolo3-facedetection/dataset/2907createData/rotatedData/"
#     undectedAddress = "/data/user/larry5/2021_summer/face/keras-yolo3-facedetection/dataset/2907createData/undetected/"
#     detectedAddress = "/data/user/larry5/2021_summer/face/keras-yolo3-facedetection/dataset/2907createData/detected/"
#     croppedAddress = "/data/user/larry5/2021_summer/face/keras-yolo3-facedetection/dataset/2907createData/faceData/"

#     imgAddress = "/data/user/larry5/2021_summer/face/keras-yolo3-facedetection/dataset/2907createData/test_rotate/nonpain/"
#     undectedAddress = "/data/user/larry5/2021_summer/face/keras-yolo3-facedetection/dataset/2907createData/test/undetected/"
#     detectedAddress = "/data/user/larry5/2021_summer/face/keras-yolo3-facedetection/dataset/2907createData/test/detected/"
#     croppedAddress = "/data/user/larry5/2021_summer/face/keras-yolo3-facedetection/dataset/2907createData/test_faced_extend/nonpain/"

    imgAddress = "/data/user/larry5/USF_video/images/pain/"
    undectedAddress = "/data/user/larry5/USF_video/face_data/undetected/"
    detectedAddress = "/data/user/larry5/USF_video/face_data/detected/"
    croppedAddress = "/data/user/larry5/USF_video/face_data/pain"
    img_list = [imgAddress + i for i in os.listdir(imgAddress)]
    
    
    
    for img in img_list:

        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image, boxes, target = yolo.detect_image(image)
            print("\n\n\n\n\n\n",boxes)
            print("\n\n\n\n\n\n")

            if target == "u":
                r_image.save(undectedAddress + img.split("/")[-1])
            else:
                top, left, bottom, right = boxes[0]
                height = bottom - top
                width = right - left
                
                tempImg = cv2.imread(img)
#                 tempImg = cv2.rotate(tempImg, cv2.ROTATE_90_CLOCK)
                
                top = max(0, np.floor(top - 0.15 * height).astype('int32'))
                left = max(0, np.floor(left - 0.15 * width).astype('int32'))
                bottom = min(tempImg.shape[1], np.floor(bottom + 0.15 * height).astype('int32'))
                right = min(tempImg.shape[0], np.floor(right + 0.15 * width).astype('int32'))
#                 croppedImg = tempImg[left:right, top:bottom]
                croppedImg = tempImg[int(top):int(bottom), int(left):int(right)]
                cv2.imwrite(croppedAddress + img.split("/")[-1], croppedImg)
                r_image.save(detectedAddress + img.split("/")[-1])
            # r_image.show()
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        print("i'm here")
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
