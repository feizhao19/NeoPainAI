import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import os
import cv2
import numpy as np

def detect_img(yolo):
    dir_file = "/data/user/larry5/phd_projects/PainDetector/data/USF/keyframe/"
    
    imgAddress_pain = dir_file + "pain/"
#     imgAddress_no_pain = dir_file + "no_pain/"
    
    face = dir_file + "face_dir/"
    
    undectedAddress = face + "undetected/"
    detectedAddress = face + "detected/"
    
    croppedAddress_pain = face + "pain/"
#     croppedAddress_no_pain = face + "no_pain/"
    os.mkdir(face)
    os.mkdir(undectedAddress)
    os.mkdir(detectedAddress)
    os.mkdir(croppedAddress_pain)
#     os.mkdir(croppedAddress_no_pain)
    img_list_pain = [imgAddress_pain + i for i in os.listdir(imgAddress_pain)]
#     img_list_no_pain = [imgAddress_no_pain + i for i in os.listdir(imgAddress_no_pain)]
#     img_list = img_list_pain + img_list_no_pain
    img_list = img_list_pain
    
    for img in img_list:
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image, boxes, target = yolo.detect_image(image)
            print("\n\n\n\n\n\n",boxes)
            if target == "u":
                r_image.save(undectedAddress + img.split("/")[-1])
            else:
                top, left, bottom, right = boxes[0]
                height = bottom - top
                width = right - left
                
                tempImg = cv2.imread(img)                
                top = max(0, np.floor(top - 0.15 * height).astype('int32'))
                left = max(0, np.floor(left - 0.15 * width).astype('int32'))
                bottom = min(tempImg.shape[1], np.floor(bottom + 0.15 * height).astype('int32'))
                right = min(tempImg.shape[0], np.floor(right + 0.15 * width).astype('int32'))
                croppedImg = tempImg[int(top):int(bottom), int(left):int(right)]
                if "no_pain" in img:
                    pass
                else:              
                    cv2.imwrite(croppedAddress_pain + img.split("/")[-1], croppedImg)
                r_image.save(detectedAddress + img.split("/")[-1])
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
