import os, shutil
import argparse
import numpy as np
import cv2
import rawpy


def run(imgs_dir, ext):
    count=1
    for root, dirs, files in os.walk(imgs_dir):
        print('\tloading: ' + root)
        for file in sorted(files): 
            if os.path.splitext(file)[1].upper() == ext.upper():
                src_file = os.path.join(root, file)
                dst_file = os.path.join(root, str(count)+ext)
                print('\t{} : {}'.format(src_file, dst_file))
                os.rename(src_file, dst_file) 
                count=count+1

def main():  
    print('>>Mate30pro: ')
    ext = '.jpg'
    imgs_dir = 'F:\\ISP_Dataset\0615_huawei_day_jpg' 
    run(imgs_dir, ext)


    print('>>Redmi8: ')
    ext = '.dng'
    imgs_dir = 'F:\\ISP_Dataset\0615_Redmi_day_raw' 
    run(imgs_dir, ext)


    print('>>oppoR17: ')
    ext = '.dng'
    imgs_dir = 'F:\\ISP_Dataset\0615_oppor17_day_raw' 
    run(imgs_dir, ext)


if __name__ == '__main__':
    main()