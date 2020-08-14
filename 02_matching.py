import os, shutil
import cv2
import argparse
import numpy as np
import splitraw as s
import imageio
import rawpy
from scipy import misc
from colour_demosaicing import (
    EXAMPLES_RESOURCES_DIRECTORY,
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)
    
def crop(redmiraw_file, redmiraw_img, huaweirgb_img, 
    redmiraw_gray, huaweirgb_cropped, output_root, tag):
    img1_raw = redmiraw_img
    img1 = redmiraw_gray
    img2 = huaweirgb_cropped

    img_name = os.path.split(redmiraw_file)[-1]
    img_name = os.path.splitext(img_name)[0]

    h, w = img1.shape
    stride=448 #patch
    pp_dir = os.path.join(output_root, tag+'_pairedpatches')
    if not os.path.exists(pp_dir):
        os.makedirs(pp_dir)

    # patches_dir
    patches_dir = os.path.join(output_root, tag+'_patches') 
    raw_patches_dir = os.path.join(patches_dir, tag.split('_')[1])
    rgb_patches_dir = os.path.join(patches_dir, tag.split('_')[-1])
    #print('\t@raw_patches_dir: {}'.format(raw_patches_dir))
    #print('\t@rgb_patches_dir: {}'.format(rgb_patches_dir))

    if not os.path.exists(patches_dir):
        os.makedirs(patches_dir)

    if not os.path.exists(raw_patches_dir):
        os.makedirs(raw_patches_dir)

    if not os.path.exists(rgb_patches_dir):
        os.makedirs(rgb_patches_dir)

    idx = 0
    for hc in range (0, h, stride):
        for wc in range(0, w, stride):
            if hc + 448 < h and wc + 448 < w:
                cropped1 = img1[0+hc:448+hc, 0+wc:448+wc]
                cropped2 = img2[0+hc:448+hc, 0+wc:448+wc]
                cropp1 = cropped1.reshape(cropped1.size, order='C')  # 
                cropp2 = cropped2.reshape(cropped2.size, order='C')
                #print("FileName: "+str(hc+448)+'x'+str(wc+448)+"  Score: "+str(np.corrcoef(cropp1, cropp2)[0, 1]))
                #threshold
                ssssss = np.corrcoef(cropp1, cropp2)[0, 1] 
                if ssssss > 0.9:
                    idx += 1
                    #print('\t@cropped1: ', np.shape(cropped1))
                    #print('\t@cropped2: ', np.shape(cropped1))
                    cropped_compare = np.hstack((cropped1, cropped2))
                    pp_file = os.path.join(pp_dir, '%s_%d_%.02f.jpg'%(img_name, idx, ssssss))
                    cv2.imwrite(pp_file, cropped_compare) 

                    # Save raw_patch & rgb_patch
                    cropped1_raw = img1_raw[0+hc:448+hc, 0+wc:448+wc]
                    rgb_patch = huaweirgb_img[0+hc:448+hc, 0+wc:448+wc]

                    h0 = np.shape(cropped1_raw)[0]
                    w0 = np.shape(cropped1_raw)[1]
                    h1 = np.shape(rgb_patch)[0] 
                    w1 = np.shape(rgb_patch)[1] 
 
                    if h0==448 and h1 == 448 and w0 == 448 and w1 == 448:
                        cropped1_raw = cropped1_raw.astype(np.uint16)

                        raw_patch_file = os.path.join(raw_patches_dir, '%s_%d.png'%(img_name, idx))
                        imageio.imwrite(raw_patch_file, cropped1_raw)
     
                        rgb_patch_file = os.path.join(rgb_patches_dir, '%s_%d.jpg'%(img_name, idx))
                        cv2.imwrite(rgb_patch_file, rgb_patch)
 
                        # Check patch: raw_rgb
                        #     Read redmi8 raw patch
                        raw_image = np.asarray(imageio.imread(raw_patch_file)) 
                        CFA = raw_image.astype(np.float32)

                        # pip install colour-demosaicing
                        if 'redmi' in 'redmiraw_file':
                            bayyer_pattern = 'BGGR'
                        if 'oppo' in 'redmiraw_file':
                            bayyer_pattern = 'RGGB' 
                        raw_rgb = demosaicing_CFA_Bayer_bilinear(CFA, bayyer_pattern)

                        # Check patch: raw_rgb
                        #     Read huawei mate30pro rgb patch
                        rgb = np.asarray(cv2.imread(rgb_patch_file)) 

                        #print('\t@raw_rgb: ', np.shape(raw_rgb))
                        #print('\t@rgb: ', np.shape(rgb))

                        patch_compare = np.hstack((raw_rgb, rgb))
                        pp_file = os.path.join(pp_dir, '%s_%d_patch.jpg'%(img_name, idx))
                        cv2.imwrite(pp_file, patch_compare)
  
def match(redmi8_dir, redmi8_ext, huawei_dir, huawei_ext, tag): 
    # RAW files
    redmi=[]   
    for root, dirs, files in os.walk(redmi8_dir):
        print('loading: ' + root)
        for file in sorted(files):
            if os.path.splitext(file)[1].upper() == redmi8_ext.upper():
                redmi.append(file)
    print('redmi: {}'.format(redmi))

    # RGB files
    huawei=[]   
    for root, dirs, files in os.walk(huawei_dir):
        print('loading: ' + root)
        for file in sorted(files):
            if os.path.splitext(file)[1].upper() == huawei_ext.upper():
                huawei.append(file)
    print('huawei: {}'.format(huawei))

    # 
    cropped_dir = tag+"_cropped" 
    if not os.path.exists(cropped_dir):
        os.makedirs(cropped_dir)

    compare_dir = tag+"_matched"
    if not os.path.exists(compare_dir):
        os.makedirs(compare_dir)
     
    print('matching: ')
    for i in range(len(redmi)):
        huaweirgb_file = os.path.join(huawei_dir, huawei[i])
        redmiraw_file = os.path.join(redmi8_dir,  redmi[i])
        print('\t{} : {}'.format(redmiraw_file, huaweirgb_file))

        # dest_crop
        cropped_file = os.path.join(cropped_dir, huawei[i].replace('.dng','.jpg'))

        # compare_matched
        compare_file = os.path.join(compare_dir, redmi[i].replace('.dng','.jpg'))
     
        redmiraw_img,  huaweirgb_img, redmiraw_gray, huaweirgb_cropped = s.splitraw(
            redmiraw_file, huaweirgb_file, cropped_file, compare_file)
 
        # redmiraw: raw
        #    redmiraw_gray     : gray
        #    huaweirgb_cropped : matched rgb_gray
        output_root = './'
        crop(redmiraw_file, redmiraw_img, huaweirgb_img, 
            redmiraw_gray, huaweirgb_cropped, output_root, tag)
        print('\t\tdone.')

def main():
    huawei_dir = "./00_huaweiMate30pro_rgb"
    huawei_ext = '.jpg'

    # redmi8 RAW & huaweimate30pro RGB 
    redmi8_dir = './01_redmi8fc_dng' #fc: frontcamera
    redmi8_ext = '.dng' 
    tag = './processed/01_redmi8_huaweimate30pro'
    match(redmi8_dir, redmi8_ext, huawei_dir, huawei_ext, tag)
     
    if not os.path.exists('processed'):
        os.makedirs('processed')

    # oppoR17 RAW & huaweimate30pro RGB  
  #  oppor17_dir = './02_oppoR17_dng'
  #  oppor17_ext = '.dng'
  #  tag = './processed/02_oppor17_huaweimate30pro'
  #  match(oppor17_dir, oppor17_ext, huawei_dir, huawei_ext, tag)


if __name__ == '__main__':
    main()
 