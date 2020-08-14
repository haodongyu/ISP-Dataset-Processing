# ISP-Dataset-Processing
Use SIFT and RANSC algorithm to process ISP-Dataset   
This is a project from my practical company. First we need to make many photos which include JPG format and RAW format with different Mobile Phone(such as Huawei P40 Pro and Redmi Note 8). Then we need to align these photos and crop them. Finally, we can get image pairs with the size of 448*448. These pairs can be used to train the deep-learning based model.  
## How to start  
### Rename  
First put all the `.py` files to the image folder. The image folder need to be saved as folling structure:  (for example)
```
|--ISP_Dataset  
  |--00_HuaweiP40Pro_rgb  
    |--20200617094620.jpg  
    ...  
    ...  
  |--01_RedmiNote8_raw  
    |--20200617094620.jpg  
    ...  
    ...  
01_rename.py  
02_matching.py  
dng_to_png.py  
splitraw.py  

```  
Then run the `01_rename.py` file to rename the image's name. All of the images will be sorted with time sequence.  
### Matching images  
Second, run `02_matching.py`. The processed patches will be saved to a new folder: `processed`.  
## Other details  
`dng_to_png.py` file can change the `dng` format to `png`. Remember define the Bayer array in it.  
`splitraw.py` can help us make the visualization of `.raw` files by matching them with `.dng` files.
