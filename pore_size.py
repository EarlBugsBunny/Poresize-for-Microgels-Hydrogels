# What to do:

    # Obtaining the porosity profile along each principle axis (https://porespy.org/examples/metrics/tutorials/porosity_profiles.html)
    # Using regionprops_3d to analyze properties of each pore (https://porespy.org/examples/metrics/tutorials/regionprops_3d.html)
    # SNOW network extraction https://porespy.org/examples/networks/tutorials/snow_basic.html)
    
#%% 
#For one stack 

import os
import imageio
import numpy as np
from skimage.filters import rank
from skimage import morphology
from skimage import filters
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import disk
import porespy as ps
import pandas as pd

ps.visualization.set_mpl_style()
np.random.seed(1)

#%%
def read_file(file_name):
    folder_name = os.path.splitext(file_name)[0]
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully")
    else:
        print(f"Folder '{folder_name}' already exists")
        
    stack = imageio.volread(file_name)
    return stack, folder_name
    

def clahe_preprocess(stack, file_name, folder_name):
    clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
    clahe_stack = np.zeros_like(stack)
    for i in range(stack.shape[0]):
        clahe_stack[i] = clahe.apply(stack[i])
    output_path = folder_name+ "/" + file_name + "_clahe.tif"
    imageio.volsave(output_path, clahe_stack)
    return clahe_stack

def gaussian_preprocess(stack, file_name, folder_name):
    print("Gaussian")
    kernel_size = (3, 3)  # Adjust the kernel size based on your needs
    gaussian_stack = np.zeros_like(stack)
    for i in range(stack.shape[0]):
        gaussian_stack[i] = cv2.GaussianBlur(stack[i], kernel_size, 0)
        
    output_path = folder_name+ "/" + file_name + "_gaussian.tif"
    imageio.volsave(output_path, gaussian_stack)
    return gaussian_stack
    
def thresholding(stack, file_name, folder_name):
    print("Thresholding")
    radius = 100
    selem = disk(radius)
    local_otsu_stack = np.zeros_like(stack)
    thresholded_stack = np.zeros_like(stack)
    for i in range(stack.shape[0]):
        print(i)
        local_otsu_stack[i] = rank.otsu(stack[i], selem)
        thresholded_stack[i] = np.where(stack[i] >= local_otsu_stack[i], 0, 255)
    
    output_path = folder_name+ "/" + file_name + "_thresholded.tif"
    imageio.volsave(output_path, thresholded_stack)
    output_path = folder_name+ "/" + file_name + "_local_otsu.tif"
    imageio.volsave(output_path, local_otsu_stack)
    
    return thresholded_stack

def postprocess(stack, file_name, folder_name):
    print("Postprocess")
    post_process_stack = np.zeros_like(stack)
    for i in range(stack.shape[0]):
        for n in range(3):
            post_process_stack[i] = filters.median(stack[i])
        post_process_stack[i] = morphology.remove_small_objects(post_process_stack[i] , min_size=20)
        # Apply morphological opening to remove small white speckles
        kernel_opening = np.ones((3, 3), np.uint8)
        post_process_stack[i] = cv2.morphologyEx(post_process_stack[i], cv2.MORPH_OPEN, kernel_opening)
        # Apply morphological closing to fill in gaps in the objects
        kernel_closing = np.ones((8, 8), np.uint8)
        post_process_stack[i] = cv2.morphologyEx(post_process_stack[i], cv2.MORPH_CLOSE, kernel_closing)
        
    output_path = folder_name+ "/" + file_name + "_post_processed.tif"
    imageio.volsave(output_path, post_process_stack)
    return post_process_stack
    
    

file_name = "240212 void space confocal2.lif - 5 percent_A1 Region1 Merged.tif"
print(file_name)
stack, folder_name = read_file(file_name)
stack = clahe_preprocess(stack, file_name, folder_name)
stack = gaussian_preprocess(stack, file_name, folder_name)
stack = thresholding(stack, file_name, folder_name)
stack = postprocess(stack, file_name, folder_name)

#%%
print("Max Ball Size")
df_list = []
for i in range(stack.shape[0]):
    print(i)
    
    folder_regions = folder_name + "/" + "regions" 
    
    if not os.path.exists(folder_regions):
        os.makedirs(folder_regions)
        
        
    im = cv2.bitwise_not(stack[i])
    im = im > 0
    snow = ps.filters.snow_partitioning(im=im, r_max=5, sigma=3)
    
    fig, ax = plt.subplots()
    ax.imshow(ps.tools.randomize_colors(snow.regions))

    # Hide axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Hide the title
    ax.set_title('')
    fig.savefig(folder_regions+"/region_slide_"+str(i)+".png",bbox_inches='tight', pad_inches=0, transparent=True, dpi=600)
    
    regions = snow.regions*snow.im
    props = ps.metrics.regionprops_3D(regions)
    df = ps.metrics.props_to_DataFrame(props)
    df_list.append(df)
    
    folder_spheres = folder_name + "/" + "spheres" 
    
    if not os.path.exists(folder_spheres):
        os.makedirs(folder_spheres)

    sph = ps.metrics.prop_to_image(regionprops=props, shape=im.shape, prop='inscribed_sphere')
    fig, ax = plt.subplots()
    ax.imshow(sph + 0.5*(~im) , cmap=plt.cm.inferno)
    
    # Hide axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Hide the title
    ax.set_title('')
    fig.savefig(folder_spheres+"/spheres_slide_"+str(i)+".png",bbox_inches='tight', pad_inches=0, transparent=True, dpi=600)
    

sheets = {f'slide_{i + 1}': df for i, df in enumerate(df_list)}

with pd.ExcelWriter(folder_name+'/output.xlsx') as writer:  
    for i, df in enumerate(df_list):
        df.to_excel(writer, sheet_name="slide_"+str(i))


#%%
"for all images"



#%%
import os
import imageio
import numpy as np
from skimage.filters import rank
from skimage import morphology
from skimage import filters
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import disk
import porespy as ps
import pandas as pd

ps.visualization.set_mpl_style()
np.random.seed(1)

#%%
def read_file(file_name):
    folder_name = os.path.splitext(file_name)[0]
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully")
    else:
        print(f"Folder '{folder_name}' already exists")
        
    stack = imageio.volread(file_name)
    return stack, folder_name
    

def clahe_preprocess(stack, file_name, folder_name):
    clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
    clahe_stack = np.zeros_like(stack)
    for i in range(stack.shape[0]):
        clahe_stack[i] = clahe.apply(stack[i])
    output_path = folder_name+ "/" + file_name + "_clahe.tif"
    imageio.volsave(output_path, clahe_stack)
    return clahe_stack

def gaussian_preprocess(stack, file_name, folder_name):
    print("Gaussian")
    kernel_size = (3, 3)  # Adjust the kernel size based on your needs
    gaussian_stack = np.zeros_like(stack)
    for i in range(stack.shape[0]):
        gaussian_stack[i] = cv2.GaussianBlur(stack[i], kernel_size, 0)
        
    output_path = folder_name+ "/" + file_name + "_gaussian.tif"
    imageio.volsave(output_path, gaussian_stack)
    return gaussian_stack
    
def thresholding(stack, file_name, folder_name):
    print("Thresholding")
    radius = 100
    selem = disk(radius)
    local_otsu_stack = np.zeros_like(stack)
    thresholded_stack = np.zeros_like(stack)
    for i in range(stack.shape[0]):
        print(i)
        local_otsu_stack[i] = rank.otsu(stack[i], selem)
        thresholded_stack[i] = np.where(stack[i] >= local_otsu_stack[i], 0, 255)
    
    output_path = folder_name+ "/" + file_name + "_thresholded.tif"
    imageio.volsave(output_path, thresholded_stack)
    output_path = folder_name+ "/" + file_name + "_local_otsu.tif"
    imageio.volsave(output_path, local_otsu_stack)
    
    return thresholded_stack

def postprocess(stack, file_name, folder_name):
    print("Postprocess")
    post_process_stack = np.zeros_like(stack)
    for i in range(stack.shape[0]):
        for n in range(3):
            post_process_stack[i] = filters.median(stack[i])
        post_process_stack[i] = morphology.remove_small_objects(post_process_stack[i] , min_size=20)
        # Apply morphological opening to remove small white speckles
        kernel_opening = np.ones((3, 3), np.uint8)
        post_process_stack[i] = cv2.morphologyEx(post_process_stack[i], cv2.MORPH_OPEN, kernel_opening)
        # Apply morphological closing to fill in gaps in the objects
        kernel_closing = np.ones((8, 8), np.uint8)
        post_process_stack[i] = cv2.morphologyEx(post_process_stack[i], cv2.MORPH_CLOSE, kernel_closing)
        
    output_path = folder_name+ "/" + file_name + "_post_processed.tif"
    imageio.volsave(output_path, post_process_stack)
    return post_process_stack
    
current_directory = os.getcwd()
for file_name in os.listdir(current_directory): 
    #file_name = "240212 void space confocal2.lif - 5 percent_A1 Region1 Merged.tif"
    print(file_name)
    stack, folder_name = read_file(file_name)
    stack = clahe_preprocess(stack, file_name, folder_name)
    stack = gaussian_preprocess(stack, file_name, folder_name)
    stack = thresholding(stack, file_name, folder_name)
    stack = postprocess(stack, file_name, folder_name)
    
    print("Max Ball Size")
    df_list = []
    for i in range(stack.shape[0]):
        print(i)
        
        folder_regions = folder_name + "/" + "regions" 
        if not os.path.exists(folder_regions):
            os.makedirs(folder_regions)
        im = cv2.bitwise_not(stack[i])
        im = im > 0
        snow = ps.filters.snow_partitioning(im=im, r_max=5, sigma=3)
        fig, ax = plt.subplots()
        ax.imshow(ps.tools.randomize_colors(snow.regions))
        # Hide axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # Hide the title
        ax.set_title('')
        fig.savefig(folder_regions+"/region_slide_"+str(i)+".png",bbox_inches='tight', pad_inches=0, transparent=True, dpi=600)
        plt.close()
        regions = snow.regions*snow.im
        props = ps.metrics.regionprops_3D(regions)
        df = ps.metrics.props_to_DataFrame(props)
        df_list.append(df)
        
        folder_spheres = folder_name + "/" + "spheres" 
        if not os.path.exists(folder_spheres):
            os.makedirs(folder_spheres)
        sph = ps.metrics.prop_to_image(regionprops=props, shape=im.shape, prop='inscribed_sphere')
        fig, ax = plt.subplots()
        ax.imshow(sph + 0.5*(~im) , cmap=plt.cm.inferno)
        # Hide axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Hide the title
        ax.set_title('')
        fig.savefig(folder_spheres+"/spheres_slide_"+str(i)+".png",bbox_inches='tight', pad_inches=0, transparent=True, dpi=600)
        plt.close()
    sheets = {f'slide_{i + 1}': df for i, df in enumerate(df_list)}
    with pd.ExcelWriter(folder_name+'/output.xlsx') as writer:  
        for i, df in enumerate(df_list):
            df.to_excel(writer, sheet_name="slide_"+str(i))


