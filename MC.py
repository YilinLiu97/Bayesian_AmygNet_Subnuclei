import sys
import csv
import numpy as np
import os
from os import listdir
from os import path
from os.path import isfile,join
import nibabel as nib
import tensorflow as tf


## ========  For Nifti files  ========= ##
def saveImageAsNifti(imageToSave,
                     imageName,
                     imageType,
                     image_proxy):

    printFileNames = False

    if printFileNames == True:
        print(" ... Saving image in {}".format(imageName))

    # Generate the nii file
    niiToSave = nib.Nifti1Image(imageToSave,image_proxy.affine)
    niiToSave.set_data_dtype(imageType)
    print('imageType: ', imageType)

    nib.save(niiToSave, imageName)

    print ("... Image succesfully saved in ", imageName)


#------- Recursively get prob file under each 'run_X' folder -----#
def get_prob_files(probs_path,suffix):
    prob_files = []
    prob_files = np.array([nib.load(join(root, f)).get_data() for root,sub,filenames in os.walk(probs_path)
                                                                for f in filenames if f.endswith(suffix)])
    return prob_files

# prob - prediction probability for each class(C). Shape: (N, C)
# returns - Shape: (N)
def predictive_entropy(prob):
        return -1 * np.sum(np.log(prob) * prob, axis=-1)

def do(argv):
    Affine_path = '../ISMRM_Dataset'
    Data_path = join(Affine_path,argv[0])
    outprob_homepath = 'models/dilated_deepmedic'
    foldername = argv[1]
    suffix = ".nii.gz"
    probs_path = join(outprob_homepath,foldername)

    prob_files = get_prob_files(probs_path,suffix)
    T = len(prob_files)
    print('Total number  of files: ',T)

    # Statistics
    probs = []
    for t in xrange(T):
        probs += [prob_files[t]]

    predictive_mean = np.mean(probs, axis=0) # the label map used as segmentation result

    predictive_variance = np.var(probs, axis=0)
#    predictive_variance = np.apply_along_axis(predictive_entropy, axis=-1, arr=predictive_mean)


    # Generate folders to store the the MC outputs
    BASE_DIR = os.getcwd()
    path_Temp = join(BASE_DIR, 'MC_outputs')

    segImageName = join(path_Temp,'MC_segResults')
    uncertaintyName = join(path_Temp,'Uncertainty')

    if not os.path.exists(segImageName):
       os.makedirs(segImageName)

    if not os.path.exists(uncertaintyName):
       os.makedirs(uncertaintyName)

    #Save the seg results
    image_proxy = nib.load(Data_path)
    nameToSave = segImageName
    imageTypeToSave_seg = np.dtype(np.int16)
    imageTypeToSave_uncer = np.dtype(np.float32)
    segmentationImage = np.argmax(predictive_mean, axis=-1).astype(np.int16) #get the segmentation output

    uncertaintyMap = np.mean(predictive_variance, axis=-1) #get the uncertainty estimation



    seg_output = saveImageAsNifti(segmentationImage,nameToSave,imageTypeToSave_seg,image_proxy)
    uncertainty_output = saveImageAsNifti(uncertaintyMap,uncertaintyName,imageTypeToSave_uncer,image_proxy)


#    print('mean(uncertainty) ', np.mean(int(0 if uncertainty_output is None else uncertainty_output)))
    # Save the uncertainty matrix
   # with open(join(uncertaintyName,'uncertainty.csv'),'wb') as f:
   #	  csv.writer(f, delimiter=' ').writerows(np.unique(predictive_variance,axis=-1))

    total_uncertainty = np.sqrt(np.sum(np.array(uncertaintyMap)))
    mean_uncertainty = np.mean(np.array(uncertaintyMap))
    max_uncertainty = np.max(np.array(uncertaintyMap))
    print('total_uncertainty: ', total_uncertainty)
    print('mean_uncertainty: ', mean_uncertainty)
    print('max_uncertainty: ', max_uncertainty)

    predictive_std = np.std(probs, axis=0)
    print(predictive_std.shape)
    # Uncertainty for each class
    for i in xrange(1,11):
        print('(Summed)uncertainty for each class_{}'.format(i), np.sqrt(np.sum(predictive_variance[:,:,:,:,i])))
#        print('uncertainty for each clas_{}'.format(i), np.divide(np.std(probs[:,:,:,:,i]),mean_uncertainty))
        print('(Mean)uncertainty for each class_{}'.format(i), np.mean(predictive_variance[:,:,:,:,i])==i)
        print('(Std/Mean)uncertainty for each class_{}'.format(i),np.std(predictive_variance[:,:,:,:,i]==i)/np.mean(predictive_variance[:,:,:,:,i]==i))
if __name__ == '__main__':
  do(sys.argv[1:])
