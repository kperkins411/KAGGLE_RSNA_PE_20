# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current sessionb
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torchvision
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import cv2
# import albumentations as albu
import functools
import glob
# from albumentations.pytorch.transforms import ToTensorV2

from tqdm.auto import tqdm
import gc
from PIL import Image
import pydicom
class config:
    '''
    configuration information
    '''
    model_name="resnet18"
    batch_size = 256
    WORKERS = 4
    numb_classes =14
    resume = False
    epochs = 10
    MODEL_PATH ='log/cpt'
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    MODEL_PARAMS_LOC = os.path.join(MODEL_PATH,'resnet18_best.pth')
    MODEL_PARAMS_NEW_LOC = os.path.join(MODEL_PATH,'resnet18_best_new.pth')
    BAIL_AFTER_THIS_MANY_VALIDATION_INCREASES = 10

    df_cols={
    'StudyInstanceUID':     0,    # - unique ID for each study(exam) in the data.
    'SeriesInstanceUID':    1,   # - unique ID for each series within the study.
    'SOPInstanceUID':       2,    # - unique ID for each image within the study ( and data).
    'pe_present_on_image':  3, # - image-level, notes whether any form of PE is present on the image.
    'negative_exam_for_pe': 4, # - exam-level, whether there are any images in the study that have PE present.
    'qa_motion':            5,     # - informational, indicates whether radiologists noted an issue with motion in the study.
    'qa_contrast':          6,   # - informational, indicates whether radiologists noted an issue with contrast in the study.
    'flow_artifact':        7, # - informational
    'rv_lv_ratio_gte_1':    8, # - exam-level, indicates whether the RV / LV ratio present in the study is >= 1
    'rv_lv_ratio_lt_1':     9,  # - exam-level, indicates whether the RV / LV ratio present in the study is < 1
    'leftsided_pe':         10, #- exam-level, indicates that there is PE present on the left side of the images in the study
    'chronic_pe':           11, #- exam-level, indicates that the PE in the study is chronic
    'true_filling_defect_not_pe': 12, # - informational, indicates a defect that is NOT PE
    'rightsided_pe':        13, # - exam-level, indicates that there is PE present on the right side of the images in the study
    'acute_and_chronic_pe': 14, # - exam-level, indicates that the PE present in the study is both acute AND chronic
    'central_pe':           15, # - exam-level, indicates that there is PE present in the center of the images in the study
    'indeterminate':        16  # -exam-level, indicates that while the study is not negative for PE, an ultimate set of exam-level labels could not be created, due to QA issues
    }

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.  
        ex. train_df = reduce_mem_usage(train_df)
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

import pandas as pd
class SInstUID_tracker():
    exam_level_cols = list(config.df_cols.keys())
    index_col=exam_level_cols[0]

    exam_level_col_numbers = [ config.df_cols['pe_present_on_image'],
                            config.df_cols['negative_exam_for_pe'],
                            config.df_cols['rv_lv_ratio_gte_1'],
                            config.df_cols['rv_lv_ratio_lt_1'],
                            config.df_cols['leftsided_pe'],
                            config.df_cols['chronic_pe'],
                            config.df_cols['rightsided_pe'],
                            config.df_cols['acute_and_chronic_pe'],
                            config.df_cols['central_pe'],
                            config.df_cols['indeterminate']]
    #
    # exam_level_col_names=['pe_present_on_image','negative_exam_for_pe','rv_lv_ratio_gte_1','rv_lv_ratio_lt_1' ]
    # # now subtract 3 because we are not getting study, series and SOP columns in pred
    # exam_level_col_numbers = [x-3 for x in exam_level_col_numbers]

    #
    # exam_level_col_names = {v: k for k, v in config.df_cols.items()}
    # exam_level_col_names = {v: k for k, v in config.df_cols.items()}

    def __init__(self,test_df, index_col=index_col, col_numbers=exam_level_col_numbers, default_val=0.0):
        '''
        index: list, maincolumn to search for
        listOfStudyID: bunch of studies to put in index
        cols: other columns
        default_val: what cols are initialized to

        '''
        self.index_col = index_col
        self.col_numbers = col_numbers
        self.default_val = default_val
        # self.all = self.index_col
        # self.all.extend(self.cols)

        self.exam_level_col_names = list(config.df_cols.keys())
        # self.exam_level_col_names = [exam_level_cols[i] for i in SInstUID_tracker.exam_level_col_numbers]

        #get unique index_cols
        self.listOfStudyID = test_df[self.index_col].unique()
        self.df = pd.DataFrame({self.index_col: self.listOfStudyID}, columns=self.exam_level_col_names).fillna(self.default_val)

    def record(self, studyid, pred):
        '''
        compares the values in pred against the values in cols and keeps the largest
        studyid: the current study we are working on
        pred: output of model

        '''
        # for index in self.df.index:
        #     if self.df.loc[index,self.index_col] == studyid:
        #         break
        index = np.flatnonzero([self.df[self.index_col] == studyid])
        if(len(index)>1):
            strindex = ' '.join([str(elem) for elem in index])
            print("OH NO! "+ studyid + " found at indexs " + strindex)

        index =index[0]
        for col in range(len(pred)):
            self.df.iloc[index, col+3 ] = np.maximum(self.df.iloc[0, col+3 ], pred[col])

    def getrow(self, studyid):
        index = np.flatnonzero([self.df[self.index_col] == studyid])
        if (len(index) > 1):
            strindex = ' '.join([str(elem) for elem in index])
            print("OH NO! " + studyid + " found at indexs " + strindex)
        return self.df.loc[index[0]]

class ValMonitor():
    '''
    stops validation when loss has increased max_consecutive_increases
    also indicates if its time to save the model
    '''
    def __init__(self, max_consecutive_increases):
        '''
        :param max_consecutive_increases: how many val increases before we should stop
        '''
        self.max_consecutive_increases=max_consecutive_increases
        self.stop_counter=0
        self.old_val = None
        self.should_stop_training=False

    def time_to_save(self, new_val):
        '''
        determines if its time to save model
        :param new_val:
        :return:
        '''
        should_save_model = False
        if( self.old_val is None):
            self.old_val = new_val+1

        if(new_val<self.old_val):
            self.old_val=new_val
            if(self.stop_counter>0):
                self.stop_counter-=1
            should_save_model = True    #save if validation is better than last
        else:
            self.stop_counter+=1

        #stop if validtion loss is rising
        self.should_stop_training= (self.stop_counter>=self.max_consecutive_increases)
        return should_save_model

    def time_to_stop(self):
        '''
        determines if its time to stop training
        :param new_val:
        :return:
        '''
        return self.should_stop_training

    
import time
class timer_kp:
    def __init__(self):
        self._start_time=time.perf_counter()
    def __call__(self):      
        print(f"Elapsed time: {(time.perf_counter() - self._start_time):0.4f} seconds")
        
transforms_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
])
transforms_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
])

class CTDatasetJPEG(Dataset):
    def __init__(self,df,path,transforms=None,preprocessing=None,size=254,mode='val'):
        
        #get a numpy representation of the pandas dataframe
        self.df_main = df.values
        self.path = path
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.size=size
    
        #either use all the validation data as given
        #or generate a balanced set
        if mode=='val':
            self.df = self.df_main
        else:
            self.generate_balanced_set()

    def __getitem__(self, idx):
        '''
        returns the image and a label
        '''
        row = self.df[idx]   #row is a numpy.ndarray

        #assummes there may be more than 1 but still just uses the first one (???)
        img = cv2.imread(glob.glob(f'{self.path}/{row[0]}/{row[1]}/*{row[2]}.jpg')[0])
        
        #lets flip from BGR to RGB
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      
        #get the last 14 values starting with pe_present_on_image, even the purely informational ones
        label = row[3:].astype(int) 
        
        #TODO ?
        label[2:] = label[2:] if label[0] == 1 else 0
         
        if self.transforms:
            img = self.transforms(img)
        if self.preprocessing:
            img = self.preprocessing(img)         
        return  row[0],row[2],img,torch.from_numpy(label)

    def __len__(self):
        return len(self.df)
    
    #this function gets a balanced set, 1/2 have pe_present_on_image=1, 1/2 have pe_present_on_image=0
    #note that we discard a bunch of images that have no pe present
    def generate_balanced_set(self):
        df0 = self.df_main[self.df_main[:,config.df_cols['pe_present_on_image']]==0]
        df1 = self.df_main[self.df_main[:,config.df_cols['pe_present_on_image']]==1]
        np.random.shuffle(df0)
        self.df = np.concatenate([df0[:len(df1)],df1],axis=0)

IGNORE_THIS_VALUE=-1
class CTDatasetDicom(Dataset):
    def __init__(self,df,path,transforms=None,preprocess_to_train_format=None,size=254,mode='val'):
        
        #get a numpy representation of the pandas dataframe
        self.df = df.values   
        self.path = path
        self.transforms = transforms
#         self.preprocessing = preprocessing
        self.preprocess_to_train_format = preprocess_to_train_format
        self.size=size


    #window just like we did on training set
    def _window(self,img, WL=50, WW=350):
        upper, lower = WL+WW//2, WL-WW//2
        X = np.clip(img.copy(), lower, upper)
        X = X - np.min(X)
        X = X / np.max(X)
        X = (X*255.0).astype('uint8')
        return X

    def __getitem__(self, idx):
        row = self.df[idx] 
#         print(row)
          
        #get the dicom data, preprocess it to the same format that we used in training
        dcm_data = pydicom.dcmread(f"{self.path}/{row[0]}/{row[1]}/{row[2]}.dcm")
        image = dcm_data.pixel_array * int(dcm_data.RescaleSlope) + int(dcm_data.RescaleIntercept)
        image = np.stack([self._window(image, WL=-600, WW=1500),
                          self._window(image, WL=40, WW=400),
                          self._window(image, WL=100, WW=700)], 2)
        
#         if self.preprocessing:
#             image = self.preprocessing(image)
 
        if self.transforms:
            image = self.transforms(image)
            
        #return study id and imageid
        return row[0],row[2],image,IGNORE_THIS_VALUE

    def __len__(self):
        return len(self.df)
    
    #this function gets a balanced set, 1/2 have pe_present_on_image=1, 1/2 have pe_present_on_image=0
    #note that we discard a bunch of images that have no pe present
#     def generate_balanced_set(self):
#         df0 = self.df_main[self.df_main[:,3]==0]
#         df1 = self.df_main[self.df_main[:,3]==1]
#         np.random.shuffle(df0)
#         self.df = np.concatenate([df0[:len(df1)],df1],axis=0)
        

# def norm(img):
#     img-=img.min()
#     return img/img.max()
    