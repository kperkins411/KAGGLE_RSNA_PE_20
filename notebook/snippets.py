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
    MODEL_PATH = 'log/cpt/resnet18_best.pth'

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

class SInstUID_tracker():
    index_col = ['StudyInstanceUID']
    exam_level_features = ['negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',
                           'leftsided_pe', 'chronic_pe', 'rightsided_pe',
                           'acute_and_chronic_pe', 'central_pe', 'indeterminate']
    def __init__(self,test_df, index_col=index_col, cols=exam_level_features, default_val=0.0):
        '''
        index: list, maincolumn to search for
        listOfStudyID: bunch of studies to put in index
        cols: other columns
        default_val: what cols are initialized to

        '''
        self.index_col = index_col
        self.cols = cols
        self.default_val = default_val
        self.all = self.index_col
        self.all.extend(self.cols)

        #get unique index_cols
        self.listOfStudyID = test_df[self.index_col[0]].unique()
        self.df = pd.Dataframe({self.index_col[0]: self.listOfStudyID}, columns=self.all)

    def record(self, studyid, pred):
        '''
        compares the values in pred against the values in cols and keeps the largest
        studyid: the current study we are working on
        pred: output of model

        '''
        for i, col in enumerate(self.cols):
            self.df.loc[studyid, col] = np.maximun(df.loc[studyid, col], pred[i])

    def getrow(self, studyid):
        return self.df.loc(studyid)

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
