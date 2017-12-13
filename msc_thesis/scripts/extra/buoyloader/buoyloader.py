# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 09:23:56 2017

@author: kwant

This script reads CSV buoy data and creates a plot.
"""

import datetime
import numpy as np
import os
import pandas as pd


# ~~~~~~~~~~~ Functions ~~~~~~~~~~~~~~~~~

def roundTime(dt=None, dateDelta=datetime.timedelta(minutes = 60)):
    """Round a datetime object to a multiple of a timedelta
    dt : datetime.datetime object, default now.
    dateDelta : timedelta object, we round to a multiple of this, default 1 minute.
    Author: Thierry Husson 2012 - Use it as you want but don't blame me.
            Stijn Nevens 2014 - Changed to use only datetime objects as variables
    """
    roundTo = dateDelta.total_seconds()

    if dt == None : dt = datetime.datetime.now()
    seconds = (dt - dt.min).seconds
    # // is a floor division, not a comment on following line:
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)



def convert_buoydata(filepath, spec):
        
    '''
    
    This script reads raw buoy data from RWS in the North Sea, it is used to open and convert data to a more convenient CSV format.
    It opens data with pandas. Next some opearations remove uneccesary information and reshape the data. 
    The result is a pandas df, with datetime as index and frequency bands as columns. 
    The df is saved as a new CSV file, so this script only has to be used once.
    
    spec = dirspec of espec
    '''
    
    if spec == 'espec':
        df = pd.read_csv(filepath ,sep = ';' , header = 796)
        
        
        #removing and reshaping the data
        del df['bpgcod'] # remove empty column
        del df['kwlcod'] # remove empty column
        
        dfi = pd.to_datetime(df['datum'] +' '+ df['tijd'], format='%d %m %Y %H:%M')
        df['datetime'] = dfi
        df['waarde'] = df['waarde'] / 100**2
        
        del df['datum']
        del df['tijd']
        
        columns = np.linspace(0.025,0.515, 50)
        dfc = pd.DataFrame(columns, columns=['col'])
        
        no = len(dfi)/50
        dfc = pd.concat([dfc]*no, ignore_index=True)
        
        df['columns'] = dfc['col']
        
        
        dfn = df.pivot(index = 'datetime',columns = 'columns', values = 'waarde')
        
        del dfn[0.515] # delete last two rows (invalid values)
        del dfn[0.505] # delete last two rows (invalid values)
        
        # save dataframe to csv on original folderV
        head, tail =  os.path.split(filepath)
        tailn = tail[:-4] + '_new.csv'
        locn = os.path.join(head,tailn)
        dfn.to_csv(locn)
    
    if spec == 'dirspec':
            
        df = pd.read_csv(filepath,sep = ';' , header = 1485)
        
        names = ['datum', 'tijd','bpgcod','waarde','kwlcod']
        
        df.columns = names
        
        #removing and reshaping the data
        del df['bpgcod'] # remove empty column
        del df['kwlcod'] # remove empty column
        
        dfi = pd.to_datetime(df['datum'] +' '+ df['tijd'], format='%d %m %Y %H:%M')
        df['datetime'] = dfi
        
        del df['datum']
        del df['tijd']
        
        
        columns = range(96)
        dfc = pd.DataFrame(columns, columns=['col'])
        
        no = len(dfi)/96
        dfc = pd.concat([dfc]*no, ignore_index=True)
        
        df['columns'] = dfc['col']
        
        
        dfn = df.pivot(index = 'datetime',columns = 'columns', values = 'waarde')
        
        # save dataframe to csv on original folderV
        head, tail =  os.path.split(filepath)
        tailn = tail[:-4] + '_new.csv'
        locn = os.path.join(head,tailn)
        dfn.to_csv(locn)
        
        
        print(locn + ' Done')
    
    return locn


def portugal(datetime_object, buoy = 1, path = r'd:\data\buoy\PORTUGAL\MONICAN'):
    '''
    This script selects and loads buoy data from portugal at the time of a datetime object.
    For monican 1 and 2, buoynum. Monican 1 is standard and most offshore
    '''
      
    
    dtime = roundTime(datetime_object)
    
    if buoy == 1:
        if dtime > datetime.datetime(2014,10,1,0,0) and dtime < datetime.datetime(2014,10,6,0,0):
            filen = os.path.join(path,'Monican01_Spec_2014Oct01_2014Oct05.txt')

        if dtime > datetime.datetime(2014,10,29,0,0) and dtime < datetime.datetime(2015,2,4,0,0):
            filen = os.path.join(path,'Monican01_Spec_2014Oct29_2015Feb03.txt')

        if dtime > datetime.datetime(2015,6,30,0,0) and dtime < datetime.datetime(2015,10,4,0,0):
            filen = os.path.join(path,'Monican01_Spec_2015Jun30_2015Oct03.txt')
            
    if buoy == 2:
        if dtime > datetime.datetime(2014,10,29,0,0) and dtime < datetime.datetime(2015,3,29,0,0):
            filen = os.path.join(path,'Monican02_Spec_2014Oct29_2015Mar28.txt')

        if dtime > datetime.datetime(2015,10,1,0,0) and dtime < datetime.datetime(2016,3,2,0,0):
            filen = os.path.join(path,'Monican02_Spec_2015Oct01_2016Mar02.txt')
    
    
    df = pd.read_csv(filen, header=1)
    
    df['Time'] = pd.to_datetime(df['Time'])  
    df = df.set_index(['Time'])
    
    buoydata = df.loc[dtime].values
    data=buoydata.reshape(3,50)
    freqs = np.linspace(0.01,0.5,50)
    data = np.vstack([freqs,data])
    
    return data


def northsea(datetime_object, buoy = 'k13', path = r'd:\data\buoy\NORTHSEA\M170513184\in'):
    
    '''
     buoy = k13, ijm, d15
     
     This script loads buoy data from the North sea from one of the three wave 
     buoys above. Note, directional information is missing for d15 and might not work properly. 
     
    '''  
    
    
    dtime = roundTime(datetime_object, datetime.timedelta(minutes = 10))
    dtime = dtime.replace(second=0)
    
    if buoy == 'k13':
        file_e = os.path.join(path, 'k13apfm_espec_new.csv')
        file_dir = os.path.join(path, 'k13apfm_dirspec_new.csv')
        
    if buoy == 'ijm':
        file_e = os.path.join(path, 'ijmuide_espec_new.csv')
        file_dir = os.path.join(path, 'ijmuide_dirspec_new.csv')
        
    if buoy == 'd15': 
        file_e = os.path.join(path, 'D15_espec_new.csv')
    
    
    dfe = pd.read_csv(file_e, header=0)
    dfd = pd.read_csv(file_dir, header=0)
    
    
    dfe['datetime'] = pd.to_datetime(dfe['datetime'])  
    dfe = dfe.set_index(['datetime'])
    
    dfd['datetime'] = pd.to_datetime(dfd['datetime'])  
    dfd = dfd.set_index(['datetime'])
    
    espec = dfe.loc[dtime].values
    dirspec = dfd.loc[dtime].values
    
    buoydata = np.hstack([espec, dirspec])
    
    data=buoydata.reshape(3,48)
    freqs = np.linspace(0.025,0.495, 48)
    freqs[0] = 0.03
    data = np.vstack([freqs,data])
    
    return data


# ~~~~~~~~~~~~~ Exectute ~~~~~~~~~~~~~~~

if __name__ == '__main__':
    dateob = datetime.datetime(2014, 11, 25, 19, 25, 10)

    '''
    S1A_IW_SLC__1SDV_20141125T183457
    
    The following locations are possible:
    ['k13', 'D15', 'ijm']

    load datetime list of mathing files  from

    d:\data\raw_sar\k13\sat_sel_datetimes.pkl
    '''
    data_po = portugal(dateob, buoy = 1)
    data_ns = northsea(dateob, buoy = 'k13')

    
    

