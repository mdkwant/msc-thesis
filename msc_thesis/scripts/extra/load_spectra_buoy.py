# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 09:23:56 2017

@author: kwant

This script reads CSV buoy data and creates a plot.
"""

import csv
import datetime
import numpy as np
import os
import pickle
import pandas as pd
import glob

# ~~~~~~~~~~~~~Input~~~~~~~~~~~~~~

'''
The following locations are possible:
['k13', 'D15', 'ijm']

load datetime list of mathing files  from

d:\data\raw_sar\k13\sat_sel_datetimes.pkl
'''
    
# ~~~~~~~~~~~ Functions ~~~~~~~~~~~~~~~~~


# open and load csv file
def read_CSV(files):
    files = files[0]
    file_data = []
    with open(files, 'rb') as csvfile:
        filereader = csv.reader(csvfile, delimiter=',', quotechar='.')
        for row in filereader:
            file_data.append(row)

    return file_data


def select_data(file_data, spec, datestr, timestr):
    if spec == 'espec':
        ind = 50
        step = ind * 6 * 24
    elif spec == 'dirspec':
        ind = 96
        step = ind * 6 * 24

    # select data at date and time str
    date_index = next((int(i)
                       for i, flist in enumerate(file_data) if datestr in flist))
    date_data = file_data[date_index:date_index + step]

    time_index = next((int(i)
                       for i, flist in enumerate(date_data) if timestr in flist))
    time_data = date_data[time_index:time_index + ind]

    return time_data


def check_nan(vals):
    vals = np.asarray(vals)
    vals[vals == 999999999.000] = np.nan
    return vals


def nans(shape, dtype=float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a


def save_object(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)



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
    
    This script reads raw buoy data fron NORTHSEA and opens it with pandas. Next some opearations remove uneccesary information and reshape the data. 
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
        
        no = int(len(dfi)/50)
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
        
        no = int(len(dfi)/96)
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

def load_A12date():
    path = r'd:\data\buoy\NORTHSEA\M170513184\in'
    datetime_object=datetime.datetime(2017,1,16,17,44,21)
    
    dtime = roundTime(datetime_object, datetime.timedelta(minutes = 10))
    dtime = dtime.replace(second=0)
    buoy = 'A12'
    
    file_e = os.path.join(path, 'A12_espec.csv')
    file_dir = os.path.join(path, 'A12_dirspec.csv')
    
    dfe = pd.read_csv(file_e, header=None, sep='; ;', engine='python')
    dfd = pd.read_csv(file_dir, header=None, sep='; ;', engine='python')
    
    dfe = dfe.sort_index()
    dflist = dfe[0].tolist()
    
    dflist2 = [date[:-1] for date in dflist if date[-1]=='0' or date[-1]=='9']
    dflist3 = [date[:-1] for date in dflist2 if date[-1]=='9']
    dflist2 = [date for date in dflist2 if date[-1]==';']
    dfli = dflist2+dflist3
    
    dflstr = [datetime.datetime.strptime(date, ';%d-%m-%y %H:%M;') for date in dfli]
    
    dfsort = sorted(dflstr)
    
    del dfe[0]
    del dfe[49] # delete last two rows (invalid values)
    del dfe[50] # delete last two rows (invalid values)
    dfe2 = dfe.iloc[:,0:48] / 100**2
    
    dfe1 = pd.DataFrame(dfsort, columns={'datetime'})
    dfn = pd.concat([dfe1, dfe2], axis=1)
    
    head, tail =  os.path.split(file_e)
    tailn = tail[:-4] + '_new.csv'
    locn = os.path.join(head,tailn)
    dfn.to_csv(locn, index=False)
    
    #prodess directional data
    
    dfd = dfd.sort_index()
    dflist = dfd[0].tolist()
    
    dflist2 = [date[:-1] for date in dflist if date[-1]=='0' or date[-1]=='9']
    dflist3 = [date[:-1] for date in dflist2 if date[-1]=='9']
    dflist2 = [date for date in dflist2 if date[-1]==';']
    dfli = dflist2+dflist3
    
    dflstr = [datetime.datetime.strptime(date, ';%d-%m-%y %H:%M;') for date in dfli]
    
    dfsort = sorted(dflstr)
    dfd[0] = dfsort
    dfd =dfd.rename(columns={0:'datetime'})
    
    
    dfval = dfd[96].tolist()
    dfval2 = [int(val[:-2]) for val in dfval]
    dfd[96] = dfval2
    dfn = dfd
    
    head, tail =  os.path.split(file_dir)
    tailn = tail[:-4] + '_new.csv'
    locn = os.path.join(head,tailn)
    dfn.to_csv(locn, index=False)
    return dfn
    

def buoy_portugal_load(datetime_object, buoy = 1):
    '''
    This script selects and loads buoy data from portugal at the time of a datetime object.
    For monican 1 and 2, buoynum. Monican 1 is standard and most offshore
    '''
      
    path = r'd:\data\buoy\PORTUGAL\MONICAN'
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


def buoy_northsea_load(datetime_object, buoy = 'k13'):
    
    '''
     buoy = k13, ijm, d15
     
     This script loads buoy data from the North sea from one of the three wave 
     buoys above. Note, directional information is missing for d15 and might not work properly. 
     
    '''
    
    path = r'd:\data\buoy\NORTHSEA\M170513184\in'
    
    dtime = roundTime(datetime_object, datetime.timedelta(minutes = 10))
    dtime = dtime.replace(second=0)
    buoy = 'k13'
    
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
    
    #data_po = buoy_portugal_load(dateob, buoy = 1)
    #data_ns = buoy_northsea_load(dateob, buoy = 'k13')
    
    filepath = r'd:\data\buoy\NORTHSEA\M170513184\in\old\ijmuide_espec.csv'
    spec = 'espec'
    
    #convert_buoydata(filepath, spec)
    
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
        
        no = int(len(dfi)/50)
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
    

