# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 09:24:12 2017

@author: kwant
name: sentinel load

This script batch downloads Sentinel-1 data from ESA scihub using the sentinelsat package. 
Information on the use installation and dependencies is found on: https://github.com/ibamacsr/sentinelsat
and http://krstn.eu/download-sentinel-images-with-python/

create geojson file at http://geojson.io/#map=2/20.1/0.0 (click new, draw square, click save)

"""
# ~~~~~Import~~~~~
from sentinelsat.sentinel import SentinelAPI, get_coordinates
import datetime
import glob
import zipfile
import os
import dateutil.parser
import pandas
import pprint
import pickle

# ~~~~~Input~~~~~

startdate = '20141001'

enddate = '20170320'

datadir = r'D:\data\raw_sar'
zipdir = r'D:\data\unzipped'

loc = ['d15', 'k13', 'muns', 'map']  # geojson objects stored in d:\scripts\geojson\
loc = loc[1☺]
# ~~~~~Functions~~~~~


def logger():
    # create a log file
    import logging

    logger = logging.getLogger('sentinelsat')
    logger.setLevel('INFO')

    h = logging.StreamHandler()
    h.setLevel('INFO')
    fmt = logging.Formatter('%(message)s')
    h.setFormatter(fmt)
    logger.addHandler(h)


class Sentinelscihub(object):

    def __init__(self, startdate, enddate, datadir, loc):

        self.startdate = startdate
        self.enddate = enddate
        self.datadir = datadir
        self.zipdir = zipdir
        self.loc = loc

    def datestamp(self):
        # load sentinel data

        user = r'mdkwant'
        password = r'c0p3rn1cus1'
        geojson = os.path.join(r'd:\scripts\geojson', loc + '.geojson')
        api = SentinelAPI(user, password, 'https://scihub.copernicus.eu/dhus')

        # search by polygon, time, and SciHub query keywords
        # https://scihub.copernicus.eu/userguide/3FullTextSearch for key words

        products = api.query(get_coordinates(geojson),
                             startdate, enddate,
                             platformname='Sentinel-1',
                             producttype='SLC',
                             sensoroperationalmode='IW')

        # collect date stamps from product information
        prod_dates = []
        for i, item in enumerate(products):
            f = str(products[i]['date'][1]['content'])
            date = dateutil.parser.parse(f, ignoretz=True)
            prod_dates.append(date)

        #metadata = api.to_geojson(products) # GeoJSON FeatureCollection
        # containing footprints and metadata of the scenes

        self.prod_dates = prod_dates

        return prod_dates

    def compare_dates(self, prod_dates):

        # load buoy dataframe
        if loc == 'k13':
            buoy_df = pandas.read_pickle('d:\\data\\buoy\\data_selection_130_loc_platform%20k13a.pkl')
        elif loc == 'd15':
            buoy_df = pandas.read_pickle(
                r'd:\\data\\buoy\\data_selection_130_loc_d151.pkl')
        b_df = buoy_df

        b_df['index_buoy'] = b_df.index

        # change datetime to date object and set dater as index
        prod_dates2 = [item.date() for item in prod_dates]

        b_df['time2'] = [item.date() for item in b_df['time']]
        b_df = b_df.set_index('time2')

        # create df object of sat data
        ident = range(len(prod_dates))
        s1_df = pandas.DataFrame({'identifier': ident, 'time': prod_dates2})
        s1_df = s1_df.sort('time')
        s1_df = s1_df.set_index('time')

        # remove duplicate dates, select first value
        df1 = s1_df
        df1 = df1[~df1.index.duplicated(keep='first')]
        df2 = b_df
        df2 = df2[~df2.index.duplicated(keep='first')]

        # compare both dataframes
        df = pandas.concat([df1, df2], axis=1)
        df3 = df.dropna(axis=0, how='any')

        # select data, create list with indexes
        sat_select = df3['identifier'].values.astype(int).tolist()
        buoy_select = df3['index_buoy'].values.astype(int).tolist()
        buoy_data = buoy_df.ix[buoy_select]

        self.sat_select = sat_select
        buoy_data.to_pickle(os.path.join(self.datadir, loc, "buoy_data.pkl"))

        return buoy_data, sat_select

    def dataload(self, sat_select):
        '''
        
        '''
        # load sentinel data
        self.sat_select = sat_select
        user = r'mdkwant'
        password = r'c0p3rn1cus1'
        geojson = os.path.join(r'd:\scripts\geojson', loc + '.geojson')
        api = SentinelAPI(user, password, 'https://scihub.copernicus.eu/dhus')
        # search by polygon, time, and SciHub query keywords
        # https://scihub.copernicus.eu/userguide/3FullTextSearch for key words
        products = api.query(get_coordinates(geojson),
                             startdate, enddate,
                             platformname='Sentinel-1',
                             producttype='SLC',
                             sensoroperationalmode='IW')

        prod = list(products[i] for i in sat_select)
        
        save_obj(prod,os.path.join(datadir, loc, "sat_metadata_list") )    
        
        
        # download all results from the search
        #api.download_all(prod, directory_path=os.path.join(datadir, loc))
         # GeoJSON FeatureCollection

        # containing footprints and metadata of the scenes
        
        return prod

    def extract(self):

        files = sorted(glob.glob(os.path.join(datadir,'*.zip')))
        for item in files:
            zip = zipfile.ZipFile(item)
            zip.extractall(zipdir)
            zip.close()
            # os.remove(item)
            print(item)
            print(datetime.datetime.utcnow())


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_datetime(prod):
    '''
    this script reads a list of sentinel scihub products and retrieves the file names and the datetime objects.
    and saves the objects in a list. 
    '''
    
    file_names = []     # This is to retrieve the original scihub filename
    
    for item in prod:    
        fname = item['title']
        file_names.append(fname)
      
    prod_dates_sel = [] # this is to retrieve a list of datetime objects corresponding to satellite images
    
    for i, item in enumerate(prod):
        f = str(prod[i]['date'][1]['content'])
        date = dateutil.parser.parse(f, ignoretz=True)
        prod_dates_sel.append(date)
        
    save_obj(prod_dates_sel,os.path.join(datadir, loc, "sat_sel_datetimes"))
    
    print(file_names)  
    return file_names, prod_dates_sel
     
# ~~~~~Execute~~~~~

if __name__ == "__main__":
    s1_data = Sentinelscihub(startdate, enddate, datadir, loc)
    prod_dates = s1_data.datestamp()
    buoy_data, sat_select = s1_data.compare_dates(prod_dates)
    prod = s1_data.dataload(sat_select)
#    s1_data.extract()
    
    fnames,prod_dates_sel2 = get_datetime(prod)
    
    
    pp = pprint.PrettyPrinter()
    pp.pprint(fnames)
