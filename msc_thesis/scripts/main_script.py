# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 11:04:10 2017

@author: kwant
"""
import preprocessing as prep
import process as proc
import buoy
import latlon

import datetime
import os
import numpy as np
import shutil
import glob
import gc


if __name__ ==  '__main__':

    #    Insert sat data

    imdir = r'd:\data\images\plot'                  # location were the plotted images are saves
    locpath = r'd:\data\unzipped\k13'               # location of UNZIPPED .SAFE folders containing satellite products
    locs = glob.glob(os.path.join(locpath, '*'))    # creating a list of .SAFE folders
    path = 'd:\data\processed\python'               # location were processed data (npy array) is saved
    location = 'k13'                                # location which is processed

    print(locs)

    for loc in locs:
        print('------------------Location------------------')
        print(loc)
        swathlist = ['IW1', 'IW2', 'IW3']
        for swath in swathlist:
             # input paths
            print('------------------Swath------------------')
            print('--------------------'+swath+'--------------------')
            dfile = os.path.split(loc)[1][:32]
            date = dfile[17:]
            saveloc = os.path.join(path, dfile, swath)

            # -------------------Preparation -------------------------------------
            
            meta, burst = prep.read_XML(loc, swath)              # Retrieve meta data from satellite product
            georef = latlon.azra_locator(loc, swath, location)   # determine if the wave buoy is located within a sub-swath and store imagette position of the wave buoy

            if georef == 1:
                print('swath ' + swath + ' does not contain buoy data and is not processed.')
                continue
            
            heading = latlon.calc_degrees(georef)                # calculate the heading of the satellite
            print(heading)
            print(georef['pass'])
            if georef['pass']=='Ascending':                     # additionally you can define the heading manually in this part
                meta['heading'] = 11.2
            else:
                meta['heading'] = -11.2 + 180

            print(meta['heading'])
            meta['specsize'] = 300                              # size (in pixels) of the cross spectrum
            meta['imsize'] = 1024                               # imagette size
            meta['smoothing'] = 11                              # smoothing window size in pixels (between 3 and 11)

            # ------------------- buoy data -----------------------------------------------------
            dateobs = datetime.datetime.strptime(date, "%Y%m%dT%H%M%S")     
            buoy_obj = buoy.loading(dateobs)                                #create buoy object from time stamp
            
            if location =='k13':                                            # load buoy data at location: North Sea
                meta['radius'] = 8                                          # radius for filtering out low frequency noise
                depth = 26.7                                                # waterdepth of wave buoy
                meta['depth'] = depth
                if dateobs > datetime.datetime(2016,12,1,0,0):
                    if dateobs > datetime.datetime(2016,12,31,0,0):
                        data = buoy_obj.buoy_northsea('A12')                # loading of buoy data from A12, IJM or K13
                    else:
                        data = buoy_obj.buoy_northsea('ijm')
                    
                    
                else:
                    data = buoy_obj.buoy_northsea('k13')
                    print(dateobs)

                if dateobs.date()==datetime.datetime(2016,10,1).date():     # load buoy data for specific dates (if buoy data was missing)
                    dateobs = datetime.datetime(2016,10,1,7,10)
                    buoy_obj = buoy.loading(dateobs)
                    data = buoy_obj.buoy_northsea('k13')
                elif dateobs.date()==datetime.datetime(2016,12,10).date():
                    dateobs = datetime.datetime(2016,12,10,19,10)
                    buoy_obj = buoy.loading(dateobs)
                    data = buoy_obj.buoy_northsea('k13')
                    

                if type(data) == int:
                    print('ERROR: no buoy data available for this date.')
                    continue
            elif location =='portugal':                                         # load buoy data for portugal
                depth = 2000
                meta['depth'] = depth
                meta['radius'] = 4
                data = buoy_obj.buoy_portugal()
                if type(data) == int:
                    data = buoy_obj.buoy_portugal(2)
                    print('Data not available: loading data from Monican 2 wave buoy')
                    if type(data) == int:
                        print('ERROR: no buoy data available for this date.')
                        continue

            thetasar = data[2, np.argmax(data[1])] - meta['heading']
               
            print('thetasar = '+ str(thetasar))                
            dates = np.array([dateobs])                        # Save buoy data for OCEANSAR processing
            np.savez(os.path.join(r'd:\data\buoy\OCEANSAR', 'spectra_'+ date[:8]+'.npz'), arr0 = data, dates = dates)
            
            impath = prep.path_exists(loc, os.path.join(imdir,location), swath)  # create file path, where buoy data is stored
            pathout = prep.copy_meta_data(loc, path, swath, meta, burst)         # copy meta-data to file path
            #print(date)
            buoy.kx_plot(data, depth, impath, date, location, ascending=True, heading=meta['heading']) # create 2D plot of wave buoy data
            buoy.plot_1D(data, location, impath, date)                                                 # create 1D plot of wave buoy data


            # --------------preprocessing S1 data -----------------------------------------------
            
            stamp = prep.path_exists(loc, path, swath)              # create file path, where processed satellite data is stored
            prep.load_data(stamp, burst, loc, meta, swath)          # load satellite image calibrate data
            prep.deramp_function(stamp, burst, meta, swath)         # apply deramping and demodulation on image, save as npy array

            # ---------------processing S1 data-------------------------------------------------
            rl_n = proc.multi_process(saveloc, meta, swath)         # start processing entrire sub-swath, calculate cross-spectra for every imagette
            proc.multi_plot(rl_n, impath, meta, dfile, swath)       # plot cross-spectra on sub-swath image
            proc.plot_variance(meta, impath, saveloc, swath)        # plot intensity image and calculate variance per imagette

            # Determine imagette location to be further processed. Imagette location defined in azimuth and range direction.
            if date=='20141125T183457':
                meta['az_loc'] = 8
                meta['ra_loc'] = 10
            elif date=='20150124T183456':
                meta['az_loc'] = 8
                meta['ra_loc'] = 10
            elif date=='20151027T183506':
                meta['az_loc'] = 8
                meta['ra_loc'] = 10
            elif date=='20150426T173313':
                meta['az_loc'] = 7
                meta['ra_loc'] = 18
            elif date=='20160519T174119':
                meta['az_loc'] = 8
                meta['ra_loc'] = 18
            elif date=='20161031T060557':
                meta['az_loc'] = 2
                meta['ra_loc'] = 8
            elif date=='20161129T172515':
                meta['az_loc'] = 6
                meta['ra_loc'] = 15
            elif date=='20170109T173321':
                meta['az_loc'] = 7
                meta['ra_loc'] = 9                
            elif date=='20170116T172512':
                meta['az_loc'] = 6
                meta['ra_loc'] = 13
            elif date=='20170121T173321':
                meta['az_loc'] = 7
                meta['ra_loc'] = 5
            elif date=='20161001T060515':
                meta['az_loc'] = 1
                meta['ra_loc'] = 16
            elif date=='20161030T172429': 
                meta['az_loc'] = 7
                meta['ra_loc'] = 9
            elif date=='20161210T173232':
                meta['az_loc'] = 7
                meta['ra_loc'] = 15
            elif date=='20170108T174048':
                meta['az_loc'] = 7
                meta['ra_loc'] = 16
            else:
                var, azi, rai = proc.calc_stats(saveloc, meta, swath) # if not in the list, take imagette with lowest variance.
                


            slc_buoy, az, ra = proc.single_image(saveloc, meta, georef, impath, swath, az=meta['az_loc'], ra=meta['ra_loc']) # process single SAR imagette
            proc.cross_spec_3_looks(slc_buoy, georef, meta, impath, az, ra)                         # plot SAR imagette cross-spectrum
            data = proc.SAR_1D(slc_buoy, data, meta, location, impath, date, az, ra, thetasar)    # plot 1D cross-spectra from SAR imagette
            #shutil.rmtree(os.path.join(path, dfile))                                              # delete path with processed satellite data. (for bath processing of images)
            print(georef['pass'])
            print(np.mean(georef['geolocationGrid'][2]))
            print(np.mean(georef['geolocationGrid'][3]))
            gc.collect()