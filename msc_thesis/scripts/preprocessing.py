# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:42:54 2017

@author: kwant

http://forum.step.esa.int/t/snappy-np-help/5246/2
"""


# ~~~~~Import~~~~~
import os
import errno
import numpy as np
import scipy.constants as cons
import datetime
import scipy
import shutil

from snappy import ProductIO
from snappy import HashMap
from snappy import GPF

import dateutil.parser
import glob
from xml.etree import ElementTree
import matplotlib.pyplot as plt
from scipy import interpolate
from osgeo import gdal



# ~~~~~Functions~~~~~


class deramp():

    def __init__(self, loc, date, swath, path):
        self.path = path
        self.loc = loc
        self.swath = swath


def path_exists(loc, path, swath):

    date_stamp = os.path.split(loc)[1][:32]
    stamp = os.path.join(path, date_stamp, swath)

    try:
        os.makedirs(stamp)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return stamp


def load_data(stamp, burst, loc, meta, swath):
    '''
    This script calibrates and loads input SLC images.

    '''
    print('Reading...')
    time_1 = datetime.datetime.utcnow()
    
    if swath == 'IW1':
        str2 = '-004.xml'
        datastr = '-004.tiff'
    elif swath == 'IW2':
        str2 = '-005.xml'
        datastr = '-005.tiff'
    elif swath == 'IW3':
        str2 = '-006.xml'
        datastr = '-006.tiff'
    
    dataloc = glob.glob(loc + r'\measurement\*' + datastr)
    dataloc = dataloc[0]
    
    name = glob.glob(loc + r'\annotation\*' + str2)
    name = name[0]
    # Open element tree and retrieve parameters
    tree = ElementTree.parse(name)
    
    olp = tree.findall(".//imageAnnotation/imageInformation/numberOfSamples")
    mylist = [t.text for t in olp]
    no_pixels = int(mylist[0])
    olp = tree.findall(".//imageAnnotation/imageInformation/numberOfLines")
    mylist = [t.text for t in olp]
    no_lines = int(mylist[0])
    
    
    name = glob.glob(loc + r'\annotation\calibration\calibration*' + str2)
    name = name[0]
    # Open element tree and retrieve parameters
    tree = ElementTree.parse(name)
    # Load meta data in dictionary meta

    line = []
    olp = tree.findall(".//calibrationVectorList/calibrationVector")
    for ele in olp:
        i = ele.findall('line')
        linei = [t.text for t in i]
        line.append(int(linei[0]))
    y = np.asarray(line)
    
    olp = tree.findall(".//calibrationVectorList/calibrationVector/pixel")
    for t in olp:
        t3 = t.text
        t1 = t3.split()
        t2 = [float(t) for t in t1]
        x = np.asarray(t2)
    
    olp = tree.findall(".//calibrationVectorList/calibrationVector/sigmaNought")
    for i,t in enumerate(olp):
        t3 = t.text
        t1 = t3.split()
        t2 = [float(t) for t in t1]
        
        if i ==0:
            t_arr = np.asarray(t2)
        else:
            t_arr0 = t_arr
            t_arr1 = np.asarray(t2)
            t_arr = np.vstack([t_arr0,t_arr1])
    
    
    f = interpolate.interp2d(x, y, t_arr, kind='linear')
    xnew = np.arange(0, no_pixels, 1)
    ynew = np.arange(0, no_lines, 1)
    znew = f(xnew, ynew)
    
        
    ds = gdal.Open(dataloc)
    slc = np.array(ds.GetRasterBand(1).ReadAsArray())
      
    ints = np.abs(slc)
    phase = np.angle(slc)
    
    s0 = np.sqrt(ints**2 / (znew**2)) #https://sentinel.esa.int/web/sentinel/radiometric-calibration-of-level-1-products
    slc_new = s0 * np.exp(1j * phase)

    print('Saving.....')
        # ~~~~Save data~~~~~
    
    np.save(os.path.join(stamp, 'data_' + swath + '_slc.npy'), slc_new)
    time_2 = datetime.datetime.utcnow()
    diff_t = round((time_2 - time_1).total_seconds())
    print('Calibration is finished in    ' + str(diff_t) + '   seconds.')
    print('------------------------------------------------------------')
    

def getText(nodelist):
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)


def nearest(items, pivot):
    imin = min(items, key=lambda x: abs(x - pivot))
    return items.index(imin)


def read_XML(loc, swath):

    # This script reads the SLC image xml file, in order to do deramping
    # this script retrieves parameters needed for deramping from an XML file.
    # as explained in table 1 from the TOPS deramping function.

    if swath == 'IW1':
        str2 = '-004.xml'
    elif swath == 'IW2':
        str2 = '-005.xml'
    elif swath == 'IW3':
        str2 = '-006.xml'

    name = glob.glob(loc + r'\annotation\*' + str2)
    name = name[0]

    # Open element tree and retrieve parameters
    tree = ElementTree.parse(name)

    # Load meta data in dictionary meta
    meta = {}

    olp = tree.findall(
        ".//generalAnnotation/productInformation/azimuthSteeringRate")
    mylist = [t.text for t in olp]
    meta['azimuthSteeringRate'] = float(mylist[0])

    olp = tree.findall(
        ".//dopplerCentroid/dcEstimateList/dcEstimate/dataDcPolynomial")
    mylist = []
    for t in olp:
        t3 = t.text
        t1 = t3.split()
        t2 = [float(t) for t in t1]
        mylist.append(t2)
    meta['dataDcPolynomial'] = mylist

    olp = tree.findall(
        ".//dopplerCentroid/dcEstimateList/dcEstimate/azimuthTime")
    mylist = []
    for t in olp:
        t2 = t.text
        date = dateutil.parser.parse(t2)

        mylist.append(date)
    meta['dcAzimuthtime'] = mylist

    olp = tree.findall(
        ".//generalAnnotation/azimuthFmRateList/azimuthFmRate/t0")
    mylist = [float(t.text) for t in olp]
    meta['dcT0'] = mylist

    olp = tree.findall(
        ".//rangePixelSpacing")
    mylist = [float(t.text) for t in olp]
    meta['rangePixelSpacing'] = mylist[0]

    olp = tree.findall(
        ".//azimuthPixelSpacing")
    mylist = [float(t.text) for t in olp]
    meta['azimuthPixelSpacing'] = mylist[0]

    olp = tree.findall(
        ".//generalAnnotation/azimuthFmRateList/azimuthFmRate/azimuthFmRatePolynomial")
    mylist = []
    for t in olp:
        t3 = t.text
        t1 = t3.split()
        t2 = [float(t) for t in t1]
        mylist.append(t2)
    meta['azimuthFmRatePolynomial'] = mylist

    olp = tree.findall(
        ".//generalAnnotation/azimuthFmRateList/azimuthFmRate/azimuthTime")
    mylist = []
    for t in olp:
        t2 = t.text
        date = dateutil.parser.parse(t2)

        mylist.append(date)
    meta['azimuthFmRateTime'] = mylist

    olp = tree.findall(
        ".//generalAnnotation/azimuthFmRateList/azimuthFmRate/t0")
    mylist = [float(t.text) for t in olp]
    meta['azimuthFmRateT0'] = mylist

    olp = tree.findall(
        ".//generalAnnotation/productInformation/radarFrequency")
    mylist = [t.text for t in olp]
    meta['radarFrequency'] = float(mylist[0])

    olp = tree.findall(".//generalAnnotation/orbitList/orbit/velocity")
    mylist = []
    for ele in olp:
        x1 = ele.findall('x')
        x = [t.text for t in x1]
        y1 = ele.findall('y')
        y = [t.text for t in y1]
        z1 = ele.findall('z')
        z = [t.text for t in z1]

        vel = [float(x[0]), float(y[0]), float(z[0])]
        mylist.append(vel)
    meta['velocity'] = mylist

    olp = tree.findall(".//generalAnnotation/orbitList/orbit/time")
    mylist = [dateutil.parser.parse(t.text) for t in olp]
    meta['velocityTime'] = mylist

    olp = tree.findall(".//swathTiming/linesPerBurst")
    mylist = [t.text for t in olp]
    meta['linesPerBurst'] = int(mylist[0])

    olp = tree.findall(
        ".//imageAnnotation/imageInformation/azimuthTimeInterval")
    mylist = [t.text for t in olp]
    meta['azimuthTimeInterval'] = float(mylist[0])

    olp = tree.findall(
        ".//generalAnnotation/productInformation/rangeSamplingRate")
    mylist = [t.text for t in olp]
    # divide by 1, as explained in the manual --> to get delta t
    meta['rangeSamplingRate'] = 1 / float(mylist[0])

    olp = tree.findall(".//imageAnnotation/imageInformation/slantRangeTime")
    mylist = [t.text for t in olp]
    meta['slantRangeTime'] = float(mylist[0])

    olp = tree.findall(".//samplesPerBurst")
    mylist = [t.text for t in olp]
    meta['samplesPerBurst'] = int(mylist[0])


    # load burst information in dictionary burst
    burst = {}

    olp = tree.findall(".//swathTiming/burstList/burst/azimuthTime")
    mylist = []
    for t in olp:
        t2 = t.text
        date = dateutil.parser.parse(t2)

        mylist.append(date)
    burst['azimuthTime'] = mylist

    meta['no_burst'] = len(burst['azimuthTime'])
    return meta, burst


def read_pass(loc, swath):
    '''
    This script scans the XML annotation for pass information (ascending/descending) and georeference points.

    '''

    if swath == 'IW1':
        str2 = '-004.xml'
    elif swath == 'IW2':
        str2 = '-005.xml'
    elif swath == 'IW3':
        str2 = '-006.xml'

    name = glob.glob(loc + r'\annotation\*' + str2)
    name = name[0]

    # Open element tree and retrieve parameters
    tree = ElementTree.parse(name)

    # Load meta data in dictionary meta
    georef = {}

    olp = tree.findall(
        ".//generalAnnotation/productInformation/pass")
    mylist = [t.text for t in olp]
    georef['pass'] = str(mylist[0])

    olp = tree.findall(".//geolocationGrid/geolocationGridPointList/geolocationGridPoint")

    line = []
    pixel = []
    lat = []
    lon = []
    in_angle = []
    for ele in olp:
        i = ele.findall('line')
        linei = [t.text for t in i]
        line.append(int(linei[0]))
        i = ele.findall('pixel')
        pixeli = [t.text for t in i]
        pixel.append(int(pixeli[0]))
        i = ele.findall('latitude')
        lati = [t.text for t in i]
        lat.append(float(lati[0]))
        i = ele.findall('longitude')
        loni = [t.text for t in i]
        lon.append(float(loni[0]))
        i = ele.findall('incidenceAngle')
        in_anglei = [t.text for t in i]
        in_angle.append(float(in_anglei[0]))

    mylist = [line, pixel, lat, lon, in_angle]
    georef['geolocationGrid'] = mylist

    a = np.array(georef['geolocationGrid'][0])
    b = np.array(georef['geolocationGrid'][1])
    c = np.array(georef['geolocationGrid'][2])

    plt.scatter(b, a, c=c, cmap=plt.cm.RdYlGn, vmin=np.min(c), vmax=np.max(c))
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.show
    return georef


def image_statistics(slc_image):
    '''
    See S-1 algorithm Definition, page 21. following statistics are calculated:
    u = mean intensity
    sigma = Normalized variance
    Bs = Skewness
    Bk Curtosis
    '''
    ic = slc_image
    u = np.mean(abs(ic)**2)
    sigma = (1 / (u**2)) * np.mean((abs(ic)**2 - u)**2)

    Bs = np.mean((abs(ic)**2 - u)**3)**2 / np.mean((abs(ic)**2 - u)**2)**3

    Bk = np.mean((abs(ic)**2 - u)**4) / np.mean((abs(ic)**2 - u)**2)**2

    stat = {}
    stat['mean'] = u
    stat['variance'] = sigma
    stat['skewness'] = Bs
    stat['curtosis'] = Bk

    return stat





def deramp_algorithm(stamp, burst_i, meta, burst):
    '''
    deramping Algorithm as defined in the algorith description.

     step 1: calculate azimuth time at midburst

    '''
    
    az_start = burst['azimuthTime'][burst_i]
    diff_t = meta['azimuthTimeInterval'] * meta['linesPerBurst'] / 2
    # according to equ 9
    az_mid = az_start + datetime.timedelta(milliseconds=diff_t * 1000)

    # azimuth time function
    az_time = np.arange(-diff_t, diff_t, meta['azimuthTimeInterval'])

    if len(az_time) != meta['linesPerBurst']:
        print('error: time vector not valid!')

    # step 2: calculate velocity at midburst time
    vel = meta['velocity']
    vel_s = [np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
             for x in vel]    # calculate Vs according to equ.10

    vel_time = meta['velocityTime']

    # find value where az_mid velocity is larger than velocity time
    sel = nearest(vel_time, az_mid)

    # use value to select 3 before and 2 after this time.
    vel_times = vel_time[sel - 2:sel + 3]
    vel_1 = vel_s[sel - 2:sel + 3]

    arr1 = np.array(vel_times)
    arr2 = np.array(vel_1)

    def to_float(d, epoch=arr1[0]):
        return (d - epoch).total_seconds()

    # calculate Vs by interpolation around nearby points○
    Vs = np.interp(to_float(az_mid), np.array(list(map(to_float, arr1))), arr2)

    # Calculate doppler Centroid rate (ks) according to equ.4
    k_s = ((2 * Vs) / scipy.constants.c) * \
        meta['radarFrequency'] * meta['azimuthSteeringRate'] * \
        scipy.constants.pi / 180

    # step 3: Calculate Doppler FM Rate (ka)
    t_0 = meta['slantRangeTime']
    t_dt = meta['rangeSamplingRate']

    ind = nearest(meta['azimuthFmRateTime'], az_mid)
    dc_ind = nearest(meta['dcAzimuthtime'], az_mid)

    t0 = meta['azimuthFmRateT0'][ind]
    c0 = meta['azimuthFmRatePolynomial'][ind][0]
    c1 = meta['azimuthFmRatePolynomial'][ind][1]
    c2 = meta['azimuthFmRatePolynomial'][ind][2]

    dc_t0 = meta['dcT0'][dc_ind]
    dc_c0 = meta['dataDcPolynomial'][dc_ind][0]
    dc_c1 = meta['dataDcPolynomial'][dc_ind][1]
    dc_c2 = meta['dataDcPolynomial'][dc_ind][2]

    # step 4 and 5: calculate ka and kt per range pixel point

    i = np.array(range(meta['samplesPerBurst']))
    i = np.expand_dims(i, 1)
    tau = t_0 + (t_dt * i)
    ka = c0 + c1 * (tau - t0) + c2 * ((tau - t0)**2)  # equ.11
    kt = (ka * k_s) / (ka - k_s)  # equ. 2

    # step 5: doppler centroid frequency
    fnc = dc_c0 + dc_c1 * (tau - dc_t0) + (dc_c2 * ((tau - dc_t0)**2))

    # step 6: reference zero doppler time
    nc = -fnc / ka
    nref = nc - nc[0]

    az_time = np.expand_dims(az_time, 1)    # make az time nx1 vector
    # make variables same size as original image
    az_time = np.repeat(az_time, len(nref), axis=1)
    nref = np.transpose(np.repeat(nref, len(az_time), axis=1))
    kt = np.transpose(np.repeat(kt, len(az_time), axis=1))
    fnc = np.transpose(np.repeat(fnc, len(az_time), axis=1))

    # Final step: calculate deramp and demodulation phase
    # o_i = np.exp(- 1j * cons.pi * kt * ((az_time - nref)**2) - 1j * 2 * cons.pi * fnc * (az_time - nref))
    o_i = (- cons.pi * kt * ((az_time - nref)**2) - 2 * cons.pi * fnc * (az_time - nref))
    return o_i


def deramp_function(stamp, burst, meta, swath):
    time_1 = datetime.datetime.utcnow()
    for burst_i in range(len(burst['azimuthTime'])):
        
        if burst_i == 0:
            o_i = deramp_algorithm(stamp, burst_i, meta, burst)
            o = o_i
        else:
            o_i = deramp_algorithm(stamp, burst_i, meta, burst)
            o = np.vstack((o, o_i))

    np.save(os.path.join(stamp, 'data_' + swath + '_o.npy'), o)

    #deramped image is constructed with slc_dr = slc * np.exp(1j * o)

    time_2 = datetime.datetime.utcnow()
    diff_t = round((time_2 - time_1).total_seconds())
    print('Deramping is finished in    ' + str(diff_t) + '   seconds.')
    print('------------------------------------------------------------')


def copy_meta_data(loc, path, swath, meta, burst):
    '''
    This function copies the metadata from the original SAFE file to the processed folder.

    Dependency: path_exist

    '''
    
    list1 = glob.glob(loc + '\*')
    list1.sort()
    listas = list(list1[i] for i in [1, 4, 5])
    listas2 = list(list1[i] for i in [0, 2])

    stamp = path_exists(loc, path, swath)
    stri = os.path.split(loc)[-1]
    pathout = os.path.join(stamp, stri)

    for i in listas:
        itemi = os.path.split(i)[-1]
        savepath = os.path.join(pathout, itemi)
        if os.path.isdir(savepath) == False:
            shutil.copytree(i, savepath)

    for i in listas2:
        shutil.copy(i, pathout)
    print('Copying of Meta data succesful.')
    print('------------------------------------------------------------')
    return pathout

# ~~~~~Calculation~~~~~

if __name__ == "__main__":

    loc = r'd:\data\unzipped\portugal\S1A_IW_SLC__1SDV_20141125T183457_20141125T183524_003442_00405B_3F04.SAFE'
    path = 'd:\data\processed\python'
    swathl = ['IW2']
  