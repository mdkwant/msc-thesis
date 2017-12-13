# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 09:37:19 2017

@author: kwant
"""

import numpy as np
import glob
from xml.etree import ElementTree
import math
# ~~~~~~~~~~~~~~~~~Fucntions~~~~~~~~~~~~~~~~~~~

def read_pass(loc, swath='IW1'):
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

    mylist = [line,pixel,lat,lon,in_angle]
    georef['geolocationGrid'] = mylist
    
    return georef


def get_data(lat_input, lon_input, lat, lon, azi, ran):

    lat_index  = np.nanargmin((lat-lat_input)**2)
    lon_index = np.nanargmin((lon-lon_input)**2)
    
    a = azi.flatten()[lat_index]
    r = ran.flatten()[lon_index]    
    return a, r


def calc_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


def calc_degrees(georef):
    '''
    This script calculates the degrees away from north where CW is positive. 
    A positive degree means adding this number to direction results in the output with North arrow at zero degrees.
    
    '''
    yi = next(i for i, x in enumerate(georef['geolocationGrid'][0]) if x >= 1)
    xi = int(len(georef['geolocationGrid'][0]) / yi)

    lat = np.array(georef['geolocationGrid'][2]).reshape(xi, yi)
    lon = np.array(georef['geolocationGrid'][3]).reshape(xi, yi)
    
    satpass = georef['pass']
    
    if satpass=='Descending':
        a1 = np.fliplr(lat).ravel()
        b1 = np.fliplr(lon).ravel()
    
    elif satpass=='Ascending':   
        a1 = np.flipud(lat).ravel()
        b1 = np.flipud(lon).ravel()
    
    glist = []
    for i, item in enumerate(a1):
        if i < 21:
            #do nothing
            i = i
        else:
            pointA = (a1[i], b1[i])
            pointB = (a1[i-21], b1[i-21]) 
            g = calc_bearing(pointA, pointB)
            glist.append(g)
    
    degree = np.mean(glist)
    return degree

def azra_locator(loc, swath, buoy_loc):
    '''
    locates the AZimuth and RAnge pixels which correspont to the location of a certain buoy, using  the geolocation grid from the satellite meta data.
    
    Possible buoy locations: 
        
    k13 : K13 Alpha platform
    muns: IJmuiden Munitie stort
    d15: D151 platform
    portugal: Monican1 buoy
    portugal2: Monican2 buoy
    '''
    
    georef = read_pass(loc, swath)
    # reshape    
    yi = next(i for i, x in enumerate(georef['geolocationGrid'][0]) if x >= 1)
    xi = int(len(georef['geolocationGrid'][0]) / yi)
    
    azi = np.array(georef['geolocationGrid'][0]).reshape(xi, yi)
    ran = np.array(georef['geolocationGrid'][1]).reshape(xi, yi)
    lat = np.array(georef['geolocationGrid'][2]).reshape(xi, yi)
    lon = np.array(georef['geolocationGrid'][3]).reshape(xi, yi)

    if buoy_loc =='k13':
        lat_input = 53.218117
        lon_input = 3.219036   
    elif buoy_loc =='muns':
        lat_input = 52.550000
        lon_input = 4.058333   
    elif buoy_loc =='d15':
        lat_input = 54.324861
        lon_input = 2.934321           
    elif buoy_loc =='portugal':
        lat_input = 39.522
        lon_input = -9.648
    elif buoy_loc =='portugal2':
        lat_input = 39.569
        lon_input = -9.208   
        
    else: 
        print('ERROR: Location ' + buoy_loc + ' is not in the list of available buoys.')
        print('------------------------------------------------------------')
        return
    
    if np.min(lat) <= lat_input <= np.max(lat):
        if np.min(lon) <= lon_input <= np.max(lon):
            a, r = get_data(lat_input, lon_input, lat, lon, azi, ran)
            r_size = ran[1,1]
            a_size = azi [1,1]
            georef['lowrange'] = r
            georef['lowazimuth'] = a
            georef['sizerange'] = r_size
            georef['sizeazimuth'] = a_size
            georef['az_loc'] = int(a / a_size)
            georef['ra_loc'] = int((r + r_size / 2) / 1024)
            print(buoy_loc +' buoy located in swath ' + swath + ' at location (a,r)' + str(georef['az_loc']) +  str(georef['ra_loc']))
        else:
            print('ERROR: buoy is not located within this area.')
            print('------------------------------------------------------------')
            errorloc = 1
            return errorloc
    
    return georef
# ~~~~~~~~~~~~~~~~~~~~Execute~~~~~~~~~~~~~~~~~~~

#%%
if __name__ == '__main__':

    loc = r'd:\data\unzipped\k13\S1A_IW_SLC__1SDV_20150426T173313_20150426T173340_005658_00741A_8482.SAFE'
    swath = ['IW2']
    
    for swath in swath:
        georef = azra_locator(loc, swath, 'k13')

    #import latlon
    #georef, ar = latlon.indices(loc, swath, 'k13')
    
    yi = next(i for i, x in enumerate(georef['geolocationGrid'][0]) if x >= 1)
    xi = int(len(georef['geolocationGrid'][0]) / yi)

    lat_input = 53.218117
    lon_input = 3.219036  
    azi = np.array(georef['geolocationGrid'][0]).reshape(xi, yi)
    ran = np.array(georef['geolocationGrid'][1]).reshape(xi, yi)
    lat = np.array(georef['geolocationGrid'][2]).reshape(xi, yi)
    lon = np.array(georef['geolocationGrid'][3]).reshape(xi, yi)
    
    a, r = get_data(lat_input, lon_input, lat, lon, azi, ran)
    
#%%
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    map = Basemap(projection='merc', lat_0 = 57, lon_0 = -135,
    resolution = 'h', area_thresh = 0.1,
    llcrnrlon=np.min(lon), llcrnrlat=np.min(lat),
    urcrnrlon=np.max(lon), urcrnrlat=np.max(lat))
     
    map.drawcoastlines()
    map.drawcountries()
    map.fillcontinents(color = 'coral')
    map.drawmapboundary()
     
    x,y = map(lon, lat)
    map.plot(x, y, 'bo', markersize=5)
    
    x2,y2 = map(lon_input, lat_input)
    map.plot(x2, y2, 'ro', markersize=6)
    
    plt.show()
    