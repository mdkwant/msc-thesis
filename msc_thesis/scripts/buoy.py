import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy import linspace
from scipy import pi, sqrt, exp
from scipy.special import erf
import pickle
import os
from pandas import DataFrame
import bisect
import datetime
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import linspace
from scipy import pi, sqrt, exp
from scipy.special import erf, gamma
from scipy import interpolate as interp
import pickle
import datetime
import os
import numpy.ma as ma
import bisect
from pandas import DataFrame
import colorcet as cc

"""

Created on Fri Jun 16 13:23:27 2017

@author: kwant

This script plots directional spectrum from wave buoy data

Additionally it contains scripts to convert spectra from Ef to E kx ky.

Update: 
    
original file: process_spectra21

- deleted function: polar_to_cart
- created "loading" class with load functions.
- merged with Paco's file

This file contains code to read buoy wave spectra (previously pickled) and transform it into a a directional
wave-spectrum. 

@author:Martijn Kwant and Paco López Dekker
"""
# ~~~~~ Functions ~~~~~~

class loading(object):
        
    def __init__(self, datetime_object):

        self.datetime_object = datetime_object

        
    def roundTime(self, dateDelta=datetime.timedelta(minutes = 60)):
        """Round a datetime object to a multiple of a timedelta
        dt : datetime.datetime object, default now.
        dateDelta : timedelta object, we round to a multiple of this, default 1 minute.
        Author: Thierry Husson 2012 - Use it as you want but don't blame me.
                Stijn Nevens 2014 - Changed to use only datetime objects as variables
        """
        dt = self.datetime_object
        
        roundTo = dateDelta.total_seconds()
    
        if dt == None : dt = datetime.datetime.now()
        seconds = (dt - dt.min).seconds
        # // is a floor division, not a comment on following line:
        rounding = (seconds+roundTo/2) // roundTo * roundTo
        return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)
    
    
    def buoy_portugal(self, buoy = 1):
        '''
        This script selects and loads buoy data from portugal at the time of a datetime object.
        For monican 1 and 2, buoynum. Monican 1 is standard and most offshore
        '''
          
        path = r'd:\data\buoy\PORTUGAL\MONICAN'
        dtime = self.roundTime()
        
        if buoy == 1:
            if dtime > datetime.datetime(2014,10,1,0,0) and dtime < datetime.datetime(2014,10,6,0,0):
                filen = os.path.join(path,'Monican01_Spec_2014Oct01_2014Oct05.txt')
    
            elif dtime > datetime.datetime(2014,10,29,0,0) and dtime < datetime.datetime(2015,2,4,0,0):
                filen = os.path.join(path,'Monican01_Spec_2014Oct29_2015Feb03.txt')
    
            elif dtime > datetime.datetime(2015,6,30,0,0) and dtime < datetime.datetime(2015,10,4,0,0):
                filen = os.path.join(path,'Monican01_Spec_2015Jun30_2015Oct03.txt')
            else:
                data = 1
                return data
                
        if buoy == 2:
            if dtime > datetime.datetime(2014,10,29,0,0) and dtime < datetime.datetime(2015,3,29,0,0):
                filen = os.path.join(path,'Monican02_Spec_2014Oct29_2015Mar28.txt')
    
            elif dtime > datetime.datetime(2015,10,1,0,0) and dtime < datetime.datetime(2016,3,2,0,0):
                filen = os.path.join(path,'Monican02_Spec_2015Oct01_2016Mar02.txt')
            else:
                data = 1
                return data   
        
        df = pd.read_csv(filen, header=1)
        
        df['Time'] = pd.to_datetime(df['Time'])  
        df = df.set_index(['Time'])
        
        buoydata = df.loc[dtime].values
        data=buoydata.reshape(3,50)
        data[0,:], data[1,:] = data[1,:], data[0,:].copy()
        
        freqs = np.linspace(0.01,0.5,50)
        data = np.vstack([freqs,data])
        #data[1,:][data[1,:] > 4] = 0
        return data
    
    
    def buoy_northsea(self, buoy):
        
        '''
         buoy = k13, ijm, d15
         
         This script loads buoy data from the North sea from one of the three wave 
         buoys above. Note, directional information is missing for d15 and might not work properly. 
         
        '''
        
        path = r'd:\data\buoy\NORTHSEA\M170513184\in'
        
        dtime = self.roundTime(datetime.timedelta(minutes = 10))
        dtime = dtime.replace(second=0)
        
        
        if buoy == 'k13':
            file_e = os.path.join(path, 'k13apfm_espec_new.csv')
            file_dir = os.path.join(path, 'k13apfm_dirspec_new.csv')
        elif buoy == 'ijm':
            file_e = os.path.join(path, 'ijmuide_espec_new.csv')
            file_dir = os.path.join(path, 'ijmuide_dirspec_new.csv')
        elif buoy == 'd15': 
            file_e = os.path.join(path, 'D15_espec_new.csv')
        elif buoy == 'A12':
            file_e = os.path.join(path, 'A12_espec_new.csv')
            file_dir = os.path.join(path, 'A12_dirspec_new.csv')
        
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



def rotate(self, degrees, transform): 
    '''
    
    rotates and transforms spectrum according to satellite flight direction.
    ammount of degrees vary per subswath. SAR images are flipped, hence the -1

    IW2:
    10.1420290713
    10.1549238803

    IW3:
    9.12905294013
    9.15353541312
    
    IW1:
    '''
    asc_cor = 11.1939368663
    des_cor = 11.2057052335
    mode = self.mode

    if mode == 'ascending':
        if transform == 'sar-buoy':
            deg = (degrees * -1) + (180 + asc_cor) * 2 * np.pi / 360
            
        if transform == 'buoy-sar':
            deg = (degrees+ (180 + asc_cor ) * 2 * np.pi / 360) * -1 
            
    elif mode == 'descending':
        if transform == 'sar-buoy':
            deg = (degrees * -1) - des_cor * 2 * np.pi / 360
            
        if transform == 'buoy-sar':
            deg = (degrees - des_cor * 2 * np.pi / 360) * -1
            
    return deg
        
        




def alpha(omega, depth):
    """pp 124 Eq 5.4.21"""
    g = 9.81
    a = (omega ** 2 * depth / g)
    return a


def beta(a):
    b = a * (np.tanh(a)) ** (-1 / 2)
    return b


def n(k_d):
    """calculate n"""
    n_i = (1 + ((2 * k_d) / np.sinh(2 * k_d))) / 2
    return n_i


def kd(omega, depth):
    """pp 124 Eq 5.4.22"""
    a = alpha(omega, depth)
    b = beta(a)

    for x in range(1):
        k_d = (a + b ** 2 * (np.cosh(b)) ** -2) / \
              (np.tanh(b) + b * (np.cosh(b)) ** -2)
        a = k_d
        b = beta(a)
    return k_d


def c(omega, k_d):
    """pp. 125 Eq 5.4.23"""
    g = 9.81
    c_i = (g / omega) * np.tanh(k_d)
    return c_i


def cg(w1, w2, k1, k2):
    """pp. 127, Eq 5.4.30"""
    c_g = (np.abs(w1 - w2) / 2) / (np.abs(k1 - k2) / 2)
    return c_g


class BuoySpectra():
    def __init__(self, bdata, ascending, depth=26.7, heading=-11.2):
        """
        
        :param bdata: 
        :param depth: 
        :param heading: heading of SAR, East of North (in degree)
        """
        self.bdata = bdata
        self.heading = heading
        self.depth = depth
        self.ascending = ascending
        # Just to know the are there
        self.Sk = lambda k: 0
        self.kmin = 0.01
        self.kmax = 0.1
        self.k = np.array([0.01, 0.1])
        self.init_buoy2Sk(self.depth)
        # We rotate the direction so that it goes in the direction of the wave propagation
        self.dir = interp.interp1d(self.k, np.angle(np.exp(-1j * (np.radians(bdata[2]) + 3 * np.pi / 2))),
                                   bounds_error=False, fill_value=0,
                                   kind='nearest')
        self.spread = interp.interp1d(self.k, np.radians(bdata[3]), bounds_error=False, fill_value=1)

    def init_buoy2Sk(self, depth=26.7):
        bdata = self.bdata
        freq = bdata[0]
        fstart = freq - 0.005
        fstart[0] = freq[0]
        fend = freq + 0.005
        fend[-1] = freq[-1]
        fbin = fend - fstart
        f_vec = np.vstack((fstart, fend, fbin))
        Ef_1d = bdata[1]
        Eftot = np.sum(Ef_1d * f_vec[2])
        #print('Total Ef spec = ' + str(Eftot) + ' [m2]')

        # calculate rad f spectrum
        Ew_spec = Ef_1d / (2 * np.pi)
        w_vec = f_vec * 2 * np.pi
        Ew_tot = np.sum(Ew_spec * w_vec[2])
        #print('Total Ew spec = ' + str(Ew_tot) + ' [m2]')
        k_1 = kd(w_vec[0], depth) / depth
        k_2 = kd(w_vec[1], depth) / depth
        k_vec = np.vstack((k_1, k_2, k_2 - k_1))
        dwdk = w_vec[2] / k_vec[2]
        Ek_spec = Ew_spec * dwdk
        
        self.k = (k_1 + k_2) / 2
        self.Sk = interp.interp1d(self.k, Ek_spec, bounds_error=False, fill_value=0)
        self.kmin = self.k.min()
        self.kmax = self.k.max()

    def dirspread(self, k, theta):

        theta = 1 * (theta + np.radians(self.heading)) #minus to change direction where waves are going to.


        wtheta = np.angle(np.exp(1j * (theta - self.dir(k))))
        s = 2 / self.spread(k)**2 - 1
        # (2 / spr_i) - 1
        D = 2 ** (2 * s - 1) / np.pi * gamma(s + 1.) ** 2 / gamma(2. * s + 1.) * np.cos(wtheta / 2.) ** (2. * s)
        return D

    def Sk2(self, kx, ky):
        th = np.arctan2(ky, kx)
        k = np.sqrt(kx**2 + ky**2)
        k_inv = np.where(k != 0, 1/k, 0)
        return self.Sk(k) * k_inv * self.dirspread(k, th) 


def load_buoydata(file, date=None):
    """    
    :param file: npz file with buoy data
    :param date: Optional datetime.datetime variable, if given it looks for the data closest to that date
    :return: 
    """
    data = np.load(file, encoding='bytes')
    dates = data['dates']
    numd = dates.size
    arr0 = data['arr0']
    shp = (numd,) + arr0.shape
    data_all = np.zeros(shp)
    data_all[0, :, :] = arr0
    for ind in range(1, numd):
        data_all[ind, :, :] = data[('arr%i' % ind)]
    if date is None or type(date) is not datetime.datetime:
        return dates, data_all
    else:
        ind = np.argmin(np.abs(dates - date))
        dmin = (dates[ind] - date).total_seconds()/60
        if np.abs(dmin) > 10:
            print('load_buoydata: offset with respect to target time is', (dates[ind] - date))
        return dates[ind], data_all[ind]






def plot_kxky_spec(k_vec, kx, ky, E_kxky, impath, savestr, j=5, size=0.02):
    '''
    plot a kx ky spectrum from buoy data.

    j is for the location of the circles.
    size is the size of the plot
    '''
    # calc input
    l_vec = 2 * np.pi / k_vec    # wave lengths
    

    # create plot
    fig, ax = plt.subplots(ncols=1, figsize=(8, 6))
    im = ax.contourf(kx, ky, E_kxky, cmap=cm.jet)
    im.set_clim(0.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('[$m^2$/ $m^2$]', rotation=90)
    
    ax.axis([-size, size, -size, size])
    x = np.linspace(-1, 1, 100)
    ax.plot(x, x, '--', color=(0.5, 0.5, 0.5))
    ax.plot(-x, x, '--', color=(0.5, 0.5, 0.5))
    ax.text(k_vec[j], -k_vec[j] / 2, str(round(l_vec[j])) + ' m', color='grey')
    ax.text(k_vec[j + 4], -k_vec[j + 6] / 1.5,
            str(round(l_vec[j + 6])) + ' m', color='grey')

    # plot circles with wave length
    circle1 = plt.Circle((0, 0), k_vec[j], color='grey', fill=False)
    circle2 = plt.Circle((0, 0), k_vec[j + 6], color='grey', fill=False)
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.grid(color='k', linestyle='--')
    ax.set_title('Wave-number spectrum E(kx,ky)')
    ax.set_xlabel('kx [$m^{-1}$]')
    ax.set_ylabel('ky [$m^{-1}$]')
    plt.tight_layout()
    plt.savefig(os.path.join(impath, savestr + "_Ekxky_spec.png"))
    
    return l_vec


def plot_1D(data, location, impath, datestr):
    fig = plt.figure(figsize=(12,10))
    ax = plt.subplot() 
    ax.plot(data[0,:], data[1,:])
    ax.set_xlim([0,0.2])
    ax.set_xlabel('Frequency [Hz]', fontsize=30)
    ax.set_ylabel('Variance density [m2/Hz]', fontsize=30)
    ax.set_title('Wave buoy - 1D - '+datestr, fontsize=30)
    ax.grid(color='k', linestyle='--')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    plt.tight_layout()
    plt.savefig(os.path.join(impath,'1D_buoy_'+location+'_'+datestr ))
    

def do_plot(fig, n, f, title, axtick, heading):
    #plt.clf()
    
    ax1 = plt.subplot(1, 1, n)
    im1 = ax1.imshow(f, origin = 'lower', cmap=plt.cm.inferno_r , extent = axtick) #plt.cm.viridis_r,
    
    ax1.set_title(title, fontsize=22)
    ax1.set_xlabel('wavenumber kx [$m^{-1}$]', fontsize=22)
    ax1.set_ylabel('wavenumber ky [$m^{-1}$]', fontsize=22)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Variance density [$m^{2}$ / $m^{2}$]  (x 100)', rotation=90, size=22)
    cbar1.ax.tick_params(labelsize=18) 
    ax1.grid(color='k', linestyle='--')
    
    ax1.set_xlim((-0.011, 0.011))
    ax1.set_ylim((-0.011, 0.011))
    ax1.xaxis.set_ticks(np.arange(-0.01, 0.01+0.0025, 0.0025))
    ax1.yaxis.set_ticks(np.arange(-0.01, 0.01+0.0025, 0.0025))
    
    # diagonal lines
    x = np.linspace(-1, 1, 100)
    ax1.plot(x, x, '--', color=(0.5, 0.5, 0.5), alpha=0.7)
    ax1.plot(-x, x, '--', color=(0.5, 0.5, 0.5), alpha=0.7)
    
    # fancy circles with length
    k = np.array([0.003, 0.007])
    l = 1 / k
    kx = (k+0.0008) * np.cos(np.pi*3/2)
    ky = (k+0.0008) * np.sin(np.pi*3/2)
    circle1 = plt.Circle((0, 0), k[0], color='grey', fill=False, alpha=0.7)
    circle2 = plt.Circle((0, 0), k[1], color='grey', fill=False, alpha=0.7)
    ax1.add_artist(circle1)
    ax1.add_artist(circle2)
    ax1.text(kx[0], ky[0], str(round(l[0])) + ' m', color='grey')
    ax1.text(kx[1], ky[1], str(round(l[1])) + ' m', color='grey')
    ax1.arrow( 0, 0, 0.009*np.sin(np.radians(heading)), 0.009*np.cos(np.radians(heading)), fc="k", ec="k",width = 0.0001, head_width=0.0005, head_length=0.001 )
    ax1.arrow( -0.009*np.sin(np.radians(heading+90)), -0.009*np.cos(np.radians(heading+90)), 0.018*np.sin(np.radians(heading+90)), 0.018*np.cos(np.radians(heading+90)), fc="k", ec="k",width = 0.00005, head_width=0.00005, head_length=0)
    ax1.arrow( -0.009*np.sin(np.radians(heading)), -0.009*np.cos(np.radians(heading)), 0.018*np.sin(np.radians(heading)), 0.018*np.cos(np.radians(heading)), fc="k", ec="k",width = 0.0001, head_width=0.0005, head_length=0.001 )

    for tick in ax1.get_xticklabels():
        tick.set_rotation(270)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)

def kx_plot(data, depth, impath, savestr, location, ascending=True, heading=-11.2):
    # calculate linear wavenumber
    time_1 = datetime.datetime.utcnow()
    
    if np.sum(data[3]) > 999999999.000:
        fig = plt.figure(figsize=(12,7))
        ax1 = plt.subplot(1, 1, 1)
        ax1.plot(data[0], data[1])
        ax1.xaxis.set_ticks(np.arange(0, 0.5, 0.02))
        ax1.grid(color='k', linestyle='--')
        ax1.set_xlabel('Frequency [Hz]', fontsize=30)
        ax1.set_ylabel('Variancy Density [m2/ Hz]', fontsize=30)
        plt.savefig(os.path.join(impath, savestr + "_Ef_spec.png"))
        
    else: 

        fig = plt.figure(figsize=(12,7))
        ax1 = plt.subplot(1, 1, 1)
        ax1.plot(data[0], data[1])
        ax1.xaxis.set_ticks(np.arange(0, 0.5, 0.02))
        ax1.grid(color='k', linestyle='--')
        ax1.set_xlabel('Frequency [Hz]', fontsize=30)
        ax1.set_ylabel('Variancy Density [m2/ Hz]', fontsize=30)
        plt.savefig(os.path.join(impath, savestr + "_Ef_spec.png"))

        bS = BuoySpectra(data, ascending, depth, heading)
        kx = np.linspace(-bS.kmax, bS.kmax, 1001).reshape((1, 1001))
        ky = np.linspace(-bS.kmax, bS.kmax, 1001).reshape((1001, 1))
        S2 = bS.Sk2(kx, ky)
        
        krad = 2 * np.pi / 1
        kxr = kx / krad
        S2_lin = (S2 * krad**2) / 100
        # print('Total Ekxky spec = '+ str(np.sum(S2_lin * (kxr[0,0]-kxr[0,1])**2)) + ' [m]')
        
        fig = plt.figure(figsize=(8,7))
        do_plot(fig, 1, S2_lin,'Buoy ocean wavespectrum - 2D' , [-bS.kmax/krad, bS.kmax/krad, -bS.kmax/krad, bS.kmax/krad], heading)
        plt.tight_layout()
        plt.savefig(os.path.join(impath, savestr + "_Ekxky_spec.png")) 
        
    time_2 = datetime.datetime.utcnow()
    diff_t = round((time_2 - time_1).total_seconds())

    print('Calculating buoy data is finished in    ' + str(diff_t) + '   seconds.')
    print('------------------------------------------------------------')
#%%
# ~~~~~~ Execute ~~~~~~

if __name__ == '__main__':
    
    locpath = r'd:\data\buoy\NORTHSEA\M170513184\out\buoyspectra_k13.pkl'
    load data
    datan = load_obj(locpath)
    dates = datan['dates']
    data_all = datan['data']
    
    impath = r'd:\data\images'
            
    dateobs = datetime.datetime(2017, 1, 16, 17, 22)
    datestr = dateobs.strftime("%Y%m%dT%H%M%S")
    date = loading(dateobs)
    location = 'k13'

    if location == 'k13':
        depth = 26.7
        
        data = date.buoy_northsea('ijm')
        np.savez(os.path.join(r'd:\data\buoy\OCEANSAR', 'spectra_'+ datestr[:8]+'.npz'), arr_0 = data)
               
        savestr = '1'
        kx_plot(data, depth, impath, savestr, location, heading=-11.2)
        





        


