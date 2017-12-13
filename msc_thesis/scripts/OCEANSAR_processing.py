
# coding: utf-8

# In[18]:

# ~~~~~Import~~~~~

import numpy as np
import scipy
from scipy import signal

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import pprint
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


from scipy.optimize import leastsq



# ~~~~~Calculate~~~~~


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

def smooth(data, window_len=9, window='flat', axis=None):
    """ Smooth the data using a window with requested size.

        This method is based on the convolution of a scaled window with the signal.
        Works with 1-D and 2-D arrays.

        :param data: Input data
        :param window_len: Dimension of the smoothing window; should be an odd integer
        :param window: Type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'.
                       Flat window will produce a moving average smoothing.
        :param axis: if set, then it smoothes only over that axis

        :returns: the smoothed signal
    """

    if data.ndim > 2:
        raise ValueError('Arrays with ndim > 2 not supported')

    if window_len < 3:
        return data

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError('Window type not supported')

    # Calculate Kernel
    if window == 'flat':
        w = np.ones(window_len)
    else:
        w = eval('np.' + window + '(window_len)')

    # Smooth
    if data.ndim > 1:
        if axis is None:
            w = np.sqrt(np.outer(w, w))
        elif axis == 0:
            w = w.reshape((w.size, 1))
        else:
            w = w.reshape((1, w.size))

    y = signal.fftconvolve(data, w / w.sum(), mode='same')

    return y

def calc_stats(loc, meta, swath):
    '''
    This script can calculate the cross spectrum per subimage for an entire SLC image.
    It also looks at the mean variance to check if the image is suitable.

    '''

    # gather image information

    # number of sub_im in azimuth (1 per burst)

    size = meta['imsize']
    no_az = meta['no_burst']
    no_ra = int(meta['samplesPerBurst'] / size)  # number of sub_im in range
    lines = meta['linesPerBurst']

    start_x = (meta['samplesPerBurst'] - (no_ra * size)) / 2
    start_y = (lines / 2) - size / 2

    slc_or = np.load(os.path.join(        loc, 'data_' + swath + '_slc.npy'), mmap_mode='r')
    phase_or = np.load(os.path.join(        loc, 'data_' + swath + '_o.npy'), mmap_mode='r')

    # divide in subimages
    var = []
    var2 = []
    ran = []
    azn = []
    for az in range(no_az):  # no_az
        for ra in range(no_ra):  # no_ra

            # find pixel borders
            up = int(start_y + az * lines)
            down = int(up + size)
            left = int(start_x + ra * size)
            right = int(left + size)

            slcor = np.array(slc_or[up:down, left:right])
            o = np.array(phase_or[up:down, left:right])
            slc = slcor * np.exp(1j * o)

            stat = image_statistics(slc)
            sigma = stat['variance']

            var.append([str(az) + str(ra), sigma, az ,ra])
            var2.append(sigma)
            azn.append(az)
            ran.append(ra)

    
    var2 = np.asarray(var2)
    id = np.nanargmin(var2)
    azi = azn[id]
    rai = ran[id]
    meta['az_loc'] = azi
    meta['ra_loc'] = rai
    print('mimimum variance at azimuth = '+str(azi)+ ' and range = '+str(rai))

    return var, azi, rai


def plot_variance(meta, impath, loc, swath):
    '''
    This function plots the variance between 1 and 2 to see if retrieval can be applied.
    '''
    time_1 = datetime.datetime.utcnow()

    size = meta['imsize']
    var, azi, rai = calc_stats(loc, meta, swath)


    no_az = meta['no_burst']  # number of sub_im in azimuth (1 per burst)
    no_ra = int(meta['samplesPerBurst'] / size)  # number of sub_im in range
    linesi = meta['linesPerBurst']

    x = np.arange(meta['samplesPerBurst'] / 32)
    y = np.arange(linesi * no_az / 32)
    z = np.zeros([len(y), len(x)])

    samples = meta['samplesPerBurst'] / 32
    lines = linesi / 32
    size = size / 32

    start_x = (samples - (no_ra * size)) / 2
    start_y = (lines / 2) - size / 2

    for az in range(no_az):
        for ra in range(no_ra):

            # find pixel borders
            up = int(start_y + az * lines)
            down = int(up + size)
            left = int(start_x + ra * size)
            right = int(left + size)

            for i in range(no_az * no_ra):
                if var[i][0] == (str(az) + str(ra)):
                    sigma = var[i][1]

            z[up:down, left:right] = sigma

    z[z == 0] = np.nan
    m = np.ma.masked_where(np.isnan(z), z)

    # create Intensity image:
    slc_or = np.load(os.path.join(
        loc, 'data_' + swath + '_slc.npy'), mmap_mode='r')
    phase_or = np.load(os.path.join(
        loc, 'data_' + swath + '_o.npy'), mmap_mode='r')

    slcor = np.array(slc_or[::32, ::32])
    o = np.array(phase_or[::32, ::32])
    slc = slcor * np.exp(1j * o)
    x0 = np.arange(slc.shape[1])
    y0 = np.arange(slc.shape[0])[::-1]
    mn = np.mean(abs(slc))
    m = np.flipud(m)
    slcfl = np.flipud(abs(slc))
    y = y[::-1]
    # Plotting:
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(18, 6), sharey=True)

    im0 = ax0.pcolormesh(x0 * 32, y0 * 32, slcfl,
                     cmap=plt.cm.gray, vmin=0, vmax=mn * 2)
    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    cbar0.set_label('Intensity [dB]', size=20)
    cbar0.ax.tick_params(labelsize=16)
    ax0.set_xlabel('[range]', fontsize=20)
    ax0.set_ylabel('[azimuth]', fontsize=20)
    for tick in ax0.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax0.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    

    im1 = ax1.pcolormesh(x * 32, y * 32, m, cmap=plt.cm.RdYlGn_r,
                     vmin=1.0, vmax=2.0)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Variance (norm.) [-]', size=20)
    cbar1.ax.tick_params(labelsize=16)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    

    fig.tight_layout()
    fig.savefig(os.path.join(impath, os.path.split(
        loc)[1] + '-sigma.png'), dpi=300)
   
    time_2 = datetime.datetime.utcnow()
    diff_t = round((time_2 - time_1).total_seconds())
    print('Calculating variance is finished in    ' + str(diff_t) + '   seconds.')
    return print('done')



def cross_spec_3_looks(slc, georef, meta, impath, brst=0, n=0):
        # first running vertically downwards across rows (axis 0), and the
        # second running horizontally across columns (axis 1)
    satpass = georef['pass']
    print(satpass)
    slc_n, z1, z2 = remove_filter(slc)
    s1, slc_L1_f  = apply_window(slc_n, z1, z2, look = 1)
    s2, slc_L2_f  = apply_window(slc_n, z1, z2, look = 2)
    s3, slc_L3_f  = apply_window(slc_n, z1, z2, look = 3)
    
    f1 = intensity_detection(s1)
    f2 = intensity_detection(s2)
    f3 = intensity_detection(s3)
    
    im12, rl12, axtick, cspec12 = cross_spec(f1, f2, meta)
    im23, rl23, axtick, cspec23 = cross_spec(f2, f3, meta)
    im13, rl13, axtick, cspec13 = cross_spec(f1, f3, meta)
    im33, rl33, axtick, cspec33 = cross_spec(f3, f3, meta)

    # plotting
    def do_plot_1(n, f, title, cm, vmi, vma):
        #plt.clf()
        ax1 = plt.subplot(3, 2, n)
        im1 = ax1.imshow(f, cmap = cm, extent = axtick, vmin = vmi, vmax = vma)
        ax1.set_title(title)
        ax1.set_xlabel('range wavenumber kx [$m^{-1}$]')
        ax1.set_ylabel('azimuth wavenumber ky [$m^{-1}$]')
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('intensity (norm.) [-]', rotation=90)
        ax1.grid(color='k', linestyle='--')
        ax1.invert_yaxis()
        
    fig = plt.figure(figsize=(17,20))
    do_plot_1(1, normalize(rl12), 'Cross-spectrum look 1-2, real part', plt.cm.inferno_r, 0, 1)
    do_plot_1(2, im12, 'Cross-spectrum look 1-2, imag part', plt.cm.YlGnBu, np.min(im12) , np.max(im12))
    do_plot_1(3, normalize(rl23), 'Cross-spectrum look 2-3, real part', plt.cm.inferno_r, 0, 1)
    do_plot_1(4, im23, 'look 2-3, imag part', plt.cm.YlGnBu, np.min(im23) , np.max(im23))
    do_plot_1(5, normalize(rl13), 'Cross-spectrum look 1-3, real part', plt.cm.inferno_r , 0, 1)
    do_plot_1(6, im13, 'look 1-3, imag part', plt.cm.YlGnBu, np.min(im13) , np.max(im13))
    plt.tight_layout()
    plt.savefig(os.path.join(impath, 'cross_spectrum' + str(brst) + str(n) + '.png'), dpi=300)
    print('done')
    
    return cspec12, cspec23, cspec13


def single_image(loc, meta, georef, impath, swath, az=0, ra=0):
    '''
    This script can calculate the cross spectrum per subimage for an entire SLC image.
    It also looks at the mean variance to check if the image is suitable.

    '''
    print('Processing single image')

    size = meta['imsize']

    if az==0:
        az = georef['az_loc']
        ra = georef['ra_loc'] 


    slc_or = np.load(os.path.join(loc, 'data_' + swath + '_slc.npy'), mmap_mode='r')
    phase_or = np.load(os.path.join(loc, 'data_' + swath + '_o.npy'), mmap_mode='r')

    # shift 463 pixels to match multi_process function
 
 
    no_ra = int(meta['samplesPerBurst'] / size)  # number of sub_im in range
    lines = meta['linesPerBurst']
    
    start_x = (meta['samplesPerBurst'] - (no_ra * size)) / 2
    start_y = (lines / 2) - size / 2
    
    
    up = int(start_y + az * lines)
    down = int(up + size)
    left = int(start_x + ra * size)
    right = int(left + size)
      

    slcor = np.array(slc_or[up:down, left:right])
    o = np.array(phase_or[up:down, left:right])
    slc_buoy = slcor * np.exp(1j * o)

    stat = image_statistics(slc_buoy)
    sigma = stat['variance']
    

    if 1.0 < sigma < 1.5:
        print('variance is ' + str(sigma))
    else:
        print('Variance too high, selecting next image in range...')
        ra = ra + 1
        left = int(start_x + ra * size)
        right = int(left + size)
        
    
        slcor = np.array(slc_or[up:down, left:right])
        o = np.array(phase_or[up:down, left:right])
        slc_buoy = slcor * np.exp(1j * o)
        
        stat = image_statistics(slc_buoy)
        sigma = stat['variance']

    ext = [left, right, down, up]

    slc_sig = np.abs(slc_buoy)**2
    slc_db = np.where(slc_sig>0, 10 * np.log10(slc_sig), 0)

    def do_plot2(n, f, title, exten):
        #plt.clf()
        ax1 = plt.subplot(1, 1, n)
        im1 = ax1.imshow(f, vmin = -26, vmax =-7, extent=ext)
        ax1.set_title(title)
        ax1.set_xlabel('range pixel [x]')
        ax1.set_ylabel('azimuth pixel [y]')
        ax1.invert_yaxis()
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('intensity dB [-]', rotation=90)
        
    fig = plt.figure(figsize=(8,5))
    do_plot2(1, slc_db, 'Intensity image', ext)
    plt.tight_layout()
    plt.savefig(os.path.join(impath, 'SLC_int_db_VV.png'), dpi=300)
        
        
    print(up,down,left,right)
    print('done')
    return slc_buoy, az, ra
    


# new functions: 

def normalize(f, a=0, b=1):
    f = np.ma.array(f, mask=np.isnan(f))

    f_n = a + (f - np.min(f))*(b - a) / (np.max(f) - np.min(f))
    return f_n


def hamming(n, N):
    # n is x, N is width)
    a = 0.53836
    b = 0.46164
    return a - (b * np.cos((2 * np.pi * n) / (N - 1))) 


def fit_cos(data):

    t = np.linspace(0, 2*np.pi, len(data))
    guess_mean = np.mean(data)
    guess_std = 3*np.std(data)/(2**0.5)
    guess_phase = 0

    # Define the function to optimize, in this case, we want to minimize the difference
    # between the actual data and our "guessed" parameters
    optimize_func = lambda x: x[0]*np.sin(t+x[1]) + x[2] - data
    est_std, est_phase, est_mean = leastsq(optimize_func, [guess_std, guess_phase, guess_mean])[0]

    # recreate the fitted curve using the optimized parameters
    data_fit = est_std*np.sin(t+est_phase) + est_mean

    return data_fit


def remove_filter(slc):

    slc_f = np.fft.fftshift(np.fft.fft(slc, axis=0))  # vertical FFT, over azimuth.
    # determinde Doppler Bandwidth: check how many values are below 0.5
    sl = np.mean(abs(slc_f), axis=1)
    sl2 = np.flipud(sl)
    
    z1 = next(i for i,v in enumerate(sl) if v > 0.4)
    z2 = slc_f.shape[0]-next(i for i,v in enumerate(sl2) if v > 0.4)
    bd = sl[z1+1:z2-1]
    
    data_fit = fit_cos(bd)
    bd_fit = np.zeros(slc_f.shape[0])
    bd_fit[z1+1:z2-1] = data_fit
    sl[0:z1+1]=0
    sl[z2-1:]=0
    diff = np.zeros([slc_f.shape[0]])
    diff[z1+1:z2-1] = sl[z1+1:z2-1]/bd_fit[z1+1:z2-1]
    
    fit_2d = np.tile(bd_fit, (slc_f.shape[1],1)).transpose()
    slc_1 = slc_f.copy()
    
    
    slc_phase = np.angle(slc_1)
    # + 1j * fit_2d * np.sin(slc_phase)
    slc_n = np.zeros([slc_f.shape[0],slc_f.shape[1]], dtype=np.complex64)
    slc_n[z1+1:z2-1] = (np.abs(slc_1[z1+1:z2-1]) / fit_2d[z1+1:z2-1]) * np.exp(1j * slc_phase[z1+1:z2-1])
    
    return slc_n, z1, z2


def apply_window(slc_n, z1, z2, look):
    # set a window function on the data, in this case a hamming window.

    slc_L_f = slc_n.copy()
    diff = z2 - z1
    db = int(diff / 3)

    start = z1 + db * (look - 1)
    end = z1 + db * look


    x = np.zeros(slc_n.shape[0])
    n = np.array(range(db))

    window = hamming(n, db) 
    x[start:end]=window
    window_2d = np.tile(x, (slc_n.shape[1],1)).transpose()

    a  = np.angle(slc_L_f) # retrieve phase
    M = np.abs(slc_L_f)
    slc_L_f = window_2d * M * np.exp(1j * a)
    s1 = np.fft.ifft(np.fft.fftshift(slc_L_f, axes=0), axis=0)
    
    return s1, slc_L_f 

def intensity_detection(s):
    mod = abs(s)**2 # gather intensity image , eq 14 in L2_OSW
    # step 6, estimation of co and cross variance function
    # use 2 pi or M*N? (slc.shape[0]**2)
    f = np.fft.fftshift(np.fft.fft2(mod))
    return f


def cross_spec_old(f1, f2, meta, size=40, shape=1024):
    cspec1 = (f1 * np.conj(f2))    
    index = [int(cspec1.shape[0]/2), int(cspec1.shape[1]/2)]
    radius = 4
    cspec2 = cmask(index,radius,cspec1)
    #cspec = smooth(cspec2, 5, window='hamming')
    cspec = cspec2 / (np.mean(abs(f1)**2) * np.mean(abs(f2)**2))
    rl = normalize(cspec.real)
    rl = smooth(rl, 5, window='hamming')
    im = normalize(cspec.imag)
    im = smooth(im, 5, window='hamming')

    # smoothing and plotting
    centre = int(shape / 2)
    kx = np.fft.fftshift(np.fft.fftfreq(shape, d=meta['rangePixelSpacing']))
    ky = np.fft.fftshift(np.fft.fftfreq(shape, d=meta['azimuthPixelSpacing']))
    x1 = int(centre - size)
    x2 = int(centre + size)
    y1 = int(centre - (size * meta['azimuthPixelSpacing'] / meta['rangePixelSpacing']))
    y2 = int(centre + (size * meta['azimuthPixelSpacing'] / meta['rangePixelSpacing']))
    
    # size in seconds around centre (0)
    im = im[y1:y2, x1:x2]
    rl = rl[y1:y2, x1:x2]
    axtick = [kx[x1], kx[x2], ky[y1], ky[y2]]

    return im, rl, axtick, cspec1

def cross_spec(f1, f2, meta):

    size = meta['specsize']
    shape = meta['imsize']
    radius = meta['radius']
    smoothing = meta['smoothing']

    cspec1 = (f1 * np.conj(f2))  
    index = [int(cspec1.shape[0]/2), int(cspec1.shape[1]/2)]
    cspec = cmask(index,radius,cspec1)
    
    cspec = smooth(cspec, smoothing, window='hamming')
    cspec = cspec / (np.mean(abs(f1)**2) * np.mean(abs(f2)**2))
    
    
    
    rl = cspec.real    
    im = cspec.imag

    # smoothing and plotting
    centre_az = int(cspec.shape[0]/2)
    centre_ra = int(cspec.shape[1]/2)
    
    kx = np.fft.fftshift(np.fft.fftfreq(int(cspec.shape[1]), d=meta['rangePixelSpacing']))
    ky = np.fft.fftshift(np.fft.fftfreq(int(cspec.shape[0]), d=meta['azimuthPixelSpacing']))
    y1 = int(centre_az - size)
    y2 = int(centre_az + size)
    x1 = int(centre_ra - (size / (meta['azimuthPixelSpacing'] / meta['rangePixelSpacing']))) # 1/m so less pixels in range needed
    x2 = int(centre_ra + (size / (meta['azimuthPixelSpacing'] / meta['rangePixelSpacing'])))
    
    # size in 1/m
    im = im[y1:y2, x1:x2]
    rl = rl[y1:y2, x1:x2]
    axtick = [kx[x1], kx[x2], ky[y1], ky[y2]] #switch y1 and y2 for plot
    return im, rl, axtick, cspec1

def cross_variance(cspec, meta, size=40, shape=1024):
    
    centre = int(shape / 2)
    
    x1 = int(centre - (size * meta['azimuthPixelSpacing'] / meta['rangePixelSpacing']))
    x2 = int(centre + (size * meta['azimuthPixelSpacing'] / meta['rangePixelSpacing']))
    y1 = int(centre - size)
    y2 = int(centre + size)
    
    x = np.linspace(-512 * meta['rangePixelSpacing'], 511 * meta['rangePixelSpacing'], 1024)
    y = np.linspace(-512 * meta['azimuthPixelSpacing'], 511 * meta['azimuthPixelSpacing'], 1024)
    
    axtick2 = [x[x1], x[x2], y[y1], y[y2]]

    cross_var = np.fft.ifft2(cspec)
    cross_var1 = abs(np.fft.fftshift(cross_var))**2
    cross_var2 = normalize(cross_var1) 
    cross_var_out = cross_var2[y1:y2, x1:x2]
    return cross_var_out, axtick2


def spectral_estimation(): 
    '''
    Not checked yet. 
    '''
    no_az = meta['no_burst']  # number of sub_im in azimuth (1 per burst)
    no_ra = int(meta['samplesPerBurst'] / size)  # number of sub_im in range
    lines = meta['linesPerBurst']
    
    start_x = (meta['samplesPerBurst'] - (no_ra * size)) / 2
    start_y = (lines / 2) - size / 2
    
    slc_or = np.load(os.path.join(loc, 'data_' + swath + '_slc.npy'), mmap_mode='r')
    phase_or = np.load(os.path.join(loc, 'data_' + swath + '_o.npy'), mmap_mode='r')
    
    # divide in subimages
    var = []
    
    az = 8
    
    ra = 6
    # find pixel borders
    up = int(start_y + az * lines)
    down = int(up + size)
    left = int(start_x + ra * size)
    right = int(left + size)
    
    slcor = np.array(slc_or[up:down, left:right])
    o = np.array(phase_or[up:down, left:right])
    slc = slcor * np.exp(1j * o)
    
    stat = image_statistics(slc)
    
    slc_n, z1, z2 = remove_filter(slc)
    s1, slc_L1_f  = apply_window(slc_n, z1, z2, look = 1)
    s2, slc_L2_f  = apply_window(slc_n, z1, z2, look = 2)
    s3, slc_L3_f  = apply_window(slc_n, z1, z2, look = 3)
    
    
    f1 = intensity_detection(s1)
    f2 = intensity_detection(s2)
    f3 = intensity_detection(s3)
    
    im12, rl12, axtick, cspec12 = cross_spec(f1, f2, meta, size=15, shape=1024)
    im23, rl23, axtick, cspec23 = cross_spec(f2, f3, meta, size=15, shape=1024)
    im13, rl13, axtick, cspec13 = cross_spec(f1, f3, meta, size=15, shape=1024)
    im33, rl33, axtick, cspec33 = cross_spec(f3, f3, meta, size=15, shape=1024)
    
    cross_var1, axtick2 = cross_variance(cspec12, meta)
    cross_var2, axtick2 = cross_variance(cspec23, meta)
    cross_var3, axtick2 = cross_variance(cspec13, meta)
    return d






#%%

def cmask(index,radius,array):
    a,b = index
    nx,ny = array.shape
    y,x = np.ogrid[-a:nx-a,-b:ny-b]
    mask = (x*x)  + y*y / (3**2) <= radius*radius
    array[mask] = 0
    return(array)


def multi_process(loc, meta, swath):
    '''
    This script can calculate the cross spectrum per subimage for an entire SLC image.
    Out put is rl_n, which is the real part of the cross spectrum between look 1 and 3 and is stacked for every subimage.

    '''

    # gather image information
    time_1 = datetime.datetime.utcnow()

    size = meta['imsize']
    spec_size = meta['specsize']
    no_az = meta['no_burst']  # number of sub_im in azimuth (1 per burst)
    no_ra = int(meta['samplesPerBurst'] / size)  # number of sub_im in range
    lines = meta['linesPerBurst']
    
    start_x = (meta['samplesPerBurst'] - (no_ra * size)) / 2
    start_y = (lines / 2) - size / 2

    slc_or = np.load(os.path.join(loc, 'data_' + swath + '_slc.npy'), mmap_mode='r')
    phase_or = np.load(os.path.join(loc, 'data_' + swath + '_o.npy'), mmap_mode='r')

    # divide in subimages
    centre = int(size / 2)
    x1 = int(centre - spec_size)
    x2 = int(centre + spec_size)
    y1 = int(centre - (spec_size / (meta['azimuthPixelSpacing'] / meta['rangePixelSpacing']))) # 1/m so less pixels in range needed
    y2 = int(centre + (spec_size / (meta['azimuthPixelSpacing'] / meta['rangePixelSpacing'])))
    y = y2-y1
    x = x2-x1
    cross_rl = np.empty((x,y))
    cross_rl[:] = np.nan

    for az in range(no_az):  # no_az
        for ra in range(no_ra):  # no_ra
             # find pixel borders
            up = int(start_y + az * lines)
            down = int(up + size)
            left = int(start_x + ra * size)
            right = int(left + size)
            slcor = np.array(slc_or[up:down, left:right])
            o = np.array(phase_or[up:down, left:right])
            slc = slcor * np.exp(1j * o)
            stat = image_statistics(slc)
            sigma = stat['variance']

            if 1.0 < sigma < 1.5:

                slc_n, z1, z2 = remove_filter(slc)
                s1, slc_L1_f  = apply_window(slc_n, z1, z2, look = 1)
                s3, slc_L3_f  = apply_window(slc_n, z1, z2, look = 3)
                           
                f1 = intensity_detection(s1)
                f3 = intensity_detection(s3)
                
                im13, rl13, axtick, cspec13 = cross_spec(f1, f3, meta)

            else:
                rl13 = np.empty((x,y))
                rl13[:] = np.nan

            if az==0 and ra==0:
                rl_n = np.dstack((cross_rl, np.flipud(rl13)))
            else: 
                rl_n2 = np.dstack((rl_n, np.flipud(rl13)))
                rl_n = rl_n2

    rl_n = rl_n[:, :, 1:]
    
    time_2 = datetime.datetime.utcnow()
    diff_t = round((time_2 - time_1).total_seconds())
    print('Cross-spectra calculated in    ' + str(diff_t) + '   seconds.')
    return rl_n


def multi_plot(rl_n, impath, meta, dfile, swath):
    '''
    This script can plot the cross spectrum per subimage for an entire SLC image.
    It also looks at the mean variance to check if the image is suitable.

    '''

    # gather image information
    
    size = meta['imsize']
    no_az = meta['no_burst']# number of sub_im in azimuth (1 per burst)
    no_ra = int(meta['samplesPerBurst'] / size)  # number of sub_im in range
    lines = meta['linesPerBurst']
      
    start_x = (meta['samplesPerBurst'] - (no_ra * size)) / 2
    start_y = (lines / 2) - size / 2
    azn = np.repeat(np.arange(no_az), no_ra)
    ran = np.tile(np.arange(no_ra), no_az)
    
    # divide in subimages
    fig = plt.figure(figsize=(25,17))
    ax = plt.subplot()  
    ax.yaxis.tick_left()
    ax.tick_params(axis='y', colors='black', labelsize=30)
    ax.tick_params(axis='x', colors='black', labelsize=30)
    
    print('min ' + str(np.min(rl_n)))
    print('max ' + str(np.max(rl_n)))   

    for n in range(rl_n.shape[2]):
        az = azn[n]
        ra = ran[n]
        rl13 = rl_n[:, :, n]

        #rldb = np.where(rl13>0, 10 * np.log10(rl13), 0)

        up = int(start_y + az * lines+10)
        down = int(up + size-10)
        left = int(start_x + ra * size+10)
        right = int(left + size-10)
        xloc = left + 300
        yloc = down + 350

        im = ax.imshow(normalize(rl13), aspect='auto', extent=[left,right,down,up], cmap = plt.cm.inferno_r, vmin =0, vmax = 1, origin='lower')#18 to 28
        ax.text(xloc, yloc, str(az) +' '+ str(ra))
    ax.set_xlabel('range [pixel]', fontsize=60)
    ax.set_xlim([0,meta['samplesPerBurst']])
    ax.set_ylabel('azimuth [pixel]', fontsize=60)
    ax.set_ylim([0, no_az * lines + 1000 ])
    cbar1 = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar1.set_label('Intensity (norm.) [-]', size=60)
    cbar1.ax.tick_params(labelsize=45) 
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(45)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(45)
    fig.tight_layout()
    fig.savefig(os.path.join(impath, dfile+'_'+swath+'spec.png'), dpi=300)
    return rl_n



def ambiguity_mask(rl, im, th, theta=999):
    ''' 
    creates a mask to solve the 180 degree ambiguity. 
    It finds the index of the location with max value of imag spectrum
    next it replaces 180 degree around this point with nan.
    '''
    
    
    th = np.fliplr(np.flipud(th))

    if theta==999:
        num_largest = 5
        indx = np.unravel_index(im.argmax(), im.shape)
        th_sel = th[indx]
    else: 
        if theta > 180:
            theta = theta-360 

        th_sel = np.radians(theta)

    start = th_sel -  np.pi/2 
    end = th_sel +  np.pi/2  # or np.pi/2 for 180 degree 

    if th_sel > np.pi / 2:
        th[(th < start) & (th > (-np.pi + start))] = np.nan    
    elif th_sel < - np.pi / 2:
        th[(th > end) & (th < (np.pi + end))] = np.nan
    else:
        th[(start > th) | (th > end)] = np.nan

    rl[np.isnan(th)==True] = np.nan
    return rl


def cmasknan(index,radius,array):
    a,b = index
    nx,ny = array.shape
    y,x = np.ogrid[-a:nx-a,-b:ny-b]
    mask = x*x + y*y / (2**2) <= radius*radius
    array[mask] = np.nan
    return(array)


def alpha(omega, depth):
    """pp 124 Eq 5.4.21"""
    g = 9.81
    a = (omega ** 2 * depth / g)
    return a


def beta(a):
    b = a * (np.tanh(a)) ** (-1 / 2)
    return b


def kd(omega, depth):
    """pp 124 Eq 5.4.22"""
    a = alpha(omega, depth)
    b = beta(a)

    for x in range(1):
        k_d = (a + b ** 2 * (np.cosh(b)) ** -2) /               (np.tanh(b) + b * (np.cosh(b)) ** -2)
        a = k_d
        b = beta(a)
    return k_d


def SAR_1D(slc, buoy_spec, meta, location, impath, datestr, az, ra, thetasar=0):
    
    '''
    This script calculates the cross spectra of a single image, between look 1 and 3. 
    Next it calculates a histogram of intensity and k values and transform to E-frequnecy domain.
    output can be plotted. 
    '''
    
    meta['smoothing'] = 0
    size = meta['specsize']
    shape = meta['imsize']
    depth = meta['depth']
    

    slc_n, z1, z2 = remove_filter(slc)
    s1, slc_L1_f  = apply_window(slc_n, z1, z2, look = 1)
    s2, slc_L2_f  = apply_window(slc_n, z1, z2, look = 2)
    s3, slc_L3_f  = apply_window(slc_n, z1, z2, look = 3) 
    f1 = intensity_detection(s1)
    f2 = intensity_detection(s2)
    f3 = intensity_detection(s3)
    im12, rl12, axtick, cspec12 = cross_spec(f1, f2, meta)
    im13, rl13, axtick, cspec13 = cross_spec(f1, f3, meta)
    # process 2D spectra
    array_in = smooth(rl12, 0, window='hamming')

    index = [int(array_in.shape[0]/2), int(array_in.shape[1]/2)]
    rl = cmasknan(index,3,array_in)
    im13 = cmasknan(index,3,smooth(im13, 9, window='hamming'))
    rl13 = cmasknan(index,3,smooth(rl13, 9, window='hamming'))

    centre = int(shape / 2)
    kx = np.fft.fftshift(np.fft.fftfreq(shape, d=meta['rangePixelSpacing']))
    ky = np.fft.fftshift(np.fft.fftfreq(shape, d=meta['azimuthPixelSpacing']))
    y1 = int(centre - size)
    y2 = int(centre + size)
    x1 = int(centre - (size / (meta['azimuthPixelSpacing'] / meta['rangePixelSpacing']))) # 1/m so less pixels in range needed
    x2 = int(centre + (size / (meta['azimuthPixelSpacing'] / meta['rangePixelSpacing'])))

    # size in seconds around centre (0)

    kx1 = kx[x1:x2]
    ky1 = ky[y1:y2]
    kx2d = np.tile(kx1, (len(ky1),1))
    ky2d = np.flipud(np.tile(ky1, (len(kx1),1)).transpose())

    th = np.arctan2(kx2d, ky2d)
    k = np.sqrt(kx2d**2 + ky2d**2)
    
    # find indices of maximum 
    #array = ambiguity_mask(rl, im13, th)
    array = rl

    print('heading is ' + str(meta['heading']))

    def plot_SAR2D(n, f, f2, title, cm, vmi, vma, heading=-meta['heading']):
        #plt.clf()
        ax1 = plt.subplot(1, 1, n)
        im1 = ax1.imshow(f, cmap=cm ,  extent = axtick, vmin = vmi, vmax = vma)
        
        ax1.arrow( 0.009*np.sin(np.radians(heading+90)), -0.009*np.cos(np.radians(heading+90)),- 0.018*np.sin(np.radians(heading+90)), 0.018*np.cos(np.radians(heading+90)), fc="k", ec="k",width = 0.00005, head_width=0.00005, head_length=0)
        ax1.arrow( 0.009*np.sin(np.radians(heading)), -0.009*np.cos(np.radians(heading)), -0.018*np.sin(np.radians(heading)), 0.018*np.cos(np.radians(heading)), fc="k", ec="k",width = 0.0001, head_width=0.0005, head_length=0.001 )
        
        ax1.set_title(title, fontsize=22)
        ax1.set_xlabel('range wavenumber kx [$m^{-1}$]', fontsize=22)
        ax1.set_ylabel('azimuth wavenumber ky [$m^{-1}$]', fontsize=22)
        
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('intensity (norm.) [-]', rotation=90, size=22)
        cbar1.ax.tick_params(labelsize=18) 

        ax1.grid(color='k', linestyle='--')

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


        ax1.set_xlim((-0.011, 0.011))
        ax1.set_ylim((-0.011, 0.011))
        ax1.xaxis.set_ticks(np.arange(-0.01, 0.01+0.0025, 0.0025))
        ax1.yaxis.set_ticks(np.arange(-0.01, 0.01+0.0025, 0.0025))
        for tick in ax1.get_xticklabels():
            tick.set_rotation(270)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(18)

        #ax1.set_ylim(ax1.get_ylim()[::-1])
        #ax1.invert_yaxis()
        

    array2 = normalize(rl)

    fig = plt.figure(figsize=(8,7))
    plot_SAR2D(1, array2, array2, 'Cross-spectrum look 1-2, real part', plt.cm.inferno_r , 0,1)
    plt.tight_layout()
    plt.savefig(os.path.join(impath,'2D_SAR_'+location+'_'+datestr), dpi=300)


    # ----- Plot Imaginairy part of cross spectrumn ------
    array_rl = normalize(rl13)
    array_im = normalize(im13,-1,1)

    def plot_SAR2D2(n, f, f2, title, cm, vmi, vma, heading=-meta['heading']):
        #plt.clf()
        ax1 = plt.subplot(1, 2, n)

        if n==2:
            im1 = ax1.imshow(f, cmap=cm ,  extent = axtick, vmin = vmi, vmax = vma)
            im2 = ax1.contour(f2, 5, colors='k' , linewidths=0.7, extent = axtick, origin='upper')
        else:
            im1 = ax1.imshow(f, cmap=cm ,  extent = axtick, vmin = vmi, vmax = vma)
        
        ax1.arrow( 0.009*np.sin(np.radians(heading+90)), -0.009*np.cos(np.radians(heading+90)),- 0.018*np.sin(np.radians(heading+90)), 0.018*np.cos(np.radians(heading+90)), fc="k", ec="k",width = 0.00005, head_width=0.00005, head_length=0)
        ax1.arrow( 0.009*np.sin(np.radians(heading)), -0.009*np.cos(np.radians(heading)), -0.018*np.sin(np.radians(heading)), 0.018*np.cos(np.radians(heading)), fc="k", ec="k",width = 0.0001, head_width=0.0005, head_length=0.001 )
        
        ax1.set_title(title, fontsize=22)
        ax1.set_xlabel('range wavenumber kx [$m^{-1}$]', fontsize=22)
        ax1.set_ylabel('azimuth wavenumber ky [$m^{-1}$]', fontsize=22)
        
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('intensity (norm.) [-]', rotation=90, size=22)
        cbar1.ax.tick_params(labelsize=18) 

        ax1.grid(color='k', linestyle='--')

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


        ax1.set_xlim((-0.011, 0.011))
        ax1.set_ylim((-0.011, 0.011))
        ax1.xaxis.set_ticks(np.arange(-0.01, 0.01+0.0025, 0.0025))
        ax1.yaxis.set_ticks(np.arange(-0.01, 0.01+0.0025, 0.0025))
        for tick in ax1.get_xticklabels():
            tick.set_rotation(270)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(18)

    fig = plt.figure(figsize=(13,7))
    plot_SAR2D2(1, array_rl, array_rl, 'Look 1-3, real part', plt.cm.inferno_r , 0,1)
    plot_SAR2D2(2, array_im, array_rl, 'Look 1-3, imaginairy part', plt.cm.YlGnBu , -1,1)
    plt.tight_layout()
    plt.savefig(os.path.join(impath,'2D_SAR_im_'+location+'_'+datestr), dpi=300)
    
    #  ------------------plot 1D spectra ----------------------
    array_in = rl12  # NO smoothing

    index = [int(array_in.shape[0]/2), int(array_in.shape[1]/2)]
    rl = cmasknan(index,3,array_in)
    
    centre = int(shape / 2)
    kx = np.fft.fftshift(np.fft.fftfreq(shape, d=meta['rangePixelSpacing']))
    ky = np.fft.fftshift(np.fft.fftfreq(shape, d=meta['azimuthPixelSpacing']))
    y1 = int(centre - size)
    y2 = int(centre + size)
    x1 = int(centre - (size / (meta['azimuthPixelSpacing'] / meta['rangePixelSpacing']))) # 1/m so less pixels in range needed
    x2 = int(centre + (size / (meta['azimuthPixelSpacing'] / meta['rangePixelSpacing'])))

    # size in seconds around centre (0)

    kx1 = kx[x1:x2]
    ky1 = ky[y1:y2]
    kx2d = np.tile(kx1, (len(ky1),1))
    ky2d = np.flipud(np.tile(ky1, (len(kx1),1)).transpose())

    th = np.arctan2(kx2d, ky2d)
    k = np.sqrt(kx2d**2 + ky2d**2)
    
    # find indices of maximum 
    array = ambiguity_mask(array, im13, th, thetasar)
    
    fig = plt.figure(figsize=(8,7))
    plot_SAR2D(1, normalize(array),array, 'Cross-spectrum look 1-2, real part', plt.cm.inferno_r, 0,1)
    plt.tight_layout()
    plt.savefig(os.path.join(impath,'3D_SAR_'+location+'_'+datestr ), dpi=300)

    freqs = np.linspace(0.005,0.505, 200)
    fstart = freqs[0:len(freqs)-1]
    fend = freqs[1:]
    fbin = fend - fstart

    f_vec = np.vstack((fstart, fend, fbin))
    w_vec = f_vec * 2 * np.pi
    k_1 = kd(w_vec[0], depth) / depth
    k_2 = kd(w_vec[1], depth) / depth
    k_vec = np.vstack((k_1, k_2, k_2 - k_1))
    kx_vec = k_vec / (2 * np.pi)

    vec = np.vstack((1/f_vec[0],f_vec[0], w_vec[0], k_vec[0], kx_vec[0]))
    vec2 = np.vstack((1/f_vec[1],f_vec[1], w_vec[1], k_vec[1], kx_vec[1]))
    vec_dt = np.vstack((1/f_vec[2],f_vec[2], w_vec[2], k_vec[2], kx_vec[2]))

    ar_1d = array.flatten()
    k_1d = k.flatten()
    k_sar = k_1d[~np.isnan(ar_1d)]
    ar_sar = ar_1d[~np.isnan(ar_1d)]

    hist, bin_edges = np.histogram(k_sar, bins=vec[4], weights = ar_sar)
    vec = vec[:, 0:-1]

    kx_sar = hist
    kr_sar = kx_sar / (2 * np.pi)

    dkdw = k_vec[2, 0:-1] / w_vec[2, 0:-1]
    w_sar = kr_sar* dkdw
    f_sar = w_sar * 2 * np.pi

    data = np.vstack((vec[1], f_sar))

    fig = plt.figure(figsize=(13,10))
    ax = plt.subplot() 
    ax.plot(data[0,:], data[1,:], label = 'SAR spectrum')
    ax.set_xlim([0,0.2])
    ax.set_xlabel('Frequency [Hz]', fontsize=30)
    ax.set_ylabel('Intensity [1/Hz]', fontsize=30)
    ax.set_title('SAR spectrum - 1D -'+datestr, fontsize=30)
    ax.grid(color='k', linestyle='--')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    plt.tight_layout()
    plt.savefig(os.path.join(impath,'1D_SAR_'+location+'_'+datestr ))

    fig = plt.figure(figsize=(12,10))
    ax = plt.subplot() 
    im1 = ax.plot(data[0,:], data[1,:], color='black', label = 'SAR spectrum')
    
    ax2 = ax.twinx()
    im2 = ax2.plot(buoy_spec[0,:], buoy_spec[1,:], '--', color= 'C0', label = 'Buoy spectrum')
    ax2.set_ylabel('Variance density [$m^{2}$/Hz]', color='C0', fontsize=30)
    ax.set_zorder(ax2.get_zorder()+1) 
    ax.patch.set_visible(False) 
    for tl in ax2.get_yticklabels():
        tl.set_color('C0')
        tl.set_fontsize(25)
    lns = im1+im2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize = 'xx-large')

    ax.set_xlim([0,0.2])
    ax.set_xlabel('Frequency [Hz]', fontsize=30)
    ax.set_ylabel('Intensity [1/Hz]', fontsize=30)
    ax.set_title('Combined spectra -'+datestr, fontsize=30)
    ax.grid(color='k', linestyle='--')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    plt.tight_layout()
    plt.savefig(os.path.join(impath,'1D_norm_'+location+'_'+datestr ))

    return data

# ~~~~~~~~~~~~~ Execute ~~~~~~~~~~~~~~~
print('done')


# In[19]:

from netCDF4 import Dataset
date = '20150124'

sar_loc = os.path.join(r'd:\data\images\oceansar',date,'proc_data.nc')
nc = Dataset(sar_loc, 'r')
slci = nc['slc_i'][0,0,:,:]
slcr = nc['slc_r'][0,0,:,:]    

slc0 = slcr + 1j * slci
slc = slc0[20:-20,10:-10]

meta={}
meta['specsize'] = 150 # size for cross spectrum calculation
meta['imsize'] = slc.shape[0] #size of image, normally 1024, now slc shape
meta['radius'] = 0
meta['smoothing'] = 0
slcsize = slc.shape
meta['depth'] = 27
meta['rangePixelSpacing']= 5.4508
meta['azimuthPixelSpacing']= 15.0516
meta['heading'] = -11.2

print(meta['rangePixelSpacing'])
print(meta['azimuthPixelSpacing'])


georef = {}
georef['pass'] = 'Ascending'
georef['az_loc'] = 0
georef['ra_loc'] = 0

siz = 256
#slc = slc[25:25+siz, 40:40+siz]

stat = image_statistics(slc)
print(stat)
print(slc.shape)
#GROUND RANGE GRID SPACING = 5.4508
#AZIMUTH GRID SPACING = 15.0516

fig = plt.figure(figsize=(20,5))
ax1 = plt.subplot(1, 1, 1)
im1 = ax1.imshow(abs(slc)**2, vmin = 0, vmax = 0.1)
ax1.set_title('sub-look 1')
ax1.set_xlabel('range pixel [x]')
ax1.set_ylabel('azimuth pixel [y]')
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('intensity [-]', rotation=90)
plt.tight_layout()
#plt.savefig(os.path.join('d:\data\images\method', 'sublook_intensity_OSAR.png'), dpi=300)
plt.show()


# In[20]:

listi = glob.glob(r'd:\data\images\oceansar\*')
for ist in listi:
    i = os.path.split(ist)[1]
    date=i

    sar_loc = os.path.join(r'd:\data\images\oceansar',date,'proc_data.nc')
    nc = Dataset(sar_loc, 'r')
    slci = nc['slc_i'][0,0,:,:]
    slcr = nc['slc_r'][0,0,:,:]    

    slc0 = slcr + 1j * slci
    slc = slc0[20:-20,10:-10]

    meta={}
    meta['specsize'] = 150 # size for cross spectrum calculation
    meta['imsize'] = slc.shape[0] #size of image, normally 1024, now slc shape
    meta['radius'] = 0
    meta['smoothing'] = 0
    slcsize = slc.shape
    meta['depth'] = 27
    meta['rangePixelSpacing']= 5.4508
    meta['azimuthPixelSpacing']= 15.0516
    meta['heading'] = -11.2

    print(meta['rangePixelSpacing'])
    print(meta['azimuthPixelSpacing'])


    georef = {}
    georef['pass'] = 'Ascending'
    georef['az_loc'] = 0
    georef['ra_loc'] = 0

    siz = 256
    #slc = slc[25:25+siz, 40:40+siz]

    stat = image_statistics(slc)
    print(stat)
    print(slc.shape)
    #GROUND RANGE GRID SPACING = 5.4508
    #AZIMUTH GRID SPACING = 15.0516

    fig = plt.figure(figsize=(20,5))
    ax1 = plt.subplot(1, 1, 1)
    im1 = ax1.imshow(abs(slc)**2, vmin = 0, vmax = 0.1)
    ax1.set_title('sub-look 1')
    ax1.set_xlabel('range pixel [x]')
    ax1.set_ylabel('azimuth pixel [y]')
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('intensity [-]', rotation=90)
    plt.tight_layout()
    #plt.savefig(os.path.join('d:\data\images\method', 'sublook_intensity_OSAR.png'), dpi=300)
    plt.show()


# In[28]:


def process_oceansar_data(date):

    '''
    This script calculates the cross spectra of a single image, between look 1 and 3. 
    Next it calculates a histogram of intensity and k values and transform to E-frequnecy domain.
    output can be plotted. 
    '''
    sar_loc = os.path.join(r'd:\data\images\oceansar',date,'proc_data.nc')
    nc = Dataset(sar_loc, 'r')
    slci = nc['slc_i'][0,0,:,:]
    slcr = nc['slc_r'][0,0,:,:]    

    slc0 = slcr + 1j * slci
    slc = slc0[20:-20,20:-20]
    
    meta={}
    meta['specsize'] = 120 # size for cross spectrum calculation
    meta['imsize'] = slc.shape[0] #size of image, normally 1024, now slc shape
    meta['radius'] = 0
    meta['smoothing'] = 0
    slcsize = slc.shape
    meta['depth'] = 27
    meta['rangePixelSpacing']= 5.4508
    meta['azimuthPixelSpacing']= 15.0516
    meta['heading'] = 11.2
    if date=='20161001':
        meta['heading'] = 168.8
    elif date=='20161031':
        meta['heading'] = 168.8

    print(meta['rangePixelSpacing'])
    print(meta['azimuthPixelSpacing'])


    georef = {}
    georef['pass'] = 'Ascending'
    georef['az_loc'] = 0
    georef['ra_loc'] = 0


    size = meta['specsize']
    shape = meta['imsize']
    depth = meta['depth']
    impath = r'd:\data\images\plot\osar_images'
    
    
    slc_n, z1, z2 = remove_filter(slc)
    s1, slc_L1_f  = apply_window(slc_n, z1, z2, look = 1)
    s2, slc_L2_f  = apply_window(slc_n, z1, z2, look = 2)
    s3, slc_L3_f  = apply_window(slc_n, z1, z2, look = 3) 
    f1 = intensity_detection(s1)
    f2 = intensity_detection(s2)
    f3 = intensity_detection(s3)
    im12, rl12, axtick, cspec12 = cross_spec(f1, f2, meta)
    im13, rl13, axtick, cspec13 = cross_spec(f1, f3, meta)
    # process 2D spectra
    array_in = smooth(rl12, 3, window='hamming')

    index = [int(array_in.shape[0]/2), int(array_in.shape[1]/2)]
    rl = cmasknan(index,3,array_in)
    im13 = cmasknan(index,3,smooth(im13, 3, window='hamming'))
    rl13 = cmasknan(index,3,smooth(rl13, 3, window='hamming'))

    centre = int(shape / 2)
    kx = np.fft.fftshift(np.fft.fftfreq(shape, d=meta['rangePixelSpacing']))
    ky = np.fft.fftshift(np.fft.fftfreq(shape, d=meta['azimuthPixelSpacing']))
    y1 = int(centre - size)
    y2 = int(centre + size)
    x1 = int(centre - (size / (meta['azimuthPixelSpacing'] / meta['rangePixelSpacing']))) # 1/m so less pixels in range needed
    x2 = int(centre + (size / (meta['azimuthPixelSpacing'] / meta['rangePixelSpacing'])))

    # size in seconds around centre (0)

    kx1 = kx[x1:x2]
    ky1 = ky[y1:y2]
    kx2d = np.tile(kx1, (len(ky1),1))
    ky2d = np.flipud(np.tile(ky1, (len(kx1),1)).transpose())

    th = np.arctan2(kx2d, ky2d)
    k = np.sqrt(kx2d**2 + ky2d**2)
    
    # find indices of maximum 
    #array = ambiguity_mask(rl, im13, th)
    array = rl

    print('heading is ' + str(meta['heading']))

    def plot_SAR2D(n, f, f2, title, cm, vmi, vma, heading=-meta['heading']):
        #plt.clf()
        ax1 = plt.subplot(1, 1, n)
        im1 = ax1.imshow(f, cmap=cm ,  extent = axtick, vmin = vmi, vmax = vma)
        
        ax1.arrow( 0.009*np.sin(np.radians(heading+90)), -0.009*np.cos(np.radians(heading+90)),- 0.018*np.sin(np.radians(heading+90)), 0.018*np.cos(np.radians(heading+90)), fc="k", ec="k",width = 0.00005, head_width=0.00005, head_length=0)
        ax1.arrow( 0.009*np.sin(np.radians(heading)), -0.009*np.cos(np.radians(heading)), -0.018*np.sin(np.radians(heading)), 0.018*np.cos(np.radians(heading)), fc="k", ec="k",width = 0.0001, head_width=0.0005, head_length=0.001 )
        
        ax1.set_title(title, fontsize=22)
        ax1.set_xlabel('range wavenumber kx [$m^{-1}$]', fontsize=22)
        ax1.set_ylabel('azimuth wavenumber ky [$m^{-1}$]', fontsize=22)
        
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('intensity (norm.) [-]', rotation=90, size=22)
        cbar1.ax.tick_params(labelsize=18) 

        ax1.grid(color='k', linestyle='--')

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


        ax1.set_xlim((-0.011, 0.011))
        ax1.set_ylim((-0.011, 0.011))
        ax1.xaxis.set_ticks(np.arange(-0.01, 0.01+0.0025, 0.0025))
        ax1.yaxis.set_ticks(np.arange(-0.01, 0.01+0.0025, 0.0025))
        for tick in ax1.get_xticklabels():
            tick.set_rotation(270)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(18)

        #ax1.set_ylim(ax1.get_ylim()[::-1])
        #ax1.invert_yaxis()
        

    array2 = normalize(rl)

    fig = plt.figure(figsize=(8,7))
    plot_SAR2D(1, array2, array2, 'Cross-spectrum look 1-2, real part', plt.cm.inferno_r , 0,1)
    plt.tight_layout()
    plt.show()
    #plt.savefig(os.path.join(impath,'2Dspectra','2D_OSAR_'+date+'.png'), dpi=300)


    # ----- Plot Imaginairy part of cross spectrumn ------
    array_rl = normalize(rl13)
    array_im = normalize(im13,-1,1)

    def plot_SAR2D2(n, f, f2, title, cm, vmi, vma, heading=-meta['heading']):
        #plt.clf()
        ax1 = plt.subplot(1, 2, n)

        if n==2:
            im1 = ax1.imshow(f, cmap=cm ,  extent = axtick, vmin = vmi, vmax = vma)
            im2 = ax1.contour(f2, 5, colors='k' , linewidths=0.7, extent = axtick, origin='upper')
        else:
            im1 = ax1.imshow(f, cmap=cm ,  extent = axtick, vmin = vmi, vmax = vma)
        
        ax1.arrow( 0.009*np.sin(np.radians(heading+90)), -0.009*np.cos(np.radians(heading+90)),- 0.018*np.sin(np.radians(heading+90)), 0.018*np.cos(np.radians(heading+90)), fc="k", ec="k",width = 0.00005, head_width=0.00005, head_length=0)
        ax1.arrow( 0.009*np.sin(np.radians(heading)), -0.009*np.cos(np.radians(heading)), -0.018*np.sin(np.radians(heading)), 0.018*np.cos(np.radians(heading)), fc="k", ec="k",width = 0.0001, head_width=0.0005, head_length=0.001 )
        
        ax1.set_title(title, fontsize=22)
        ax1.set_xlabel('range wavenumber kx [$m^{-1}$]', fontsize=22)
        ax1.set_ylabel('azimuth wavenumber ky [$m^{-1}$]', fontsize=22)
        
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('intensity (norm.) [-]', rotation=90, size=22)
        cbar1.ax.tick_params(labelsize=18) 

        ax1.grid(color='k', linestyle='--')

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


        ax1.set_xlim((-0.011, 0.011))
        ax1.set_ylim((-0.011, 0.011))
        ax1.xaxis.set_ticks(np.arange(-0.01, 0.01+0.0025, 0.0025))
        ax1.yaxis.set_ticks(np.arange(-0.01, 0.01+0.0025, 0.0025))
        for tick in ax1.get_xticklabels():
            tick.set_rotation(270)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(18)

    fig = plt.figure(figsize=(13,7))
    plot_SAR2D2(1, array_rl, array_rl, 'Look 1-3, real part', plt.cm.inferno_r , 0,1)
    plot_SAR2D2(2, array_im, array_rl, 'Look 1-3, imaginairy part', plt.cm.YlGnBu , -1,1)
    plt.tight_layout()
    #plt.savefig(os.path.join(impath,'2DOSAR','2D_OSAR_im_'+date+'.png'), dpi=300)
    
    #  ------------------plot 1D spectra ----------------------
    array_in = rl12  # NO smoothing

    index = [int(array_in.shape[0]/2), int(array_in.shape[1]/2)]
    rl = cmasknan(index,3,array_in)
    
    centre = int(shape / 2)
    kx = np.fft.fftshift(np.fft.fftfreq(shape, d=meta['rangePixelSpacing']))
    ky = np.fft.fftshift(np.fft.fftfreq(shape, d=meta['azimuthPixelSpacing']))
    y1 = int(centre - size)
    y2 = int(centre + size)
    x1 = int(centre - (size / (meta['azimuthPixelSpacing'] / meta['rangePixelSpacing']))) # 1/m so less pixels in range needed
    x2 = int(centre + (size / (meta['azimuthPixelSpacing'] / meta['rangePixelSpacing'])))

    # size in seconds around centre (0)

    kx1 = kx[x1:x2]
    ky1 = ky[y1:y2]
    kx2d = np.tile(kx1, (len(ky1),1))
    ky2d = np.flipud(np.tile(ky1, (len(kx1),1)).transpose())

    th = np.arctan2(kx2d, ky2d)
    k = np.sqrt(kx2d**2 + ky2d**2)
    
    # find indices of maximum 
    array = ambiguity_mask(array, im13, th, 999)
    
    freqs = np.linspace(0.005,0.505, 200)
    fstart = freqs[0:len(freqs)-1]
    fend = freqs[1:]
    fbin = fend - fstart

    f_vec = np.vstack((fstart, fend, fbin))
    w_vec = f_vec * 2 * np.pi
    k_1 = kd(w_vec[0], depth) / depth
    k_2 = kd(w_vec[1], depth) / depth
    k_vec = np.vstack((k_1, k_2, k_2 - k_1))
    kx_vec = k_vec / (2 * np.pi)

    vec = np.vstack((1/f_vec[0],f_vec[0], w_vec[0], k_vec[0], kx_vec[0]))
    vec2 = np.vstack((1/f_vec[1],f_vec[1], w_vec[1], k_vec[1], kx_vec[1]))
    vec_dt = np.vstack((1/f_vec[2],f_vec[2], w_vec[2], k_vec[2], kx_vec[2]))

    ar_1d = array.flatten()
    k_1d = k.flatten()
    k_sar = k_1d[~np.isnan(ar_1d)]
    ar_sar = ar_1d[~np.isnan(ar_1d)]

    hist, bin_edges = np.histogram(k_sar, bins=vec[4], weights = ar_sar)
    vec = vec[:, 0:-1]

    kx_sar = hist
    kr_sar = kx_sar / (2 * np.pi)

    dkdw = k_vec[2, 0:-1] / w_vec[2, 0:-1]
    w_sar = kr_sar* dkdw
    f_sar = w_sar * 2 * np.pi

    data = np.vstack((vec[1], f_sar))
    np.save(os.path.join(impath,'1Dspectra','data'+date+'.npy'), data)
    
    fig = plt.figure(figsize=(13,10))
    ax = plt.subplot() 
    ax.plot(data[0,:], data[1,:], label = 'SAR spectrum')
    ax.set_xlim([0,0.2])
    ax.set_xlabel('Frequency [Hz]', fontsize=30)
    ax.set_ylabel('Intensity [1/Hz]', fontsize=30)
    ax.set_title('SAR spectrum - 1D -'+date, fontsize=30)
    ax.grid(color='k', linestyle='--')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    plt.tight_layout()
    #plt.savefig(os.path.join(impath,'1Dspectra','1D_OSAR_'+date+'.png'), dpi=300)





    
plt.savefig(os.path.join(r'd:\data\images\plot\osar_images',date+'.png'), dpi=300)
plt.close()


# In[29]:

import glob
listi = glob.glob(r'd:\data\images\oceansar\*')
print(listi)


# In[32]:

date = '20150426'
listi = glob.glob(r'd:\data\images\oceansar\*')
listi = listi
for ist in listi:
    i = os.path.split(ist)[1]
    print(i)
    process_oceansar_data(i)


# In[ ]:



