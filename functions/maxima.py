#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:44:14 2019

@author: martin
"""



        #return out

#import trackpy as tp
import numpy as np
from scipy.ndimage import filters, measurements
from matplotlib.path import Path
import scipy.cluster.hierarchy as hcluster
import skimage

def count_maxima(par_obj, time_pt, fileno, reset_max=False):
    """Choose type of maxima finding based on par_obj.count_maxima_laplace
    and number of dimensions (2/3)
    """
    #count maxima won't work properly if have selected a random set of Z
    imfile = par_obj.filehandlers[fileno]
    if par_obj.min_distance[2] == 0 or imfile.max_z == 0 and not par_obj.count_maxima_laplace:
        #2d hessian case
        count_maxima_2d(par_obj, time_pt, fileno, reset_max)
        return
    if par_obj.count_maxima_laplace == 'cluster':
        #pseudo-3D case
        count_maxima_2d_v2(par_obj, time_pt, fileno, reset_max)
        return
    elif par_obj.count_maxima_laplace is True:
        #2/3d laplacian case
        count_maxima_laplace(par_obj, time_pt, fileno, reset_max)
        return
    else:
        #standard hessian 3d case
        maxima_hessian_3D(par_obj, time_pt, fileno, reset_max)
        return
    
def maxima_hessian_3D(par_obj, time_pt, fileno, reset_max = False):
    
    imfile = par_obj.filehandlers[fileno]
    
    radius = [par_obj.min_distance[0], par_obj.min_distance[1], par_obj.resize_factor*par_obj.min_distance[2]/imfile.z_calibration]

    predMtx = np.zeros((par_obj.height, par_obj.width, imfile.max_z+1))

    for i in range(imfile.max_z+1):
        if time_pt in par_obj.data_store['pred_arr'][fileno]:
            if i in par_obj.data_store['pred_arr'][fileno][time_pt]:
                predMtx[:, :, i] = par_obj.data_store['pred_arr'][fileno][time_pt][i]
    """
    with par_obj.data.predictions[fileno].reader() as predictions:
        
        for i in range(imfile.max_z+1):
            predMtx[:, :, i] = predictions.read_plane(t=time_pt,z=i,f=fileno)
    """
    det3 = skimage.util.apply_parallel(det_hess_3d, predMtx, chunks=(250, 250, par_obj.max_z+1),extra_arguments=[radius], depth=0, mode=None)
    #det3 = det_hess_3d(predMtx, radius)

    #if not already set, create. This is then used for the entire image and all subsequent training. 
    #Based on the reasonable assumption that the training set contains a higher P, better gaussian fit
    #than in the test set - generalisation error makes this likely
    #A little hacky, but otherwise the normalisation screws everything up
    if not par_obj.max_det or reset_max:
        par_obj.max_det = np.max(det3)
        
    # normalise, preventing divide by zero
    det3 = det3/max(par_obj.max_det,0.0000000001)
    """
    with par_obj.data.hessians[fileno].writer() as hessians:
        for i in range(imfile.max_z+1):
           hessians.write_plane((det3[:, :, i]*255).astype('uint8'),time_pt,i,fileno)
    """
    par_obj.data_store['maxi_arr'][fileno][time_pt] = {}
    for i in range(imfile.max_z+1):
        par_obj.data_store['maxi_arr'][fileno][time_pt][i] = det3[:, :, i]
        
    #now deal with actual point detections
    pts = peak_local_max(det3, min_distance=radius, threshold_abs=par_obj.abs_thr,threshold_rel=0)
    
    pts = [np.array([pt[0],pt[1],pt[2],1]) for pt in pts]
    
    pts = filter_points_roi(par_obj,pts,fileno,time_pt)

    par_obj.data_store['pts'][fileno][time_pt] = pts
    
    par_obj.show_pts = 1

def filter_points_roi(par_obj,pts,fileno,time_pt):
    #Filter those points which are not inside the region.
    pts2keep = []
    if par_obj.data_store['roi_stkint_x'][fileno][time_pt].__len__() > 0:
        

        for i in par_obj.data_store['roi_stkint_x'][fileno][time_pt]:
            for pt2d in pts:

                if pt2d[2] == i:
                    #Find the region of interest.
                    ppt_x = par_obj.data_store['roi_stkint_x'][fileno][time_pt][i]
                    ppt_y = par_obj.data_store['roi_stkint_y'][fileno][time_pt][i]
                    #Reformat to make the path object.
                    pot = []
                    for b in range(0, ppt_x.__len__()):
                        pot.append([ppt_x[b], ppt_y[b]])
                    p = Path(pot)
                    if p.contains_point([pt2d[1], pt2d[0]]) is True:
                        pts2keep.append(pt2d)
    else:
        return pts
    
    return pts2keep


def count_maxima_laplace(par_obj, time_pt, fileno, reset_max=False):
    #count maxima won't work properly if have selected a random set of Z
    min_d = par_obj.min_distance
    imfile = par_obj.filehandlers[fileno]
    #if par_obj.min_distance[2] == 0 or par_obj.max_z == 0:
    #    count_maxima_2d(par_obj, time_pt, fileno)
    #    return
    predMtx = np.zeros((par_obj.height, par_obj.width, imfile.max_z+1))
    for i in range(imfile.max_z+1):
        predMtx[:, :, i] = par_obj.data_store['pred_arr'][fileno][time_pt][i]

    laplace = -filters.gaussian_laplace(predMtx, min_d, mode='constant')

    #if not already set, create. This is then used for the entire image and all subsequent training.
    #A little hacky, but otherwise the normalisation screws everything up
    if not par_obj.max_det:
        par_obj.max_det = np.max(laplace)
    elif reset_max:
        par_obj.max_det = np.max(laplace)

    laplace = laplace/par_obj.max_det
    par_obj.data_store['maxi_arr'][fileno][time_pt] = {}
    for i in range(imfile.max_z+1):
        par_obj.data_store['maxi_arr'][fileno][time_pt][i] = laplace[:, :, i]

    pts = peak_local_max(laplace, min_distance=min_d, threshold_abs=par_obj.abs_thr)

    pts2keep = []
    for pt2d in pts:
        #determinants of submatrices
        pts2keep.append([pt2d[0], pt2d[1], pt2d[2], 1])
    pts = pts2keep
    par_obj.show_pts = 1

    #Filter those which are not inside the region.
    if par_obj.data_store['roi_stkint_x'][fileno][time_pt].__len__() > 0:
        pts2keep = []

        for i in par_obj.data_store['roi_stkint_x'][fileno][time_pt]:
            for pt2d in pts:

                if pt2d[2] == i:
                    #Find the region of interest.
                    ppt_x = par_obj.data_store['roi_stkint_x'][fileno][time_pt][i]
                    ppt_y = par_obj.data_store['roi_stkint_y'][fileno][time_pt][i]
                    #Reformat to make the path object.
                    pot = []
                    for b in range(0, ppt_x.__len__()):
                        pot.append([ppt_x[b], ppt_y[b]])
                    p = Path(pot)
                    if p.contains_point([pt2d[1], pt2d[0]]) is True:
                        pts2keep.append(pt2d)
        pts = pts2keep

    par_obj.data_store['pts'][fileno][time_pt] = pts
def count_maxima_laplace_variable(par_obj, time_pt, fileno, reset_max=False):
    #count maxima won't work properly if have selected a random set of Z
    min_d = [x for x in par_obj.min_distance]
    imfile = par_obj.filehandlers[fileno]
    #if par_obj.min_distance[2] == 0 or par_obj.max_z == 0:
    #    count_maxima_2d(par_obj, time_pt, fileno)
    #    return
    predMtx = np.zeros((par_obj.height, par_obj.width, imfile.max_z+1))
    for i in range(imfile.max_z+1):
        predMtx[:, :, i] = par_obj.data_store['pred_arr'][fileno][time_pt][i]

    l1=filters.gaussian_laplace(predMtx, [0,0,min_d[2]], mode='constant')
    l2=filters.gaussian_laplace(predMtx, [0,0,min_d[2]*2], mode='constant') *2
    l3=filters.gaussian_laplace(predMtx, [0,0,min_d[2]*.5], mode='constant')  *.5
    '''l1=-filters.gaussian_laplace(predMtx, min_d, mode='constant')
    l2=-filters.gaussian_laplace(predMtx, [x*.5 for x in min_d], mode='constant')
    l3=-filters.gaussian_laplace(predMtx, [x*2 for x in min_d], mode='constant')
    '''
    l3=filters.gaussian_laplace((l1+l2+l3), [min_d[0],min_d[1],0], mode='constant')/3
    #if not already set, create. This is then used for the entire image and all subsequent training.
    #A little hacky, but otherwise the normalisation screws everything up
    if not par_obj.max_det or reset_max:
        par_obj.max_det = np.max(l3)

    l3 = l3/par_obj.max_det
    par_obj.data_store['maxi_arr'][fileno][time_pt] = {}
    for i in range(imfile.max_z+1):
        par_obj.data_store['maxi_arr'][fileno][time_pt][i] = l3[:, :, i]

    pts = peak_local_max(l3, min_distance=min_d, threshold_abs=par_obj.abs_thr)

    pts2keep = []
    for pt2d in pts:
        #determinants of submatrices
        pts2keep.append([pt2d[0], pt2d[1], pt2d[2], 1])
    pts = pts2keep
    par_obj.show_pts = 1

    #Filter those which are not inside the region.
    if par_obj.data_store['roi_stkint_x'][fileno][time_pt].__len__() > 0:
        pts2keep = []

        for i in par_obj.data_store['roi_stkint_x'][fileno][time_pt]:
            for pt2d in pts:

                if pt2d[2] == i:
                    #Find the region of interest.
                    ppt_x = par_obj.data_store['roi_stkint_x'][fileno][time_pt][i]
                    ppt_y = par_obj.data_store['roi_stkint_y'][fileno][time_pt][i]
                    #Reformat to make the path object.
                    pot = []
                    for b in range(0, ppt_x.__len__()):
                        pot.append([ppt_x[b], ppt_y[b]])
                    p = Path(pot)
                    if p.contains_point([pt2d[1], pt2d[0]]) is True:
                        pts2keep.append(pt2d)
        pts = pts2keep

    par_obj.data_store['pts'][fileno][time_pt] = pts
    
def det_hess_3d(predMtx, min_distance):
    
    #silently fails if image is too small to work on
    if predMtx.shape[0]<3 or predMtx.shape[1]<3 or predMtx.shape[2]<3:
        return np.zeros(predMtx.shape)
    
    gau_stk = filters.gaussian_filter(predMtx.astype('float32'), min_distance, mode='constant')

    y, x, z = np.gradient(gau_stk, 1)
    xy, xx, xz = np.gradient(x)
    yy, yz = np.gradient(y,axis=(0,2))
    zz = np.gradient(z,axis=2)
    det3 = -1*((((yy*zz)-(yz*yz))*xx)-(((xy*zz)-(yz*xz))*xy)+(((xy*yz)-(yy*xz))*xz))
    det2 = xx*yy-xy*xy
    det1 = -1*xx
    det3*=(det3>0)
    det3*=(det2>0)
    det3*=(det1>0)
    #det3[det3<0] = 0
    #det3[det2<0] = 0
    #det3[det1<0] = 0
    return det3

def count_maxima_thresh(par_obj, time_pt, fileno,reset_max=False):

    #count maxima won't work properly if have selected a random set of Z
    imfile = par_obj.filehandlers[fileno]
    if par_obj.min_distance[2] == 0 or imfile.max_z == 0:
        count_maxima_2d(par_obj, time_pt, fileno,reset_max)
        return
    predMtx = np.zeros((par_obj.height, par_obj.width, imfile.max_z+1))
    for i in range(imfile.max_z+1):
        predMtx[:, :, i] = par_obj.data_store['pred_arr'][fileno][time_pt][i]

    radius = [par_obj.min_distance[0], par_obj.min_distance[1], par_obj.resize_factor*par_obj.min_distance[2]/imfile.z_calibration]

    [det3, det2, det1] = det_hess_3d(predMtx, radius)


    #if not already set, create. This is then used for the entire image and all subsequent training.
    #A little hacky, but otherwise the normalisation screws everything up
    if not par_obj.max_det:
        par_obj.max_det = np.max(det3)
    # normalise
    det3 = det3/par_obj.max_det
    det3=det3>par_obj.abs_thr
    par_obj.data_store['maxi_arr'][fileno][time_pt] = {}
    for i in range(imfile.max_z+1):
        par_obj.data_store['maxi_arr'][fileno][time_pt][i] = det3[:, :, i]

    det_bin = det3>par_obj.abs_thr

    det_label,no_obj = measurements.label(det_bin)

    #par_obj.pts = v2._prune_blobs(par_obj.pts, min_distance=[int(self.count_txt_1.text()),int(self.count_txt_2.text()),int(self.count_txt_3.text())])
    pts2keep = []
    print (no_obj)

    det_com = measurements.center_of_mass(det_bin, det_label, range(1,no_obj+1))

    for pt2d in det_com:
        ptuple = tuple(np.round(x).astype('uint') for x in pt2d)

        #determinants of submatrices
        dp = det1[ptuple]
        dp2 = det2[ptuple]
        #dp3 = det3[ptuple]
            #negative definite, therefore maximum (note signs in det calculation)
        #if dp >= 0 and dp2 >= 0: # and dp3>=par_obj.abs_thr:
            #print 'point retained', det[ptuple]<0 , det2[ptuple]<0 , det3[ptuple]<0
        pts2keep.append([ptuple[0], ptuple[1], ptuple[2], 1])
    pts = pts2keep
    par_obj.show_pts = 1

    #Filter those which are not inside the region.
    if par_obj.data_store['roi_stkint_x'][fileno][time_pt].__len__() > 0:
        pts2keep = []

        for i in par_obj.data_store['roi_stkint_x'][fileno][time_pt]:
            for pt2d in pts:

                if pt2d[2] == i:
                    #Find the region of interest.
                    ppt_x = par_obj.data_store['roi_stkint_x'][fileno][time_pt][i]
                    ppt_y = par_obj.data_store['roi_stkint_y'][fileno][time_pt][i]
                    #Reformat to make the path object.
                    pot = []
                    for b in range(0, ppt_x.__len__()):
                        pot.append([ppt_x[b], ppt_y[b]])
                    p = Path(pot)
                    if p.contains_point([pt2d[1], pt2d[0]]) is True:
                        pts2keep.append(pt2d)
        pts = pts2keep

    par_obj.data_store['pts'][fileno][time_pt] = pts


def det_hess_2d(predIm, min_distance):
    gau_Im = filters.gaussian_filter(predIm, (min_distance[0], min_distance[1]))
    y, x = np.gradient(gau_Im, 1)
    xy, xx = np.gradient(x)
    yy, yx = np.gradient(y)

    det = xx*yy-xy*yx

    return det, xx

def count_maxima_2d(par_obj, time_pt, fileno, reset_max):
    #count maxima won't work properly if have selected a random set of Z
    imfile = par_obj.filehandlers[fileno]
    par_obj.min_distance[2] = 0
    det = np.zeros((par_obj.height, par_obj.width, imfile.max_z+1))
    xx = np.zeros((par_obj.height, par_obj.width, imfile.max_z+1))
    for i in range(imfile.max_z+1):
        predIm = par_obj.data_store['pred_arr'][fileno][time_pt][i].astype('float32')
        [deti, xxi] = det_hess_2d(predIm, par_obj.min_distance)
        det[:, :, i] = deti
        xx[:, :, i] = xxi

    # if not already set, create. This is then used for the entire image and all
    #subsequent training. A little hacky, but otherwise the normalisation screws everything up
    if not par_obj.max_det or reset_max==True:
        par_obj.max_det = np.max(det)

    detn = det/par_obj.max_det
    par_obj.data_store['maxi_arr'][fileno][time_pt] = {}
    for i in range(imfile.max_z+1):

        #par_obj.data_store[time_pt]['maxi_arr'][i] = np.sqrt(detn[:,:,i]*par_obj.data_store[time_pt]['pred_arr'][i])
        par_obj.data_store['maxi_arr'][fileno][time_pt][i] = detn[:, :, i]

    pts = peak_local_max(detn, min_distance=par_obj.min_distance, threshold_abs=par_obj.abs_thr)

    pts2keep = []
    for pt2d in pts:

        T = xx[pt2d[0], pt2d[1], pt2d[2]]
        #D=det[pt2d[0],pt2d[1],pt2d[2]]
        # Removes points that are positive definite and therefore minima
        if T > 0: # and D>0:
            print ('point removed')
            pass

        else:
            pts2keep.append([pt2d[0], pt2d[1], pt2d[2], 1])

    pts = pts2keep


    #Filter those which are not inside the region.
    if par_obj.data_store['roi_stkint_x'][fileno][time_pt].__len__() > 0:
        pts2keep = []

        for i in par_obj.data_store['roi_stkint_x'][fileno][time_pt]:
            for pt2d in pts:
                if pt2d[2] == i:
                    #Find the region of interest.
                    ppt_x = par_obj.data_store['roi_stkint_x'][fileno][time_pt][i]
                    ppt_y = par_obj.data_store['roi_stkint_y'][fileno][time_pt][i]
                    #Reformat to make the path object.
                    pot = []
                    for b in range(0, ppt_x.__len__()):
                        pot.append([ppt_x[b], ppt_y[b]])
                    p = Path(pot)
                    if p.contains_point([pt2d[1], pt2d[0]]) is True:
                        pts2keep.append(pt2d)

        pts = pts2keep


    par_obj.data_store['pts'][fileno][time_pt] = pts

def count_maxima_2d_v2(par_obj, time_pt, fileno, reset_max):
    #count maxima won't work properly if have selected a random set of Z
    imfile = par_obj.filehandlers[fileno]
    par_obj.min_distance[2] = 0
    det = np.zeros((par_obj.height, par_obj.width, imfile.max_z+1))
    xx = np.zeros((par_obj.height, par_obj.width, imfile.max_z+1))
    for i in range(imfile.max_z+1):
        predIm = par_obj.data_store['pred_arr'][fileno][time_pt][i]
        [deti, xxi] = det_hess_2d(predIm, par_obj.min_distance)
        det[:, :, i] = deti
        xx[:, :, i] = xxi

    # if not already set, create. This is then used for the entire image and all
    #subsequent training. A little hacky, but otherwise the normalisation screws everything up
    if not par_obj.max_det or reset_max==True:
        par_obj.max_det = np.max(det)

    detn = det/par_obj.max_det
    par_obj.data_store['maxi_arr'][fileno][time_pt] = {}
    for i in range(imfile.max_z+1):
        #par_obj.data_store[time_pt]['maxi_arr'][i] = np.sqrt(detn[:,:,i]*par_obj.data_store[time_pt]['pred_arr'][i])
        par_obj.data_store['maxi_arr'][fileno][time_pt][i] = detn[:, :, i]

    pts = peak_local_max(detn, min_distance=par_obj.min_distance, threshold_abs=par_obj.abs_thr)

    pts2keep = []
    for pt2d in pts:

        T = xx[pt2d[0], pt2d[1], pt2d[2]]
        #D=det[pt2d[0],pt2d[1],pt2d[2]]
        # Removes points that are positive definite and therefore minima
        if T > 0: # and D>0:
            pass
            #print 'point removed'
        else:
            pts2keep.append([pt2d[0], pt2d[1], pt2d[2], 1])

    pts = pts2keep


    #Filter those which are not inside the region.
    if par_obj.data_store['roi_stkint_x'][fileno][time_pt].__len__() > 0:
        pts2keep = []

        for i in par_obj.data_store['roi_stkint_x'][fileno][time_pt]:
            for pt2d in pts:
                if pt2d[2] == i:
                    #Find the region of interest.
                    ppt_x = par_obj.data_store['roi_stkint_x'][fileno][time_pt][i]
                    ppt_y = par_obj.data_store['roi_stkint_y'][fileno][time_pt][i]
                    #Reformat to make the path object.
                    pot = []
                    for b in range(0, ppt_x.__len__()):
                        pot.append([ppt_x[b], ppt_y[b]])
                    p = Path(pot)
                    if p.contains_point([pt2d[1], pt2d[0]]) is True:
                        pts2keep.append(pt2d)

        pts = pts2keep

    thresh = 2

    clusters = hcluster.fclusterdata(pts, thresh, criterion='distance')

    pts2keep = []

    pts = np.array(pts,dtype='float')
    pts[:,2] = pts[:,2] * par_obj.z_cal/1

    for clno in range(1,clusters.max()+1):

        cluster_pts = pts[np.where(clusters == clno)[0],:]
        centroid = np.mean(cluster_pts,axis=0)
        centroid[2] = centroid[2]/par_obj.z_cal*1
        centroid = np.round(centroid).astype('int')
        pts2keep.append([centroid[0],centroid[1],centroid[2],centroid[3]])

    pts=pts2keep

    par_obj.data_store['pts'][fileno][time_pt] = pts

def det_hess_2_5d(predIm, min_distance,):
    gau_Im = filters.gaussian_filter(predIm, min_distance)-filters.gaussian_filter(predIm, [min_distance[0]*2,min_distance[0]*2,min_distance[2]])
    deti=np.zeros_like(predIm)
    xxi=np.zeros_like(predIm)
    for i in range(predIm.shape[2]):
        y, x = np.gradient(gau_Im[:,:,i], 1)
        xy, xx = np.gradient(x)
        yy, yx = np.gradient(y)

        deti[:,:,i] = xx*yy-xy*yx
        xxi[:,:,i] = xx
#    z = np.gradient(deti,axis=2)
    #zz = np.gradient(deti,axis=2)*gau_Im
    return deti, xxi

def count_maxima_2_5d(par_obj, time_pt, fileno, reset_max):
    #count maxima won't work properly if have selected a random set of Z
    imfile = par_obj.filehandlers[fileno]

    predMtx = np.zeros((par_obj.height, par_obj.width, imfile.max_z+1))
    for i in range(imfile.max_z+1):
        predMtx[:, :, i] = par_obj.data_store['pred_arr'][fileno][time_pt][i]


    [det, xx] = det_hess_2_5d(predMtx, par_obj.min_distance)
    #[det2, xx] = det_hess_2_5d(predMtx, [par_obj.min_distance[0:2],par_obj.min_distance[2]/2])
    #[det3, xx] = det_hess_2_5d(predMtx, [par_obj.min_distance[0:2],par_obj.min_distance[2]*2])
    #det=det2+det+det3

    # if not already set, create. This is then used for the entire image and all
    #subsequent training. A little hacky, but otherwise the normalisation screws everything up
    if not par_obj.max_det or reset_max==True:
        par_obj.max_det = np.max(det)

    detn = det/par_obj.max_det
    par_obj.data_store['maxi_arr'][fileno][time_pt] = {}
    for i in range(imfile.max_z+1):
        #par_obj.data_store[time_pt]['maxi_arr'][i] = np.sqrt(detn[:,:,i]*par_obj.data_store[time_pt]['pred_arr'][i])
        par_obj.data_store['maxi_arr'][fileno][time_pt][i] = detn[:, :, i]

    pts = peak_local_max(detn, min_distance=par_obj.min_distance, threshold_abs=par_obj.abs_thr)

    pts2keep = []
    for pt2d in pts:

        T = xx[pt2d[0], pt2d[1], pt2d[2]]
        #D = z[pt2d[0],pt2d[1],pt2d[2]]
        # Removes points that are positive definite and therefore minima
        if T > 0:# and D>0:
            pass
            #print 'point removed'
        else:
            pts2keep.append([pt2d[0], pt2d[1], pt2d[2], 1])

    pts = pts2keep


    #Filter those which are not inside the region.
    if par_obj.data_store['roi_stkint_x'][fileno][time_pt].__len__() > 0:
        pts2keep = []

        for i in par_obj.data_store['roi_stkint_x'][fileno][time_pt]:
            for pt2d in pts:
                if pt2d[2] == i:
                    #Find the region of interest.
                    ppt_x = par_obj.data_store['roi_stkint_x'][fileno][time_pt][i]
                    ppt_y = par_obj.data_store['roi_stkint_y'][fileno][time_pt][i]
                    #Reformat to make the path object.
                    pot = []
                    for b in range(0, ppt_x.__len__()):
                        pot.append([ppt_x[b], ppt_y[b]])
                    p = Path(pot)
                    if p.contains_point([pt2d[1], pt2d[0]]) is True:
                        pts2keep.append(pt2d)

        pts = pts2keep


    par_obj.data_store['pts'][fileno][time_pt] = pts

"""rankorder.py - convert an image of any type to an image of ints whose
pixels have an identical rank order compared to the original image

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentstky
"""
'''
def rank_order(image):
    """Return an image of the same shape where each pixel is the
    index of the pixel value in the ascending order of the unique
    values of `image`, aka the rank-order value.

    Parameters
    ----------
    image: ndarray

    Returns
    -------
    labels: ndarray of type np.uint32, of shape image.shape
        New array where each pixel has the rank-order value of the
        corresponding pixel in `image`. Pixel values are between 0 and
        n - 1, where n is the number of distinct unique values in
        `image`.

    original_values: 1-D ndarray
        Unique original values of `image`

    Examples
    --------
    >>> a = np.array([[1, 4, 5], [4, 4, 1], [5, 1, 1]])
    >>> a
    array([[1, 4, 5],
           [4, 4, 1],
           [5, 1, 1]])
    >>> rank_order(a)
    (array([[0, 1, 2],
           [1, 1, 0],
           [2, 0, 0]], dtype=uint32), array([1, 4, 5]))
    >>> b = np.array([-1., 2.5, 3.1, 2.5])
    >>> rank_order(b)
    (array([0, 1, 2, 1], dtype=uint32), array([-1. ,  2.5,  3.1]))
    """
    flat_image = image.ravel()
    sort_order = flat_image.argsort().astype('uint32')
    flat_image = flat_image[sort_order]
    sort_rank = np.zeros_like(sort_order)
    is_different = flat_image[:-1] != flat_image[1:]
    np.cumsum(is_different, out=sort_rank[1:])
    original_values = np.zeros((sort_rank[-1] + 1,), image.dtype)
    original_values[0] = flat_image[0]
    original_values[1:] = flat_image[1:][is_different]
    int_image = np.zeros_like(sort_order)
    int_image[sort_order] = sort_rank
    return (int_image.reshape(image.shape), original_values)

def _blob_overlap(blob1, blob2, min_distance):
    """Finds the overlapping area fraction between two blobs.
    Returns a float representing fraction of overlapped area.
    """

    d1 = abs(blob1[0] - blob2[0]) > min_distance[0]
    d2 = abs(blob1[1] - blob2[1]) > min_distance[1]
    d3 = abs(blob1[2] - blob2[2]) > min_distance[2]

    if d1 is False or d2 is False or d3 is False:
        #overlap detected

        return True

    return False

def _prune_blobs(blobs_array, min_distance):
    """Eliminated blobs with area overlap.

    Parameters
    ----------
    blobs_array : ndarray
        A 2d array with each row representing 3 values, ``(y,x,sigma)``
        where ``(y,x)`` are coordinates of the blob and ``sigma`` is the
        standard deviation of the Gaussian kernel which detected the blob.
    overlap : float
        A value between 0 and 1. If the fraction of area overlapping for 2
        blobs is greater than `overlap` the smaller blob is eliminated.

    Returns
    -------
    A : ndarray
        `array` with overlapping blobs removed.
    """

    # iterating again might eliminate more blobs, but one iteration suffices
    # for most cases
    for blob1, blob2 in itt.combinations(blobs_array, 2):

        if _blob_overlap(blob1, blob2, min_distance) is True:
            blob2[2] = -1

            #if blob1[2] > blob2[2]:
            #    blob2[2] = -1
            #else:
            #    blob1[2] = -1

    # return blobs_array[blobs_array[:, 2] > 0]
    return np.array([b for b in blobs_array if b[2] > 0])
'''
def peak_local_max(image, min_distance=10, threshold_abs=0, threshold_rel=0.1,
                   exclude_border=False, indices=True, num_peaks=np.inf,
                   footprint=None, labels=None):
    """
    Find peaks in an image, and return them as coordinates or a boolean array.

    Peaks are the local maxima in a region of `2 * min_distance + 1`
    (i.e. peaks are separated by at least `min_distance`).

    NOTE: If peaks are flat (i.e. multiple adjacent pixels have identical
    intensities), the coordinates of all such pixels are returned.

    Parameters
    ----------
    image : ndarray of floats
        Input image.
    min_distance : int
        Minimum number of pixels separating peaks in a region of `2 *
        min_distance + 1` (i.e. peaks are separated by at least
        `min_distance`). If `exclude_border` is True, this value also excludes
        a border `min_distance` from the image boundary.
        To find the maximum number of peaks, use `min_distance=1`.
    threshold_abs : float
        Minimum intensity of peaks.
    threshold_rel : float
        Minimum intensity of peaks calculated as `max(image) * threshold_rel`.
    exclude_border : bool
        If True, `min_distance` excludes peaks from the border of the image as
        well as from each other.
    indices : bool
        If True, the output will be an array representing peak coordinates.
        If False, the output will be a boolean array shaped as `image.shape`
        with peaks present at True elements.
    num_peaks : int
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` peaks based on highest peak intensity.
    footprint : ndarray of bools, optional
        If provided, `footprint == 1` represents the local region within which
        to search for peaks at every point in `image`.  Overrides
        `min_distance`, except for border exclusion if `exclude_border=True`.
    labels : ndarray of ints, optional
        If provided, each unique region `labels == value` represents a unique
        region to search for peaks. Zero is reserved for background.

    Returns
    -------
    output : ndarray or ndarray of bools

        * If `indices = True`  : (row, column, ...) coordinates of peaks.
        * If `indices = False` : Boolean array shaped like `image`, with peaks
          represented by True values.

    Notes
    -----
    The peak local maximum function returns the coordinates of local peaks
    (maxima) in a image. A maximum filter is used for finding local maxima.
    This operation dilates the original image. After comparison between
    dilated and original image, peak_local_max function returns the
    coordinates of peaks where dilated image = original.

    Examples
    --------
    >>> img1 = np.zeros((7, 7))
    >>> img1[3, 4] = 1
    >>> img1[3, 2] = 1.5
    >>> img1
    array([[ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  1.5,  0. ,  1. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]])

    >>> peak_local_max(img1, min_distance=1)
    array([[3, 2],
           [3, 4]])

    >>> peak_local_max(img1, min_distance=2)
    array([[3, 2]])

    >>> img2 = np.zeros((20, 20, 20))
    >>> img2[10, 10, 10] = 1
    >>> peak_local_max(img2, exclude_border=False)
    array([[10, 10, 10]])

    """
    out = np.zeros_like(image, dtype=np.bool)


    if np.all(image == image.flat[0]):
        if indices is True:
            return []
        else:
            return out

    image = image.copy()
    # Non maximum filter
    if footprint is not None:
        image_max = filters.maximum_filter(image, footprint=footprint, mode='constant')
    else:
        size = np.array(min_distance)*2.3548
        image_max = filters.maximum_filter(image, size=size, mode='constant')
    mask = (image == image_max)
    image *= mask

    if exclude_border:
        # zero out the image borders
        for i in range(image.ndim):

            image = image.swapaxes(0, i)
            min_d = np.floor(min_distance[i])

            image[:min_d] = 0
            image[-min_d:] = 0
            image = image.swapaxes(0, i)

    # find top peak candidates above a threshold
    peak_threshold = max(np.max(image.ravel()) * threshold_rel, threshold_abs)

    # get coordinates of peaks
    coordinates = np.argwhere(image > peak_threshold)

    if coordinates.shape[0] > num_peaks:
        intensities = image.flat[np.ravel_multi_index(coordinates.transpose(), image.shape)]
        idx_maxsort = np.argsort(intensities)[::-1]
        coordinates = coordinates[idx_maxsort][:num_peaks]

    if indices is True:
        return coordinates
    else:
        nd_indices = tuple(coordinates.T)
        out[nd_indices] = True


        return out
