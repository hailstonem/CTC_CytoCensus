"""
Created on Wed Mar 30 14:25:35 2016

@author: martin

Feature calculation methods (moved from v2_functions for clarity)
 makes adding new features more straightforward
 might be nice to add subclasses to this
"""
import scipy

from skimage import feature as skfeat

from skimage import exposure

import skimage
import numpy as np

# from v2_functions import get_tiff_slice
from scipy.ndimage.interpolation import shift

from sklearn import ensemble
from sklearn import tree
from sklearn.pipeline import Pipeline

from scipy.ndimage.filters import generic_filter


def RF(par_obj, RF_type="ETR"):
    """Choose regression method. Must implement fit and predict. Can use sklearn pipeline"""
    if RF_type == "ETR":
        method = ensemble.ExtraTreesRegressor(
            par_obj.num_of_tree,
            max_depth=par_obj.max_depth,
            min_samples_split=par_obj.min_samples_split,
            min_samples_leaf=par_obj.min_samples_leaf,
            max_features=par_obj.max_features,
            bootstrap=True,
            n_jobs=-1,
        )
    elif RF_type == "GBR":
        method = ensemble.GradientBoostingRegressor(
            loss="ls",
            learning_rate=0.01,
            n_estimators=par_obj.num_of_tree,
            max_depth=par_obj.max_depth,
            min_samples_split=par_obj.min_samples_split,
            min_samples_leaf=par_obj.min_samples_leaf,
            max_features=par_obj.max_features,
        )
    elif RF_type == "GBR2":
        method = ensemble.GradientBoostingRegressor(
            loss="lad",
            learning_rate=0.1,
            n_estimators=par_obj.num_of_tree,
            max_depth=par_obj.max_depth,
            min_samples_split=par_obj.min_samples_split,
            min_samples_leaf=par_obj.min_samples_leaf,
            max_features=par_obj.max_features,
        )
    elif RF_type == "ABR":
        method = ensemble.AdaBoostRegressor(
            base_estimator=ensemble.ExtraTreesRegressor(
                10,
                max_depth=par_obj.max_depth,
                min_samples_split=par_obj.min_samples_split,
                min_samples_leaf=par_obj.min_samples_leaf,
                max_features=par_obj.max_features,
                bootstrap=True,
                n_jobs=-1,
            ),
            n_estimators=3,
            learning_rate=1.0,
            loss="square",
        )
    return method


def get_feature_lengths(feature_type):
    """Choose feature sets, accepts feature types 'basic' 'pyramid' 'imhist' 'dual'"""
    # dictionary of feature sets
    # can add arbitrary feature sets by defining a name, length, and function
    # that accepts two arguments
    feature_dict = {
        "basic": [13, local_shape_features_basic],
        "fine": [21, local_shape_features_fine],
        "pyramid": [26, local_shape_features_pyramid],
        "histeq": [26, local_shape_features_fine_imhist],
        "radial": [26, local_shape_features_radial],
        "gradient": [21, local_shape_features_gradient],
        "std": [26, local_shape_features_std],
        "daisy": [200, local_shape_features_daisy],
        "minmax": [31, local_shape_features_minmax],
    }

    if feature_type in feature_dict:
        feat_length = feature_dict[feature_type][0]
        feat_func = feature_dict[feature_type][1]
    else:
        raise Exception("Feature set not found")

    return feat_length, feat_func


def feature_create_threadable(par_obj, imRGB):
    """Creates features based on input image"""
    # get number of features
    [feat_length, feat_func] = get_feature_lengths(par_obj.feature_type)

    # preallocate array
    """feat = np.zeros(((int(par_obj.crop_y2) - int(par_obj.crop_y1)), (int(par_obj.crop_x2) -
    int(par_obj.crop_x1)), feat_length * (par_obj.ch_active.__len__())))"""
    feat = np.zeros(
        (par_obj.height, par_obj.width, feat_length * (par_obj.ch_active.__len__()))
    )

    if par_obj.numCH == 1:
        imG = imRGB[:, :].astype(np.float32) / par_obj.tiffarray_typemax

        feat = feat_func(imG, par_obj.feature_scale)
    else:
        for b in range(0, par_obj.ch_active.__len__()):
            imG = imRGB[:, :, b].astype(np.float32) / par_obj.tiffarray_typemax
            feat[:, :, (b * feat_length) : ((b + 1) * feat_length)] = feat_func(
                imG, par_obj.feature_scale
            )
    """        if b==1:#dirty hack to test
                imG = imRGB[:,:,b].astype(np.float32)*imRGB[:,:,b].astype(np.float32)
                feat[:,:,(2*feat_length):((2+1)*feat_length)]=feat_func(imG,par_obj.feature_scale)"""

    return feat


def feature_create_threadable_auto(par_obj, imno, tpt, zslice):
    # allows auto-context based features to be included
    # currently slow, could optimise to calculate more efficiently
    [_feat_length, feat_func] = get_feature_lengths(par_obj.feature_type)
    feat = feat_func(
        par_obj.data_store["pred_arr"][imno][tpt][zslice], par_obj.feature_scale
    )
    return feat


def auto_context_features(feat_array):
    # pattern=[1,2,3,5,7,10,12,15,20,25,30,35,40,45,50,60,70,80,90,100]
    pattern = [1, 2, 4, 8, 16, 32, 64, 128]
    feat_list = []
    for xshift in pattern:
        for yshift in pattern:
            a = shift(feat_array, (xshift, 0), order=0, cval=0)
            b = shift(feat_array, (-xshift, 0), order=0, cval=0)
            c = shift(feat_array, (yshift, 0), order=0, cval=0)
            d = shift(feat_array, (-yshift, 0), order=0, cval=0)
            feat_list.append(a)
            feat_list.append(b)
            feat_list.append(c)
            feat_list.append(d)
    return feat_list


def calculate_skfeat_eigenvalues(im, s):
    # designed to be similar to vigra use of vigra.filters.structureTensorEigenvalues
    # slight differences in results
    gim = scipy.ndimage.filters.gaussian_filter(im, 0.5 * s, mode="reflect", truncate=4)
    x, y, z = skfeat.structure_tensor(gim, 1 * s, mode="reflect")
    st_skfeat = skfeat.structure_tensor_eigvals(x, y, z)

    return ((st_skfeat[0] / (50 + s)), (st_skfeat[1] / (50 + s)))


def local_shape_features_fine(im, scaleStart):
    """ Creates features. Exactly as in the Luca Fiaschi paper but on 5 scales, and a truncated gaussian"""
    s = scaleStart

    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC, imSizeR, 21))

    st08 = calculate_skfeat_eigenvalues(im, s * 2)
    st16 = calculate_skfeat_eigenvalues(im, s * 4)
    st32 = calculate_skfeat_eigenvalues(im, s * 8)
    st64 = calculate_skfeat_eigenvalues(im, s * 16)
    st128 = calculate_skfeat_eigenvalues(im, s * 32)

    f[:, :, 0] = im
    f[:, :, 1] = scipy.ndimage.gaussian_gradient_magnitude(im, s, truncate=2.5)
    f[:, :, 2] = st08[0]
    f[:, :, 3] = st08[1]
    f[:, :, 4] = scipy.ndimage.gaussian_laplace(im, s, truncate=2.5)

    f[:, :, 5] = scipy.ndimage.gaussian_gradient_magnitude(im, s * 2, truncate=2.5)
    f[:, :, 6] = st16[0]
    f[:, :, 7] = st16[1]
    f[:, :, 8] = scipy.ndimage.gaussian_laplace(im, s * 2, truncate=2.5)

    f[:, :, 9] = scipy.ndimage.gaussian_gradient_magnitude(im, s * 4, truncate=2.5)
    f[:, :, 10] = st32[0]
    f[:, :, 11] = st32[1]
    f[:, :, 12] = scipy.ndimage.gaussian_laplace(im, s * 4, truncate=2.5)

    f[:, :, 13] = scipy.ndimage.gaussian_gradient_magnitude(im, s * 8, truncate=2.5)
    f[:, :, 14] = st64[0]
    f[:, :, 15] = st64[1]
    f[:, :, 16] = scipy.ndimage.gaussian_laplace(im, s * 8, truncate=2.5)

    f[:, :, 17] = scipy.ndimage.gaussian_gradient_magnitude(im, s * 16, truncate=2.5)
    f[:, :, 18] = st128[0]
    f[:, :, 19] = st128[1]
    f[:, :, 20] = scipy.ndimage.gaussian_laplace(im, s * 16, truncate=2.5)

    return f


def local_shape_features_gradient(im, scaleStart):
    """ Creates features. Exactly as in the Luca Fiaschi paper but on 5 scales, and a truncated gaussian"""
    s = scaleStart

    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC, imSizeR, 21))

    f[:, :, 0] = im
    f[:, :, 1] = scipy.ndimage.gaussian_gradient_magnitude(im, s, truncate=2.5)
    f[:, :, 2] = np.gradient(scipy.ndimage.gaussian_filter(im, s), axis=0)
    f[:, :, 3] = np.gradient(scipy.ndimage.gaussian_filter(im, s), axis=1)
    f[:, :, 4] = scipy.ndimage.gaussian_laplace(im, s, truncate=2.5)

    f[:, :, 5] = scipy.ndimage.gaussian_gradient_magnitude(im, s * 2, truncate=2.5)
    f[:, :, 6] = np.gradient(scipy.ndimage.gaussian_filter(im, s * 2), axis=0)
    f[:, :, 7] = np.gradient(scipy.ndimage.gaussian_filter(im, s * 2), axis=1)
    f[:, :, 8] = scipy.ndimage.gaussian_laplace(im, s * 2, truncate=2.5)

    f[:, :, 9] = scipy.ndimage.gaussian_gradient_magnitude(im, s * 4, truncate=2.5)
    f[:, :, 10] = np.gradient(scipy.ndimage.gaussian_filter(im, s * 4), axis=0)
    f[:, :, 11] = np.gradient(scipy.ndimage.gaussian_filter(im, s * 4), axis=1)
    f[:, :, 12] = scipy.ndimage.gaussian_laplace(im, s * 4, truncate=2.5)

    f[:, :, 13] = scipy.ndimage.gaussian_gradient_magnitude(im, s * 8, truncate=2.5)
    f[:, :, 14] = np.gradient(scipy.ndimage.gaussian_filter(im, s * 8), axis=0)
    f[:, :, 15] = np.gradient(scipy.ndimage.gaussian_filter(im, s * 8), axis=1)
    f[:, :, 16] = scipy.ndimage.gaussian_laplace(im, s * 8, truncate=2.5)

    f[:, :, 17] = scipy.ndimage.gaussian_gradient_magnitude(im, s * 16, truncate=2.5)
    f[:, :, 18] = np.gradient(scipy.ndimage.gaussian_filter(im, s * 16), axis=0)
    f[:, :, 19] = np.gradient(scipy.ndimage.gaussian_filter(im, s * 16), axis=1)
    f[:, :, 20] = scipy.ndimage.gaussian_laplace(im, s * 16, truncate=2.5)

    return f


def frst2d(image, radii, alpha, stdFactor, mode, discard=0):

    image = image.astype("float32")
    sizex, sizey = image.shape

    gx = scipy.ndimage.filters.sobel(image, axis=0)
    gy = scipy.ndimage.filters.sobel(image, axis=1)

    dark = False
    bright = False

    if mode == "bright":
        bright = True
    elif mode == "dark":
        dark = True
    elif mode == "both":
        bright = True
        dark = True
    else:
        raise Exception("invalid mode!")

    maxRadius = int(np.ceil(max(radii)))

    filtered = np.zeros((sizex + 2 * maxRadius, sizey + 2 * maxRadius))
    S_n = np.zeros((sizex + 2 * maxRadius, sizey + 2 * maxRadius))
    # S = np.zeros((len(radii), sizex + 2 * maxRadius, sizey + 2 * maxRadius))
    # t0=time.time()
    gnormMat = np.hypot(gx, gy)

    glx = gx / (gnormMat + 0.0000001)
    gly = gy / (gnormMat + 0.0000001)

    indices = np.nonzero(gnormMat > (gnormMat.max() * discard))
    for radiusIndex, n in enumerate(radii):
        glxn = np.round(n * glx[indices[0], indices[1]]).astype("int")
        glyn = np.round(n * gly[indices[0], indices[1]]).astype("int")

        O_n = np.zeros_like(filtered)
        M_n = np.zeros_like(filtered)
        # O_n[idx2change.flat,idy2change.flat] = O_n[idx2change.flat,idy2change.flat] +1
        # M_n[idx2change.flat,idy2change.flat] = M_n[idx2change.flat,idy2change.flat] +gnormMat[indices[0],indices[1]]
        # O_n[idx2change_neg.flat,idx2change_neg.flat] = O_n[idx2change_neg.flat,idx2change_neg.flat] -1
        # M_n[idx2change_neg.flat,idx2change_neg.flat] = M_n[idx2change_neg.flat,idx2change_neg.flat] -gnormMat[indices[0],indices[1]]
        if bright:
            idx2change = indices[0] + glxn + maxRadius
            idy2change = indices[1] + glyn + maxRadius
            np.add.at(O_n, (idx2change.flat, idy2change.flat), 1)
            np.add.at(M_n, (idx2change.flat, idy2change.flat), gnormMat[indices])

        if dark:
            idx2change_neg = indices[0] - glxn + maxRadius
            idy2change_neg = indices[1] - glyn + maxRadius
            np.add.at(O_n, (idx2change_neg.flat, idy2change_neg.flat), -1)
            np.add.at(
                M_n, (idx2change_neg.flat, idy2change_neg.flat), -gnormMat[indices]
            )

        O_n = np.abs(O_n)
        M_n = np.abs(M_n)
        if n == 1:
            O_n = O_n / 10
        else:
            O_n = O_n / 10
        # print O_n.max()
        # print O_n.mean()
        S_n += np.power(O_n, alpha) * M_n

    output = S_n[maxRadius : sizex + maxRadius, maxRadius : sizey + maxRadius]

    return output


def local_shape_features_radial(im, scaleStart):
    """ Creates features based on those in the Luca Fiaschi paper but on 5 scales independent of object size,
    but using a gaussian pyramid to calculate at multiple scales more efficiently
    and then linear upsampling to original image scale
    also includes gaussian smoothed image
    """
    # Smoothing and scale parameters chosen to approximate those in 'fine'
    # features

    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC, imSizeR, 26))
    f[:, :, 0] = im
    frst = frst2d(im, range(1, 50, 3), 1.5, 0.1, "both", discard=0.01)
    # create pyramid structure
    pyr = skimage.transform.pyramid_gaussian(
        im, sigma=1.5, max_layer=5, downscale=2, multichannel=False
    )

    a = im
    for layer in range(0, 5):

        # calculate scale
        scale = [
            float(im.shape[0]) / float(a.shape[0]),
            float(im.shape[1]) / float(a.shape[1]),
        ]

        # create features
        lap = scipy.ndimage.filters.laplace(a)

        [m, n] = np.gradient(a)
        ggm = np.hypot(m, n)

        x, y, z = skfeat.structure_tensor(a, 1)
        st = skfeat.structure_tensor_eigvals(x, y, z)

        # upsample features to original image
        lap = scipy.ndimage.interpolation.zoom(lap, scale, order=1)
        ggm = scipy.ndimage.interpolation.zoom(ggm, scale, order=1)
        st0 = scipy.ndimage.interpolation.zoom(st[0], scale, order=1)
        st1 = scipy.ndimage.interpolation.zoom(st[1], scale, order=1)

        # up = scipy.ndimage.interpolation.zoom(a, scale, order=1)

        f[:, :, layer * 5 + 1] = lap
        f[:, :, layer * 5 + 2] = ggm
        f[:, :, layer * 5 + 3] = st0
        f[:, :, layer * 5 + 4] = st1
        f[:, :, layer * 5 + 5] = scipy.ndimage.gaussian_filter(frst, (layer + 1) * 2)

        # get next layer
        a = next(pyr)

    return f


def strided_sliding_std_dev(data, radius=5):
    windowed = rolling_window(data, (2 * radius, 2 * radius))
    shape = windowed.shape
    windowed = windowed.reshape(shape[0], shape[1], -1)
    return windowed.std(axis=-1)


def rolling_window(a, window):
    """Takes a numpy array *a* and a sequence of (or single) *window* lengths
    and returns a view of *a* that represents a moving window."""
    if not hasattr(window, "__iter__"):
        return rolling_window_lastaxis(a, window)
    for i, win in enumerate(window):
        if win > 1:
            a = a.swapaxes(i, -1)
            a = rolling_window_lastaxis(a, win)
            a = a.swapaxes(-2, i)
    return a


def rolling_window_lastaxis(a, window):
    """Directly taken from Erik Rigtorp's post to numpy-discussion.
    <http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>"""
    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > a.shape[-1]:
        raise ValueError("`window` is too long.")
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def local_shape_features_std(im, scaleStart):
    """ Creates features based on those in the Luca Fiaschi paper but on 5 scales independent of object size,
    but using a gaussian pyramid to calculate at multiple scales more efficiently
    and then linear upsampling to original image scale
    also includes gaussian smoothed image
    """
    # Smoothing and scale parameters chosen to approximate those in 'fine'
    # features

    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC, imSizeR, 26))
    f[:, :, 0] = im

    # create pyramid structure
    pyr = skimage.transform.pyramid_gaussian(
        im, sigma=1.5, max_layer=5, downscale=2, multichannel=False
    )

    a = im
    for layer in range(0, 5):

        # calculate scale
        scale = [
            float(im.shape[0]) / float(a.shape[0]),
            float(im.shape[1]) / float(a.shape[1]),
        ]

        # create features
        lap = scipy.ndimage.filters.laplace(a)

        [m, n] = np.gradient(a)
        ggm = np.hypot(m, n)

        x, y, z = skfeat.structure_tensor(a, 1)
        st = skfeat.structure_tensor_eigvals(x, y, z)

        up = np.pad(strided_sliding_std_dev(a, radius=3), (3, 0), "edge")

        # upsample features to original image
        lap = scipy.ndimage.interpolation.zoom(lap, scale, order=1)
        ggm = scipy.ndimage.interpolation.zoom(ggm, scale, order=1)
        st0 = scipy.ndimage.interpolation.zoom(st[0], scale, order=1)
        st1 = scipy.ndimage.interpolation.zoom(st[1], scale, order=1)

        up = scipy.ndimage.interpolation.zoom(up, scale, order=1)

        f[:, :, layer * 5 + 1] = lap
        f[:, :, layer * 5 + 2] = ggm
        f[:, :, layer * 5 + 3] = st0
        f[:, :, layer * 5 + 4] = st1
        f[:, :, layer * 5 + 5] = up

        # get next layer
        a = next(pyr)

    return f


def local_shape_features_minmax(im, scaleStart):
    """ Creates features based on those in the Luca Fiaschi paper but on 5 scales independent of object size,
    but using a gaussian pyramid to calculate at multiple scales more efficiently
    and then linear upsampling to original image scale
    also includes gaussian smoothed image
    """
    # Smoothing and scale parameters chosen to approximate those in 'fine'
    # features

    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC, imSizeR, 31))
    f[:, :, 0] = im

    # create pyramid structure
    pyr = skimage.transform.pyramid_gaussian(
        im, sigma=1.5, max_layer=5, downscale=2, multichannel=False
    )

    a = im
    for layer in range(0, 5):

        # calculate scale
        scale = [
            float(im.shape[0]) / float(a.shape[0]),
            float(im.shape[1]) / float(a.shape[1]),
        ]

        # create features
        lap = scipy.ndimage.filters.laplace(a)

        [m, n] = np.gradient(a)
        ggm = np.hypot(m, n)

        x, y, z = skfeat.structure_tensor(a, 1)
        st = skfeat.structure_tensor_eigvals(x, y, z)

        maxim = scipy.ndimage.maximum_filter(a, 3)
        minim = scipy.ndimage.minimum_filter(a, 3)

        # upsample features to original image
        lap = scipy.ndimage.interpolation.zoom(lap, scale, order=1)
        ggm = scipy.ndimage.interpolation.zoom(ggm, scale, order=1)
        st0 = scipy.ndimage.interpolation.zoom(st[0], scale, order=1)
        st1 = scipy.ndimage.interpolation.zoom(st[1], scale, order=1)
        maxim = scipy.ndimage.interpolation.zoom(maxim, scale, order=1)
        minim = scipy.ndimage.interpolation.zoom(minim, scale, order=1)

        f[:, :, layer * 6 + 1] = lap
        f[:, :, layer * 6 + 2] = ggm
        f[:, :, layer * 6 + 3] = st0
        f[:, :, layer * 6 + 4] = st1
        f[:, :, layer * 6 + 5] = minim
        f[:, :, layer * 6 + 6] = maxim
        # get next layer
        a = next(pyr)

    return f


def local_shape_features_daisy(im, scaleStart):
    radius = 6
    feat = skfeat.daisy(
        np.pad(im, radius, mode="edge"), step=2, radius=radius, normalization="off"
    )
    feat = feat.reshape((feat.shape[0], feat.shape[1], -1))
    print(feat.shape)
    f = scipy.ndimage.interpolation.zoom(
        feat,
        (float(im.shape[0]) / feat.shape[0], float(im.shape[1]) / feat.shape[1], 1),
        order=1,
    )

    return f


def local_shape_features_pyramid(im, scaleStart):
    """ Creates features based on those in the Luca Fiaschi paper but on 5 scales independent of object size,
    but using a gaussian pyramid to calculate at multiple scales more efficiently
    and then linear upsampling to original image scale
    also includes gaussian smoothed image
    """
    # Smoothing and scale parameters chosen to approximate those in 'fine'
    # features

    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC, imSizeR, 26))
    f[:, :, 0] = im

    # create pyramid structure
    pyr = skimage.transform.pyramid_gaussian(
        im, sigma=1.5, max_layer=5, downscale=2, multichannel=False
    )

    a = im
    for layer in range(0, 5):

        # calculate scale
        scale = [
            float(im.shape[0]) / float(a.shape[0]),
            float(im.shape[1]) / float(a.shape[1]),
        ]

        # create features
        lap = scipy.ndimage.filters.laplace(a)

        [m, n] = np.gradient(a)
        ggm = np.hypot(m, n)

        x, y, z = skfeat.structure_tensor(a, 1)
        st = skfeat.structure_tensor_eigvals(x, y, z)

        # upsample features to original image
        lap = scipy.ndimage.interpolation.zoom(lap, scale, order=1)
        ggm = scipy.ndimage.interpolation.zoom(ggm, scale, order=1)
        st0 = scipy.ndimage.interpolation.zoom(st[0], scale, order=1)
        st1 = scipy.ndimage.interpolation.zoom(st[1], scale, order=1)

        up = scipy.ndimage.interpolation.zoom(a, scale, order=1)

        f[:, :, layer * 5 + 1] = lap
        f[:, :, layer * 5 + 2] = ggm
        f[:, :, layer * 5 + 3] = st0
        f[:, :, layer * 5 + 4] = st1
        f[:, :, layer * 5 + 5] = up

        # get next layer
        a = next(pyr)

    return f


def local_shape_features_fine_imhist(im, scaleStart):
    """As per pyramid features but with histogram equalisation"""

    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC, imSizeR, 26))
    im = exposure.equalize_hist(im)
    # im=exposure.equalize_adapthist(im, kernel_size=5)
    f[:, :, 0] = im

    # set up pyramid
    pyr = skimage.transform.pyramid_gaussian(
        im, sigma=1.5, max_layer=5, downscale=2, multichannel=False
    )
    a = im

    for layer in range(0, 5):

        scale = [
            float(im.shape[0]) / float(a.shape[0]),
            float(im.shape[1]) / float(a.shape[1]),
        ]

        lap = scipy.ndimage.filters.laplace(a)

        [m, n] = np.gradient(a)
        ggm = np.hypot(m, n)

        x, y, z = skfeat.structure_tensor(a, 1)
        st = skfeat.structure_tensor_eigvals(x, y, z)

        lap = scipy.ndimage.interpolation.zoom(lap, scale, order=1)
        ggm = scipy.ndimage.interpolation.zoom(ggm, scale, order=1)
        st0 = scipy.ndimage.interpolation.zoom(st[0], scale, order=1)
        st1 = scipy.ndimage.interpolation.zoom(st[1], scale, order=1)
        up = scipy.ndimage.interpolation.zoom(a, scale, order=1)

        f[:, :, layer * 5 + 1] = lap
        f[:, :, layer * 5 + 2] = ggm
        f[:, :, layer * 5 + 3] = st0
        f[:, :, layer * 5 + 4] = st1
        f[:, :, layer * 5 + 5] = up
        a = next(pyr)
    return f


def local_shape_features_basic(im, scaleStart):
    """Exactly as in the Luca Fiaschi paper. Calculates features at 3 scales dependent on scale parameter"""
    s = scaleStart

    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC, imSizeR, 13))

    st08 = calculate_skfeat_eigenvalues(im, s * 2)
    st16 = calculate_skfeat_eigenvalues(im, s * 4)
    st32 = calculate_skfeat_eigenvalues(im, s * 8)

    f[:, :, 0] = im
    f[:, :, 1] = scipy.ndimage.gaussian_gradient_magnitude(im, s, truncate=2.5)
    f[:, :, 2] = st08[0]
    f[:, :, 3] = st08[1]
    f[:, :, 4] = scipy.ndimage.gaussian_laplace(im, s, truncate=2.5)

    f[:, :, 5] = scipy.ndimage.gaussian_gradient_magnitude(im, s * 2, truncate=2.5)
    f[:, :, 6] = st16[0]
    f[:, :, 7] = st16[1]
    f[:, :, 8] = scipy.ndimage.gaussian_laplace(im, s * 2, truncate=2.5)

    f[:, :, 9] = scipy.ndimage.gaussian_gradient_magnitude(im, s * 4, truncate=2.5)
    f[:, :, 10] = st32[0]
    f[:, :, 11] = st32[1]
    f[:, :, 12] = scipy.ndimage.gaussian_laplace(im, s * 4, truncate=2.5)

    return f

