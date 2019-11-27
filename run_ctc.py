import os
import pickle
import argparse

import numpy as np
import scipy
import scipy.ndimage.filters as filters
import multiprocessing
import multiprocessing.dummy

multiprocessing.freeze_support()
import tifffile
import skimage
import skimage.exposure
import skimage.segmentation
from skimage.transform import resize


# CytoCensus imports
from features import local_features as lf
from parameters import parameter_object
from functions.maxima import det_hess_3d, peak_local_max


def main():

    parser = argparse.ArgumentParser(
        description="Run CytoCensus Detection using pretrained model on CTC dataset"
    )
    parser.add_argument("CURR_DIR", metavar="D", type=str)
    parser.add_argument("FOLDER", metavar="F", type=str)
    parser.add_argument("SEGMENTATION_OBJECT_SIZE", metavar="O", type=int)
    parser.add_argument("SEGMENTATION_ITERATIONS", metavar="I", type=int)
    parser.add_argument("SPLIT", metavar="W", type=bool, default=True)
    parser.add_argument("DOWNSAMPLE", metavar="S", type=int, default=2)
    parser.add_argument("Z_ADJUST", metavar="Z", type=int)
    parser.add_argument("MODELNAME", metavar="M", type=str)
    args = parser.parse_args()

    # parameters
    CURR_DIR = args.CURR_DIR
    FOLDER = args.FOLDER
    SEGMENTATION_OBJECT_SIZE = args.SEGMENTATION_OBJECT_SIZE
    SEGMENTATION_ITERATIONS = args.SEGMENTATION_ITERATIONS
    SPLIT = args.SPLIT
    DOWNSAMPLE = args.DOWNSAMPLE
    Z_ADJUST = args.Z_ADJUST
    MODELNAME = args.MODELNAME

    directory = CURR_DIR + FOLDER + "\\"
    filenames = os.listdir(directory)
    print("Files to be processed: " + str(filenames))

    # Create parameter object
    par_obj = parameter_object.ParameterClass()
    # Load in Model
    load_model(par_obj, MODELNAME)
    pool = multiprocessing.dummy.Pool(int(multiprocessing.cpu_count() / 2))
    pool.imap(
        detect_and_segment,
        [
            [
                it,
                filename,
                par_obj,
                CURR_DIR,
                FOLDER,
                directory,
                SEGMENTATION_OBJECT_SIZE,
                SEGMENTATION_ITERATIONS,
                SPLIT,
                DOWNSAMPLE,
                Z_ADJUST,
            ]
            for it, filename in enumerate(filenames)
        ],
    )
    pool.close()
    pool.join()
    print("done")


def detect_and_segment(params):
    [
        it,
        filename,
        par_obj,
        CURR_DIR,
        FOLDER,
        directory,
        SEGMENTATION_OBJECT_SIZE,
        SEGMENTATION_ITERATIONS,
        SPLIT,
        DOWNSAMPLE,
        Z_ADJUST,
    ] = params
    # Resize image in XY
    image_out = (1, 1 / DOWNSAMPLE, 1 / DOWNSAMPLE)
    file_array = tifffile.imread(directory + filename)
    dtype = file_array.dtype
    original_file_shape = file_array.shape
    file_array = resize(
        file_array,
        (np.array(file_array.shape) * image_out).astype("int"),
        preserve_range=True,
        order=1,
    ).astype("uint16")

    # Set image size
    par_obj.height = file_array.shape[1]
    par_obj.width = file_array.shape[2]
    par_obj.numCH = 1
    # 12bit
    if dtype == "uint8":
        par_obj.tiffarray_typemax = np.iinfo(dtype).max
    else:
        par_obj.tiffarray_typemax = 4095

    # Correct error in the normalisation of CE dataset
    if "Fluo-N3DH-CE" in directory:
        # print("Fluo-N3DH-CE")
        par_obj.tiffarray_typemax = 4095

    ###Create features and predict###
    predMtx = np.zeros_like(file_array, dtype="uint16")

    for n, plane in enumerate(file_array):
        feat = lf.feature_create_threadable(
            par_obj, plane.astype("float32") / par_obj.tiffarray_typemax
        )
        # Predict from features
        mimg_lin = np.reshape(feat, (par_obj.height * par_obj.width, feat.shape[2]))
        linPred = par_obj.RF[0].predict(mimg_lin).astype("uint16")
        predMtx[n] = linPred.reshape(par_obj.height, par_obj.width)

    ###Count maxima###
    z_calibration = 0.75

    # Use custom version so don't need to worry about saving in par_object structure. Also different Z ordering
    radius = [
        par_obj.resize_factor * par_obj.min_distance[2] / z_calibration,
        par_obj.min_distance[0],
        par_obj.min_distance[1],
    ]
    det3 = det_hess_3d(predMtx, radius)

    # normalise, preventing divide by zero
    det3 = det3 / max(par_obj.max_det, 0.0000000001)

    # now deal with actual point detections
    pts = peak_local_max(
        det3, min_distance=radius, threshold_abs=par_obj.abs_thr, threshold_rel=0
    )
    pts = [np.array([pt[0], pt[1], pt[2], 1]) for pt in pts]

    ###Create mask with appropriate centres###
    mask = np.zeros_like(predMtx, dtype="uint16")
    for pt in pts:
        [z, x, y, W] = pt
        if W:
            mask[z, x, y] = 255
    mask = filters.maximum_filter(mask, size=(3, 3, 3))
    # Perform Segmentation, and save
    if not os.path.isdir(CURR_DIR + FOLDER + "_RES" + "\\"):
        os.mkdir(CURR_DIR + FOLDER + "_RES" + "\\")

    result_directory = CURR_DIR + FOLDER + "_RES" + "\\"
    read_resize_segment_faster_label_save(
        [
            it,
            mask,
            file_array,
            filename,
            result_directory,
            Z_ADJUST,
            DOWNSAMPLE,
            SEGMENTATION_OBJECT_SIZE,
            SEGMENTATION_ITERATIONS,
            SPLIT,
            original_file_shape,
        ]
    )
    print(str(it) + " segmentation saved: " + str(pts.__len__()) + " cells")


def load_model(par_obj, MODELNAME):
    # run_ctc_cytocensus required for running outside folder
    par_obj.selectedModel = "./run_ctc_cytocensus/models/" + MODELNAME
    par_obj.evaluated = False

    save_file = pickle.load(open(par_obj.selectedModel + str(".mdla"), "rb"))
    print(par_obj.selectedModel + str(".mdla"))

    par_obj.modelName = save_file["name"]
    par_obj.modelDescription = save_file["description"]
    par_obj.RF = save_file["model"]
    local_time = save_file["date"]
    par_obj.M = save_file["M"]
    par_obj.c = save_file["c"]
    par_obj.feature_type = save_file["feature_type"]
    par_obj.feature_scale = save_file["feature_scale"]
    par_obj.sigma_data = save_file["sigma_data"]
    par_obj.ch_active = save_file["ch_active"]
    par_obj.limit_ratio_size = save_file["limit_ratio_size"]
    par_obj.max_depth = save_file["max_depth"]
    par_obj.min_samples_split = save_file["min_samples"]
    par_obj.min_samples_leaf = save_file["min_samples_leaf"]
    par_obj.max_features = save_file["max_features"]
    par_obj.num_of_tree = save_file["num_of_tree"]

    par_obj.resize_factor = save_file["resize_factor"]
    par_obj.min_distance = save_file["min_distance"]
    par_obj.abs_thr = save_file["abs_thr"]
    par_obj.rel_thr = save_file["rel_thr"]
    par_obj.count_maxima_laplace = save_file["count_maxima_laplace"]
    par_obj.max_det = save_file["max_det"]
    save_im = save_file["imFile"]
    print("Status: Model loaded. ")


# Segment for CTC
def read_resize_segment_faster_label_save(params):
    it, mask, image, filename, directory, Z_ADJUST, DOWNSAMPLE, SEGMENTATION_OBJECT_SIZE, SEGMENTATION_ITERATIONS, SPLIT, original_file_shape = (
        params
    )

    image_array = image
    # image_array = skimage.exposure.equalize_hist(image_array, nbins=256)

    labels = scipy.ndimage.measurements.label(mask)[0].astype("uint16")
    # sampling corrects for downsampling compared to original. Also use to correct for non-isotropic resolution
    dt = scipy.ndimage.morphology.distance_transform_edt(
        255 - mask, sampling=(Z_ADJUST / DOWNSAMPLE * 0.5, 1, 1)
    )

    file_array = dt < (SEGMENTATION_OBJECT_SIZE / 2)
    objects = skimage.segmentation.morphological_chan_vese(
        np.pad(image_array, SEGMENTATION_ITERATIONS, mode="reflect"),
        iterations=SEGMENTATION_ITERATIONS,
        init_level_set=np.pad(file_array, SEGMENTATION_ITERATIONS, mode="reflect"),
        lambda2=2,
    )[
        SEGMENTATION_ITERATIONS:-SEGMENTATION_ITERATIONS,
        SEGMENTATION_ITERATIONS:-SEGMENTATION_ITERATIONS,
        SEGMENTATION_ITERATIONS:-SEGMENTATION_ITERATIONS,
    ]

    # optional splitting
    if SPLIT == True:
        wt = skimage.morphology.watershed(dt, labels, mask=objects > 0).astype("uint16")
    else:
        wt = scipy.ndimage.measurements.label(objects)[0].astype("uint16")
    file_array = resize(wt, original_file_shape, preserve_range=True, order=0).astype(
        "uint16"
    )
    # file_array = measure.label(file_array, background=0,return_num=True)[0].astype('float32')

    tifffile.imsave(directory + "mask" + filename[1:4] + ".tif", file_array)


if __name__ == "__main__":
    main()
