"""
This scriot is for interpolation between two transformation matrices 
"""

import glob
from os.path import dirname, join

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

idx_fname = 10

tf_mtxs = sorted(
    glob.glob(
        "/home/osobky/Documents/Master/Data/01_scene_01_omar/01_lidar/tf_matrix/rotation/*.rotation.csv"
    )
)

for i in range(len(tf_mtxs)):
    if i + 1 != len(tf_mtxs):
        print(
            "tf_start is at timestamp: ", tf_mtxs[i].split("/")[idx_fname].split("_")[0]
        )
        tf_start = np.loadtxt(
            tf_mtxs[i],
            delimiter=",",
        )
        print(
            "tf_end is at timestamp: ",
            tf_mtxs[i + 1].split("/")[idx_fname].split("_")[0],
        )
        tf_end = np.loadtxt(
            tf_mtxs[i + 1],
            delimiter=",",
        )

        key_rotations = R.from_quat([tf_start, tf_end])

        # print(key_rotations.as_euler("xyz", degrees=True))

        key_times = [0, 10]
        slerp = Slerp(key_times, key_rotations)

        times = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        interp_rots = slerp(times)

        # print(interp_rots.as_euler("xyz", degrees=True))
        print(interp_rots.as_matrix())

        path = "/home/osobky/Documents/Master/Data/01_scene_01_omar/01_lidar/tf_matrix/rotation/"

        for j, tf in enumerate(interp_rots):
            if (
                int(tf_mtxs[i].split("/")[idx_fname].split("_")[1])
                + 100000000 * (j + 1)
                < 1000000000
            ):
                fname = int(
                    tf_mtxs[i].split("/")[idx_fname].split("_")[1]
                ) + 100000000 * (j + 1)
                fname = (
                    tf_mtxs[i].split("/")[idx_fname].split("_")[0]
                    + "_"
                    + str(fname)
                    + "_"
                    + "_".join(tf_mtxs[i].split("/")[idx_fname].split("_")[2:])
                )
                fpath = join(dirname(path), str(fname))
                np.savetxt(fpath, np.array(tf.as_quat()), delimiter=",")
            elif (
                int(tf_mtxs[i].split("/")[idx_fname].split("_")[1])
                + 100000000 * (j + 1)
                == 1000000000
            ):
                fname = int(tf_mtxs[i].split("/")[idx_fname].split("_")[0]) + 1
                fname = (
                    str(fname)
                    + "_"
                    + "000000000"
                    + "_"
                    + "_".join(tf_mtxs[i].split("/")[idx_fname].split("_")[2:])
                )
                fpath = join(dirname(path), str(fname))
                np.savetxt(fpath, np.array(tf.as_quat()), delimiter=",")
            else:
                fname = int(tf_mtxs[i].split("/")[idx_fname].split("_")[0]) + 1
                fname = (
                    str(fname)
                    + "_"
                    + "100000000"
                    + "_"
                    + "_".join(tf_mtxs[i].split("/")[idx_fname].split("_")[2:])
                )
                fpath = join(dirname(path), str(fname))
                np.savetxt(fpath, np.array(tf.as_quat()), delimiter=",")


# Now we will interpolate for translation vectors

tsl_vct = sorted(
    glob.glob(
        "/home/osobky/Documents/Master/Data/01_scene_01_omar/01_lidar/tf_matrix/translation/*.translation.csv"
    )
)


for i in range(len(tsl_vct)):
    if i + 1 != len(tsl_vct):
        print(
            "tsl_start is at timestamp: ",
            tsl_vct[i].split("/")[idx_fname].split("_")[0],
        )
        tsl_start = np.loadtxt(
            tsl_vct[i],
            delimiter=",",
        )
        print(
            "tsl_end is at timestamp: ",
            tsl_vct[i + 1].split("/")[idx_fname].split("_")[0],
        )
        tsl_end = np.loadtxt(
            tsl_vct[i + 1],
            delimiter=",",
        )

        key_tsl = np.vstack([tsl_start, tsl_end])

        # print(key_rotations.as_euler("xyz", degrees=True))

        key_times = [0, 10]
        linfit = interp1d(key_times, key_tsl, axis=0)

        times = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        interp_tsl = linfit(times)

        path = "/home/osobky/Documents/Master/Data/01_scene_01_omar/01_lidar/tf_matrix/translation/"

        for j, tsl in enumerate(interp_tsl):
            if (
                int(tsl_vct[i].split("/")[idx_fname].split("_")[1])
                + 100000000 * (j + 1)
                < 1000000000
            ):
                fname = int(
                    tsl_vct[i].split("/")[idx_fname].split("_")[1]
                ) + 100000000 * (j + 1)
                fname = (
                    tsl_vct[i].split("/")[idx_fname].split("_")[0]
                    + "_"
                    + str(fname)
                    + "_"
                    + "_".join(tsl_vct[i].split("/")[idx_fname].split("_")[2:])
                )
                fpath = join(dirname(path), str(fname))
                np.savetxt(fpath, np.array(tsl), delimiter=",")
            elif (
                int(tsl_vct[i].split("/")[idx_fname].split("_")[1])
                + 100000000 * (j + 1)
                == 1000000000
            ):
                fname = int(tsl_vct[i].split("/")[idx_fname].split("_")[0]) + 1
                fname = (
                    str(fname)
                    + "_"
                    + "000000000"
                    + "_"
                    + "_".join(tsl_vct[i].split("/")[idx_fname].split("_")[2:])
                )
                fpath = join(dirname(path), str(fname))
                np.savetxt(fpath, np.array(tsl), delimiter=",")
            else:
                fname = int(tsl_vct[i].split("/")[idx_fname].split("_")[0]) + 1
                fname = (
                    str(fname)
                    + "_"
                    + "100000000"
                    + "_"
                    + "_".join(tsl_vct[i].split("/")[idx_fname].split("_")[2:])
                )
                fpath = join(dirname(path), str(fname))
                np.savetxt(fpath, np.array(tsl), delimiter=",")
