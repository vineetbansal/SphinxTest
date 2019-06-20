import os
import logging

import mrcfile
import numpy as np
import pyfftw
from concurrent import futures
from tqdm import tqdm

from scipy import ndimage, misc, signal
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation, center_of_mass
from sklearn import svm, preprocessing

from aspyre import config
from aspyre.apple.helper import PickerHelper

logger = logging.getLogger(__name__)


class Picker:
    """ This class does the actual picking with help from PickerHelper class. """

    def __init__(self, particle_size, max_size, min_size, query_size, tau1, tau2, moa,
                 container_size, filename, output_directory):

        self.particle_size = int(particle_size / 2)
        self.max_size = int(max_size / 2)
        self.min_size = int(min_size / 2)
        self.query_size = int(query_size / 2)
        self.query_size -= self.query_size % 2
        self.tau1 = tau1
        self.tau2 = tau2
        self.moa = int(moa / 2)
        self.container_size = int(container_size / 2)
        self.filename = filename
        self.output_directory = output_directory

        self.query_size -= self.query_size % 2

    def read_mrc(self):
        """Gets and preprocesses micrograph.

        Reads the micrograph, applies binning and a low-pass filter.

        Returns:
            Micrograph image.
        """

        with mrcfile.open(self.filename, mode='r+', permissive=True) as mrc:
            im = mrc.data.astype('float')

        # Discard outer pixels
        im = im[
            config.apple.mrc.margin_top: -config.apple.mrc.margin_bottom,
            config.apple.mrc.margin_left: -config.apple.mrc.margin_right
        ]

        # Make square
        side_length = min(im.shape)
        im = im[:side_length, :side_length]

        im = misc.imresize(im, 1/config.apple.mrc.shrink_factor, mode='F', interp='cubic')
        im = signal.correlate(
            im,
            PickerHelper.gaussian_filter(
                config.apple.mrc.gauss_filter_size,
                config.apple.mrc.gauss_filter_sigma
            ),
            'same'
        )

        return im.astype('double')

    def query_score(self, micro_img, show_progress=True):
        """Calculates score for each query image.

        Extracts query images and reference windows. Computes the cross-correlation between these
        windows, and applies a threshold to compute a score for each query image.

        Args:
            micro_img: Micrograph image.
            show_progress: Whether to show a progress bar

        Returns:
            Matrix containing a score for each query image.
        """

        query_box = PickerHelper.extract_query(micro_img, int(self.query_size / 2))

        out_shape = query_box.shape[0], query_box.shape[1], query_box.shape[2], query_box.shape[3] // 2 + 1
        query_box_a = np.empty(out_shape, dtype='complex128')
        fft_class_f = pyfftw.FFTW(np.empty_like(query_box), query_box_a, axes=(2, 3), direction='FFTW_FORWARD')
        fft_class_f(query_box, query_box_a)
        query_box = np.conj(query_box_a)

        reference_box_a = PickerHelper.extract_references(micro_img, self.query_size, self.container_size)
        out_shape2 = reference_box_a.shape[0], reference_box_a.shape[1], reference_box_a.shape[-1] // 2 + 1

        reference_box = np.empty(out_shape2, dtype='complex128')
        fft_class_f2 = pyfftw.FFTW(np.empty_like(reference_box_a), reference_box, axes=(1, 2), direction='FFTW_FORWARD')
        fft_class_f2(reference_box_a, reference_box)

        conv_map = np.zeros((reference_box.shape[0], query_box.shape[0], query_box.shape[1]))

        def _work(index):
            window_t = np.empty(query_box.shape, dtype=query_box.dtype)
            cc = np.empty((query_box.shape[0], query_box.shape[1], query_box.shape[2],
                           2 * query_box.shape[3] - 2), dtype=micro_img.dtype)
            fft_class = pyfftw.FFTW(window_t, cc, axes=(2, 3), direction='FFTW_BACKWARD')

            window_t = np.multiply(reference_box[index], query_box)
            fft_class(window_t, cc)
            return index, cc.real.max((2, 3)) - cc.real.mean((2, 3))

        n_works = reference_box.shape[0]
        n_threads = config.apple.conv_map_nthreads

        pbar = tqdm(total=n_works, disable=not show_progress)
        if n_threads > 1:

            with futures.ThreadPoolExecutor(n_threads) as executor:
                to_do = [executor.submit(_work, i) for i in range(n_works)]

                for future in futures.as_completed(to_do):
                    i, res = future.result()
                    conv_map[i, :, :] = res
                    pbar.update(1)
        else:

            for i in range(n_works):
                _, conv_map[i, :, :] = _work(i)
                pbar.update(1)

        pbar.close()

        conv_map = np.transpose(conv_map, (1, 2, 0))

        min_val = np.amin(conv_map)
        max_val = np.amax(conv_map)
        thresh = min_val + (max_val - min_val) / config.apple.response_thresh_norm_factor

        return np.sum(conv_map >= thresh, axis=2)

    def run_svm(self, micro_img, score):
        """
        Trains and uses an SVM classifier.

        Trains an SVM classifier to distinguish between noise and particle projections based on
        mean intensity and variance. Every possible window in the micrograph is then classified
        as either noise or particle, resulting in a segmentation of the micrograph.

        Args:
            micro_img: Micrograph image.
            score: Matrix containing a score for each query image.

        Returns:
            Segmentation of the micrograph into noise and particle projections.
        """

        particle_windows = np.floor(self.tau1)
        non_noise_windows = np.ceil(self.tau2)
        bw_mask_p, bw_mask_n = Picker.get_maps(self, score, micro_img, particle_windows, non_noise_windows)

        x, y = PickerHelper.get_training_set(micro_img, bw_mask_p, bw_mask_n, self.query_size)

        scaler = preprocessing.StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        classify = svm.SVC(C=1, kernel=config.apple.svm.kernel, gamma=config.apple.svm.gamma, class_weight='balanced')
        classify.fit(x, y)

        mean_all, std_all = PickerHelper.moments(micro_img, self.query_size)

        mean_all = mean_all[self.query_size - 1:-(self.query_size - 1),
                            self.query_size - 1:-(self.query_size - 1)]

        std_all = std_all[self.query_size - 1:-(self.query_size - 1),
                          self.query_size - 1:-(self.query_size - 1)]

        mean_all = np.reshape(mean_all, (np.prod(mean_all.shape), 1), 'F')
        std_all = np.reshape(std_all, (np.prod(std_all.shape), 1), 'F')
        cls_input = np.concatenate((mean_all, std_all), axis=1)
        cls_input = scaler.transform(cls_input)

        # compute classification for all possible windows in micrograph
        segmentation = classify.predict(cls_input)

        _segmentation_shape = int(np.sqrt(segmentation.shape[0]))
        segmentation = np.reshape(segmentation, (_segmentation_shape, _segmentation_shape), 'F')

        return segmentation.copy()

    def morphology_ops(self, segmentation):
        """
        Discards suspected artifacts from segmentation.

        Args:
            segmentation: Segmentation of the micrograph into noise and particle projections.

        Returns:
            Segmentation of the micrograph into noise and particle projections.
        """

        if (binary_fill_holes(segmentation) == np.ones(segmentation.shape)).all():
            segmentation[0:100, 0:100] = np.zeros((100, 100))

        segmentation = binary_fill_holes(segmentation)
        y, x = np.ogrid[-self.min_size:self.min_size+1, -self.min_size:self.min_size+1]
        element = x*x+y*y <= self.min_size * self.min_size
        segmentation_e = binary_erosion(segmentation, element)

        y, x = np.ogrid[-self.max_size:self.max_size+1, -self.max_size:self.max_size+1]
        element = x*x+y*y <= self.max_size * self.max_size
        segmentation_o = binary_erosion(segmentation, element)
        segmentation_o = np.reshape(segmentation_o,
                                    (segmentation_o.shape[0], segmentation_o.shape[1], 1), 'F')

        size_const, _ = ndimage.label(segmentation_e, np.ones((3, 3)))
        size_const = np.reshape(size_const, (size_const.shape[0], size_const.shape[1], 1), 'F')
        labels = np.unique(size_const*segmentation_o)
        idx = np.where(labels != 0)
        labels = np.take(labels, idx)
        labels = np.reshape(labels, (1, 1, np.prod(labels.shape)), 'F')

        matrix1 = np.repeat(size_const, labels.shape[2], 2)
        matrix2 = np.repeat(labels, matrix1.shape[0], 0)
        matrix2 = np.repeat(matrix2, matrix1.shape[1], 1)

        matrix3 = np.equal(matrix1, matrix2)
        matrix4 = np.sum(matrix3, 2)

        segmentation_e[np.where(matrix4 == 1)] = 0

        return segmentation_e

    def extract_particles(self, segmentation, create_jpg=False):
        """
        Saves particle centers into output .star file, after dismissing regions
        that are too big to contain a particle.

        Args:
            segmentation: Segmentation of the micrograph into noise and particle projections.
            create_jpg: whether to create jpg file of picked particles
        """
        segmentation = segmentation[self.query_size // 2 - 1:-self.query_size // 2,
                                    self.query_size // 2 - 1:-self.query_size // 2]
        labeled_segments, _ = ndimage.label(segmentation, np.ones((3, 3)))
        values, repeats = np.unique(labeled_segments, return_counts=True)

        values_to_remove = np.where(repeats > self.max_size ** 2)
        values = np.take(values, values_to_remove)
        values = np.reshape(values, (1, 1, np.prod(values.shape)), 'F')

        labeled_segments = np.reshape(labeled_segments, (labeled_segments.shape[0],
                                                         labeled_segments.shape[1], 1), 'F')
        matrix1 = np.repeat(labeled_segments, values.shape[2], 2)
        matrix2 = np.repeat(values, matrix1.shape[0], 0)
        matrix2 = np.repeat(matrix2, matrix1.shape[1], 1)

        matrix3 = np.equal(matrix1, matrix2)
        matrix4 = np.sum(matrix3, 2)

        segmentation[np.where(matrix4 == 1)] = 0
        labeled_segments, _ = ndimage.label(segmentation, np.ones((3, 3)))

        max_val = np.amax(np.reshape(labeled_segments, (np.prod(labeled_segments.shape))))
        center = center_of_mass(segmentation, labeled_segments, np.arange(1, max_val))
        center = np.rint(center)

        img = np.zeros((segmentation.shape[0], segmentation.shape[1]))
        img[center[:, 0].astype(int), center[:, 1].astype(int)] = 1
        y, x = np.ogrid[-self.moa:self.moa+1, -self.moa:self.moa+1]
        element = x*x+y*y <= self.moa * self.moa
        img = binary_dilation(img, structure=element)
        labeled_img, _ = ndimage.label(img, np.ones((3, 3)))
        values, repeats = np.unique(labeled_img, return_counts=True)
        y = np.where(repeats == np.count_nonzero(element))
        y = np.array(y)
        y = y.astype(int)
        y = np.reshape(y, (np.prod(y.shape)), 'F')
        y -= 1
        center = center[y, :]

        center = center + (self.query_size // 2 - 1) * np.ones(center.shape)
        center = center + (self.query_size // 2 - 1) * np.ones(center.shape)
        center = center + np.ones(center.shape)

        center = config.apple.mrc.shrink_factor * center

        # swap columns to align with Relion
        center = center[:, [1, 0]]

        # first column is x; second column is y - offset by margins that were discarded from the image
        center[:, 0] += config.apple.mrc.margin_left
        center[:, 1] += config.apple.mrc.margin_top

        if self.output_directory is not None:
            basename = os.path.basename(self.filename)
            name_str, ext = os.path.splitext(basename)

            applepick_path = os.path.join(self.output_directory, "{}_applepick.star".format(name_str))
            with open(applepick_path, "w") as f:
                np.savetxt(f, ["data_root\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2"], fmt='%s')
                np.savetxt(f, center, fmt='%d %d')

        if create_jpg:
            self.create_jpg(center)

        return center

    def get_maps(self, score, micro_img, particle_windows, non_noise_windows):
        """
        Gets maps of regions from which to extract particle training for the SVM classifier.

        Args:
            score: Matrix containing a score for each query image.
            micro_img: Micrograph image.
            particle_windows: Number of windows that must contain a particle.
            non_noise_windows: Number of windows that must contain noise.
        """
        idx = np.argsort(-np.reshape(score, (np.prod(score.shape)), 'F'))

        y = idx % score.shape[0]
        x = np.floor(idx/score.shape[0])

        bw_mask_p = np.zeros((micro_img.shape[0], micro_img.shape[1]))

        begin_row_idx = y*int(self.query_size / 2)
        end_row_idx = np.minimum(y * int(self.query_size / 2) + self.query_size,
                                 bw_mask_p.shape[0] * np.ones(y.shape[0]))

        begin_col_idx = x*int(self.query_size / 2)
        end_col_idx = np.minimum(x * int(self.query_size / 2) + self.query_size,
                                 bw_mask_p.shape[1] * np.ones(x.shape[0]))

        begin_row_idx = begin_row_idx.astype(int)
        end_row_idx = end_row_idx.astype(int)
        begin_col_idx = begin_col_idx.astype(int)
        end_col_idx = end_col_idx.astype(int)

        for j in range(0, particle_windows.astype(int)):
            bw_mask_p[begin_row_idx[j]:end_row_idx[j], begin_col_idx[j]:end_col_idx[j]] = np.ones(
                end_row_idx[j] - begin_row_idx[j], end_col_idx[j] - begin_col_idx[j])

        bw_mask_n = np.copy(bw_mask_p)
        for j in range(particle_windows.astype(int), non_noise_windows.astype(int)):
            bw_mask_n[begin_row_idx[j]:end_row_idx[j], begin_col_idx[j]:end_col_idx[j]] = np.ones(
                end_row_idx[j] - begin_row_idx[j], end_col_idx[j] - begin_col_idx[j])

        return bw_mask_p, bw_mask_n

    def create_jpg(self, centers):
        with mrcfile.open(self.filename, mode='r') as mrc:
            micro_img = mrc.data

        micro_img = np.double(micro_img)
        micro_img = micro_img - np.amin(micro_img)
        picks = np.ones(micro_img.shape)
        for i in range(0, centers.shape[0]):
            y = int(centers[i, 1])
            x = int(centers[i, 0])
            d = int(np.floor(self.particle_size))
            picks[y-d:y-d+5, x-d:x+d] = 0
            picks[y+d:y+d+5, x-d:x+d] = 0
            picks[y-d:y+d, x-d:x-d+5] = 0
            picks[y-d:y+d, x+d:x+d+5] = 0

        out_img = np.multiply(micro_img, picks)
        image_filename = os.path.splitext(os.path.basename(self.filename))[0] + '_result.jpg'
        image_path = os.path.join(self.output_directory, image_filename)
        misc.imsave(image_path, out_img)
