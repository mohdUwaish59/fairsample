import copy
import math
from operator import itemgetter
import numpy as np
from sklearn.cluster import KMeans
import sklearn
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import scipy


class ComplexityMetrics:
    '''
    Complexity metrics calculator for classification datasets.
    Contains all complexity score calculation methods.
    '''

    def __init__(self, X, y, meta=None):
        '''
        Initialize complexity metrics calculator.
        
        Parameters:
        -----------
        X : numpy.array
            Feature matrix (n_samples, n_features)
        y : numpy.array
            Target labels (n_samples,)
        meta : list, optional
            List indicating feature types: 0 for numerical, 1 for categorical
        '''
        self.X = np.array(X)
        self.y = np.array(y)
        self.classes = np.unique(self.y)
        
        # Auto-detect meta if not provided
        if meta is None:
            self.meta = self._is_categorical(X)
        else:
            self.meta = meta
        
        # Calculate distance matrices
        self.dist_matrix, self.unnorm_dist_matrix = self._calculate_distance_matrix(self.X)
        
        # Class statistics
        self.class_count = self._count_class_instances(self.y)
        self.class_inxs = self._get_class_inxs()
        
        # Cache for expensive computations
        self.sphere_inst_count_T1 = []
        self.radius_T1 = []
        self.sphere_tuple_ONB = []
        
        if len(self.class_count) < 2:
            raise ValueError("ERROR: Less than two classes are in the dataset.")

    def _is_categorical(self, X):
        '''Detect categorical features'''
        meta = []
        for i in range(X.shape[1]):
            column = X[:, i]
            try:
                float_column = column.astype(float)
                meta.append(0)
            except ValueError:
                meta.append(1)
        return meta

    def _count_class_instances(self, y):
        '''Count instances of each class'''
        class_count = np.zeros(len(self.classes))
        for i in range(len(self.classes)):
            count = len(np.where(y == self.classes[i])[0])
            class_count[i] += count
        return class_count

    def _get_class_inxs(self):
        '''Get indices for each class'''
        class_inds = []
        for cls in self.classes:
            cls_ind = np.where(self.y == cls)[0]
            class_inds.append(cls_ind)
        return class_inds

    def _distance_HEOM(self, X):
        '''Calculate distance matrix using HEOM metric'''
        meta = self.meta
        dist_matrix = np.zeros((len(X), len(X)))
        unnorm_dist_matrix = np.zeros((len(X), len(X)))

        # Calculate ranges
        range_max = np.max(X, axis=0)
        range_min = np.min(X, axis=0)

        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                dist = 0
                unnorm_dist = 0
                for k in range(len(X[0])):
                    # Missing value
                    if X[i][k] == None or X[j][k] == None:
                        dist += 1
                        unnorm_dist += 1
                    # Numerical
                    if meta[k] == 0:
                        if range_max[k] == range_min[k]:
                            dist += (abs(X[i][k] - X[j][k])) ** 2
                            unnorm_dist += (abs(X[i][k] - X[j][k])) ** 2
                        else:
                            dist += (abs(X[i][k] - X[j][k]) / (range_max[k] - range_min[k])) ** 2
                            unnorm_dist += abs(X[i][k] - X[j][k]) ** 2
                    # Categorical
                    if meta[k] == 1:
                        if X[i][k] != X[j][k]:
                            dist += 1
                            unnorm_dist += 1

                dist_matrix[i][j] = np.sqrt(dist)
                dist_matrix[j][i] = np.sqrt(dist)
                unnorm_dist_matrix[i][j] = np.sqrt(unnorm_dist)
                unnorm_dist_matrix[j][i] = np.sqrt(unnorm_dist)

        return dist_matrix, unnorm_dist_matrix

    def _distance_HEOM_different_arrays(self, X, X2):
        '''Calculate distance matrix between two different arrays'''
        meta = self.meta
        dist_matrix = np.zeros((len(X2), len(X)))

        range_max = np.max(X, axis=0)
        range_min = np.min(X, axis=0)

        for i in range(len(X2)):
            for j in range(len(X)):
                dist = 0
                for k in range(len(X2[0])):
                    if X2[i][k] == None or X[j][k] == None:
                        dist += 1
                    if meta[k] == 0:
                        if range_max[k] - range_min[k] == 0:
                            dist += (abs(X2[i][k] - X[j][k])) ** 2
                        else:
                            dist += (abs(X2[i][k] - X[j][k]) / (range_max[k] - range_min[k])) ** 2
                    if meta[k] == 1:
                        if X2[i][k] != X[j][k]:
                            dist += 1
                dist_matrix[i][j] = np.sqrt(dist)

        return dist_matrix

    def _calculate_distance_matrix(self, X, distance_func="HEOM"):
        '''Calculate distance matrix'''
        if distance_func == "HEOM" or distance_func == "default":
            distance_matrix, unnorm_distance_matrix = self._distance_HEOM(X)
        return distance_matrix, unnorm_distance_matrix

    def _knn(self, inx, line, k, y=None, clear_diag=True):
        '''K-nearest neighbors'''
        if y is None:
            y = self.y

        count = np.zeros(len(self.classes))

        if clear_diag:
            line[inx] = math.inf

        for i in range(k):
            index = np.where(line == min(line))[0][0]
            line[index] = math.inf
            cls_inx = np.where(self.classes == y[index])[0][0]
            count[cls_inx] += 1

        return count

    def _knn_dists(self, inx, line, k, clear_diag=True):
        '''Get distances of k-nearest neighbors of same class'''
        dists = []
        if clear_diag:
            line[inx] = math.inf
        for i in range(k):
            index = np.where(line == min(line))[0][0]
            if self.y[index] == self.y[inx]:
                dists.append(line[index])
            line[index] = math.inf
        return dists

    def _hypersphere(self, inx, sigma, distance_matrix=None, y=None):
        '''Count samples inside hypersphere'''
        if distance_matrix is None:
            distance_matrix = self.dist_matrix
        if y is None:
            y = self.y

        line = distance_matrix[inx]
        n_minus = 0
        n_plus = 0

        for i in range(len(line)):
            if line[i] <= sigma:
                if y[i] == y[inx]:
                    n_plus += 1
                else:
                    n_minus += 1
        return [n_minus, n_plus]

    def _hypersphere_sim(self, inx, sigma):
        '''Sum of distances inside hypersphere'''
        line = self.dist_matrix[inx]
        n_minus = 0
        n_plus = 0

        for i in range(len(line)):
            if line[i] <= sigma:
                if self.y[i] == self.y[inx]:
                    n_plus += line[i]
                else:
                    n_minus += line[i]
        return [n_minus, n_plus]

    # ==================== FEATURE-BASED METRICS ====================

    def F1(self):
        '''
        Fisher's discriminant ratio.
        Returns array with F1 value for each feature.
        '''
        f1s = []

        for i in range(len(self.class_inxs)):
            for j in range(i + 1, len(self.class_inxs)):
                sample_c1 = self.X[self.class_inxs[i]]
                sample_c2 = self.X[self.class_inxs[j]]

                avg_c1 = np.mean(sample_c1, 0)
                avg_c2 = np.mean(sample_c2, 0)
                std_c1 = np.std(sample_c1, 0)
                std_c2 = np.std(sample_c2, 0)

                f1 = ((avg_c1 - avg_c2) ** 2) / (std_c1 ** 2 + std_c2 ** 2)
                f1[np.isinf(f1)] = 0
                f1[np.isnan(f1)] = 0
                f1 = 1 / (1 + f1)
                f1s.append(f1)

        f1_val = np.mean(f1s, axis=0)
        return f1_val

    def F1v(self):
        '''
        Directional-vector maximum Fisher's discriminant ratio.
        Returns array with F1v value for each class pair.
        '''
        f1vs = []

        for i in range(len(self.class_inxs)):
            for j in range(i + 1, len(self.class_inxs)):
                sample_c1 = self.X[self.class_inxs[i]]
                sample_c2 = self.X[self.class_inxs[j]]

                avg_c1 = np.mean(sample_c1, 0)
                avg_c2 = np.mean(sample_c2, 0)
                W = (len(sample_c1) * np.cov(sample_c1, rowvar=False, ddof=1) + 
                     len(sample_c2) * np.cov(sample_c2, rowvar=False, ddof=1)) / (len(sample_c1) + len(sample_c2))
                B = np.outer(avg_c1 - avg_c2, avg_c1 - avg_c2)
                d = np.matmul(scipy.linalg.pinv(W), avg_c1 - avg_c2)

                f1v = (np.matmul(d.T, np.matmul(B, d))) / (np.matmul(d.T, np.matmul(W, d)))
                f1v = 1 / (1 + f1v)
                f1vs.append(f1v)

        return f1vs

    def F2(self, imb=False):
        '''
        Volume of overlapping region.
        
        Parameters:
        -----------
        imb : bool
            If True, return separate values for majority/minority classes
        '''
        f2s = []

        for i in range(len(self.class_inxs)):
            for j in range(i + 1, len(self.class_inxs)):
                sample_c1 = self.X[self.class_inxs[i]]
                sample_c2 = self.X[self.class_inxs[j]]

                maxmin = np.max([np.min(sample_c1, axis=0), np.min(sample_c2, axis=0)], axis=0)
                minmax = np.min([np.max(sample_c1, axis=0), np.max(sample_c2, axis=0)], axis=0)
                numer = np.maximum(0.0, minmax - maxmin)

                if imb:
                    if len(sample_c1) > len(sample_c2):
                        maxmax_minority = np.max(sample_c2, axis=0)
                        minmin_minority = np.min(sample_c2, axis=0)
                        maxmax_majority = np.max(sample_c1, axis=0)
                        minmin_majority = np.min(sample_c1, axis=0)
                    else:
                        maxmax_minority = np.max(sample_c1, axis=0)
                        minmin_minority = np.min(sample_c1, axis=0)
                        maxmax_majority = np.max(sample_c2, axis=0)
                        minmin_majority = np.min(sample_c2, axis=0)

                    denom_maj = (maxmax_majority - minmin_majority)
                    denom_min = (maxmax_minority - minmin_minority)

                    n_d_min = numer / denom_min
                    n_d_maj = numer / denom_maj

                    n_d_min[np.isinf(n_d_min)] = 0
                    n_d_min[np.isnan(n_d_min)] = 0
                    n_d_maj[np.isinf(n_d_maj)] = 0
                    n_d_maj[np.isnan(n_d_maj)] = 0

                    f2_min = np.prod(n_d_min)
                    f2_maj = np.prod(n_d_maj)
                    f2s.append([f2_maj, f2_min])
                else:
                    maxmax = np.max([np.max(sample_c1, axis=0), np.max(sample_c2, axis=0)], axis=0)
                    minmin = np.min([np.min(sample_c1, axis=0), np.min(sample_c2, axis=0)], axis=0)
                    denom = (maxmax - minmin)

                    n_d = numer / denom
                    n_d[np.isinf(n_d)] = 0
                    n_d[np.isnan(n_d)] = 0
                    f2 = np.prod(n_d)
                    f2s.append(f2)

        return f2s

    def _F3_counter(self, t_X, maxmin, minmax):
        '''Helper for F3 calculation'''
        overlap_count = []
        for k in range(len(t_X)):
            feature = t_X[k]
            count = 0
            for value in feature:
                if value >= maxmin[k] and value <= minmax[k]:
                    count += 1
            overlap_count.append(count)
        return overlap_count

    def F3(self, imb=False):
        '''
        Maximum individual feature efficiency.
        
        Parameters:
        -----------
        imb : bool
            If True, return separate values for majority/minority classes
        '''
        f3s = []

        for i in range(len(self.class_inxs)):
            for j in range(i + 1, len(self.class_inxs)):
                sample_c1 = self.X[self.class_inxs[i]]
                sample_c2 = self.X[self.class_inxs[j]]

                maxmin = np.max([np.min(sample_c1, axis=0), np.min(sample_c2, axis=0)], axis=0)
                minmax = np.min([np.max(sample_c1, axis=0), np.max(sample_c2, axis=0)], axis=0)

                transpose_sample_c1 = np.transpose(sample_c1)
                transpose_sample_c2 = np.transpose(sample_c2)
                transpose_X = np.transpose(self.X)

                if imb:
                    c1_count = self._F3_counter(transpose_sample_c1, maxmin, minmax)
                    c2_count = self._F3_counter(transpose_sample_c2, maxmin, minmax)

                    if len(sample_c1) > len(sample_c2):
                        maj_count = c1_count
                        min_count = c2_count
                        len_min = len(sample_c2)
                        len_maj = len(sample_c1)
                    else:
                        maj_count = c2_count
                        min_count = c1_count
                        len_min = len(sample_c1)
                        len_maj = len(sample_c2)

                    min_overlap_min = min(min_count)
                    min_overlap_maj = min(maj_count)
                    f3_min = min_overlap_min / len_min
                    f3_maj = min_overlap_maj / len_maj
                    f3 = [f3_maj, f3_min]
                else:
                    overlap_count = self._F3_counter(transpose_X, maxmin, minmax)
                    min_overlap = min(overlap_count)
                    f3 = min_overlap / (len(self.X[self.class_inxs[i]]) + len(self.X[self.class_inxs[j]]))

                f3s.append(f3)

        return f3s

    def F4(self, imb=False):
        '''
        Collective feature efficiency.
        
        Parameters:
        -----------
        imb : bool
            If True, return separate values for majority/minority classes
        '''
        f4s = []

        for i2 in range(len(self.class_inxs)):
            for j2 in range(i2 + 1, len(self.class_inxs)):
                valid_inxs_c1 = self.class_inxs[i2]
                valid_inxs_c2 = self.class_inxs[j2]
                sample_c1 = self.X[valid_inxs_c1]
                sample_c2 = self.X[valid_inxs_c2]
                sample_c1_y = self.y[valid_inxs_c1]
                sample_c2_y = self.y[valid_inxs_c2]

                X = np.concatenate((sample_c1, sample_c2), axis=0)
                y = np.concatenate((sample_c1_y, sample_c2_y), axis=0)
                valid_inxs_c1 = np.where(y == self.classes[i2])[0]
                valid_inxs_c2 = np.where(y == self.classes[j2])[0]

                transpose_X = np.transpose(X)

                while len(X) > 0:
                    if len(sample_c1) == 0 or len(sample_c2) == 0:
                        maxmin = np.full(len(X[0]), np.inf)
                        minmax = np.full(len(X[0]), -np.inf)
                    else:
                        maxmin = np.max([np.min(sample_c1, axis=0), np.min(sample_c2, axis=0)], axis=0)
                        minmax = np.min([np.max(sample_c1, axis=0), np.max(sample_c2, axis=0)], axis=0)

                    overlap_count = []
                    inx_lists = []

                    for i in range(len(transpose_X)):
                        feature = transpose_X[i]
                        inx_list = []
                        count = 0
                        for j in range(len(feature)):
                            value = feature[j]
                            if value >= maxmin[i] and value <= minmax[i]:
                                count += 1
                                inx_list.append(j)
                        overlap_count.append(count)
                        inx_lists.append(inx_list)

                    min_overlap = min(overlap_count)
                    min_inx = overlap_count.index(min_overlap)
                    min_overlap_inx = inx_lists[min_inx]

                    valid_features = list(range(0, min_inx)) + list(range(min_inx + 1, len(transpose_X)))

                    if len(min_overlap_inx) == 0 or len(valid_features) == 0:
                        new_X = []
                        new_y = []
                        for inx in range(len(X)):
                            if inx in min_overlap_inx:
                                new_X.append(X[inx])
                                new_y.append(y[inx])
                        X = np.array(new_X)
                        y = np.array(new_y)
                        transpose_X = np.transpose(X) if len(X) > 0 else []

                        valid_inxs_c1 = []
                        sample_c1 = []
                        for inx in range(len(y)):
                            if y[inx] == self.classes[i2]:
                                valid_inxs_c1.append(inx)
                                sample_c1.append(X[inx])

                        valid_inxs_c2 = []
                        sample_c2 = []
                        for inx in range(len(y)):
                            if y[inx] == self.classes[j2]:
                                valid_inxs_c2.append(inx)
                                sample_c2.append(X[inx])
                        break

                    new_X = []
                    new_y = []

                    for inx in range(len(X)):
                        if inx in min_overlap_inx:
                            sample = []
                            for ft in valid_features:
                                sample.append(X[inx, ft])
                            new_X.append(sample)
                            new_y.append(y[inx])

                    X = np.array(new_X)
                    y = np.array(new_y)
                    transpose_X = np.transpose(X) if len(X) > 0 else []

                    valid_inxs_c1 = []
                    sample_c1 = []
                    for inx in range(len(y)):
                        if y[inx] == self.classes[i2]:
                            valid_inxs_c1.append(inx)
                            sample_c1.append(X[inx])

                    valid_inxs_c2 = []
                    sample_c2 = []
                    for inx in range(len(y)):
                        if y[inx] == self.classes[j2]:
                            valid_inxs_c2.append(inx)
                            sample_c2.append(X[inx])

                    sample_c1 = np.array(sample_c1) if len(sample_c1) > 0 else np.array([])
                    sample_c2 = np.array(sample_c2) if len(sample_c2) > 0 else np.array([])

                if imb:
                    if len(sample_c2_y) > len(sample_c1_y):
                        f4_min = len(valid_inxs_c1) / len(sample_c1_y)
                        f4_maj = len(valid_inxs_c2) / len(sample_c2_y)
                    else:
                        f4_min = len(valid_inxs_c2) / len(sample_c2_y)
                        f4_maj = len(valid_inxs_c1) / len(sample_c1_y)
                    f4 = [f4_maj, f4_min]
                else:
                    f4 = (len(valid_inxs_c1) + len(valid_inxs_c2)) / (len(sample_c1_y) + len(sample_c2_y))

                f4s.append(f4)

        return f4s

    def _class_overlap(self, class_samples, other_samples):
        '''Calculate class overlap for input_noise'''
        min_class = np.min(other_samples, axis=0)
        max_class = np.max(other_samples, axis=0)
        count = 0
        for sample in class_samples:
            for i in range(len(sample)):
                feature = sample[i]
                if feature > min_class[i] and feature < max_class[i]:
                    count += 1
        return count

    def input_noise(self, imb=False):
        '''
        Input noise metric.
        
        Parameters:
        -----------
        imb : bool
            If True, return separate values for majority/minority classes
        '''
        ins = []
        for i in range(len(self.class_inxs)):
            for j in range(i + 1, len(self.class_inxs)):
                X = self.X
                valid_inxs_c1 = self.class_inxs[i]
                valid_inxs_c2 = self.class_inxs[j]
                sample_c1 = X[valid_inxs_c1]
                sample_c2 = X[valid_inxs_c2]

                if len(sample_c1) > len(sample_c2):
                    total_count_maj = self._class_overlap(sample_c1, sample_c2)
                    total_count_min = self._class_overlap(sample_c2, sample_c1)
                    sample_maj = sample_c1
                    sample_min = sample_c2
                else:
                    total_count_min = self._class_overlap(sample_c1, sample_c2)
                    total_count_maj = self._class_overlap(sample_c2, sample_c1)
                    sample_min = sample_c1
                    sample_maj = sample_c2

                if imb:
                    ins.append([
                        total_count_maj / (len(sample_maj) * len(X[0])),
                        total_count_min / (len(sample_min) * len(X[0]))
                    ])
                else:
                    total_count = total_count_maj + total_count_min
                    ins.append(total_count / (len(X) * len(X[0])))

        return ins

    # ==================== INSTANCE-BASED METRICS ====================

    def N3(self, k=1, imb=False, inst_level=False):
        '''
        Error rate of nearest neighbor classifier.
        
        Parameters:
        -----------
        k : int
            Number of nearest neighbors
        imb : bool
            If True, return separate values for each class
        inst_level : bool
            If True, return instance-level hardness values
        '''
        n3_counts = np.zeros(len(self.classes))
        inst_hardness = []

        for sample in range(len(self.X)):
            count = self._knn(sample, copy.copy(self.dist_matrix[sample]), k)
            cls_inx = np.where(self.classes == self.y[sample])[0][0]

            class_count = count[cls_inx]
            max_count = np.max(count)

            if class_count < max_count:
                n3_counts[cls_inx] += 1

            inst_hardness.append(1 - class_count / k)

        if inst_level:
            return np.array(inst_hardness)
        else:
            if imb:
                n3 = np.divide(n3_counts, self.class_count)
            else:
                n3 = sum(n3_counts) / len(self.X)
            return n3

    def _interpolate_samples(self, class_inxs=None):
        '''Create interpolated samples'''
        if class_inxs is None:
            class_inxs = self.class_inxs

        X_interp = []
        y_interp = []
        for cls_inx in class_inxs:
            new_X = self.X[cls_inx, :]
            new_y = self.y[cls_inx]
            sample1_inxs = np.random.choice(len(new_X), len(new_X))
            sample2_inxs = np.random.choice(len(new_X), len(new_X))
            sample1 = new_X[sample1_inxs, :]
            sample2 = new_X[sample2_inxs, :]

            alpha = np.random.ranf(new_X.shape)
            X_interp_cls = sample1 + (sample2 - sample1) * alpha

            y_interp = np.append(y_interp, new_y)
            if len(X_interp) == 0:
                X_interp = X_interp_cls
            else:
                X_interp = np.concatenate((X_interp, X_interp_cls), axis=0)

        return X_interp, y_interp

    def N4(self, k=1, imb=False):
        '''
        Non-linearity of nearest neighbor classifier.
        
        Parameters:
        -----------
        k : int
            Number of nearest neighbors
        imb : bool
            If True, return separate values for each class
        '''
        X_interp, y_interp = self._interpolate_samples()
        new_dist = self._distance_HEOM_different_arrays(self.X, X_interp)

        n4_counts = np.zeros(len(self.classes))

        for sample in range(len(X_interp)):
            count = self._knn(sample, copy.copy(new_dist[sample]), k, y=y_interp, clear_diag=False)
            cls_inx = np.where(self.classes == y_interp[sample])[0][0]

            class_count = count[cls_inx]
            max_count = np.max(count)

            if class_count < max_count:
                n4_counts[cls_inx] += 1

        if imb:
            n4 = np.divide(n4_counts, self.class_count)
        else:
            n4 = sum(n4_counts) / len(self.X)

        return n4

    def kDN(self, k=5, imb=False):
        '''
        k-Disagreeing Neighbors.
        
        Parameters:
        -----------
        k : int
            Number of nearest neighbors
        imb : bool
            If True, return separate values for each class
        '''
        kDN_value = np.zeros(len(self.classes))

        for i in range(len(self.dist_matrix)):
            line = self.dist_matrix[i]
            count = self._knn(i, copy.copy(line), k)
            cls_inx = np.where(self.classes == self.y[i])[0][0]
            kDN_value[cls_inx] += (k - count[cls_inx]) / k

        if imb:
            kDN_value = np.divide(kDN_value, self.class_count)
        else:
            kDN_value = sum(kDN_value) / len(self.X)

        return kDN_value

    def CM(self, k=5, imb=False):
        '''
        Classification Margin.
        
        Parameters:
        -----------
        k : int
            Number of nearest neighbors
        imb : bool
            If True, return separate values for each class
        '''
        CM_value = np.zeros(len(self.classes))

        for i in range(len(self.dist_matrix)):
            line = self.dist_matrix[i]
            count = self._knn(i, copy.copy(line), k)
            cls_inx = np.where(self.classes == self.y[i])[0]
            kDN_value = (k - count[cls_inx]) / k
            if kDN_value > 0.5:
                CM_value[cls_inx] += 1

        if imb:
            CM_value = np.divide(CM_value, self.class_count)
        else:
            CM_value = sum(CM_value) / len(self.X)

        return CM_value

    def R_value(self, k=5, theta=2, imb=False):
        '''
        Augmented R value.
        
        Parameters:
        -----------
        k : int
            Number of neighbors
        theta : int
            Threshold of neighbors
        imb : bool
            If True, return separate values for majority/minority
        '''
        r_matrix = np.zeros((len(self.classes), len(self.classes)))

        for i in range(len(self.dist_matrix)):
            line = self.dist_matrix[i]
            count = self._knn(i, copy.copy(line), k)
            cls_inx = np.where(self.classes == self.y[i])[0][0]
            for j in range(len(self.classes)):
                if theta < count[j]:
                    r_matrix[cls_inx, j] += 1

        for i in range(len(r_matrix)):
            for j in range(len(r_matrix[0])):
                r_matrix[i, j] = r_matrix[i, j] / self.class_count[i]

        r_values = []
        for i in range(len(r_matrix)):
            for j in range(i + 1, len(r_matrix)):
                if self.class_count[i] > self.class_count[j]:
                    imbalanced_ratio = self.class_count[i] / self.class_count[j]
                    overlap_ci_cj = r_matrix[i, j]
                    overlap_cj_ci = r_matrix[j, i]
                else:
                    imbalanced_ratio = self.class_count[j] / self.class_count[i]
                    overlap_ci_cj = r_matrix[j, i]
                    overlap_cj_ci = r_matrix[i, j]

                if imb:
                    r = [overlap_cj_ci, overlap_ci_cj]
                    r_values.append(r)
                else:
                    r = (1 / (imbalanced_ratio + 1)) * (overlap_ci_cj + imbalanced_ratio * overlap_cj_ci)
                    r_values.append(r)

        return r_values

    def D3_value(self, k=5):
        '''
        D3 value.
        
        Parameters:
        -----------
        k : int
            Number of neighbors
        '''
        d3_matrix = np.zeros(len(self.classes))

        for i in range(len(self.dist_matrix)):
            line = self.dist_matrix[i]
            count = self._knn(i, copy.copy(line), k)
            cls_inx = np.where(self.classes == self.y[i])[0][0]
            if 0.5 > (count[cls_inx] / k):
                d3_matrix[cls_inx] += 1

        return d3_matrix

    def SI(self, k=1, imb=False):
        '''
        Separability Index.
        
        Parameters:
        -----------
        k : int
            Number of nearest neighbors
        imb : bool
            If True, return separate values for each class
        '''
        sample_count = np.zeros(len(self.classes))

        for sample in range(len(self.X)):
            count = self._knn(sample, copy.copy(self.dist_matrix[sample]), k)
            cls_inx = np.where(self.classes == self.y[sample])[0][0]

            class_count = count[cls_inx]
            max_count = np.max(count)

            if class_count == max_count:
                sample_count[cls_inx] += 1

        if imb:
            si_measure = np.divide(sample_count, self.class_count)
        else:
            si_measure = sum(sample_count) / len(self.y)

        return si_measure

    def borderline(self, imb=False, return_all=False):
        '''
        Borderline examples metric.
        
        Parameters:
        -----------
        imb : bool
            If True, return separate values for each class
        return_all : bool
            If True, return all categories (borderline, safe, rare, outlier)
        '''
        borderline_count = np.zeros(len(self.classes))
        safe_count = np.zeros(len(self.classes))
        rare_count = np.zeros(len(self.classes))
        outlier_count = np.zeros(len(self.classes))

        classification = []
        k = 5

        for i in range(len(self.X)):
            count = self._knn(i, copy.copy(self.dist_matrix[i]), k)
            cls_inx = np.where(self.classes == self.y[i])[0][0]

            other_class_count = sum(count) - count[cls_inx]

            if other_class_count == 2 or other_class_count == 3:
                borderline_count[cls_inx] += 1
                classification.append("B")
            elif other_class_count < 2:
                safe_count[cls_inx] += 1
                classification.append("S")
            elif other_class_count == 4:
                rare_count[cls_inx] += 1
                classification.append("R")
            elif other_class_count == 5:
                outlier_count[cls_inx] += 1
                classification.append("O")

        if imb:
            borderline = np.divide(borderline_count, self.class_count)
            safe = np.divide(safe_count, self.class_count)
            rare = np.divide(rare_count, self.class_count)
            outlier = np.divide(outlier_count, self.class_count)
        else:
            borderline = sum(borderline_count) / len(self.X)
            safe = sum(safe_count) / len(self.X)
            rare = sum(rare_count) / len(self.X)
            outlier = sum(outlier_count) / len(self.X)

        if return_all:
            return borderline, safe, rare, outlier, classification
        else:
            return borderline

    def deg_overlap(self, k=5, imb=False):
        '''
        Degree of overlap.
        
        Parameters:
        -----------
        k : int
            Number of nearest neighbors
        imb : bool
            If True, return separate values for each class
        '''
        deg = np.zeros(len(self.classes))

        for i in range(len(self.X)):
            count = self._knn(i, copy.copy(self.dist_matrix[i]), k)
            cls_inx = np.where(self.classes == self.y[i])[0][0]
            if count[cls_inx] != k:
                deg[cls_inx] += 1

        if imb:
            deg_ov = np.divide(deg, self.class_count)
        else:
            deg_ov = sum(deg) / len(self.X)

        return deg_ov

    # ==================== STRUCTURAL METRICS ====================

    def _calculate_n_inter(self, dist_matrix=None, y=None, imb=False):
        '''Calculate number of inter-class edges in MST'''
        if dist_matrix is None:
            dist_matrix = self.dist_matrix
        if y is None:
            y = self.y

        minimum_spanning_tree = scipy.sparse.csgraph.minimum_spanning_tree(
            csgraph=np.triu(dist_matrix, k=1), overwrite=True
        )
        mst_array = minimum_spanning_tree.toarray().astype(float)

        vertix = []
        for i in range(len(mst_array)):
            for j in range(len(mst_array[0])):
                if mst_array[i][j] != 0:
                    if y[i] != y[j]:
                        vertix.append(i)
                        vertix.append(j)

        unique_vertix = np.unique(vertix)

        if imb == False:
            count = len(unique_vertix)
        else:
            count = np.zeros(len(self.classes))
            for inx in unique_vertix:
                cls_inx = np.where(self.classes == y[inx])[0][0]
                count[cls_inx] += 1

        return count

    def N1(self, imb=False):
        '''
        Fraction of borderline points (MST-based).
        
        Parameters:
        -----------
        imb : bool
            If True, return separate values for each class
        '''
        count = self._calculate_n_inter(imb=imb)

        if imb:
            n1 = np.divide(count, self.class_count)
        else:
            n1 = count / len(self.y)

        return n1

    def N2(self, imb=False):
        '''
        Ratio of intra/extra class nearest neighbor distance.
        
        Parameters:
        -----------
        imb : bool
            If True, return separate values for each class
        '''
        count_inter = np.zeros(len(self.classes))
        count_intra = np.zeros(len(self.classes))

        for i in range(len(self.dist_matrix)):
            min_inter = np.inf
            min_intra = np.inf

            for j in range(len(self.dist_matrix[0])):
                if self.y[i] == self.y[j] and i != j and self.dist_matrix[i][j] < min_intra:
                    min_intra = self.dist_matrix[i][j]
                if self.y[i] != self.y[j] and self.dist_matrix[i][j] < min_inter:
                    min_inter = self.dist_matrix[i][j]

            cls_inx = np.where(self.classes == self.y[i])[0][0]
            count_inter[cls_inx] += min_inter
            count_intra[cls_inx] += min_intra

        if imb:
            r = np.array([])
            for i in range(len(count_inter)):
                if count_inter[i] == 0:
                    r = np.append(r, 0)
                else:
                    r = np.append(r, (count_intra[i] / count_inter[i]))
            N2_val = np.divide(r, (1 + r))
        else:
            if sum(count_inter) == 0:
                r = 0
            else:
                r = sum(count_intra) / sum(count_inter)
            N2_val = r / (1 + r)

        return N2_val

    def _find_nearest_opposite_class(self, x_inx, x_dist):
        '''Find nearest sample of opposite class'''
        nearest_opposite_class_dist = np.inf
        nearest_opposite_class_inx = None

        for i in range(len(x_dist)):
            if x_dist[i] < nearest_opposite_class_dist and self.y[x_inx] != self.y[i]:
                nearest_opposite_class_dist = x_dist[i]
                nearest_opposite_class_inx = i

        return nearest_opposite_class_inx, nearest_opposite_class_dist

    def _find_nearest_opposite_class_all(self, dist_matrix=None):
        '''Find nearest opposite class for all samples'''
        if dist_matrix is None:
            dist_matrix = self.dist_matrix

        nearest_opposite_class_array = []
        nearest_opposite_class_dist_array = []

        for i in range(len(dist_matrix)):
            nearest_opposite_class_inx, nearest_opposite_class_dist = self._find_nearest_opposite_class(
                i, dist_matrix[i]
            )
            nearest_opposite_class_array.append(nearest_opposite_class_inx)
            nearest_opposite_class_dist_array.append(nearest_opposite_class_dist)

        return np.array(nearest_opposite_class_array), np.array(nearest_opposite_class_dist_array)

    def _find_spheres(self, ind, e_ind, e_dist, radius):
        '''Calculate hypersphere radius for T1'''
        if radius[ind] >= 0.0:
            return radius[ind]

        ind_enemy = e_ind[ind]

        if ind == e_ind[ind_enemy]:
            radius[ind_enemy] = 0.5 * e_dist[ind]
            radius[ind] = 0.5 * e_dist[ind]
            return radius[ind]

        radius[ind] = 0.0
        radius_enemy = self._find_spheres(ind_enemy, e_ind, e_dist, radius)
        radius[ind] = abs(e_dist[ind] - radius_enemy)

        return radius[ind]

    def _is_inside(self, center_a, center_b, radius_a, radius_b):
        '''Check if hypersphere a is inside hypersphere b'''
        distance_centers = np.sqrt(sum(np.square(center_a - center_b)))
        if distance_centers + radius_a <= radius_b:
            return True
        return False

    def _scale_N(self, N):
        '''Scale features to [0, 1]'''
        N_scaled = N
        if not np.allclose(1.0, np.max(N, axis=0)) or not np.allclose(0.0, np.min(N, axis=0)):
            N_scaled = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(N)
        return N_scaled

    def _remove_overlapped_spheres(self, radius):
        '''Remove hyperspheres completely inside another'''
        X = self.X
        inx_sorted = np.argsort(radius)
        inst_per_sphere = np.ones(len(self.X), dtype=int)

        for inx1, inx1_sphere in enumerate(inx_sorted[:-1]):
            for inx2_sphere in inx_sorted[:inx1:-1]:
                if self._is_inside(X[inx1_sphere], X[inx2_sphere], radius[inx1_sphere], radius[inx2_sphere]):
                    inst_per_sphere[inx2_sphere] += inst_per_sphere[inx1_sphere]
                    inst_per_sphere[inx1_sphere] = 0
                    break

        return inst_per_sphere

    def _get_sphere_count(self):
        '''Calculate sphere count for T1'''
        e_ind, e_dist = self._find_nearest_opposite_class_all(dist_matrix=self.unnorm_dist_matrix)
        radius = np.array([-1.0] * len(e_ind))

        for ind in range(len(radius)):
            if radius[ind] < 0.0:
                self._find_spheres(ind, e_ind, e_dist, radius)

        sphere_inst_count = self._remove_overlapped_spheres(radius)

        return sphere_inst_count, radius

    def T1(self, imb=False):
        '''
        Fraction of hyperspheres covering data.
        
        Parameters:
        -----------
        imb : bool
            If True, return separate values for each class
        '''
        if len(self.sphere_inst_count_T1) == 0:
            self.sphere_inst_count_T1, self.radius_T1 = self._get_sphere_count()

        sphere_inst_count = self.sphere_inst_count_T1

        if imb:
            inx = np.where(sphere_inst_count != 0)[0]
            num_inx_per_class = np.zeros(len(self.classes))
            for i in inx:
                cls_inx = np.where(self.classes == self.y[i])[0][0]
                num_inx_per_class[cls_inx] += 1
            t1 = np.divide(num_inx_per_class, self.class_count)
        else:
            t1 = len(sphere_inst_count[sphere_inst_count > 0]) / len(self.y)

        return t1

    def DBC(self, distance_func="default", imb=False, sphere_count_method="ONB"):
        '''
        Distance-based complexity.
        
        Parameters:
        -----------
        distance_func : str
            Distance function to use
        imb : bool
            If True, return separate values for each class
        sphere_count_method : str
            Method to calculate spheres ('T1' or 'ONB')
        '''
        if sphere_count_method == "T1":
            if len(self.sphere_inst_count_T1) == 0:
                self.sphere_inst_count_T1, self.radius_T1 = self._get_sphere_count()
            sphere_inst_count = self.sphere_inst_count_T1
            inx = np.where(sphere_inst_count != 0)[0]
            sphere_inst_count = sphere_inst_count[inx]

        elif sphere_count_method == "ONB":
            if len(self.sphere_tuple_ONB) == 0:
                self.sphere_tuple_ONB = self._get_ONB_spheres()
            sphere_inst_count = [x[0] for x in self.sphere_tuple_ONB]
            inx = [x[2] for x in self.sphere_tuple_ONB]
        else:
            print("Choose a valid sphere_count_method (T1 or ONB)")
            return

        num_inx_per_class = np.zeros(len(self.classes))

        if imb:
            for i in inx:
                cls_inx = np.where(self.classes == self.y[i])[0][0]
                num_inx_per_class[cls_inx] += 1

        new_X = self.X[inx]
        new_y = self.y[inx]

        new_dist_matrix, _ = self._calculate_distance_matrix(new_X, distance_func=distance_func)
        n_inter = self._calculate_n_inter(dist_matrix=new_dist_matrix, y=new_y, imb=imb)

        if imb:
            dbc_measure = np.divide(n_inter, num_inx_per_class)
        else:
            dbc_measure = n_inter / len(sphere_inst_count)

        return dbc_measure

    def LSC(self, imb=False):
        '''
        Local Set Average Cardinality.
        
        Parameters:
        -----------
        imb : bool
            If True, return separate values for each class
        '''
        nearest_enemy_inx, nearest_enemy_dist = self._find_nearest_opposite_class_all()

        if imb:
            ls_count = []
            for i in range(len(self.classes)):
                ls_count.append([])
        else:
            ls_count = []

        for i in range(len(self.dist_matrix)):
            count = 0
            for j in range(len(self.dist_matrix[i])):
                if self.y[i] == self.y[j] and self.dist_matrix[i][j] < nearest_enemy_dist[i]:
                    count += 1

            if imb:
                cls_inx = np.where(self.classes == self.y[i])[0][0]
                ls_count[cls_inx].append(count)
            else:
                ls_count.append(count)

        if imb:
            ls_sum = []
            for i in range(len(self.classes)):
                ls_sum.append(sum(ls_count[i]))
            lsc_measure = 1 - (np.divide(ls_sum, self.class_count ** 2))
        else:
            lsc_measure = 1 - (sum(ls_count) / (len(self.X) ** 2))

        return lsc_measure

    def Clust(self, imb=False):
        '''
        Clustering-based metric.
        
        Parameters:
        -----------
        imb : bool
            If True, return separate values for each class
        '''
        nearest_enemy_inx, nearest_enemy_dist = self._find_nearest_opposite_class_all()

        ls_count = []
        for i in range(len(self.class_count)):
            ls_count.append([])

        for i in range(len(self.dist_matrix)):
            count = 0
            inxs = []
            for j in range(len(self.dist_matrix[i])):
                if i != j and self.y[i] == self.y[j] and self.dist_matrix[i][j] < nearest_enemy_dist[i]:
                    count += 1
                    inxs.append(j)
            cls_inx = np.where(self.classes == self.y[i])[0][0]
            ls_count[cls_inx].append((count, i, inxs))

        core_count = np.zeros(len(self.classes))
        for i in range(len(ls_count)):
            clusters = []
            ls_count[i].sort(key=itemgetter(0), reverse=True)

            for i2 in range(len(ls_count[i])):
                inCluster = False
                for c in clusters:
                    clusterCoreLocalSet = ls_count[i][c[0]][2]
                    if ls_count[i][i2][1] in clusterCoreLocalSet:
                        c[2].append(ls_count[i][i2][1])
                        inCluster = True
                if not inCluster:
                    clusterCoreInx = ls_count[i].index(ls_count[i][i2])
                    clusterCore = ls_count[i][i2]
                    clusterMembers = [clusterCore]
                    cluster = [clusterCoreInx, clusterCore, clusterMembers]
                    clusters.append(cluster)
            core_count[i] = len(clusters)

        if imb:
            clust_measure = np.divide(core_count, self.class_count)
        else:
            clust_measure = sum(core_count) / len(self.X)

        return clust_measure

    def NSG(self, imb=False, sphere_count_method="ONB"):
        '''
        Non-linearity of support vector classifier.
        
        Parameters:
        -----------
        imb : bool
            If True, return separate values for each class
        sphere_count_method : str
            Method to calculate spheres ('T1' or 'ONB')
        '''
        if sphere_count_method == "T1":
            if len(self.sphere_inst_count_T1) == 0:
                self.sphere_inst_count_T1, self.radius_T1 = self._get_sphere_count()
            sphere_inst_count = self.sphere_inst_count_T1
            inx = np.where(sphere_inst_count != 0)[0]
            sphere_inst_count = sphere_inst_count[inx]

        elif sphere_count_method == "ONB":
            if len(self.sphere_tuple_ONB) == 0:
                self.sphere_tuple_ONB = self._get_ONB_spheres()
            sphere_inst_count = [x[0] for x in self.sphere_tuple_ONB]
            inx = [x[2] for x in self.sphere_tuple_ONB]
        else:
            print("Choose a valid sphere_count_method (T1 or ONB)")
            return

        if imb:
            num_inx_per_class = np.zeros(len(self.classes))
            for i in inx:
                cls_inx = np.where(self.classes == self.y[i])[0][0]
                num_inx_per_class[cls_inx] += 1
            nsg_measure = np.divide(self.class_count, num_inx_per_class)
        else:
            nsg_measure = sum(sphere_inst_count) / len(sphere_inst_count)

        return nsg_measure

    def ICSV(self, normalize=True, imb=False, sphere_count_method="ONB"):
        '''
        Inter-class coefficient of variation.
        
        Parameters:
        -----------
        normalize : bool
            Whether to normalize radius
        imb : bool
            If True, return separate values for each class
        sphere_count_method : str
            Method to calculate spheres ('T1' or 'ONB')
        '''
        if sphere_count_method == "T1":
            if len(self.sphere_inst_count_T1) == 0:
                self.sphere_inst_count_T1, self.radius_T1 = self._get_sphere_count()
            sphere_inst_count = self.sphere_inst_count_T1
            radius = self.radius_T1
            inx = np.where(sphere_inst_count != 0)[0]
            sphere_inst_count = sphere_inst_count[inx]
            radius_non_zero = radius[inx]

        elif sphere_count_method == "ONB":
            if len(self.sphere_tuple_ONB) == 0:
                self.sphere_tuple_ONB = self._get_ONB_spheres()
            sphere_inst_count = [x[0] for x in self.sphere_tuple_ONB]
            inx = [x[2] for x in self.sphere_tuple_ONB]
            radius_non_zero = [x[3] for x in self.sphere_tuple_ONB]
        else:
            print("Choose a valid sphere_count_method (T1 or ONB)")
            return

        radius_non_zero = np.array(radius_non_zero)
        if normalize:
            radius_non_zero = radius_non_zero / np.max(radius_non_zero)

        n = len(self.X[0])
        density = []
        volumes = []

        for i in range(len(sphere_inst_count)):
            volumes.append((math.pi ** (n / 2) / math.gamma(n / 2 + 1)) * radius_non_zero[i] ** n)

        for i in range(len(sphere_inst_count)):
            density.append(sphere_inst_count[i] / volumes[i])

        if imb:
            density_per_class = []
            for i in range(len(self.classes)):
                density_per_class.append([])

            j = 0
            for i in inx:
                cls_inx = np.where(self.classes == self.y[i])[0][0]
                density_per_class[cls_inx].append(density[j])
                j += 1

            icsv_measure = []
            for i in range(len(self.classes)):
                icsv_measure.append(np.std(density_per_class[i]))
        else:
            icsv_measure = np.std(density)

        return icsv_measure

    def _get_ONB_spheres(self):
        '''Calculate ONB spheres'''
        featu = pd.DataFrame(self.X)
        clas = pd.DataFrame(self.y)
        clas.rename(columns={0: 'class'}, inplace=True)
        dataset = featu.join(clas)

        clas_dif = self.classes
        dtf_dist = pd.DataFrame(self.dist_matrix)

        lista_el_cla = []
        el_cla = []

        for cla in clas_dif:
            lista_el_cla = lista_el_cla + [list(dataset.loc[dataset["class"] == cla].index.values)]
            el_cla = el_cla + [len(list(dataset.loc[dataset["class"] == cla].index.values))]

        cl = 0
        sphere_coverage = []

        for cla in clas_dif:
            falta = lista_el_cla[cl]

            while len(falta) != 0:
                bolaslist = []
                center = falta[0]
                r = 0

                for j in falta:
                    mininter = min(dtf_dist.loc[j, dataset["class"] != cla])
                    s = dtf_dist.iloc[j, falta] <= mininter
                    bolas = [i for i in s[s].index.values]

                    if len(bolas) > len(bolaslist):
                        center = j
                        bolaslist = bolas
                        r = mininter

                sphere_coverage.append((len(bolaslist), self.classes[cl], center, r))

                for ele in sorted(bolaslist, reverse=True):
                    del falta[falta.index(ele)]

            cl += 1

        return sphere_coverage

    def ONB(self, imb=False, is_tot=False):
        '''
        Overlap Number of Balls.
        
        Parameters:
        -----------
        imb : bool
            If True, return separate values for each class
        is_tot : bool
            If True, return total balls instead of average
        '''
        if len(self.sphere_tuple_ONB) == 0:
            self.sphere_tuple_ONB = self._get_ONB_spheres()

        b_list = self.sphere_tuple_ONB
        tot = len(b_list)

        avg = np.zeros(len(self.classes))
        for b in b_list:
            avg[b[1]] += 1

        avg = np.divide(avg, self.class_count)

        if is_tot:
            return tot / len(self.X)
        else:
            if imb:
                return avg
            else:
                return sum(avg) / len(self.classes)

    def _calculate_cells(self, resolution, transpose_X, get_labels=0):
        '''Create cells for purity and neighbourhood_separability'''
        feature_bounds = []
        steps = []

        for j in range(len(self.X[0])):
            min_feature = min(transpose_X[j])
            max_feature = max(transpose_X[j])
            step = (max_feature - min_feature) / (resolution + 1)
            steps.append(step)

            if step == 0:
                feature_bounds.append([min_feature, max_feature])
            else:
                feature_bounds.append(np.linspace(min_feature, max_feature, num=2 + resolution))

        sample_dic = {}

        for s in range(len(self.X)):
            sample = self.X[s]
            for j in range(len(self.X[0])):
                for k in range(len(feature_bounds[j])):
                    if sample[j] >= feature_bounds[j][k] and sample[j] <= feature_bounds[j][k] + steps[j]:
                        if str(s) not in sample_dic:
                            sample_dic[str(s)] = "" + str(k)
                        else:
                            sample_dic[str(s)] += "-" + str(k)
                        break

        reverse_dic = {}
        reverse_dic_labels = {}

        for k, v in sample_dic.items():
            reverse_dic[v] = reverse_dic.get(v, [])
            reverse_dic_labels[v] = reverse_dic_labels.get(v, [])

            if get_labels == 1:
                reverse_dic_labels[v].append(self.y[int(k)])

            reverse_dic[v].append(int(k))

        return reverse_dic_labels, reverse_dic

    def purity(self, max_resolution=32):
        '''
        Purity measure (multi-resolution).
        
        Parameters:
        -----------
        max_resolution : int
            Maximum resolution for partitioning
        '''
        transpose_X = np.transpose(self.X)
        purities = []

        for i in range(max_resolution):
            reverse_dic, __ = self._calculate_cells(i, transpose_X, get_labels=1)
            purity = 0

            for cell in reverse_dic:
                classes = self.classes
                class_counts = [0] * len(classes)
                num_classes = len(classes)

                for label in reverse_dic[cell]:
                    class_counts[np.where(classes == label)[0][0]] += 1

                class_sum = 0
                for count in class_counts:
                    class_sum += ((count / sum(class_counts)) - (1 / num_classes)) ** 2

                cell_purity = math.sqrt((num_classes / (num_classes - 1)) * class_sum)
                purity += cell_purity * (sum(class_counts)) / len(self.X)

            purities.append(purity)

        w_purities = []
        for i in range(len(purities)):
            new_p = purities[i] * (1 / 2 ** i)
            w_purities.append(new_p)

        norm_resolutions = [x / (max_resolution - 1) for x in list(range(max_resolution))]
        norm_purities = [(x - min(w_purities)) / (max(w_purities) - min(w_purities)) for x in w_purities]

        auc = sklearn.metrics.auc(norm_resolutions, norm_purities)
        pur = auc / 0.702

        return pur

    def neighbourhood_separability(self, max_resolution=32):
        '''
        Neighbourhood separability (multi-resolution).
        
        Parameters:
        -----------
        max_resolution : int
            Maximum resolution for partitioning
        '''
        transpose_X = np.transpose(self.X)
        neigh_sep = []

        for i in range(max_resolution):
            reverse_dic_labels, reverse_dic = self._calculate_cells(i, transpose_X, get_labels=1)
            average_ns = 0

            for cell in reverse_dic:
                average_auc = 0

                for sample in reverse_dic[cell]:
                    props = []
                    same_class_num = len(np.where(reverse_dic_labels[cell] == self.y[sample])[0]) - 1

                    if same_class_num > 11:
                        same_class_num = 11

                    for k in range(same_class_num):
                        count = self._knn(
                            reverse_dic[cell].index(sample),
                            copy.copy(self.dist_matrix[sample, reverse_dic[cell]]),
                            k + 1
                        )
                        cls_inx = np.where(self.classes == self.y[sample])[0][0]
                        class_count = count[cls_inx]
                        prop = class_count / sum(count)
                        props.append(prop)

                    if len(props) == 0 and len(reverse_dic[cell]) > 1:
                        auc = 0
                    elif len(props) == 0 and len(reverse_dic[cell]) == 1:
                        auc = 1
                    elif len(props) == 1:
                        auc = props[0]
                    else:
                        norm_k = [x / same_class_num for x in list(range(same_class_num))]
                        auc = sklearn.metrics.auc(norm_k, props)

                    average_auc += auc

                average_auc /= len(reverse_dic[cell])
                average_ns += average_auc * (len(reverse_dic[cell]) / len(self.X))

            neigh_sep.append(average_ns)

        w_neigh_sep = []
        for i in range(len(neigh_sep)):
            w_neigh_sep.append(neigh_sep[i] * (1 / 2 ** i))

        norm_resolutions = [x / (max_resolution - 1) for x in list(range(max_resolution))]
        final_auc = sklearn.metrics.auc(norm_resolutions, w_neigh_sep)

        return final_auc

    # ==================== MULTI-RESOLUTION METRICS ====================

    def _MRI_p(self, profile):
        '''Calculate MRI value of a pattern'''
        sum_val = 0
        m = len(profile)
        for j in range(m):
            w = (1 - (j / m))
            sum_val += w * (1 - profile[j])
        mri_val = sum_val / (2 * m)
        return mri_val

    def _MRI_k(self, cluster):
        '''Calculate MRI for a cluster'''
        sum_val = 0
        for i in range(len(cluster)):
            profile = cluster[i]
            sum_val += self._MRI_p(profile)
        mri_val = sum_val / len(cluster)
        return mri_val

    def MRCA(self, sigmas=[0.25, 0.5, 0.75], n_clusters=3, distance_func="default"):
        '''
        Multiresolution Clustering Analysis.
        
        Parameters:
        -----------
        sigmas : list
            Multiple hypersphere radii
        n_clusters : int
            Number of clusters for KMeans
        distance_func : str
            Distance function to use
        '''
        for i2 in range(len(self.class_inxs)):
            for j2 in range(i2 + 1, len(self.class_inxs)):
                c1 = self.classes[i2]
                c2 = self.classes[j2]
                sample_c1 = self.X[self.class_inxs[i2]]
                sample_c2 = self.X[self.class_inxs[j2]]
                sample_c1_y = self.y[self.class_inxs[i2]]
                sample_c2_y = self.y[self.class_inxs[j2]]
                new_X = np.concatenate([sample_c1, sample_c2], axis=0)
                new_y = np.concatenate([sample_c1_y, sample_c2_y], axis=0)
                new_dist_matrix, _ = self._calculate_distance_matrix(new_X, distance_func=distance_func)

                mrca = np.zeros(n_clusters)
                profiles = np.zeros((len(new_X), len(sigmas)))

                for i in range(len(new_X)):
                    for j in range(len(sigmas)):
                        sigma = sigmas[j]
                        n = self._hypersphere(i, sigma, distance_matrix=new_dist_matrix, y=new_y)

                        if new_y[i] == c1:
                            alt_y = 1
                            psi = alt_y * ((n[1] - n[0]) / (n[1] + n[0]))
                        else:
                            alt_y = -1
                            psi = alt_y * ((n[0] - n[1]) / (n[0] + n[1]))

                        profiles[i, j] = psi

                kmeans = KMeans(n_clusters=n_clusters).fit(profiles)

                for i in range(n_clusters):
                    inx = np.where(kmeans.labels_ == i)[0]
                    cluster = profiles[inx]
                    mrca[i] = self._MRI_k(cluster)

                return mrca

    def C1(self, max_k=5, imb=False):
        '''
        Entropy of class proportions.
        
        Parameters:
        -----------
        max_k : int
            Maximum number of neighbors
        imb : bool
            If True, return separate values for each class
        '''
        c1_sum = np.zeros(len(self.classes))

        for i in range(len(self.X)):
            c1_instance_sum = 0
            cls_inx = np.where(self.classes == self.y[i])[0][0]

            for k in range(1, max_k + 1):
                count = self._knn(i, copy.copy(self.dist_matrix[i]), k)
                pkj = count[cls_inx] / k
                c1_instance_sum += pkj

            c1_instance_val = 1 - (c1_instance_sum / max_k)
            c1_sum[cls_inx] += c1_instance_val

        if imb:
            c1_val = np.divide(c1_sum, self.class_count)
        else:
            c1_val = sum(c1_sum) / len(self.X)

        return c1_val

    def C2(self, max_k=5, imb=False):
        '''
        Weighted distance-based class entropy.
        
        Parameters:
        -----------
        max_k : int
            Maximum number of neighbors
        imb : bool
            If True, return separate values for each class
        '''
        c2_sum = np.zeros(len(self.classes))

        for i in range(len(self.X)):
            c2_instance_sum = 0
            cls_inx = np.where(self.classes == self.y[i])[0][0]

            for k in range(1, max_k + 1):
                dists = self._knn_dists(i, copy.copy(self.dist_matrix[i]), k)
                pkj = 0
                for d in dists:
                    if d > 1:
                        d = 1
                    pkj += 1 - d
                pkj /= k
                c2_instance_sum += pkj

            c2_instance_val = 1 - (c2_instance_sum / max_k)
            c2_sum[cls_inx] += c2_instance_val

        if imb:
            c2_val = np.divide(c2_sum, self.class_count)
        else:
            c2_val = sum(c2_sum) / len(self.X)

        return c2_val

    # ==================== ADDITIONAL METRICS ====================

    def svm_reduction(self, kernel='rbf', gamma=0.1, C=100.0):
        '''
        SVM support vector reduction ratio.
        
        Parameters:
        -----------
        kernel : str
            SVM kernel type
        gamma : float
            Kernel coefficient
        C : float
            Regularization parameter
        '''
        from sklearn.svm import SVC
        
        svm = SVC(kernel=kernel, gamma=gamma, C=C)
        svm.fit(self.X, self.y)
        
        class_count_reduced = self._count_class_instances(self.y[svm.support_])
        reduction = 1 - np.divide(class_count_reduced, self.class_count)
        
        return reduction

    def tree_depth(self, tree_type="DT", n_estimators=100):
        '''
        Decision tree or Random forest depth.
        
        Parameters:
        -----------
        tree_type : str
            'DT' for decision tree, 'RF' for random forest
        n_estimators : int
            Number of trees (for RF only)
        '''
        if tree_type == "RF":
            rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            rf.fit(self.X, self.y)
            tree_depths = [estimator.tree_.max_depth for estimator in rf.estimators_]
            avg_tree_depth = sum(tree_depths) / len(tree_depths)
            return avg_tree_depth
        
        elif tree_type == "DT":
            dt = DecisionTreeClassifier()
            dt.fit(self.X, self.y)
            tree_depth = dt.get_depth()
            return tree_depth


# ==================== WRAPPER CLASS FOR EASY USE ====================

class ComplexityMeasures:
    """
    User-friendly wrapper for ComplexityMetrics with all complexity measures.
    
    This class provides easy access to all complexity measures with options to:
    - Get all measures at once
    - Get specific categories (feature/instance/structural/multiresolution)
    - Get specific measures by name
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target vector
    distance_func : str, default="default"
        Distance function to use
    
    Example
    -------
    >>> cm = ComplexityMeasures(X, y)
    >>> 
    >>> # Get all measures
    >>> all_measures = cm.get_all_complexity_measures()
    >>> 
    >>> # Get specific category
    >>> feature_measures = cm.get_all_complexity_measures(measures='feature')
    >>> 
    >>> # Get specific measures
    >>> selected = cm.get_all_complexity_measures(measures=['N3', 'F1', 'N1'])
    >>> 
    >>> # Quick analysis
    >>> basic = cm.analyze_overlap()
    """
    
    def __init__(self, X, y, distance_func="default"):
        """Initialize complexity measures calculator."""
        self.metrics = ComplexityMetrics(X, y)
        self.X = self.metrics.X
        self.y = self.metrics.y
        self.classes = self.metrics.classes
        self.class_count = self.metrics.class_count
        self.class_inxs = self.metrics.class_inxs
        self.dist_matrix = self.metrics.dist_matrix
    
    def get_all_complexity_measures(self, k=5, imb=False, measures='all'):
        """
        Get all available complexity measures in a structured format.
        
        Parameters
        ----------
        k : int, default=5
            Number of neighbors for k-NN based measures
        imb : bool, default=False
            Whether to return class-specific values for imbalanced analysis
        measures : str or list, default='all'
            Which measures to calculate:
            - 'all': All available measures
            - 'basic': Essential measures (N3, N1, N2, F1, F2)
            - 'feature': Only feature overlap measures
            - 'instance': Only instance overlap measures
            - 'structural': Only structural overlap measures
            - 'multiresolution': Only multiresolution measures
            - list: Specific measure names (e.g., ['N3', 'F1', 'N1'])
        
        Returns
        -------
        results : dict
            Structured dictionary with all complexity measures
        """
        results = {
            'dataset_info': {
                'n_samples': len(self.X),
                'n_features': self.X.shape[1],
                'n_classes': len(self.classes),
                'class_distribution': np.bincount(self.y).tolist(),
                'imbalance_ratio': np.max(np.bincount(self.y)) / np.min(np.bincount(self.y))
            },
            'feature_overlap': {},
            'instance_overlap': {},
            'structural_overlap': {},
            'multiresolution_overlap': {}
        }
        
        # Define measure categories
        feature_measures = ['F1', 'F1v', 'F2', 'F3', 'F4', 'input_noise']
        instance_measures = ['N3', 'N4', 'kDN', 'CM', 'R_value', 'D3_value', 'SI', 
                           'borderline', 'deg_overlap']
        structural_measures = ['N1', 'N2', 'T1', 'DBC', 'LSC', 'Clust', 'NSG', 
                             'ICSV', 'ONB']
        multiresolution_measures = ['purity', 'neighbourhood_separability', 'MRCA', 
                                   'C1', 'C2']
        
        # Determine which measures to calculate
        if measures == 'basic':
            calc_measures = ['N3', 'N1', 'N2', 'F1', 'F2']
        elif measures == 'all':
            calc_measures = (feature_measures + instance_measures + 
                           structural_measures + multiresolution_measures)
        elif measures == 'feature':
            calc_measures = feature_measures
        elif measures == 'instance':
            calc_measures = instance_measures
        elif measures == 'structural':
            calc_measures = structural_measures
        elif measures == 'multiresolution':
            calc_measures = multiresolution_measures
        elif isinstance(measures, list):
            calc_measures = measures
        else:
            calc_measures = ['N3', 'N1', 'N2', 'F1', 'F2']
        
        # Calculate measures
        for measure_name in calc_measures:
            try:
                if measure_name == 'F1':
                    results['feature_overlap']['F1'] = self.metrics.F1()
                elif measure_name == 'F1v':
                    results['feature_overlap']['F1v'] = self.metrics.F1v()
                elif measure_name == 'F2':
                    results['feature_overlap']['F2'] = self.metrics.F2(imb=imb)
                elif measure_name == 'F3':
                    results['feature_overlap']['F3'] = self.metrics.F3(imb=imb)
                elif measure_name == 'F4':
                    results['feature_overlap']['F4'] = self.metrics.F4(imb=imb)
                elif measure_name == 'input_noise':
                    results['feature_overlap']['input_noise'] = self.metrics.input_noise(imb=imb)
                
                elif measure_name == 'N3':
                    results['instance_overlap']['N3'] = self.metrics.N3(k=k, imb=imb)
                elif measure_name == 'N4':
                    results['instance_overlap']['N4'] = self.metrics.N4(k=k, imb=imb)
                elif measure_name == 'kDN':
                    results['instance_overlap']['kDN'] = self.metrics.kDN(k=k, imb=imb)
                elif measure_name == 'CM':
                    results['instance_overlap']['CM'] = self.metrics.CM(k=k, imb=imb)
                elif measure_name == 'R_value':
                    results['instance_overlap']['R_value'] = self.metrics.R_value(k=k, imb=imb)
                elif measure_name == 'D3_value':
                    results['instance_overlap']['D3'] = self.metrics.D3_value(k=k)
                elif measure_name == 'SI':
                    results['instance_overlap']['SI'] = self.metrics.SI(k=k, imb=imb)
                elif measure_name == 'borderline':
                    results['instance_overlap']['borderline'] = self.metrics.borderline(imb=imb)
                elif measure_name == 'deg_overlap':
                    results['instance_overlap']['deg_overlap'] = self.metrics.deg_overlap(k=k, imb=imb)
                
                elif measure_name == 'N1':
                    results['structural_overlap']['N1'] = self.metrics.N1(imb=imb)
                elif measure_name == 'N2':
                    results['structural_overlap']['N2'] = self.metrics.N2(imb=imb)
                elif measure_name == 'T1':
                    results['structural_overlap']['T1'] = self.metrics.T1(imb=imb)
                elif measure_name == 'DBC':
                    results['structural_overlap']['DBC'] = self.metrics.DBC(imb=imb)
                elif measure_name == 'LSC':
                    results['structural_overlap']['LSC'] = self.metrics.LSC(imb=imb)
                elif measure_name == 'Clust':
                    results['structural_overlap']['Clust'] = self.metrics.Clust(imb=imb)
                elif measure_name == 'NSG':
                    results['structural_overlap']['NSG'] = self.metrics.NSG(imb=imb)
                elif measure_name == 'ICSV':
                    results['structural_overlap']['ICSV'] = self.metrics.ICSV(imb=imb)
                elif measure_name == 'ONB':
                    results['structural_overlap']['ONB'] = self.metrics.ONB(imb=imb)
                
                elif measure_name == 'purity':
                    results['multiresolution_overlap']['purity'] = self.metrics.purity()
                elif measure_name == 'neighbourhood_separability':
                    results['multiresolution_overlap']['neighbourhood_separability'] = self.metrics.neighbourhood_separability()
                elif measure_name == 'MRCA':
                    results['multiresolution_overlap']['MRCA'] = self.metrics.MRCA()
                elif measure_name == 'C1':
                    results['multiresolution_overlap']['C1'] = self.metrics.C1(imb=imb)
                elif measure_name == 'C2':
                    results['multiresolution_overlap']['C2'] = self.metrics.C2(imb=imb)
            
            except Exception as e:
                if measures != 'all':  # Only warn if not calculating all
                    print(f"Warning: Could not calculate {measure_name}: {e}")
        
        return results
    
    def analyze_overlap(self, measures='basic'):
        """
        Quick overlap analysis using essential measures.
        
        Parameters
        ----------
        measures : str, default='basic'
            Level of analysis:
            - 'basic': Essential measures (N3, N1, N2, F1, F2)
            - 'standard': Common measures
            - 'all': All available measures
        
        Returns
        -------
        results : dict
            Dictionary containing complexity measures
        """
        if measures == 'basic':
            results = {
                'n_samples': len(self.X),
                'n_features': self.X.shape[1],
                'n_classes': len(self.classes),
                'class_distribution': np.bincount(self.y).tolist(),
                'imbalance_ratio': np.max(np.bincount(self.y)) / np.min(np.bincount(self.y)),
                'N3': self._safe_calc(lambda: self.metrics.N3(k=1)),
                'N1': self._safe_calc(lambda: self.metrics.N1()),
                'N2': self._safe_calc(lambda: self.metrics.N2()),
                'F1': self._safe_calc(lambda: np.mean(self.metrics.F1())),
                'F2': self._safe_calc(lambda: np.mean(self.metrics.F2())),
            }
        elif measures == 'standard':
            all_measures = self.get_all_complexity_measures(measures='all')
            results = self._flatten_dict(all_measures)
        elif measures == 'all':
            all_measures = self.get_all_complexity_measures(measures='all')
            results = all_measures
        else:
            results = self.analyze_overlap('basic')
        
        return results
    
    def _safe_calc(self, func):
        """Safely calculate a measure."""
        try:
            return func()
        except:
            return None
    
    def _flatten_dict(self, nested_dict):
        """Flatten nested dictionary."""
        flat = {}
        for category, measures in nested_dict.items():
            if isinstance(measures, dict):
                for key, value in measures.items():
                    if isinstance(value, (list, np.ndarray)):
                        try:
                            flat[key] = float(np.mean(value))
                        except:
                            flat[key] = value
                    else:
                        flat[key] = value
            else:
                flat[category] = measures
        return flat


def compare_pre_post_overlap(X_original, y_original, X_resampled, y_resampled):
    """
    Compare complexity measures before and after resampling.
    
    Parameters
    ----------
    X_original : array-like
        Original feature matrix
    y_original : array-like
        Original target vector
    X_resampled : array-like
        Resampled feature matrix
    y_resampled : array-like
        Resampled target vector
        
    Returns
    -------
    comparison : dict
        Dictionary containing before/after comparison
    """
    cm_original = ComplexityMeasures(X_original, y_original)
    original_measures = cm_original.analyze_overlap()
    
    cm_resampled = ComplexityMeasures(X_resampled, y_resampled)
    resampled_measures = cm_resampled.analyze_overlap()
    
    comparison = {
        'original': original_measures,
        'resampled': resampled_measures,
        'improvements': {
            'N3_reduction': original_measures.get('N3', 0) - resampled_measures.get('N3', 0),
            'N1_reduction': original_measures.get('N1', 0) - resampled_measures.get('N1', 0),
            'N2_improvement': resampled_measures.get('N2', 0) - original_measures.get('N2', 0),
            'F1_improvement': resampled_measures.get('F1', 0) - original_measures.get('F1', 0),
            'F2_improvement': original_measures.get('F2', 0) - resampled_measures.get('F2', 0),
            'imbalance_improvement': original_measures.get('imbalance_ratio', 0) - resampled_measures.get('imbalance_ratio', 0)
        }
    }
    
    return comparison
