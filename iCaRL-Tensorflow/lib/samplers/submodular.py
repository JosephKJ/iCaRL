import numpy as np
import time
from scipy.spatial.distance import cdist
from operator import itemgetter

import tensorflow as tf


class SubmodularSampler:
    def __init__(self, set, subset_size, penultimate_acts, logits):
        self.set = set
        self.subset_size = subset_size
        self.penultimate_activations = penultimate_acts
        self.logits = logits    # Contains output of Softmax (not Sigmoid)
        self.detailed_logging = False

    def get_subset(self):
        return self._select_subset_items()

    def _select_subset_items(self, alpha_1=0.8, alpha_2=1, alpha_3=0.8, alpha_4=1):
        set = self.set.copy()
        index_set = range(0, len(set))  # It contains the indices of each image of the set.

        subset = []
        subset_indices = []     # Subset of indices. Keeping track to improve computational performance.

        # Computing the pairwise distance between all the points once. Used in Diversity Score computation.
        pdist = cdist(self.penultimate_activations, self.penultimate_activations)

        # Computing entropy of all the data points once. Used in Uncertainity score calculation.
        p_log_p = self.logits * tf.log(self.logits)
        H = np.sum(-p_log_p.numpy(), axis=1)

        # Computing distance of each point from the class-mean
        class_mean = np.reshape(np.mean(self.penultimate_activations, axis=0), (1, -1))
        mean_dist = cdist(self.penultimate_activations, class_mean)

        for i in range(0, self.subset_size):
            now = time.time()
            scores = []

            # Compute d_score for the whole subset. Then add the d_score of just the
            # new item to compute the total d_score.
            d_score = np.sum(self._compute_d_score(list(subset_indices), pdist))

            # Same logic for u_score
            u_score = np.sum(self._compute_u_score(list(subset_indices), H))

            # Same logic for md_score
            md_score = np.sum(self._compute_md_score(list(subset_indices), mean_dist))

            d_s = self._normalize(d_score + self._compute_d_score(index_set, pdist))
            u_s = self._normalize(u_score + self._compute_u_score(index_set, H))
            md_s = self._normalize(md_score + self._compute_md_score(index_set, mean_dist))
            r_s = self._normalize(self._compute_r_score(list(subset_indices), index_set))

            scores = alpha_1*d_s + alpha_2*u_s + alpha_3*md_s + alpha_4*r_s

            best_item_index = np.argmax(scores)
            best_item = set[best_item_index]

            subset.append(best_item)
            subset_indices.append(index_set[best_item_index])

            set = np.delete(set, best_item_index, axis=0)
            index_set = np.delete(index_set, best_item_index, axis=0)

            if self.detailed_logging:
                print('Time for processing {0}/{1} exemplar is {2}'.format(i, self.subset_size, time.time()-now))

        return np.array(subset)

    def _normalize(self, vector, eps=0.0001):
        std = np.std(vector)
        return (vector - np.mean(vector)) / (std + eps)

    def _compute_d_score(self, subset_indices, pdist):
        """
        Computes the Diversity Score: The new point should be distant from all the elements in the class.
        :param subset_indices:
        :param alpha:
        :return: d_score
        """
        if len(subset_indices) == 0:
            return 0

        dist = pdist[subset_indices]
        return np.sum(np.array(dist), axis=1)

    def _compute_u_score(self, subset_indices, H):
        """
        Compute the Uncertainity Score: The point that makes the model most confused, should be preferred.
        :param subset_indices:
        :param alpha:
        :return: u_score
        """
        if len(subset_indices) == 0:
            return 0

        scores = H[subset_indices]
        return scores

    def _compute_md_score(self, subset_indices, mean_dist):
        """
        Computes Mean Divergence score: The new datapoint should be close to the class mean.
        :param subset_indices:
        :param class_mean:
        :param class_mean:
        :param class_mean:
        :param alpha:
        :return:
        """
        if len(subset_indices) == 0:
            return 0

        scores = mean_dist[subset_indices]
        return np.reshape(-scores,(-1,))

    def _compute_r_score(self, subset_indices, index_set):
        """
        Computes Redundancy Score: The point should be distant from all the other elements in the subset.
        Select that element that maximize the minimum distance of it from all the other items.
        :param subset_indices:
        :param alpha:
        :return:
        """
        if len(subset_indices) == 0:
            return 0

        if len(index_set) == 0:
            return 0

        if len(subset_indices) == 1:
            return [np.linalg.norm(np.array(itemgetter(*index_set)(self.penultimate_activations)) -
                                   np.array(subset_indices[0]))]

        if len(index_set) == 1:
            return [np.min(np.linalg.norm(np.array(self.penultimate_activations[index_set[0]]) -
                                          np.array((itemgetter(*subset_indices)(self.penultimate_activations)))))]

        index_p_acts = np.array(itemgetter(*index_set)(self.penultimate_activations))
        subset_p_acts = np.array((itemgetter(*subset_indices)(self.penultimate_activations)))
        pdist = cdist(index_p_acts, subset_p_acts)
        r_score = np.min(pdist, axis=1)
        return r_score
