
import numpy as np
import numbers
from collections.abc import Iterable
from sklearn.utils.validation import check_random_state, check_array
from sklearn.utils import shuffle as util_shuffle


def generate_point_within_bounds(base_point, min_dist, max_dist):
    # Generate a random distance and Generate a random direction
    dist = np.random.uniform(min_dist, max_dist)
    direction = np.random.normal(0, 1, base_point.shape)
    direction /= np.linalg.norm(direction)

    # Calculate the new point
    new_point = base_point + dist * direction
    return new_point.reshape(1, -1)

def make_blobs(
        n_samples=20,  # sample per sub-cls
        n_features=2,  # int, default=2 The number of features for each sample.
        center_box=(-10.0, 10.0),  # generate cluster centers within box
        cls_dist=[6, 9],
        sub_cls_dist=[2,4],
        cls_std=2.0,
        sub_cls_std=1.0,
        n_cls=2,
        n_sub_cls=2,
        shuffle=True,
        random_state=None,
):
    generator = check_random_state(random_state)

    # ========= generate the class centers for cls
    min_dist, max_dist = cls_dist
    cls_centers = [generator.uniform(center_box[0], center_box[1], size=(1, n_features))]

    for i in range(1, n_cls):
        dist_constraint = False
        while not dist_constraint:
            _cls_center = generate_point_within_bounds(cls_centers[-1], min_dist, max_dist)
            dist_constraint = True
            for p in cls_centers:
                if not min_dist <= np.linalg.norm(_cls_center - p) <= max_dist:
                    dist_constraint = False
        cls_centers.append(_cls_center)

    cls_centers = np.concatenate(cls_centers)

    # ========= generate the class centers for sub_cls

    all_sub_centers = {}

    min_dist, max_dist = sub_cls_dist
    dist_cls_center = (min_dist+max_dist)/2
    for id_cls in range(n_cls):
        sub_cls_centers = [generate_point_within_bounds(cls_centers[id_cls], dist_cls_center * 0.8, dist_cls_center)]
        for i in range(1, n_sub_cls):
            dist_constraint = False
            while not dist_constraint:
                _sub_cls_centers = generate_point_within_bounds(cls_centers[id_cls], dist_cls_center * 0.8, dist_cls_center)
                dist_constraint = True
                for p in sub_cls_centers:
                    if not min_dist <= np.linalg.norm(_sub_cls_centers - p):
                        dist_constraint = False
            sub_cls_centers.append(_sub_cls_centers)
        sub_cls_centers = np.concatenate(sub_cls_centers, axis=0)

        all_sub_centers[id_cls] = sub_cls_centers

    # ========= generate the samples for sub_cls
    if isinstance(sub_cls_std, numbers.Real):
        sub_cls_std = np.full(n_cls*n_sub_cls, sub_cls_std)

    x, fine_y, coarse_y = [], [], []
    for id_cls in range(n_cls):
        _sub_cls_centers = all_sub_centers[id_cls]
        for id_sub_cls in range(n_sub_cls):
            _x = generator.normal(
                loc=_sub_cls_centers[id_sub_cls], scale=sub_cls_std[id_cls*n_sub_cls + id_sub_cls], size=(n_samples, n_features)
            )
            _fine_y = np.array([id_cls*n_sub_cls + id_sub_cls] * n_samples)
            _coarse_y = np.array([id_cls] * n_samples)

            x.append(_x)
            fine_y.append(_fine_y)
            coarse_y.append(_coarse_y)
    x = np.concatenate(x, axis=0)
    fine_y = np.concatenate(fine_y)
    coarse_y = np.concatenate(coarse_y)

    if shuffle:
        x, fine_y, coarse_y = util_shuffle(x, fine_y, coarse_y, random_state=generator)

    return x, fine_y, coarse_y