The Adelaide Cube FH dataset is comprised of eight stereo pairs, extracted from the AdelaideRMF, where a cube togheter with other object undergoes a rigid motion. A multi-heterogenous-model classification of the 2D corresponeces is presented. The 2D point-correspondences of these sequences are described either with different  fundamental matrices (if the corresponding 3D points lie on a generic objet) or with an homography (when the corresponding 3D points lie on one of the face of the cube).

These data are distributed in the hope that they will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


These datasets are extracted from the AdelaideRMF: Robust Model Fitting Data Set 
(available at https://cs.adelaide.edu.au/~hwong/doku.php?id=data)
W.r.t. the orginal data, points lying on the faces of the cube have been splitted in clusters described by a planar homography.


To use this dataset, please cite these pubblications:

1) L.Magri, A.Fusiello, Fitting Multiple Heterogeneous Models by Multi-class Cascaded T-linkage, CVPR 2019

2) H. S. Wong, T.-J. Chin, J. Yu and D. Suter, Dynamic and Hierarchical Multi-Structure Geometric Model Fitting. ICCV 2011.


@INPROCEEDINGS{magfus2019,
  author = {L.Magri and A.Fusiello},
  title = {Fitting Multiple Heterogeneous Models by Multi-class Cascaded T-linkage},
  booktitle = {International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}

@INPROCEEDINGS{wongiccv2011,
  author = {H. S. Wong and T.-J. Chin and J. Yu and D. Suter},
  title = {Dynamic and Hierarchical Multi-Structure Geometric Model Fitting},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year = {2011}
}

Thank you!

X is a 6xN data vector. Each column represents a pair of correspondence expressed in normalized coordinates (centroid is the origin and the mean distance from the origin is sqrt(2)).
y is a 6xN vector of correspondences in the image space.
G is a labeling vector. The label 0 corresponds to outlying points.