import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

input_size = (512, 512)
nb_samples = 1005
batch_size = 64
epochs = 150

source = "C:/Users/s4625266/PycharmProjects/coral/processed_image/"
path = 'C:/Users/s4625266/PycharmProjects/coral/pickled_data'


def feature_extractor():
    conv_base = VGG19(weights='imagenet', include_top=False)

    # define image generator with augmentation
    data_generator = ImageDataGenerator(rescale=1. / 255.,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest')

    generator = data_generator.flow_from_directory(
        source,
        seed=42,
        target_size=input_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    # generate features
    i = 0
    features = np.zeros(shape=(nb_samples, 16, 16, 512))
    for inputs_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        i += 1
        if i * batch_size >= nb_samples:
            break

    return features


def extract_levels(row_clusters):
    clusters = {}
    for row in range(row_clusters.shape[0]):
        cluster_n = row + nb_samples
        # which clusters / labels are present in this row
        glob1, glob2 = row_clusters[row, 0], row_clusters[row, 1]

        # if this is a cluster, pull the cluster
        this_clust = []
        for glob in [glob1, glob2]:
            if glob > (nb_samples - 1):
                this_clust += clusters[glob]
            # if it isn't, add the label to this cluster
            else:
                this_clust.append(glob)

        clusters[cluster_n] = this_clust
    return clusters


def plot_dendrogram(linkage_matrix, **kwargs):
    ddata = dendrogram(linkage_matrix, **kwargs)
    idx = 1005
    for i, d, c in zip(ddata['icoord'], ddata['dcoord'],
                       ddata['color_list']):
        x = 0.5 * sum(i[1:3])
        y = d[1]
        plt.plot(y, x, 'o', c=c)
        plt.annotate("%.3g" % idx, (y, x), xytext=(15, 5),
                     textcoords='offset points',
                     va='top', ha='center')
        idx += 1


def get_labels(X):
    # PCA
    pca = PCA(n_components=2).fit_transform(X)

    # Agglomerative Clustering
    AC = AgglomerativeClustering(n_clusters=None, affinity='euclidean', memory=None, connectivity=None,
                                 compute_full_tree='auto', linkage='ward', distance_threshold=0,
                                 compute_distances=True).fit(pca)

    Z = linkage(pca, "ward")
    plot_dendrogram(Z, labels=np.arange(pca.shape[0]),
                    truncate_mode='level', show_leaf_counts=False,
                    orientation='left')
    plt.show()

    clusters = extract_levels(AC.children_)
    print(f'Clusters: {clusters}\n')

    hierarchy_list = []
    for label in range(len(AC.labels_)):
        label_list = []
        for key, value in clusters.items():
            for val in value:
                if val == label:
                    label_list.append(key)
        hierarchy_list.append(label_list)
        print(f'Label: {label}, Hierarchy: {label_list}')

    hierarchy_labels = np.zeros([len(hierarchy_list), len(min(hierarchy_list, key=lambda x: len(x)))])

    cut_tree = hierarchy_labels.shape[1]
    for i, j in enumerate(hierarchy_list):
        hierarchy_labels[i][0:len(j)] = j[:cut_tree]
    print(f'\nHierarchy Labels: \n{np.array(hierarchy_labels)}\n')

    return to_categorical(hierarchy_labels - nb_samples), hierarchy_labels, clusters, hierarchy_list


def formTree(list):
    tree = {}
    for item in list:
        currTree = tree

        for key in item[::-1]:
            if key not in currTree:
                currTree[key] = {}
            currTree = currTree[key]

    return tree


def store(file, object):
    with open(f'{file}.pkl', 'wb') as f:
        pickle.dump(object, f)


features = feature_extractor()
train_features = np.reshape(features, (nb_samples, features.shape[1] * features.shape[2] * features.shape[3]))
print(f'\nShape of Features: {features.shape}, Shape of Train Features: {train_features.shape}')

y_train, hierarchy_labels, clusters, hierarchy_list = get_labels(train_features)
print(f'\nShape of Labels: {y_train.shape}')

train_features = np.reshape(train_features, (nb_samples, 256, 256, 2))

nested_dict = formTree(hierarchy_list)
print(nested_dict)

hi1 = np.zeros([nb_samples, 1])
hi2 = np.zeros([nb_samples, 1])
hi3 = np.zeros([nb_samples, 1])
hi4 = np.zeros([nb_samples, 1])
hi5 = np.zeros([nb_samples, 1])
hi6 = np.zeros([nb_samples, 1])

for i, j in enumerate(np.fliplr(hierarchy_labels - nb_samples)):
    hi1[i] = j[0]
    hi2[i] = j[1]
    hi3[i] = j[2]
    hi4[i] = j[3]
    hi5[i] = j[4]
    hi6[i] = j[5]

hi1_C, hi2_C, hi3_C, hi4_C, hi5_C, hi6_C = to_categorical(hi1), to_categorical(hi2), to_categorical(hi3), \
                                           to_categorical(hi4), to_categorical(hi5), to_categorical(hi6)

store(f'{path}/train_features', train_features)
store(f'{path}/y_train', y_train)
store(f'{path}/hierarchy_labels', hierarchy_labels)
store(f'{path}/clusters', clusters)
store(f'{path}/nested_dict', nested_dict)

store(f'{path}/hi1_C', hi1_C)
store(f'{path}/hi2_C', hi2_C)
store(f'{path}/hi3_C', hi3_C)
store(f'{path}/hi4_C', hi4_C)
store(f'{path}/hi5_C', hi5_C)
store(f'{path}/hi6_C', hi6_C)
