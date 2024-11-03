import argparse
import time
import cv2
import imageio.v2 as imageio
import scipy.sparse as sp
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
from skimage.segmentation import slic
from collections import Counter
from torch_geometric.nn import GATConv
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits import axes_grid1


def parse_arguments():
    """
    Parse command line arguments for the change detection script.

    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Detecting land-cover changes using SDCGA")
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Dimension of hidden layers.')
    parser.add_argument('--n_seg', type=int, default=10000, help='Approximate number of superpixels.')
    parser.add_argument('--cmp', type=int, default=15, help='Compactness parameter for SLIC algorithm.')
    parser.add_argument('--beta', type=float, default=1, help='Penalty coefficient for delta.')
    parser.add_argument('--regularization', type=float, default=0.00001, help='Regularization term coefficient.')
    parser.add_argument('--k_ratio', type=float, default=0.1, help='Neighbor ratio for graph construction.')
    parser.add_argument('--epoch', type=int, default=300, help='Number of training epochs.')
    parser.add_argument('--image_t1_path', type=str, required=True, help='Path to the first-time image.')
    parser.add_argument('--image_t2_path', type=str, required=True, help='Path to the second-time image.')
    parser.add_argument('--ref_gt_path', type=str, required=True, help='Path to the reference ground truth image.')
    args = parser.parse_args()
    return args


class GraphEncoder(nn.Module):
    """
    Graph Encoder module using GATConv layer.
    """
    def __init__(self, input_dim, hidden_dim, dropout, num_heads):
        super(GraphEncoder, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        return x


class GraphDecoder(nn.Module):
    """
    Graph Decoder module using GATConv layer.
    """
    def __init__(self, hidden_dim, output_dim, dropout, num_heads):
        super(GraphDecoder, self).__init__()
        self.conv1 = GATConv(hidden_dim * num_heads, output_dim, heads=1, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = F.relu(self.conv1(x, edge_index))
        return x


class GraphAutoencoder(nn.Module):
    """
    Graph Autoencoder consisting of an encoder and a decoder.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, num_heads):
        super(GraphAutoencoder, self).__init__()
        self.encoder = GraphEncoder(input_dim, hidden_dim, dropout, num_heads)
        self.decoder = GraphDecoder(hidden_dim, output_dim, dropout, num_heads)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        x_hat = self.decoder(z, edge_index)
        return x_hat


def parula_cmap():
    """
    Define and return a 'parula' colormap similar to MATLAB's parula colormap.

    Returns:
        cmap: A LinearSegmentedColormap object representing the parula colormap.
    """
    # List of RGB colors defining the parula colormap
    colors = [[0.2422, 0.1504, 0.6603],
            [0.2444, 0.1534, 0.6728],
            [0.2464, 0.1569, 0.6847],
            [0.2484, 0.1607, 0.6961],
            [0.2503, 0.1648, 0.7071],
            [0.2522, 0.1689, 0.7179],
            [0.254, 0.1732, 0.7286],
            [0.2558, 0.1773, 0.7393],
            [0.2576, 0.1814, 0.7501],
            [0.2594, 0.1854, 0.761],
            [0.2611, 0.1893, 0.7719],
            [0.2628, 0.1932, 0.7828],
            [0.2645, 0.1972, 0.7937],
            [0.2661, 0.2011, 0.8043],
            [0.2676, 0.2052, 0.8148],
            [0.2691, 0.2094, 0.8249],
            [0.2704, 0.2138, 0.8346],
            [0.2717, 0.2184, 0.8439],
            [0.2729, 0.2231, 0.8528],
            [0.274, 0.228, 0.8612],
            [0.2749, 0.233, 0.8692],
            [0.2758, 0.2382, 0.8767],
            [0.2766, 0.2435, 0.884],
            [0.2774, 0.2489, 0.8908],
            [0.2781, 0.2543, 0.8973],
            [0.2788, 0.2598, 0.9035],
            [0.2794, 0.2653, 0.9094],
            [0.2798, 0.2708, 0.915],
            [0.2802, 0.2764, 0.9204],
            [0.2806, 0.2819, 0.9255],
            [0.2809, 0.2875, 0.9305],
            [0.2811, 0.293, 0.9352],
            [0.2813, 0.2985, 0.9397],
            [0.2814, 0.304, 0.9441],
            [0.2814, 0.3095, 0.9483],
            [0.2813, 0.315, 0.9524],
            [0.2811, 0.3204, 0.9563],
            [0.2809, 0.3259, 0.96],
            [0.2807, 0.3313, 0.9636],
            [0.2803, 0.3367, 0.967],
            [0.2798, 0.3421, 0.9702],
            [0.2791, 0.3475, 0.9733],
            [0.2784, 0.3529, 0.9763],
            [0.2776, 0.3583, 0.9791],
            [0.2766, 0.3638, 0.9817],
            [0.2754, 0.3693, 0.984],
            [0.2741, 0.3748, 0.9862],
            [0.2726, 0.3804, 0.9881],
            [0.271, 0.386, 0.9898],
            [0.2691, 0.3916, 0.9912],
            [0.267, 0.3973, 0.9924],
            [0.2647, 0.403, 0.9935],
            [0.2621, 0.4088, 0.9946],
            [0.2591, 0.4145, 0.9955],
            [0.2556, 0.4203, 0.9965],
            [0.2517, 0.4261, 0.9974],
            [0.2473, 0.4319, 0.9983],
            [0.2424, 0.4378, 0.9991],
            [0.2369, 0.4437, 0.9996],
            [0.2311, 0.4497, 0.9995],
            [0.225, 0.4559, 0.9985],
            [0.2189, 0.462, 0.9968],
            [0.2128, 0.4682, 0.9948],
            [0.2066, 0.4743, 0.9926],
            [0.2006, 0.4803, 0.9906],
            [0.195, 0.4861, 0.9887],
            [0.1903, 0.4919, 0.9867],
            [0.1869, 0.4975, 0.9844],
            [0.1847, 0.503, 0.9819],
            [0.1831, 0.5084, 0.9793],
            [0.1818, 0.5138, 0.9766],
            [0.1806, 0.5191, 0.9738],
            [0.1795, 0.5244, 0.9709],
            [0.1785, 0.5296, 0.9677],
            [0.1778, 0.5349, 0.9641],
            [0.1773, 0.5401, 0.9602],
            [0.1768, 0.5452, 0.956],
            [0.1764, 0.5504, 0.9516],
            [0.1755, 0.5554, 0.9473],
            [0.174, 0.5605, 0.9432],
            [0.1716, 0.5655, 0.9393],
            [0.1686, 0.5705, 0.9357],
            [0.1649, 0.5755, 0.9323],
            [0.161, 0.5805, 0.9289],
            [0.1573, 0.5854, 0.9254],
            [0.154, 0.5902, 0.9218],
            [0.1513, 0.595, 0.9182],
            [0.1492, 0.5997, 0.9147],
            [0.1475, 0.6043, 0.9113],
            [0.1461, 0.6089, 0.908],
            [0.1446, 0.6135, 0.905],
            [0.1429, 0.618, 0.9022],
            [0.1408, 0.6226, 0.8998],
            [0.1383, 0.6272, 0.8975],
            [0.1354, 0.6317, 0.8953],
            [0.1321, 0.6363, 0.8932],
            [0.1288, 0.6408, 0.891],
            [0.1253, 0.6453, 0.8887],
            [0.1219, 0.6497, 0.8862],
            [0.1185, 0.6541, 0.8834],
            [0.1152, 0.6584, 0.8804],
            [0.1119, 0.6627, 0.877],
            [0.1085, 0.6669, 0.8734],
            [0.1048, 0.671, 0.8695],
            [0.1009, 0.675, 0.8653],
            [0.0964, 0.6789, 0.8609],
            [0.0914, 0.6828, 0.8562],
            [0.0855, 0.6865, 0.8513],
            [0.0789, 0.6902, 0.8462],
            [0.0713, 0.6938, 0.8409],
            [0.0628, 0.6972, 0.8355],
            [0.0535, 0.7006, 0.8299],
            [0.0433, 0.7039, 0.8242],
            [0.0328, 0.7071, 0.8183],
            [0.0234, 0.7103, 0.8124],
            [0.0155, 0.7133, 0.8064],
            [0.0091, 0.7163, 0.8003],
            [0.0046, 0.7192, 0.7941],
            [0.0019, 0.722, 0.7878],
            [0.0009, 0.7248, 0.7815],
            [0.0018, 0.7275, 0.7752],
            [0.0046, 0.7301, 0.7688],
            [0.0094, 0.7327, 0.7623],
            [0.0162, 0.7352, 0.7558],
            [0.0253, 0.7376, 0.7492],
            [0.0369, 0.74, 0.7426],
            [0.0504, 0.7423, 0.7359],
            [0.0638, 0.7446, 0.7292],
            [0.077, 0.7468, 0.7224],
            [0.0899, 0.7489, 0.7156],
            [0.1023, 0.751, 0.7088],
            [0.1141, 0.7531, 0.7019],
            [0.1252, 0.7552, 0.695],
            [0.1354, 0.7572, 0.6881],
            [0.1448, 0.7593, 0.6812],
            [0.1532, 0.7614, 0.6741],
            [0.1609, 0.7635, 0.6671],
            [0.1678, 0.7656, 0.6599],
            [0.1741, 0.7678, 0.6527],
            [0.1799, 0.7699, 0.6454],
            [0.1853, 0.7721, 0.6379],
            [0.1905, 0.7743, 0.6303],
            [0.1954, 0.7765, 0.6225],
            [0.2003, 0.7787, 0.6146],
            [0.2061, 0.7808, 0.6065],
            [0.2118, 0.7828, 0.5983],
            [0.2178, 0.7849, 0.5899],
            [0.2244, 0.7869, 0.5813],
            [0.2318, 0.7887, 0.5725],
            [0.2401, 0.7905, 0.5636],
            [0.2491, 0.7922, 0.5546],
            [0.2589, 0.7937, 0.5454],
            [0.2695, 0.7951, 0.536],
            [0.2809, 0.7964, 0.5266],
            [0.2929, 0.7975, 0.517],
            [0.3052, 0.7985, 0.5074],
            [0.3176, 0.7994, 0.4975],
            [0.3301, 0.8002, 0.4876],
            [0.3424, 0.8009, 0.4774],
            [0.3548, 0.8016, 0.4669],
            [0.3671, 0.8021, 0.4563],
            [0.3795, 0.8026, 0.4454],
            [0.3921, 0.8029, 0.4344],
            [0.405, 0.8031, 0.4233],
            [0.4184, 0.803, 0.4122],
            [0.4322, 0.8028, 0.4013],
            [0.4463, 0.8024, 0.3904],
            [0.4608, 0.8018, 0.3797],
            [0.4753, 0.8011, 0.3691],
            [0.4899, 0.8002, 0.3586],
            [0.5044, 0.7993, 0.348],
            [0.5187, 0.7982, 0.3374],
            [0.5329, 0.797, 0.3267],
            [0.547, 0.7957, 0.3159],
            [0.5609, 0.7943, 0.305],
            [0.5748, 0.7929, 0.2941],
            [0.5886, 0.7913, 0.2833],
            [0.6024, 0.7896, 0.2726],
            [0.6161, 0.7878, 0.2622],
            [0.6297, 0.7859, 0.2521],
            [0.6433, 0.7839, 0.2423],
            [0.6567, 0.7818, 0.2329],
            [0.6701, 0.7796, 0.2239],
            [0.6833, 0.7773, 0.2155],
            [0.6963, 0.775, 0.2075],
            [0.7091, 0.7727, 0.1998],
            [0.7218, 0.7703, 0.1924],
            [0.7344, 0.7679, 0.1852],
            [0.7468, 0.7654, 0.1782],
            [0.759, 0.7629, 0.1717],
            [0.771, 0.7604, 0.1658],
            [0.7829, 0.7579, 0.1608],
            [0.7945, 0.7554, 0.157],
            [0.806, 0.7529, 0.1546],
            [0.8172, 0.7505, 0.1535],
            [0.8281, 0.7481, 0.1536],
            [0.8389, 0.7457, 0.1546],
            [0.8495, 0.7435, 0.1564],
            [0.86, 0.7413, 0.1587],
            [0.8703, 0.7392, 0.1615],
            [0.8804, 0.7372, 0.165],
            [0.8903, 0.7353, 0.1695],
            [0.9, 0.7336, 0.1749],
            [0.9093, 0.7321, 0.1815],
            [0.9184, 0.7308, 0.189],
            [0.9272, 0.7298, 0.1973],
            [0.9357, 0.729, 0.2061],
            [0.944, 0.7285, 0.2151],
            [0.9523, 0.7284, 0.2237],
            [0.9606, 0.7285, 0.2312],
            [0.9689, 0.7292, 0.2373],
            [0.977, 0.7304, 0.2418],
            [0.9842, 0.733, 0.2446],
            [0.99, 0.7365, 0.2429],
            [0.9946, 0.7407, 0.2394],
            [0.9966, 0.7458, 0.2351],
            [0.9971, 0.7513, 0.2309],
            [0.9972, 0.7569, 0.2267],
            [0.9971, 0.7626, 0.2224],
            [0.9969, 0.7683, 0.2181],
            [0.9966, 0.774, 0.2138],
            [0.9962, 0.7798, 0.2095],
            [0.9957, 0.7856, 0.2053],
            [0.9949, 0.7915, 0.2012],
            [0.9938, 0.7974, 0.1974],
            [0.9923, 0.8034, 0.1939],
            [0.9906, 0.8095, 0.1906],
            [0.9885, 0.8156, 0.1875],
            [0.9861, 0.8218, 0.1846],
            [0.9835, 0.828, 0.1817],
            [0.9807, 0.8342, 0.1787],
            [0.9778, 0.8404, 0.1757],
            [0.9748, 0.8467, 0.1726],
            [0.972, 0.8529, 0.1695],
            [0.9694, 0.8591, 0.1665],
            [0.9671, 0.8654, 0.1636],
            [0.9651, 0.8716, 0.1608],
            [0.9634, 0.8778, 0.1582],
            [0.9619, 0.884, 0.1557],
            [0.9608, 0.8902, 0.1532],
            [0.9601, 0.8963, 0.1507],
            [0.9596, 0.9023, 0.148],
            [0.9595, 0.9084, 0.145],
            [0.9597, 0.9143, 0.1418],
            [0.9601, 0.9203, 0.1382],
            [0.9608, 0.9262, 0.1344],
            [0.9618, 0.932, 0.1304],
            [0.9629, 0.9379, 0.1261],
            [0.9642, 0.9437, 0.1216],
            [0.9657, 0.9494, 0.1168],
            [0.9674, 0.9552, 0.1116],
            [0.9692, 0.9609, 0.1061],
            [0.9711, 0.9667, 0.1001],
            [0.973, 0.9724, 0.0938],
            [0.9749, 0.9782, 0.0872],
            [0.9769, 0.9839, 0.0805]]
    
    return LinearSegmentedColormap.from_list("parula", colors)


def otsu(data, num=1000):
    """
    Compute Otsu's threshold for a given data array.

    Args:
        data (np.ndarray): The data array.
        num (int): Number of bins to consider.

    Returns:
        best_threshold (float): The threshold value that maximizes inter-class variance.
    """
    max_value = np.max(data)
    min_value = np.min(data)
    total_num = data.shape[0]
    step_value = (max_value - min_value) / num
    value = min_value + step_value
    best_threshold = min_value
    best_inter_class_var = 0

    while value <= max_value:
        data_1 = data[data < value]
        data_2 = data[data >= value]
        w1 = data_1.shape[0] / total_num
        w2 = data_2.shape[0] / total_num

        mean_1 = data_1.mean() if data_1.size > 0 else 0
        mean_2 = data_2.mean() if data_2.size > 0 else 0

        inter_class_var = w1 * w2 * np.power((mean_1 - mean_2), 2)
        if best_inter_class_var < inter_class_var:
            best_inter_class_var = inter_class_var
            best_threshold = value
        value += step_value
    return best_threshold


def label2idx(sup_img):
    """
    Convert superpixel labels to indices.

    Args:
        sup_img (np.ndarray): Superpixel label image.

    Returns:
        idx_list (list): List of arrays, where each array contains indices of pixels belonging to a superpixel.
    """
    return [np.where(sup_img.ravel() == i)[0] for i in range(1, np.max(sup_img) + 1)]


def MSMfeature_extraction(sup_img, image_t1, image_t2):
    """
    Extract Multi-scale Mean (MSM) features from superpixels.

    Args:
        sup_img (np.ndarray): Superpixel label image.
        image_t1 (np.ndarray): First-time image (H x W x C1).
        image_t2 (np.ndarray): Second-time image (H x W x C2).

    Returns:
        node_t1 (np.ndarray): Features for image_t1 superpixels.
        node_t2 (np.ndarray): Features for image_t2 superpixels.
    """
    # Get the size of the images
    h, w, b1 = image_t1.shape
    _, _, b2 = image_t2.shape

    # Convert the superpixels to indices
    idx_t1 = label2idx(sup_img)

    # Get the number of superpixels
    nbr_sp = sup_img.max()

    # Reshape the images into vectors
    re_image_t1 = image_t1.reshape((h * w, b1))
    re_image_t2 = image_t2.reshape((h * w, b2))

    # Initialize the feature matrices
    node_t1 = np.zeros((nbr_sp, 3 * b1))
    node_t2 = np.zeros((nbr_sp, 3 * b2))

    # Loop over each superpixel
    for i in range(nbr_sp):
        # Get the indices of the pixels in the current superpixel
        index_vector = idx_t1[i]

        # Compute the features for the first image
        sub_superpixel_t1 = re_image_t1[index_vector, :]
        mean_feature_t1 = np.mean(sub_superpixel_t1, axis=0)
        median_feature_t1 = np.median(sub_superpixel_t1, axis=0)
        var_feature_t1 = np.var(sub_superpixel_t1, axis=0)
        node_t1[i, :] = np.concatenate([mean_feature_t1, median_feature_t1, var_feature_t1], axis=0)

        # Compute the features for the second image
        sub_superpixel_t2 = re_image_t2[index_vector, :]
        mean_feature_t2 = np.mean(sub_superpixel_t2, axis=0)
        median_feature_t2 = np.median(sub_superpixel_t2, axis=0)
        var_feature_t2 = np.var(sub_superpixel_t2, axis=0)
        node_t2[i, :] = np.concatenate([mean_feature_t2, median_feature_t2, var_feature_t2], axis=0)

    # Transpose the feature matrices
    node_t1 = node_t1.T
    node_t2 = node_t2.T

    # Replace NaN values with 0
    node_t1 = np.nan_to_num(node_t1)
    node_t2 = np.nan_to_num(node_t2)

    # Normalize the features to [0, 1]
    node_t1_max = np.max(node_t1, axis=1, keepdims=True) + np.finfo(float).eps
    node_t1 = node_t1 / node_t1_max
    node_t2_max = np.max(node_t2, axis=1, keepdims=True) + np.finfo(float).eps
    node_t2 = node_t2 / node_t2_max

    # Transpose back to (superpixels x features)
    node_t1 = node_t1.T
    node_t2 = node_t2.T

    return node_t1, node_t2


def Adj_Laplacian_matrix_calculation(X, K=None):
    """
    Calculate the adjacency and Laplacian matrices for the graph.

    Args:
        X (np.ndarray): Data matrix (N x D).
        K (int): Number of neighbors.

    Returns:
        Adj (torch.LongTensor): Adjacency matrix indices.
        Laplacian (torch.FloatTensor): Laplacian matrix.
        S (np.ndarray): Weight matrix.
    """
    if K is None:
        K = int(np.round(np.sqrt(X.shape[0])))

    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)

    N = X.shape[0]
    S = np.zeros((N, N))
    K = K + 1  # Include self

    nbrs = NearestNeighbors(n_neighbors=K).fit(X)
    distances, indices = nbrs.kneighbors(X)

    degree = Counter(indices.ravel())
    Kmat = np.array([v for k, v in degree.items()])

    kmax = K
    kmin = int(np.round(kmax / 10)) + 1
    Kmat[Kmat >= kmax] = kmax
    Kmat[Kmat <= kmin] = kmin

    if len(Kmat) < N:
        Kmat = np.append(Kmat, np.full(N - len(Kmat), kmin))

    for i in range(N):
        Ki = Kmat[i]
        id_x = indices[i, 1:Ki]
        S[i, id_x] = np.ones((1, len(id_x)))

    # Symmetrize the adjacency matrix
    Adj = (S + S.T) / 2
    Adj[Adj != 0] = 1
    Adj = sp.coo_matrix(Adj)

    # Compute Laplacian matrix
    Laplacian = sp.csgraph.laplacian(Adj, normed=True)
    row = torch.from_numpy(Adj.row).long()
    col = torch.from_numpy(Adj.col).long()
    Adj = torch.stack([row, col], dim=0)

    # Convert Laplacian to dense tensor
    Laplacian = torch.from_numpy(Laplacian.todense()).float()

    return Adj, Laplacian, S


def norm_img(img):
    """
    Normalize image to [0, 1].

    Args:
        img (np.ndarray): Input image of shape (H, W, C).

    Returns:
        nm_img (np.ndarray): Normalized image.
    """
    img_height, img_width, channel = img.shape
    img = img.reshape(-1, channel)
    max_value = np.max(img, axis=0, keepdims=True)
    min_value = np.min(img, axis=0, keepdims=True)
    diff_value = max_value - min_value + np.finfo(float).eps
    nm_img = (img - min_value) / diff_value
    nm_img = nm_img.reshape(img_height, img_width, channel)
    return nm_img


def perfor_multivalue(im, imref):
    """
    Compute performance metrics for change detection.

    Args:
        im (np.ndarray): Binary change map (0 or 255).
        imref (np.ndarray): Ground truth reference map (0 or 255).

    Returns:
        fp (int): False positives.
        fn (int): False negatives.
        oe (int): Overall errors.
        pcc (float): Percent correct classification.
        kappa (float): Kappa coefficient.
        img_multivalue (np.ndarray): Error map.
        F1 (float): F1 score.
    """
    im = im.astype(float)
    imref = imref.astype(float)

    # Map 0 to 2 and 255 to 254 in imref
    imref[imref == 0] = 2
    imref[imref == 255] = 254

    A, B = im.shape
    N = A * B
    img_multivalue = np.zeros((A, B))  # Error observation map

    # Number of unchanged pixels in reference image
    Nu = np.sum(imref == 2)

    # Number of changed pixels in reference image
    Nc = np.sum(imref == 254)

    img_temp = im - imref

    # Initialize counts
    tn = len(np.where(img_temp == -2)[0])    # True negatives
    tp = len(np.where(img_temp == 1)[0])     # True positives
    fn = len(np.where(img_temp == -254)[0])  # False negatives
    fp = len(np.where(img_temp == 253)[0])   # False positives

    img_multivalue[img_temp == -2] = 0      # True negatives
    img_multivalue[img_temp == 1] = 255     # True positives
    img_multivalue[img_temp == -254] = 180  # False negatives
    img_multivalue[img_temp == 253] = 100   # False positives

    oe = fp + fn
    pcc = (tp + tn) / N

    # Kappa coefficient
    pre = ((tp + fp) * Nc + (fn + tn) * Nu) / (N ** 2)
    kappa = (pcc - pre) / (1 - pre)

    P = tp / (tp + fp) if (tp + fp) != 0 else 0
    R = tp / (tp + fn) if (tp + fn) != 0 else 0
    F1 = 2 * P * R / (P + R) if (P + R) != 0 else 0

    return fp, fn, oe, pcc, kappa, img_multivalue, F1


def Gray2Color(img, uni_value, color):
    """
    Convert a grayscale label image to a color image.

    Args:
        img (np.ndarray): Grayscale image.
        uni_value (list): Unique grayscale values.
        color (list): Corresponding RGB colors.

    Returns:
        img_color (np.ndarray): Color image.
    """
    img_color = np.stack([img]*3, axis=-1)  # Create 3-channel image
    for val, col in zip(uni_value, color):
        mask = img == val
        img_color[mask] = col
    img_color = img_color.astype(np.uint8)
    return img_color


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """
    Add a vertical colorbar to an image plot.

    Args:
        im (AxesImage): Image to which the colorbar is added.
        aspect (float): Ratio of long to short dimensions.
        pad_fraction (float): Fraction of original axes to use as padding.
        kwargs: Additional keyword arguments for colorbar.

    Returns:
        Colorbar object.
    """
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def remove_outlier(y):
    """
    Remove outliers in data by replacing values that are more than 4 standard deviations
    from the mean with the maximum of the non-outlier values.

    Args:
        y (np.ndarray): Input data.

    Returns:
        x (np.ndarray): Data with outliers replaced.
    """
    x = np.copy(y)
    mean_y = np.mean(y)
    std_y = np.std(y)
    threshold = mean_y + 4 * std_y
    outliers = y > threshold
    x[outliers] = np.max(x[~outliers])
    return x


def normalize_dif(matrix):
    """
    Normalize a N x 1 matrix to [0, 1].

    Args:
        matrix (np.ndarray): Input data.

    Returns:
        normalized_matrix (np.ndarray): Normalized data.
    """
    normalized_matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min() + np.finfo(float).eps)
    return normalized_matrix


def train_modal(feature_nums, sup_nums, node_t1, node_t2, Adj_x, Adj_y, Laplacian_x, Laplacian_y, args):
    """
    Train the graph autoencoder model.

    Args:
        feature_nums (int): Number of features.
        sup_nums (int): Number of superpixels (nodes).
        node_t1 (np.ndarray): Features for time 1.
        node_t2 (np.ndarray): Features for time 2.
        Adj_x (torch.LongTensor): Adjacency indices for time 1.
        Adj_y (torch.LongTensor): Adjacency indices for time 2.
        Laplacian_x (torch.FloatTensor): Laplacian matrix for time 1.
        Laplacian_y (torch.FloatTensor): Laplacian matrix for time 2.
        args: Training arguments.

    Returns:
        recon_node_t1 (np.ndarray): Reconstructed features for time 1.
        delt_t1 (np.ndarray): Delta values for time 1.
        recon_node_t2 (np.ndarray): Reconstructed features for time 2.
        delt_t2 (np.ndarray): Delta values for time 2.
    """
    # Initialize delta parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    delt_t1 = torch.ones(sup_nums, feature_nums, requires_grad=True, device=device)
    delt_t2 = torch.ones(sup_nums, feature_nums, requires_grad=True, device=device)

    # Convert data to tensors
    node_t1 = torch.tensor(node_t1, dtype=torch.float32, device=device)
    node_t2 = torch.tensor(node_t2, dtype=torch.float32, device=device)
    Adj_x = Adj_x.to(device)
    Adj_y = Adj_y.to(device)
    Laplacian_x = Laplacian_x.to(device)
    Laplacian_y = Laplacian_y.to(device)

    # Initialize model
    model = GraphAutoencoder(input_dim=feature_nums, hidden_dim=args.hidden_dim,
                             output_dim=feature_nums, dropout=args.dropout,
                             num_heads=args.num_heads).to(device)
    model.train()

    # Define optimizer
    optimizer = optim.Adam([delt_t1, delt_t2] + list(model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    loss_values = []
    epochs = []

    for epoch in range(args.epoch):
        optimizer.zero_grad()

        # Forward pass
        recon_node_t1 = model(node_t2, Adj_x)
        recon_node_t2 = model(node_t1, Adj_y)

        # Compute structure loss
        structure_loss_t1 = 2 * torch.trace(recon_node_t1.t() @ Laplacian_x @ recon_node_t1)
        structure_loss_t2 = 2 * torch.trace(recon_node_t2.t() @ Laplacian_y @ recon_node_t2)
        structure_loss = structure_loss_t1 + structure_loss_t2

        # Compute reconstruction loss
        reconstruct_loss_t1 = torch.norm(node_t2 - recon_node_t1 + delt_t1, p='fro')**2
        reconstruct_loss_t2 = torch.norm(node_t1 - recon_node_t2 + delt_t2, p='fro')**2
        reconstruct_loss = reconstruct_loss_t1 + reconstruct_loss_t2

        # Delta penalty term
        delt_loss = torch.norm(delt_t1, p='fro')**2 + torch.norm(delt_t2, p='fro')**2

        # Total loss
        loss = structure_loss + reconstruct_loss + args.beta * delt_loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{args.epoch}, Loss: {loss.item():.4f}')

        loss_values.append(loss.item())
        epochs.append(epoch + 1)

    # Plot loss curve
    plt.figure()
    plt.plot(epochs, loss_values, 'b-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.show()

    # Get the results
    recon_node_t1 = recon_node_t1.cpu().detach().numpy()
    delt_t1 = delt_t1.cpu().detach().numpy()
    recon_node_t2 = recon_node_t2.cpu().detach().numpy()
    delt_t2 = delt_t2.cpu().detach().numpy()

    return recon_node_t1, delt_t1, recon_node_t2, delt_t2


def main():
    """Main function for the SDCGA multimodal change detection."""
    # Start timing
    start_time = time.time()

    # Parse command line arguments
    args = parse_arguments()

    # ============================ Load and Preprocess Images ============================================
    # Load images
    image_t1 = imageio.imread(args.image_t1_path)
    image_t2 = imageio.imread(args.image_t2_path)
    Ref_gt = imageio.imread(args.ref_gt_path)

    # Adjust image shapes if necessary
    if image_t2.ndim == 3 and image_t2.shape[0] <= 4:
        image_t2 = np.transpose(image_t2, (1, 2, 0))

    if image_t1.ndim == 2:
        image_t1 = np.expand_dims(image_t1, axis=2)
    image_t1 = np.repeat(image_t1, 3, axis=2)

    # Normalize images
    image_t1 = norm_img(image_t1.astype(np.float32))
    image_t2 = norm_img(image_t2.astype(np.float32))

    # Get image dimensions
    height, width, channel_t1 = image_t1.shape
    _, _, channel_t2 = image_t2.shape

    # ======================= Feature Extraction Using Superpixels ====================================
    # Compute average image for superpixel segmentation
    image_t1t2 = (image_t1 + image_t2) / 2

    # Perform SLIC superpixel segmentation
    sup_img = slic(image_t1t2, n_segments=args.n_seg, compactness=args.cmp)
    sup_img = sup_img + 1  # Ensure labels start from 1

    # Extract MSM features for each superpixel
    node_t1, node_t2 = MSMfeature_extraction(sup_img, image_t1, image_t2)

    # Optionally, select a subset of features (e.g., first 3)
    # node_t1 = node_t1[:, :3]
    # node_t2 = node_t2[:, :3]

    # ============ Construct Graphs =====================================================================
    sup_nums = node_t1.shape[0]
    feature_nums = node_t1.shape[1]
    Kmax = int(round(args.k_ratio * sup_nums))

    # Calculate adjacency and Laplacian matrices for both times
    Adj_x, Laplacian_x, _ = Adj_Laplacian_matrix_calculation(node_t1, Kmax)
    Adj_y, Laplacian_y, _ = Adj_Laplacian_matrix_calculation(node_t2, Kmax)

    # =================== Train the Model ==============================================================
    recon_node_t1, delt_t1, recon_node_t2, delt_t2 = train_modal(
        feature_nums, sup_nums, node_t1, node_t2, Adj_x, Adj_y, Laplacian_x, Laplacian_y, args)

    # ============ Compute Change Map ==================================================================
    # Initialize change intensity maps
    dif_fw = np.zeros((height, width))
    dif_bw = np.zeros((height, width))

    # Compute change intensity for each superpixel
    for i in range(sup_nums):
        diff_value_fw = np.sum(np.square(delt_t1[i, :]))
        diff_value_bw = np.sum(np.square(delt_t2[i, :]))
        dif_fw[sup_img == i+1] = diff_value_fw
        dif_bw[sup_img == i+1] = diff_value_bw

    # Flatten change intensity maps
    dif_fw_flat = dif_fw.reshape(-1, 1)
    dif_bw_flat = dif_bw.reshape(-1, 1)

    # Remove outliers and normalize
    dif_fw_flat = remove_outlier(dif_fw_flat)
    dif_bw_flat = remove_outlier(dif_bw_flat)
    dif_fw_flat = normalize_dif(dif_fw_flat)
    dif_bw_flat = normalize_dif(dif_bw_flat)

    # Fuse the change intensity maps (e.g., sum or other strategy)
    dif_fusion_flat = normalize_dif(dif_fw_flat + dif_bw_flat)

    # Reshape back to image
    dif_fusion = dif_fusion_flat.reshape(height, width)

    # Threshold using Otsu's method
    threshold = otsu(dif_fusion_flat)
    CM = np.zeros((height, width), dtype=np.uint8)
    CM[dif_fusion > threshold] = 255
    CM[dif_fusion <= threshold] = 0

    # ============ Compute Reconstructed Images =========================================================
    # Reconstruct images using the reconstructed node features
    RegImg_t1 = np.zeros((height, width, channel_t2))
    RegImg_t2 = np.zeros((height, width, channel_t1))

    for i in range(sup_nums):
        RegImg_t1[sup_img == i+1] = recon_node_t1[i, :channel_t2]
        RegImg_t2[sup_img == i+1] = recon_node_t2[i, :channel_t1]

    # ============ Visualization =========================================================================
    # Plot original images
    plt.figure()
    plt.imshow(image_t1)
    plt.title("Original Image T1")
    plt.axis('off')

    plt.figure()
    plt.imshow(image_t2)
    plt.title("Original Image T2")
    plt.axis('off')

    # Plot reconstructed images
    plt.figure()
    plt.imshow(RegImg_t1)
    plt.title("Reconstructed Image T1")
    plt.axis('off')

    plt.figure()
    plt.imshow(RegImg_t2)
    plt.title("Reconstructed Image T2")
    plt.axis('off')

    # Plot change intensity map
    plt.figure()
    im = plt.imshow(dif_fusion, cmap=parula_cmap())
    add_colorbar(im)
    plt.title("Change Intensity Map")
    plt.axis('off')

    # Plot change map
    plt.figure()
    plt.imshow(CM, cmap='gray')
    plt.title("Change Map")
    plt.axis('off')

    plt.show()

    # ============ Evaluate Performance ==================================================================
    fp, fn, oe, oa, kappa, multivalue, F1 = perfor_multivalue(CM, Ref_gt)
    print(f'Overall Accuracy (OA): {oa:.4f}')
    print(f'Kappa Coefficient (KC): {kappa:.4f}')
    print(f'F1 Score: {F1:.4f}')

    # Visualize error map
    uni_value = [0, 100, 180, 255]
    color = [np.array([255, 250, 236]),
             np.array([255, 78, 0]),
             np.array([28, 28, 28]),
             np.array([29, 121, 192])]
    img_color = Gray2Color(multivalue, uni_value, color)

    plt.figure()
    plt.imshow(img_color)
    plt.title("Error Map")
    plt.axis('off')
    plt.show()

    # ============ End Timing ============================================================================
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

    print('Processing complete!')


if __name__ == '__main__':
    main()
