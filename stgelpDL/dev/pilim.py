import PIL.Image as pilim

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from predictor.utility import msg2log

def rgba2rgb(rgba_file:str, rgb_file:str):
    rgba_image = pilim.open(rgba_file)
    rgb_image  = rgba_image.convert('RGB')
    im = np.array(rgb_image)
    result = pilim.fromarray(im.astype(np.uint8))
    result.save(rgb_file)
    return

def png2jpg(png_file:str, jpg_file:str):
    png_image = pilim.open(png_file)
    png_image.save(jpg_file)

    return

def bw_bb()->(np.array,np.array,pilim,pilim):
    img_bw = np.ones([500, 335], dtype=np.uint8) * 255 #Allaice Alphonse First Communion of Anaemic Young Girls is the Snow,1883
    img_bb = np.ones([727, 626], dtype=np.uint8) # Allais Alphonse Negroes Fighting in a tunnel by Nigth, 1884
    image_bw = pilim.fromarray(img_bw)
    # image_bw.show()
    image_bb = pilim.fromarray(img_bb)
    # image_bb.show()
    hist_bb, bin_edges = np.histogram(img_bb[:, :], bins=256, range=(0, 256), density=True)
    hist_bw, bin_edges = np.histogram(img_bw[:, :], bins=256, range=(0, 256), density=True)
    return hist_bw,hist_bb,image_bw, image_bb

def kMeanCluster(name: str, X: np.array, labelX: list, cluster_max:int = 4,type_features: str ='pca', n_component: int=2, \
                 file_png_path: str="", f: object = None) -> (dict, dict, dict, dict):
    """

    :param name:  - title for logs
    :param X: np.array((n_samples, n_features)
    :param labelX: -list of n_samples labels for each row of X
    :param type_features: 'pca' -principal components (default) or 'feat' -original features.
    :param n_component: number components for clusterisation
    :param file_png_path:
    :param f:
    :return:
    """
    pass
    n_cluster_max = cluster_max
    cluster_centers = {}
    cluster_labels = {}
    cluster_contains_blocks = {}
    block_belongs_to_cluster = {}
    Path(file_png_path).mkdir(parents=True, exist_ok=True)

    for n_clusters in range(1, n_cluster_max):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X[:,:])
        lst_lbl=[]
        for i in range(n_clusters):
            lst_lbl.append([])
        for i in range(len(kmeans.labels_)):
            ind=kmeans.labels_[i]
            lst_lbl[ind].append(labelX[i])
        message = f"""
                Data maped on {n_component} components (or first features) 
                Number of clusters : {n_clusters}
                Cluster centers    : 
{kmeans.cluster_centers_}
                Cluster labels     : 
{kmeans.labels_}
{lst_lbl}

        """
        msg2log(kMeanCluster.__name__, message, f)
        fl_name = "{}_festures_{}_clusters_{}.png".format(name, n_component, n_clusters)
        file_png = Path(Path(file_png_path) / Path(fl_name))
        plotClusters(kmeans, X, file_png)

        cluster_centers[n_clusters]  = kmeans.cluster_centers_
        cluster_labels[n_clusters]   = kmeans.labels_


    return cluster_centers, cluster_labels

def plotClusters(kmeans: KMeans, X: np.array, file_png:Path):
    """
    The plot shows 2 first component of X
    :param kmeans: -sclearn.cluster.Kmeans object
    :param X: matrix n_samples * n_features or principal component n_samples * n_components.
    :param file_png:
    :return:
    """
    plt.scatter(X[:, 0].tolist(), X[:, 1].tolist(), c=kmeans.labels_.astype(float), s=50, alpha=0.5)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=50)
    plt.savefig(file_png)
    plt.close("all")
    return

if __name__ == "__main__":
    rgba_image=pilim.open("/home/dmitryv/Pictures/Magritte_ObstackleVoid.png")
    rgb_image=rgba_image.convert('RGB')

    im=np.array(rgb_image)

    result = pilim.fromarray(im.astype(np.uint8))
    result.save('a.jpg')
    rgbArray = plt.imread('a.jpg')
    pass

