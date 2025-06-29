from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain.schema.runnable import RunnableParallel
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model1 = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)

model2 = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)


prompt1 = PromptTemplate(
    template="explain me the topic in form of notes of text : {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="generate quiz of the text as multiple choice question  {topic}"
    , input_variables=['topic']
)

prompt3 = PromptTemplate(
    template='merge the provided notes and  quiz  into a single document \n notes {notes} and quiz {quiz}'
    , input_variables=['notes', 'quiz']

)

parser = StrOutputParser()

parellel_chain = RunnableParallel(
    {
        'notes':prompt1|model1|parser,
        'quiz':prompt2|model2|parser 
    }
)

merge_chain = prompt3| model1 | parser

chain = parellel_chain|merge_chain

text = """
K-means is often referred to as Lloyd’s algorithm. In basic terms, the algorithm has three steps. The first step chooses the initial centroids, with the most basic method being to choose 
 samples from the dataset 
. After initialization, K-means consists of looping between the two other steps. The first step assigns each sample to its nearest centroid. The second step creates new centroids by taking the mean value of all of the samples assigned to each previous centroid. The difference between the old and the new centroids are computed and the algorithm repeats these last two steps until this value is less than a threshold. In other words, it repeats until the centroids do not move significantly.

../_images/sphx_glr_plot_kmeans_digits_001.png
K-means is equivalent to the expectation-maximization algorithm with a small, all-equal, diagonal covariance matrix.

The algorithm can also be understood through the concept of Voronoi diagrams. First the Voronoi diagram of the points is calculated using the current centroids. Each segment in the Voronoi diagram becomes a separate cluster. Secondly, the centroids are updated to the mean of each segment. The algorithm then repeats this until a stopping criterion is fulfilled. Usually, the algorithm stops when the relative decrease in the objective function between iterations is less than the given tolerance value. This is not the case in this implementation: iteration stops when centroids move less than the tolerance.

Given enough time, K-means will always converge, however this may be to a local minimum. This is highly dependent on the initialization of the centroids. As a result, the computation is often done several times, with different initializations of the centroids. One method to help address this issue is the k-means++ initialization scheme, which has been implemented in scikit-learn (use the init='k-means++' parameter). This initializes the centroids to be (generally) distant from each other, leading to probably better results than random initialization, as shown in the reference. For detailed examples of comparing different initialization schemes, refer to A demo of K-Means clustering on the handwritten digits data and Empirical evaluation of the impact of k-means initialization.

K-means++ can also be called independently to select seeds for other clustering algorithms, see sklearn.cluster.kmeans_plusplus for details and example usage.

The algorithm supports sample weights, which can be given by a parameter sample_weight. This allows to assign more weight to some samples when computing cluster centers and values of inertia. For example, assigning a weight of 2 to a sample is equivalent to adding a duplicate of that sample to the dataset 
.

Examples

Clustering text documents using k-means: Document clustering using KMeans and MiniBatchKMeans based on sparse data

An example of K-Means++ initialization: Using K-means++ to select seeds for other clustering algorithms.

2.3.2.1. Low-level parallelism
KMeans benefits from OpenMP based parallelism through Cython. Small chunks of data (256 samples) are processed in parallel, which in addition yields a low memory footprint. For more details on how to control the number of threads, please refer to our Parallelism notes.

Examples

Demonstration of k-means assumptions: Demonstrating when k-means performs intuitively and when it does not

A demo of K-Means clustering on the handwritten digits data: Clustering handwritten digits

2.3.2.2. Mini Batch K-Means
The MiniBatchKMeans is a variant of the KMeans algorithm which uses mini-batches to reduce the computation time, while still attempting to optimise the same objective function. Mini-batches are subsets of the input data, randomly sampled in each training iteration. These mini-batches drastically reduce the amount of computation required to converge to a local solution. In contrast to other algorithms that reduce the convergence time of k-means, mini-batch k-means produces results that are generally only slightly worse than the standard algorithm.

The algorithm iterates between two major steps, similar to vanilla k-means. In the first step, 
 samples are drawn randomly from the dataset, to form a mini-batch. These are then assigned to the nearest centroid. In the second step, the centroids are updated. In contrast to k-means, this is done on a per-sample basis. For each sample in the mini-batch, the assigned centroid is updated by taking the streaming average of the sample and all previous samples assigned to that centroid. This has the effect of decreasing the rate of change for a centroid over time. These steps are performed until convergence or a predetermined number of iterations is reached.

MiniBatchKMeans converges faster than KMeans, but the quality of the results is reduced. In practice this difference in quality can be quite small, as shown in the example and cited reference.

../_images/sphx_glr_plot_mini_batch_kmeans_001.png
Examples

Comparison of the K-Means and MiniBatchKMeans clustering algorithms: Comparison of KMeans and MiniBatchKMeans

Clustering text documents using k-means: Document clustering using KMeans and MiniBatchKMeans based on sparse data

Online learning of a dictionary of parts of faces

2.3.3. Affinity Propagation
AffinityPropagation creates clusters by sending messages between pairs of samples until convergence. A dataset is then described using a small number of exemplars, which are identified as those most representative of other samples. The messages sent between pairs represent the suitability for one sample to be the exemplar of the other, which is updated in response to the values from other pairs. This updating happens iteratively until convergence, at which point the final exemplars are chosen, and hence the final clustering is given.

../_images/sphx_glr_plot_affinity_propagation_001.png
Affinity Propagation can be interesting as it chooses the number of clusters based on the data provided. For this purpose, the two important parameters are the preference, which controls how many exemplars are used, and the damping factor which damps the responsibility and availability messages to avoid numerical oscillations when updating these messages.

The main drawback of Affinity Propagation is its complexity. The algorithm has a time complexity of the order 
, where 
 is the number of samples and 
 is the number of iterations until convergence. Further, the memory complexity is of the order 
 if a dense similarity matrix is used, but reducible if a sparse similarity matrix is used. This makes Affinity Propagation most appropriate for small to medium sized datasets.

Examples

Demo of affinity propagation clustering algorithm: Affinity Propagation on a synthetic 2D datasets with 3 classes

Visualizing the stock market structure Affinity Propagation on financial time series to find groups of companies

2.3.4. Mean Shift
MeanShift clustering aims to discover blobs in a smooth density of samples. It is a centroid based algorithm, which works by updating candidates for centroids to be the mean of the points within a given region. These candidates are then filtered in a post-processing stage to eliminate near-duplicates to form the final set of centroids.

The algorithm automatically sets the number of clusters, instead of relying on a parameter bandwidth, which dictates the size of the region to search through. This parameter can be set manually, but can be estimated using the provided estimate_bandwidth function, which is called if the bandwidth is not set.

The algorithm is not highly scalable, as it requires multiple nearest neighbor searches during the execution of the algorithm. The algorithm is guaranteed to converge, however the algorithm will stop iterating when the change in centroids is small.

Labelling a new sample is performed by finding the nearest centroid for a given sample.

../_images/sphx_glr_plot_mean_shift_001.png
Examples

A demo of the mean-shift clustering algorithm: Mean Shift clustering on a synthetic 2D datasets with 3 classes.

2.3.5. Spectral clustering
SpectralClustering performs a low-dimension embedding of the affinity matrix between samples, followed by clustering, e.g., by KMeans, of the components of the eigenvectors in the low dimensional space. It is especially computationally efficient if the affinity matrix is sparse and the amg solver is used for the eigenvalue problem (Note, the amg solver requires that the pyamg module is installed.)

The present version of SpectralClustering requires the number of clusters to be specified in advance. It works well for a small number of clusters, but is not advised for many clusters.

For two clusters, SpectralClustering solves a convex relaxation of the normalized cuts problem on the similarity graph: cutting the graph in two so that the weight of the edges cut is small compared to the weights of the edges inside each cluster. This criteria is especially interesting when working on images, where graph vertices are pixels, and weights of the edges of the similarity graph are computed using a function of a gradient of the image.

noisy_img segmented_img

Warning

Transforming distance to well-behaved similarities

Note that if the values of your similarity matrix are not well distributed, e.g. with negative values or with a distance matrix rather than a similarity, the spectral problem will be singular and the problem not solvable. In which case it is advised to apply a transformation to the entries of the matrix. For instance, in the case of a signed distance matrix, is common to apply a heat kernel:

similarity = np.exp(-beta * distance / distance.std())
See the examples for such an application.

Examples

Spectral clustering for image segmentation: Segmenting objects from a noisy background using spectral clustering.

Segmenting the picture of greek coins in regions: Spectral clustering to split the image of coins in regions.

2.3.5.1. Different label assignment strategies
Different label assignment strategies can be used, corresponding to the assign_labels parameter of SpectralClustering. "kmeans" strategy can match finer details, but can be unstable. In particular, unless you control the random_state, it may not be reproducible from run-to-run, as it depends on random initialization. The alternative "discretize" strategy is 100% reproducible, but tends to create parcels of fairly even and geometrical shape. The recently added "cluster_qr" option is a deterministic alternative that tends to create the visually best partitioning on the example application below.

assign_labels="kmeans"

assign_labels="discretize"

assign_labels="cluster_qr"

coin_kmeans

coin_discretize

coin_cluster_qr

2.3.5.2. Spectral Clustering Graphs
Spectral Clustering can also be used to partition graphs via their spectral embeddings. In this case, the affinity matrix is the adjacency matrix of the graph, and SpectralClustering is initialized with affinity='precomputed':

from sklearn.cluster import SpectralClustering
sc = SpectralClustering(3, affinity='precomputed', n_init=100,
                        assign_labels='discretize')
sc.fit_predict(adjacency_matrix)  
2.3.6. Hierarchical clustering
Hierarchical clustering is a general family of clustering algorithms that build nested clusters by merging or splitting them successively. This hierarchy of clusters is represented as a tree (or dendrogram). The root of the tree is the unique cluster that gathers all the samples, the leaves being the clusters with only one sample. See the Wikipedia page for more details.

The AgglomerativeClustering object performs a hierarchical clustering using a bottom up approach: each observation starts in its own cluster, and clusters are successively merged together. The linkage criteria determines the metric used for the merge strategy:

Ward minimizes the sum of squared differences within all clusters. It is a variance-minimizing approach and in this sense is similar to the k-means objective function but tackled with an agglomerative hierarchical approach.

Maximum or complete linkage minimizes the maximum distance between observations of pairs of clusters.

Average linkage minimizes the average of the distances between all observations of pairs of clusters.

Single linkage minimizes the distance between the closest observations of pairs of clusters.

AgglomerativeClustering can also scale to large number of samples when it is used jointly with a connectivity matrix, but is computationally expensive when no connectivity constraints are added between samples: it considers at each step all the possible merges.

FeatureAgglomeration

The FeatureAgglomeration uses agglomerative clustering to group together features that look very similar, thus decreasing the number of features. It is a dimensionality reduction tool, see Unsupervised dimensionality reduction.

2.3.6.1. Different linkage type: Ward, complete, average, and single linkage
AgglomerativeClustering supports Ward, single, average, and complete linkage strategies.

../_images/sphx_glr_plot_linkage_comparison_001.png
Agglomerative cluster has a “rich get richer” behavior that leads to uneven cluster sizes. In this regard, single linkage is the worst strategy, and Ward gives the most regular sizes. However, the affinity (or distance used in clustering) cannot be varied with Ward, thus for non Euclidean metrics, average linkage is a good alternative. Single linkage, while not robust to noisy data, can be computed very efficiently and can therefore be useful to provide hierarchical clustering of larger datasets. Single linkage can also perform well on non-globular data.

Examples

Various Agglomerative Clustering on a 2D embedding of digits: exploration of the different linkage strategies in a real dataset.

Comparing different hierarchical linkage methods on toy datasets: exploration of the different linkage strategies in toy datasets.

2.3.6.2. Visualization of cluster hierarchy
It’s possible to visualize the tree representing the hierarchical merging of clusters as a dendrogram. Visual inspection can often be useful for understanding the structure of the data, though more so in the case of small sample sizes.

../_images/sphx_glr_plot_agglomerative_dendrogram_001.png
Examples

Plot Hierarchical Clustering Dendrogram

2.3.6.3. Adding connectivity constraints
An interesting aspect of AgglomerativeClustering is that connectivity constraints can be added to this algorithm (only adjacent clusters can be merged together), through a connectivity matrix that defines for each sample the neighboring samples following a given structure of the data. For instance, in the swiss-roll example below, the connectivity constraints forbid the merging of points that are not adjacent on the swiss roll, and thus avoid forming clusters that extend across overlapping folds of the roll.

unstructured structured

These constraint are useful to impose a certain local structure, but they also make the algorithm faster, especially when the number of the samples is high.

The connectivity constraints are imposed via an connectivity matrix: a scipy sparse matrix that has elements only at the intersection of a row and a column with indices of the dataset that should be connected. This matrix can be constructed from a-priori information: for instance, you may wish to cluster web pages by only merging pages with a link pointing from one to another. It can also be learned from the data, for instance using sklearn.neighbors.kneighbors_graph to restrict merging to nearest neighbors as in this example, or using sklearn.feature_extraction.image.grid_to_graph to enable only merging of neighboring pixels on an image, as in the coin example."""

result = chain.invoke({'text':text , 'topic': text})


print(result)
chain.get_graph().print_ascii()