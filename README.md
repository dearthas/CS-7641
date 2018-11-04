egastineau3

The project has 2 folder : seed and leaf. Each folder are constructed in the same way. In a folder you find all the codes used to draw the graphs and the csv data. 
Some codes just create CSV that will be read by other code to draw graphs. 
The folders GAUSSIAN,ICA, LDA and PCA contains the transformed data with different number of component kept.

The codes : ica, gaussian_random, lda and pca are the dimensionality reduction algorithms. 
kmeans2 et em2 are the clustering algorithms to find the best parameters. 
kmeans_multiple and em_multiple are the codes used to draw the graphs that represent the clustering accuracy depending on the number of components kept after dimensionality reduction algorithms ( the data are in the folders ).
kurtosis is the code to calculate the kurtosis.
neural is the code for the neural network.
---_neural_14 or ---_neural_6 are the data used to test the neural network on the newly projected data.
leaf_em, leaf_kmeans, seeds_em and seeds_kmeans are the data used to test the neural network on the data after clustering.

Of course leaf.csv and seeds.csv are the original data.

I uploaded my project on my personnel github account : dearthas

https://github.com/dearthas/CS-7641
