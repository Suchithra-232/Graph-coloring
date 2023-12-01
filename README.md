# Graph Coloring via Eigen values of Hermitian Matrix and K-Means Clustering
Graph coloring with the eigen values of Hermitian matrix and K-means Clustering with multiprocessing using Ray in Python
Generating colored graph using matplotlib
Here's a step-by-step explanation of the code:

1. Ray Initialization:
   - Initialize the Ray framework with 2 CPUs.

2. Generate Sample Graph:
   - Create a sample graph using the Erdos-Renyi model with 7 nodes and a probability of 0.5 for edge creation.

3. Sequential Spectral Graph Coloring:
   - Calculate the Laplacian matrix of the graph.
   - Compute the second smallest eigenvalue and its corresponding eigenvector.
   - Assign colors to vertices based on the sign of the eigenvector.
   - Convert the colors to integers.
   - Print the colored vertices and the time taken for sequential spectral graph coloring.

4. Parallel Spectral Graph Coloring using Ray:
   - Create a remote actor (SpectralGraphColoringActor) to parallelize the spectral graph coloring.
   - Measure the time taken for parallel spectral graph coloring.
   - Print the colored vertices and the time taken for parallel spectral graph coloring.

5. Convert Colored Vertices to Array for K-Means Clustering:
   - Prepare data for K-Means clustering from both sequential and parallel processes.

6. Sequential K-Means Clustering:
   - Apply K-Means clustering sequentially to the data obtained from sequential spectral graph coloring.
   - Print the K-Means clustering result and the time taken for sequential K-Means clustering.

7. Parallel K-Means Clustering using Ray:
   - Create a remote actor (KMeansActor) to parallelize K-Means clustering.
   - Measure the time taken for parallel K-Means clustering.
   - Print the K-Means clustering result and the time taken for parallel K-Means clustering.

8. Shutdown Ray:
   - Terminate the Ray framework.

9. Combine Colored Vertices from Sequential and Parallel Processes:
   - Concatenate the colored vertices obtained from sequential spectral graph coloring and both processes of K-Means clustering.

10. Create a Colored Graph:
    - Copy the original graph and assign colors to nodes based on the combined colored vertices.
  
11. Adjust Colors to Avoid Conflicts:
    - Check for adjacent nodes with the same color and assign a new color to avoid conflicts.

12. Draw the Colored Graph:
    - Layout the graph using the spring layout algorithm.
    - Assign colors to nodes and draw the final colored graph.

13. End of Code Execution.

