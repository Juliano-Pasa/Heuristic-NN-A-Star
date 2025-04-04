# Learning Heuristics for Topographic Path Planning in Agent-Based Simulations

## About

This project had the objective of optimizing path planning calculations for agents in a large topographic terrain through modifications in the heuristic functions. Together with @Henrique-Liesenfeld-Krever and @ThiagoRSL, we developed several deep neural networks to replace heuristic functions. We found that training these DNNs with terrain data to predict the distance between a starting and an ending point yields better results than using generic functions such as the euclidean distance. The DNNs were capable of learning terrain specific topography and how they affect the distance between two points. The results of this work were published in [this paper](https://www.scitepress.org/Papers/2023/121299/121299.pdf) and landed us an indication to best student paper during [SIMULTECH 2023](https://simultech.scitevents.org/?y=2023). More work was developed after the publication to further improve the performance of the DNNs.

## Gathering Data for Training

The first step was to generate data about the terrain to train the DNNs. [1_generate_dataset.py](1_generate_dataset.py) runs a parallel Dijkstra in the GPU to calculate the real distance from points A to points B. We used 8 maps with dimensions of 300x300 pixels, sampling 10% of points. This resulted in a dataset with ~16.5 GB and took ~48 hours to train. The distance between all calculated points were our target value, since they are the value we expect a heuristic function to output.

[2_shuffle_split_dataset.py](2_shuffle_split_dataset.py) prepares the data for training. It shuffles the files and splits them in training, validation and test set.

## Training the DNNs

There was nothing special during the training of the DNNs. Each DNN had 7 features: *mapId, startX, startY, startZ, goalX, goalY, goalZ* and 1 target *D(S, G)*, where *D(S, G)* was the distance between the points *S (start)* and *G (goal)*. [3_train.py](3_train.py) executes the training on all models.

## Testing the DNNs

We evaluated the DNNs in multiple dimensions. More importantly than minimizing the loss function, we needed to have a DNN that could speed up the path planning between two points. For thousands of paths computed, we measured number of openned nodes, path cost (total distance) and execution time. The tests consisted of computing the path using A* with euclidean distance and comparing it to A* with the DNNs. Using the DNNs, path costs was slightly higher (not ideal), but number of openned nodes and execution time were significantly lower. Unfortunately, execution time was only lower if the values were precomputed. [This cursed file](4_test.py) runs all tests.

## Solving the Execution Time problem

During testing, we noticed that calling the DNNs to calculate on value at a time was extremely slow. This made the overall execution time slower, making this approach worse than using a simple euclidean distance. Precomputing and caching the distances among all points in the map also was not viable. This would only be useful if lots of paths had to be calculated, and even then, the memory required to store all this data increases to the factorial of the map size. 

To solve this problem, we developed a middle ground between calling the DNN for each point and caching all results. Everytime A* requested the heuristic value between a point P and the goal G, it checks if this value is cached or not. If it is, it is simply returned and the algorithm continues. If the distance between them is not cached, the DNN computes the distance from P to G, and also the distances from the neighbours of P to G. Since A* will likely request the heuristic value of a neighbour of P, this approach basically precomputes values only for points that will probably be part of the final path. With this modification, using A* with the DNNs used less memory and took less time to execute when compared to A* using euclidean distance.
