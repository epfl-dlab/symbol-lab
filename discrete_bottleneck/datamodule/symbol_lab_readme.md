## 1. Grid Dataset
Samples in this environment can be thought of as a 2-D grid of cells, each cell being either empty or containing an object. Objects can be of different types, and objects of the same type can have different colors. For the simplest case, let us discard the color and let all objects be the same color but may be of different types. We can denote empty cells by 0 and index objects by their type. For instance, if there are only 3 object types \{square, circle, triangle\}, indexed by \{1,2,3\}, then the following grid could be a generated sample containing 6 squares, 2 circles, and 3 triangles:

		1	1	0	0	2	3
		0	2	1	0	1	3
		3	0	0	0	1	1

This grid of objects can be thought of as a long sequence if the rows are put beside each other. The string representation of such a flattened sequence will be a sample of the Grid dataset.
The motivation behind defining this environment is that if such a sequence is to be reconstructed from a learned latent representation, then a discrete compositional representation can have an advantage in terms of the required capacity, interpretability, and generalization. Consider a seq2seq model for this task; it has to encode all the grid information into a high dimensional representation. Now, if we just slowly start to increase the number of objects and their types and increase each dimension by 1, it is very easy to see how fast the latent dimension should grow to be able to still encode all the possibilities. Such representation of knowledge is clearly inefficient. The situation is better for VQ-VAE yet not perfect. However, consider a discrete sequential representation, i.e., a sequence of words. In principle, it has the capacity to describe the environment efficiently through the composition of basic elements. And it is not confined to just one representation. To describe the above grid with words, either of the following leads to the same reconstruction:
-	squares 1,2,9,11,17,18, circles 5,8, triangles 6,12,13 (counting positions row-wise)
-	squares 1,4,8,14,15,18, circles 5,13, triangles 3,16,17 (counting positions column-wise)
-	squares: two neighbors at 1, two neighbors at 17, two separated by an empty at 9, circles and triangles: like before

The point is that if the grid gets larger or objects are added, the compositional representation should still capture it without a hassle.

## 2. Structured Discrete Latent
In the Grid dataset, we have a discrete space but we do not specify or directly control the latent from which the grid is instantiated. In fact as was discussed above, there could be several latents describing the same output. However in this environment, we are going to explicitly specify the discrete latent structure ($Z$) and define a decoder ($f$) for that and control several aspects of them. This environment is useful to get an intuition on controlled setups, discovering weaknesses and advantages of different methods, and evaluating theoretical analysis.

$f$ can have a combination of the following properties:
- deterministic vs. stochastic
- invertible vs. non-invertible
- real number output vs. sequence outputs

$Z$ can be any of the following structures:
- One discrete variable with 2 possible values (1 bit of information)
- One discrete variable with n possible values
- M discrete variables, each with n possible values (cross-product space)

These properties and parameters specifying them can be observed in [controlled_decoder.yaml](../../configs/datamodule/controlled_decoder.yaml). Here in this [notebook](../../notebooks/dataset_playground.ipynb) we will instantiate datasets for several combinations of these properties and give several samples to illustrate the discrete latent structure and its decoding in each scenario. For stochastic vs. deterministic (S vs. D) and invertible vs. non-invertible (I vs. NI), we have created 4 config files and will use them in the notebook.