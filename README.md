# Neural Network from Scratch

"Neural Network from Scratch" is my implementation of a neural network starting from first principles. It includes both forward and backward propagations, along with a variety of gradient-based and non-gradient-based optimizers. The main goal of this project was to deepen my understanding of neural network mechanics. Over time, this project has evolved into a comprehensive library, equipped with tools like Docker to generate documentation, and a demo UI project designed to showcase the library's features in an accessible and interactive manner.

For more details, you can explore the [Neural Network from Scratch library documentation](library/nnfs_library.md).

## Documentation

The documentation is generated using `docker-compose.docs.yml`. A Docker container is utilized to generate document content, producing PDF and HTML pages. To access the documentation, refer to [NNFS Documentation](library/docs).

Additionally, you can find the documentation for the demo program at [Demo Documentation](program/nnfs_demo.md).

## Videos

I've prepared a video that offers a guided tour of the User Interface. This UI simplifies interactions with the neural network library by allowing modification of various network parameters. You can watch the video here: [User Interface Walkthrough](videos/Neural%20Networks%20From%20Scratch%20UI.mp4).

To visualize the network architecture, take a look at this graph: [Network Architecture Visualization](https://github.com/JHF101/nnfs/assets/63376348/8baaa871-5875-4679-acbf-a8faedb8e926).

Additionally, I've included a video showcasing the training, testing, and validation error and accuracy using a gradient-based algorithm known as the Delta-Bar-Delta optimizer. You can view the results here: [Training Results with Delta-Bar-Delta Optimizer](https://github.com/JHF101/nnfs/assets/63376348/fff8789a-c740-4f98-ac2b-0da3f8bffae9).

Lastly, there's a video demonstrating the use of a genetic algorithm, which is a non-gradient algorithm. This algorithm's evaluation parameter has been adjusted to use either error or accuracy as its fitness evaluation metric. Watch the video here: [Genetic Algorithm Demonstration](https://github.com/JHF101/nnfs/assets/63376348/d2b1fa2d-7d92-4ec3-97fd-65b4303c9e4d).

## References

[1] D. C. Plaut and A. Others, “Experiments on Learning by Back Propagation,” Jun. 1986. Accessed: Oct. 15, 2023. [Online]. Available: https://eric.ed.gov/?id=ED286930

[2] S. Sarkar and K. L. Boyer, “Quantitative Measures of Change Based on Feature Organization: Eigenvalues and Eigenvectors,” Computer Vision and Image Understanding, vol. 71, no. 1, pp. 110–136, Jul. 1998, doi: 10.1006/cviu.1997.0637.

[3] R. A. Jacobs, “Increased rates of convergence through learning rate adaptation,” Neural Networks, vol. 1, no. 4, pp. 295–307, Jan. 1988, doi: 10.1016/0893-6080(88)90003-2.

[4] R. Sutton, “Adapting Bias by Gradient Descent: An Incremental Version of Delta-Bar-Delta,” Proceedings of the Tenth National Conference on Artificial Intelligence, 1995, [Online]. Available: https://www.researchgate.net/publication/2783837_Adapting_Bias_by_Gradient_Descent_An_Incremental_Version_of_Delta-Bar-Delta

[5] M. Riedmiller, “Advanced supervised learning in multi-layer perceptrons — From backpropagation to adaptive learning algorithms,” Computer Standards & Interfaces, vol. 16, no. 3, pp. 265–278, Jul. 1994, doi: 10.1016/0920-5489(94)90017-5.

[6] M. Riedmiller and H. Braun, “A direct adaptive method for faster backpropagation learning: the RPROP algorithm,” in IEEE International Conference on Neural Networks, San Francisco, CA, USA: IEEE, 1993, pp. 586–591. doi: 10.1109/ICNN.1993.298623.

[7] M. Riedmiller, Rprop - Description and Implementation Details: Technical Report. Inst. f. Logik, Komplexit{\"a}t u. Deduktionssysteme, 1994.

[8] C. Igel and M. Hüsken, “Improving the Rprop Learning Algorithm,” Proceeding of the Second International Symposium on Neural Computation, NC’2000, pp. 115–121, 2000.

[9] R. Mahajan and G. Kaur, “Neural Networks using Genetic Algorithms,” vol. 77, pp. 6–11, 2013, doi: 10.5120/13549-1153.

[10] L. Prechelt and F. Informatik, “PROBEN1 - A Set of Neural Network Benchmark Problems and Benchmarking Rules,” 1995.

[11] D. J. Montana and L. Davis, “Training feedforward neural networks using genetic algorithms,” in Proceedings of the 11th international joint conference on Artificial intelligence - Volume 1, in IJCAI’89. San Francisco, CA, USA: Morgan Kaufmann Publishers Inc., Aug. 1989, pp. 762–767.

[12] P. C. Pendharkar and J. A. Rodger, “AN EMPIRICAL STUDY OF NON-BINARY GENETIC ALGORITHM-BASED NEURAL APPROACHES FOR CLASSIFICATION: 20th International Conference on Information Systems, ICIS 1999,” 1999, pp. 155–165. Accessed: Oct. 15, 2023. [Online]. Available: http://www.scopus.com/inward/record.url?scp=85032266822&partnerID=8YFLogxK

[13] C. M. Bishop, Pattern recognition and machine learning. in Information science and statistics. New York: Springer, 2006.
