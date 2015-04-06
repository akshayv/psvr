Parallel Support Vector Regression
==================================

Final Year Dissertation by Akshay Viswanathan at National University of Singapore.

Support Vector Regression is a variation of Support Vector Machines which produces a regression model from a subset of the input training data. Unfortunately, usage of SVRs for large data is limited by the cubic time and quadratic memory cost (in the number of training instances) incurred in the training process.  

To improve scalability and allow usage with real world data, a Parallel SVR algorithm has been proposed which scales with increased number of processor cores and utilizes low rank matrix approximations. 

The project is developed to be run on a Linux platform. Please ensure that either MPI2 or MPI3 has been installed before proceeding.

Running the project
=================== 

Preprocessing:
* Please ensure that both training and testing files have the following format:
 <br/><em>y<sub>i</sub>&nbsp;&nbsp;&nbsp;1:&nbsp;x<sub>1</sub>&nbsp;&nbsp;&nbsp;2:&nbsp;x<sub>2</sub> &nbsp;&nbsp;....&nbsp;&nbsp;n: x<sub>n</sub></em>

1. move to /trunk directory
2. compile the project by running the command: make
3. After the project has been made, you can run the training phase by running: mpirun -n {number of cores} ./svr_train [options] {training_file}
  * The training by default uses the Gaussian Kernel, to use another, refer to the command line parameters
4. Prediction can be run by: mpirun -n {number of cores} ./svr_predict [options] {prediction_file}.
  * Please note that the number of cores for prediction and training must be the same
  * The prediction results are stored in a file named 'PredictResult' by default.
  
For example:
* mpirun -n 2 ./svr_train datasets/pyrim
* mpirun -n 2 ./svr_predict datasets/pyrim
