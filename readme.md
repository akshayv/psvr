Parallel Support Vector Regression
==================================

Final Year Dissertation by Akshay Viswanathan at National University of Singapore.

This project corresponds to a Parallel Support Vector Regression Implementation.

Support Vector Regression is a variation of Support Vector Machines which produces a regression model from a subset of the input training data. Unfortunately, usage of SVRs for large data is limited by the cubic time and quadratic memory cost (in the number of training instances) incurred in the training process.  

To improve scalability and allow usage with real world data, a Parallel SVR algorithm has been proposed which scales with increased number of processor cores and utilizes low rank matrix approximations. 

The project is developed to be run on a Linux platform. Please ensure that MPI has been installed before proceeding.

Running the project
=================== 
1. move to /trunk directory and run the command: make
2. After the project has been made, you can run the training phase by running: mpirun -n {number of cores} ./svr_train {training_file}
  * The training by default uses the Gaussian Kernel, to use another, refer to the command line parameters
3. Prediction can be run by: mpirun -n {number of cores} ./svr_predict {prediction_file}.
  * Please note that the number of cores for prediction and training must be the same
  
For example:
* mpirun -n 2 ./svr_train datasets/pyrim
* mpirun -n 2 ./svr_predict datasets/pyrim


Please note: This project is currently functional but improvements will be made to it at least until the thesis is complete in April 2015. 
