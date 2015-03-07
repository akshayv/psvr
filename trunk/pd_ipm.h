/*

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef PD_IPM_H__
#define PD_IPM_H__

namespace psvr {
class PrimalDualIPMParameter;
class ParallelMatrix;
class Document;
class Model;
class LLMatrix;
// Newton method for primal-dual interior point method for SVM optimization,
class PrimalDualIPM {
 public:
  // Using Newton method to solve the optimization problem
  //   parameter: the options of interior point method
  //   h: ICF factorized matrix
  //   doc: data points
  //   model: store the optimization result
  int Solve(const PrimalDualIPMParameter& parameter,
            const ParallelMatrix& h,
            const Document& doc,
            Model* model,
            bool failsafe);

 private:
  int ComputePartialRho(const ParallelMatrix& icf,
                        const double *x, const double* x_star,
                        const int local_num_rows,
                        const double to, double *rho);

  // Compute $HH^T\alpha$, which is part of $z$, $\alpha$ is primal variable
  int ComputePartialZ(const ParallelMatrix& icf,
                      const double *rho_x, const double* rho_x_star, 
                      const double* delta_star,
                      const int local_num_rows,
                      double *z);

  int ComputePartialSigma(const ParallelMatrix& icf,
                          const double *delta, const double *delta_star,
                          const int local_num_rows, double *sigma);

  // Compute surrogate gap
  double ComputeSurrogateGap(double c_pos,
                                        double c_neg,
                                        const double *value,
                                        int local_num_rows,
                                        const double *x,
                                        const double *x_star,
                                        const double *la,
                                        const double *xi,
                                        const double *the,
                                        const double *phi);

  // Compute direction of primal vairalbe $x$
  int ComputeDeltaX(const ParallelMatrix& icf,
                    const double *d, const double *mult_factor,
                    const double *value,
                    const double dnu, const double *z,
                    const LLMatrix& lra, int local_num_rows,
                    double *dx);

  // Compute direction of primal varialbe $\nu$
  int ComputeDeltaNu(const ParallelMatrix& icf, 
          const double *d, const double *z, 
          const double *mult_factor,
          const double *x, const double *x_star,
          const double *delta, const double *delta_star,
          const double *rho_x, const double *rho_x_star,
          const LLMatrix& lra, int local_num_rows, 
          double *dnu);

  // Solve a special form of linear equation using
  // Sherman-Morrison-Woodbury formula
  int LinearSolveViaICFCol(const ParallelMatrix& icf,
                           const double *mult_factor,
                           const double *d,
                           const double *b,
                           const LLMatrix& lra,
                           const int local_num_rows,
                           double *x);

  // Loads the values of alpha, alpha*, xi, lambda, theta, phi and nu 
  //to resume from an interrupted solving process.
  void LoadVariables(const PrimalDualIPMParameter& parameter,
                     int num_local_doc, int num_total_doc, int *step,
                     double* nu, double *x, double *x_star, double *la, 
                     double *xi, double* the, double* phi);

  // Saves the values of alpha, alpha*, xi, lambda, theta, phi and nu.
  void SaveVariables(const PrimalDualIPMParameter& parameter,
                     int num_local_doc, int num_total_doc, int step,
                     double nu, double *x, double *x_star, double *la, 
                     double *xi, double* the, double* phi);
};
}  // namespace psvm

#endif
