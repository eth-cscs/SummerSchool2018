// linear algebra subroutines
// Ben Cumming @ CSCS

#include <iostream>

#include <cmath>
#include <cstdio>

#include <mpi.h>

// Just for cuBLAS
#include "cuda_helpers.h"
#include "linalg.h"
#include "operators.h"
#include "stats.h"
#include "data.h"

namespace linalg {

bool cg_initialized = false;
Field r;
Field Ap;
Field p;
Field Fx;
Field Fxold;
Field v;
Field xold;

using namespace operators;
using namespace stats;
using data::Field;

// initialize temporary storage fields used by the cg solver
// I do this here so that the fields are persistent between calls
// to the CG solver. This is useful if we want to avoid malloc/free calls
// on the device for the OpenACC implementation
void cg_init(int nx, int ny)
{
    Ap.init(nx,ny);
    r.init(nx,ny);
    p.init(nx,ny);
    Fx.init(nx,ny);
    Fxold.init(nx,ny);
    v.init(nx,ny);
    xold.init(nx,ny);

    cg_initialized = true;
}

////////////////////////////////////////////////////////////////////////////////
//  blas level 1 reductions
////////////////////////////////////////////////////////////////////////////////

// computes the inner product of x and y
// x and y are vectors on length N
double ss_dot(Field const& x, Field const& y)
{
    double result = 0;
    double result_global = 0;
    const int N = x.length();
    auto status =
        cublasDdot(
            cublas_handle(),  N,
            x.device_data(), 1,
            y.device_data(), 1,
            &result
        );

    MPI_Allreduce(&result, &result_global, 1, MPI_DOUBLE, MPI_SUM, data::domain.comm_cart);
    return result_global;
}

// computes the 2-norm of x
// x is a vector on length N
double ss_norm2(Field const& x)
{
    double result = 0;
    double result_global = 0;
    const int n = x.length();

    auto status =
        cublasDnrm2(
            cublas_handle(), n,
            x.device_data(), 1,
            &result
        );

    // take the square of the result, because we still have to sum of the local
    // partial sums before taking sqrt of the full global sum
    result *= result;
    MPI_Allreduce(&result, &result_global, 1, MPI_DOUBLE, MPI_SUM, data::domain.comm_cart);
    return sqrt(result_global);
}

// sets entries in a vector to value
// x is a vector on length N
// value is th
void ss_fill(Field& x, const double value)
{
    const int n = x.length();

    // TODO: Offload this loop to the GPU
    for (int i = 0; i < n; i++)
        x[i] = value;
}

////////////////////////////////////////////////////////////////////////////////
//  blas level 1 vector-vector operations
////////////////////////////////////////////////////////////////////////////////

// computes y := alpha*x + y
// x and y are vectors on length N
// alpha is a scalar
void ss_axpy(Field& y, const double alpha, Field const& x)
{
    const int n = y.length();

    // TODO: Offload this loop to the GPU
    for (int i = 0; i < n; i++)
        y[i] += alpha * x[i];
}

// computes y = x + alpha*(l-r)
// y, x, l and r are vectors of length N
// alpha is a scalar
void ss_add_scaled_diff(Field& y, Field const& x, const double alpha,
    Field const& l, Field const& r)
{
    const int n = y.length();

    // TODO: Offload this loop to the GPU
    for (int i = 0; i < n; i++)
        y[i] = x[i] + alpha * (l[i] - r[i]);
}

// computes y = alpha*(l-r)
// y, l and r are vectors of length N
// alpha is a scalar
void ss_scaled_diff(Field& y, const double alpha,
    Field const& l, Field const& r)
{
    const int n = y.length();

    // TODO: Offload this loop to the GPU
    for (int i = 0; i < n; i++)
        y[i] = alpha * (l[i] - r[i]);
}

// computes y := alpha*x
// alpha is scalar
// y and x are vectors on length n
void ss_scale(Field& y, const double alpha, Field& x)
{
    const int n = x.length();

    // TODO: Offload this loop to the GPU
    for (int i = 0; i < n; i++)
        y[i] = alpha * x[i];
}

// computes linear combination of two vectors y := alpha*x + beta*z
// alpha and beta are scalar
// y, x and z are vectors on length n
void ss_lcomb(Field& y, const double alpha, Field& x, const double beta,
    Field const& z)
{
    const int n = x.length();

    // TODO: Offload this loop to the GPU
    for (int i = 0; i < n; i++)
        y[i] = alpha * x[i] + beta * z[i];
}

// copy one vector into another y := x
// x and y are vectors of length N
void ss_copy(Field& y, Field const& x)
{
    const int n = x.length();

    // TODO: Offload this loop to the GPU
    for (int i = 0; i < n; i++)
        y[i] = x[i];
}

// conjugate gradient solver
// solve the linear system A*x = b for x
// the matrix A is implicit in the objective function for the diffusion equation
// the value in x constitute the "first guess" at the solution
// x(N)
// ON ENTRY contains the initial guess for the solution
// ON EXIT  contains the solution
void ss_cg(Field& x, Field const& b, const int maxiters, const double tol, bool& success)
{

    // this is the dimension of the linear system that we are to solve
    int nx = data::domain.nx;
    int ny = data::domain.ny;

    if(!cg_initialized) {
        cg_init(nx,ny);
    }

    // epsilon value use for matrix-vector approximation
    double eps     = 1.e-8;
    double eps_inv = 1. / eps;

    // allocate memory for temporary storage
    ss_fill(Fx,    0.0);
    ss_fill(Fxold, 0.0);
    ss_copy(xold, x);

    // matrix vector multiplication is approximated with
    // A*v = 1/epsilon * ( F(x+epsilon*v) - F(x) )
    //     = 1/epsilon * ( F(x+epsilon*v) - Fxold )
    // we compute Fxold at startup
    // we have to keep x so that we can compute the F(x+exps*v)
    diffusion(x, Fxold);

    // v = x + epsilon*x
    ss_scale(v, 1.0 + eps, x);

    // Fx = F(v)
    diffusion(v, Fx);

    // r = b - A*x
    // where A*x = (Fx-Fxold)/eps
    ss_add_scaled_diff(r, b, -eps_inv, Fx, Fxold);

    // p = r
    ss_copy(p, r);

    // rold = <r,r>
    double rold = ss_dot(r, r);
    double rnew = rold;

    // check for convergence
    success = sqrt(rold) < tol;
    if (success) {
        return;
    }

    int iter;
    for(iter=0; iter<maxiters; iter++) {
        // Ap = A*p
        ss_lcomb(v, 1.0, xold, eps, p);
        diffusion(v, Fx);
        ss_scaled_diff(Ap, eps_inv, Fx, Fxold);

        // alpha = rold / p'*Ap
        auto dot = ss_dot(p, Ap);

        double alpha = rold / dot;

        // x += alpha*p
        ss_axpy(x, alpha, p);

        // r -= alpha*Ap
        ss_axpy(r, -alpha, Ap);

        // find new norm
        rnew = ss_dot(r, r);

        // test for convergence
        if (sqrt(rnew) < tol) {
            success = true;
            break;
        }

        // p = r + (rnew/rold) * p
        ss_lcomb(p, 1.0, r, rnew / rold, p);

        rold = rnew;
    }
    stats::iters_cg += iter + 1;

    if (!success && !data::domain.rank) {
        std::cerr << "ERROR: CG failed to converge after " << iter
                  << " iterations, with residual " << sqrt(rnew)
                  << std::endl;
    }
}

} // namespace linalg
