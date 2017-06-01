#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "util.h"
#include <string.h>

/* compuate norm of residual */
double compute_norm(double *u, int N)
{
    int i, j;
    double norm = 0.0;
    for (i = 0; i <= N; i++) {
        for (j = 0; j <= N; j++)
            norm += u[i * (N+1) + j] * u[i * (N+1) + j];
    }
    return sqrt(norm);
}

/* set vector to zero */
void set_zero (double *u, int N) {
    int i, j;
    for (i = 0; i <= N; i++) {
        for (j = 0; j <= N; j++)
            u[i * (N+1) + j] = 0.0;
    }
}

/* debug function */
void output_to_screen (double *u, int N) {
    int i, j;
    for (i = 0; i <= N; i++) {
        for (j = 0; j <= N; j++)
            printf("%f ", u[i * (N+1) + j]);
        printf("\n");
    }
}

/* coarsen uf from (N+1)*(N+1) to (N/2+1)*(N/2+1)
 assuming N = 2^l
 */
void coarsen(double *uf, double *uc, int N) {
    int ic, jc;
    for (ic = 1; ic < N/2; ++ic)
        for (jc = 1; jc < N/2; ++jc)
            uc[ic*(N/2+1)+jc] = 0.25 * uf[2*ic*(N+1)+2*jc] + 0.125 * (uf[(2*ic-1)*(N+1)+2*jc]+uf[(2*ic)*(N+1)+(2*jc-1)]+uf[(2*ic+1)*(N+1)+(2*jc)]+uf[(2*ic)*(N+1)+(2*jc+1)]) + 0.0625 * (uf[(2*ic-1)*(N+1)+(2*jc-1)]+uf[(2*ic-1)*(N+1)+(2*jc+1)]+uf[(2*ic+1)*(N+1)+(2*jc-1)]+uf[(2*ic+1)*(N+1)+(2*jc+1)]);
}


/* refine u from (N+1)*(N+1) to (2*N+1)*(2*N+1)
 assuming N = 2^l, and add to existing uf
 */
void refine_and_add(double *u, double *uf, int N)
{
    int i, j;
    uf[1 * (2*N+1) + 1] += 0.25 * (u[0 * (N+1) + 0] + u[0 * (N+1) + 1] + u[1 * (N+1) + 0] + u[1 * (N+1) + 1]);
    for (j = 1; j < N; ++j) {
        uf[1 * (2*N+1) + 2*j] += 0.5 * (u[0 * (N+1) + j] + u[1 * (N+1) + j]);
        uf[1 * (2*N+1) + (2*j+1)] += 0.25 * (u[0 * (N+1) + j] + u[0 * (N+1) + (j+1)] + u[1 * (N+1) + j] + u[1 * (N+1) + (j+1)]);
    }
    for (i = 1; i < N; ++i) {
        uf[2*i*(2*N+1)+1] += 0.5 * (u[i * (N+1) + 0] + u[i * (N+1) + 1]);
        uf[(2*i+1)*(2*N+1)+1] += 0.25 * (u[i * (N+1) + 0] + u[i * (N+1) + 1] + u[(i+1) * (N+1) + 0] + u[(i+1) * (N+1) + 1]);
        for (j = 1; j < N; ++j) {
            uf[2*i*(2*N+1)+(2*j)] += u[i * (N+1) + j];
            uf[(2*i+1)*(2*N+1)+(2*j)] += 0.5 * (u[i * (N+1) + j] + u[(i + 1) * (N+1) + j]);
            uf[2*i*(2*N+1)+(2*j+1)] += 0.5 * (u[i * (N+1) + j] + u[i * (N+1) + (j+1)]);
            uf[(2*i+1)*(2*N+1)+(2*j+1)] += 0.25 * (u[i * (N+1) + j] + u[(i + 1) * (N+1) + j] + u[i * (N+1) + (j+1)] + u[(i+1) * (N+1) + (j+1)]);
        }
    }
}

/* compute residual vector */
void compute_residual(double *u, double *rhs, double *res, int N, double invhsq)
{
    int i, j;
    for (i = 1; i < N; i++) {
        for (j = 1; j < N; j++)
            res[i * (N+1) + j] = (rhs[i * (N+1) + j] - (4.*u[i * (N+1) + j] - u[(i-1)*(N+1)+j] - u[i*(N+1)+(j-1)] - u[(i+1)*(N+1)+j] - u[i*(N+1)+(j+1)]) * invhsq);
    }
}


/* compute residual and coarsen */
void compute_and_coarsen_residual(double *u, double *rhs, double *resc,
                                  int N, double invhsq)
{
    double *resf = calloc(sizeof(double), (N+1)*(N+1));
    compute_residual(u, rhs, resf, N, invhsq);
    coarsen(resf, resc, N);
    free(resf);
}


/* Perform Jacobi iterations on u */
void jacobi(double *u, double *rhs, int N, double hsq, int ssteps)
{
    int i, j, k;
    /* Jacobi damping parameter -- plays an important role in MG */
    double omega = 2./3.;
    double *unew = calloc(sizeof(double), (N+1)*(N+1));
    for (k = 0; k < ssteps; ++k) {
        for (i = 1; i < N; i++){
            for (j = 1; j < N; j++) {
                unew[i*(N+1)+j]  = u[i*(N+1)+j] +  omega * 0.25 * (hsq*rhs[i*(N+1)+j] + u[(i-1)*(N+1)+j] + u[i*(N+1)+(j-1)] + u[(i+1)*(N+1)+j] - u[i*(N+1)+(j+1)] - 4*u[i*(N+1)+j]);
            }
        }
        memcpy(u, unew, (N+1)*(N+1)*sizeof(double));
    }
    free (unew);
}


int main(int argc, char * argv[])
{
    int i, j, Nfine, l, iter, max_iters, levels, ssteps = 3;
    
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: ./multigrid_2d Nfine maxiter [s-steps]\n");
        fprintf(stderr, "Nfine: # of intervals, must be power of two number\n");
        fprintf(stderr, "s-steps: # jacobi smoothing steps (optional, default is 3)\n");
        abort();
    }
    sscanf(argv[1], "%d", &Nfine);
    sscanf(argv[2], "%d", &max_iters);
    if (argc > 3)
        sscanf(argv[3], "%d", &ssteps);
    
    /* compute number of multigrid levels */
    levels = floor(log2(Nfine));
    printf("Multigrid Solve using V-cycles for -u'' = f on (0,1)*(0,1)\n");
    printf("Number of intervals = %d, max_iters = %d\n", Nfine, max_iters);
    printf("Number of MG levels: %d \n", levels);
    
    /* timing */
    timestamp_type time1, time2;
    get_timestamp(&time1);
    
    /* Allocation of vectors, including left and right bdry points */
    double *u[levels], *rhs[levels];
    /* N, h*h and 1/(h*h) on each level */
    int *N = (int*) calloc(sizeof(int), levels);
    double *invhsq = (double* ) calloc(sizeof(double), levels);
    double *hsq = (double* ) calloc(sizeof(double), levels);
    double * res = (double *) calloc(sizeof(double), (Nfine+1)*(Nfine+1));
    for (l = 0; l < levels; ++l) {
        N[l] = Nfine / (int) pow(2,l);
        double h = 1.0 / N[l];
        hsq[l] = h * h;
        printf("MG level %2d, N = %8d\n", l, N[l]);
        invhsq[l] = 1.0 / hsq[l];
        u[l]    = (double *) calloc(sizeof(double), (N[l]+1)*(N[l]+1));
        rhs[l] = (double *) calloc(sizeof(double), (N[l]+1)*(N[l]+1));
    }
    /* rhs on finest mesh */
    for (i = 0; i <= N[0]; ++i) {
        for (j = 0; j <= N[0]; ++j) {
            rhs[0][i * (N+1) + j] = 1.0;
        }
    }
    
    double res_norm, res0_norm, tol = 1e-6;
    
    /* initial residual norm */
    compute_residual(u[0], rhs[0], res, N[0], invhsq[0]);
    res_norm = res0_norm = compute_norm(res, N[0]);
    printf("Initial Residual: %f\n", res0_norm);
    
    for (iter = 0; iter < max_iters && res_norm/res0_norm > tol; iter++) {
        /* V-cycle: Coarsening */
        for (l = 0; l < levels-1; ++l) {
            /* pre-smoothing and coarsen */
            jacobi(u[l], rhs[l], N[l], hsq[l], ssteps);
            compute_and_coarsen_residual(u[l], rhs[l], rhs[l+1], N[l], invhsq[l]);
            /* initialize correction for solution with zero */
            set_zero(u[l+1],N[l+1]);
        }
        /* V-cycle: Solve on coarsest grid using many smoothing steps */
        jacobi(u[levels-1], rhs[levels-1], N[levels-1], hsq[levels-1], 50);
        
        /* V-cycle: Refine and correct */
        for (l = levels-1; l > 0; --l) {
            /* refine and add to u */
            refine_and_add(u[l], u[l-1], N[l]);
            /* post-smoothing steps */
            jacobi(u[l-1], rhs[l-1], N[l-1], hsq[l-1], ssteps);
        }
        
        if (0 == (iter % 1)) {
            compute_residual(u[0], rhs[0], res, N[0], invhsq[0]);
            res_norm = compute_norm(res, N[0]);
            printf("[Iter %d] Residual norm: %2.8f\n", iter, res_norm);
        }
    }
    
    /* Clean up */
    free (hsq);
    free (invhsq);
    free (N);
    free(res);
    for (l = levels-1; l >= 0; --l) {
        free(u[l]);
        free(rhs[l]);
    }
    
    /* timing */
    get_timestamp(&time2);
    double elapsed = timestamp_diff_in_seconds(time1,time2);
    printf("Time elapsed is %f seconds.\n", elapsed);
    return 0;
}