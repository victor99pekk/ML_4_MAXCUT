// maxcut_sdp.c
//
// Compile with:
//   gcc -O2 maxcut_sdp.c -o maxcut_sdp -lcsdp -llapack -lblas -lm
//
// Usage:
//   ./maxcut_sdp adjacency.txt
//
// where adjacency.txt is formatted as:
//   n
//   w11 w12 ... w1n
//   w21 w22 ... w2n
//   ...
//   wn1 wn2 ... wnn
//
// (entries are 0 or 1)


// clang -O2 maxcut_sdp.c \
//   -I$(brew --prefix)/include \
//   -L$(brew --prefix)/lib \
//   -lcsdp \
//   -framework Accelerate \
//   -lm \
//   -o maxcut_sdp

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "csdp.h"    // http://www.math.uwaterloo.ca/~bico/CSdp/
extern void dsyev_(char*, char*, int*, double*, int*, double*, double*, int*, int*);

// simple Gaussian RNG via Box–Muller
static int have_spare = 0;
static double rand_spare;
double randn() {
    if (have_spare) {
        have_spare = 0;
        return rand_spare;
    }
    have_spare = 1;
    double u, v, s;
    do {
        u = 2.0*rand()/RAND_MAX - 1.0;
        v = 2.0*rand()/RAND_MAX - 1.0;
        s = u*u + v*v;
    } while (s == 0.0 || s >= 1.0);
    s = sqrt(-2.0*log(s)/s);
    rand_spare = v*s;
    return u*s;
}

// read adjacency
int read_adj(const char *fn, int *pn, double **Wout) {
    FILE *f = fopen(fn,"r");
    if(!f) return -1;
    int n;
    if(fscanf(f,"%d",&n)!=1) { fclose(f); return -1; }
    double *W = malloc(n*n*sizeof(double));
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            int wij;
            if(fscanf(f,"%d",&wij)!=1){ fclose(f); free(W); return -1; }
            W[i*n+j] = (double)wij;
        }
    }
    fclose(f);
    *pn = n;
    *Wout = W;
    return 0;
}

int main(int argc, char **argv){
    if(argc!=2){
        fprintf(stderr,"Usage: %s adjacency.txt\n",argv[0]);
        return 1;
    }
    srand(time(NULL));

    // 1) load W
    int n;
    double *W;
    if(read_adj(argv[1],&n,&W)){
        fprintf(stderr,"Error reading %s\n",argv[1]);
        return 1;
    }

    // 2) build SDP data for CSDP:
    //    max ¼∑ W_ij (1−X_ij)  <=>  min ∑ W_ij X_ij  + const
    //    s.t. X⪰0, X_ii=1
    int m = n;                     // # linear constraints
    int blockStruct[2] = {1, n};   // one semidef block of size n
    double *b = malloc(m*sizeof(double));
    for(int i=0;i<m;i++) b[i]=1.0;

    // each constraint k: ⟨E_kk, X⟩ = 1
    constraintMatrix *constraints = malloc(m*sizeof(constraintMatrix));
    for(int k=0;k<m;k++){
        constraints[k].blockNumber = 1;
        constraints[k].blockPtr   = calloc(n+1, sizeof(int));
        constraints[k].blockPtr[0] = 1;           // 1 non‐zero
        constraints[k].blockZ   = calloc(1, sizeof(int)); // index into entries below
        constraints[k].blockRow = calloc(1, sizeof(int));
        constraints[k].blockCol = calloc(1, sizeof(int));
        constraints[k].blockMat = calloc(1, sizeof(double));
        // store (row,col,val) = (k,k,1.0)
        constraints[k].blockRow[0] = k+1;          
        constraints[k].blockCol[0] = k+1;
        constraints[k].blockMat[0] = 1.0;
    }

    // objective matrix: C satisfying minimize ⟨C,X⟩ = ∑_{i<j} W_ij X_ij
    // we'll only store the strict upper‐triangle; CSDP expects
    int nzC = 0;
    for(int i=0;i<n;i++) for(int j=i+1;j<n;j++) if(W[i*n+j]!=0) nzC++;
    int *CblockNo = malloc(nzC*sizeof(int));
    int *iC = malloc(nzC*sizeof(int));
    int *jC = malloc(nzC*sizeof(int));
    double *vC = malloc(nzC*sizeof(double));
    int idx=0;
    for(int i=0;i<n;i++){
        for(int j=i+1;j<n;j++){
            double w = W[i*n+j];
            if(w==0) continue;
            CblockNo[idx] = 1;      // block 1
            iC[idx] = i+1;          
            jC[idx] = j+1;          
            vC[idx] = w;            
            idx++;
        }
    }

    // 3) call CSDP
    double *X = calloc(n*n, sizeof(double));
    double *y = calloc(m, sizeof(double));
    double *Z = calloc(n*n, sizeof(double));
    int sstatus = easy_sdp(
        1, blockStruct,
        m,           // # constraints
        &nzC, CblockNo, iC, jC, vC,
        b,
        0,           // 0 = minimize
        NULL,        // no free vars
        X, y, Z, NULL
    );
    if(sstatus){
        fprintf(stderr,"CSDP failed (status=%d)\n",sstatus);
        return 1;
    }

    // 4) randomized rounding via eigen-decomp of X
    //    X is stored block‐wise; here it's just an n×n dense
    char jobz='V', uplo='U';
    int info, lda=n;
    double *wvals = malloc(n*sizeof(double));
    dsyev_(&jobz, &uplo, &n, X, &lda, wvals, NULL, 0, &info);
    if(info){
        fprintf(stderr,"dsyev_ failed: %d\n", info);
        return 1;
    }
    // form V = Q * sqrt(diag(max(w,0)))
    double *V = malloc(n*n*sizeof(double));
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            double lam = wvals[j] > 0 ? sqrt(wvals[j]) : 0;
            V[i*n+j] = X[i*n+j] * lam;
        }
    }
    // pick random normal g
    double *g = malloc(n*sizeof(double));
    for(int j=0;j<n;j++) g[j] = randn();

    // compute h = V * g
    double *h = calloc(n, sizeof(double));
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            h[i] += V[i*n+j] * g[j];
        }
    }

    // 5) output partition (0/1) by sign(h_i)
    for(int i=0;i<n;i++){
        printf("%d\n", h[i]>=0 ? 1 : 0);
    }

    return 0;
}
