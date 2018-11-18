#include <vector>
#include <algorithm>
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
#include <arpa/inet.h>

#include "utils.h"

extern gsl_rng * RANDOM_NUMBER;

int compare (const void * a, const void * b) {
    return ( *(int*)a - *(int*)b );
}

double log_sum(double log_a, double log_b) {
    double v;
    if (log_a == -1) return(log_b);

    if (log_a < log_b) {
        v = log_b+log(1 + exp(log_a-log_b));
    }
    else {
        v = log_a+log(1 + exp(log_b-log_a));
    }
    return(v);
}

void minc(gsl_matrix* m, int i, int j, double x) {
    mset(m, i, j, mget(m, i, j) + x);
    return;
}

void row_sum(const gsl_matrix* m, gsl_vector* val) {
    size_t i, j;
    gsl_vector_set_zero(val);
    for (i = 0; i < m->size1; i++)
        for (j = 0; j < m->size2; j++)
            vinc(val, i, mget(m, i, j));
    return;
}

void col_sum(const gsl_matrix* m, gsl_vector* val) {
    size_t i, j;
    gsl_vector_set_zero(val);

    for (i = 0; i < m->size1; i++)
        for (j = 0; j < m->size2; j++)
            vinc(val, j, mget(m, i, j));
    return;
}


void mtx_sqrt(gsl_matrix* m) {
    size_t i, j;
    for (i = 0; i < m->size1; i++)
        for (j = 0; j < m->size2; j++)
            mset(m, i, j, sqrt(mget(m, i, j)));
    return;
}


void vct_fprintf(FILE * file, const gsl_vector * v) {
    size_t i;
    for (i = 0; i < v->size; i++)
        fprintf(file, "%10.15e ", vget(v, i));
    fprintf(file, "\n");
    return;
}

void mtx_fprintf(FILE * file, const gsl_matrix * m) {
    size_t i, j;
    for (i = 0; i < m->size1; i++) {
        for (j = 0; j < m->size2; j++)
            fprintf(file, "%10.15e ", mget(m, i, j));
        fprintf(file, "\n");
    }
    return;
}

void mtx_fscanf(FILE* file, gsl_matrix* m) {
    size_t i, j;
    double x;
    for (i = 0; i < m->size1; i++) {
        for (j = 0; j < m->size2; j++) {
            fscanf(file, "%lf", &x);
            mset(m, i, j, x);
        }
    }
    return;
}

void matrix_vector_solve(const gsl_matrix* m, const gsl_vector* b, gsl_vector* v) {
    gsl_matrix *lu;
    gsl_permutation* p;
    int signum;

    p = gsl_permutation_alloc(m->size1);
    lu = gsl_matrix_alloc(m->size1, m->size2);

    gsl_matrix_memcpy(lu, m);
    gsl_linalg_LU_decomp(lu, p, &signum);
    gsl_linalg_LU_solve(lu, p, b, v);
  
    gsl_matrix_free(lu);
    gsl_permutation_free(p);
    return;
}


void matrix_inverse(const gsl_matrix* m, gsl_matrix* inverse) {
    gsl_matrix *lu;
    gsl_permutation* p;
    int signum;
    p = gsl_permutation_alloc(m->size1);
    lu = gsl_matrix_alloc(m->size1, m->size2);
    gsl_matrix_memcpy(lu, m);
    gsl_linalg_LU_decomp(lu, p, &signum);
    gsl_linalg_LU_invert(lu, p, inverse);

    gsl_matrix_free(lu);
    gsl_permutation_free(p);
    return;
}

double log_det(const gsl_matrix* m) {
    gsl_matrix* lu;
    gsl_permutation* p;
    double result;
    int signum;
    p = gsl_permutation_alloc(m->size1);
    lu = gsl_matrix_alloc(m->size1, m->size2);
    gsl_matrix_memcpy(lu, m);
    gsl_linalg_LU_decomp(lu, p, &signum);
    result = gsl_linalg_LU_lndet(lu);

    gsl_matrix_free(lu);
    gsl_permutation_free(p);

  return result;
}


void sym_eigen(gsl_matrix* m, gsl_vector* vals, gsl_matrix* vects) {
    gsl_eigen_symmv_workspace* wk;
    gsl_matrix* mcpy;
    int r;
    mcpy = gsl_matrix_alloc(m->size1, m->size2);
    wk = gsl_eigen_symmv_alloc(m->size1);
    gsl_matrix_memcpy(mcpy, m);
    r = gsl_eigen_symmv(mcpy, vals, vects, wk);
    gsl_eigen_symmv_free(wk);
    gsl_matrix_free(mcpy);
    return;
}

void gsl_vector_apply(gsl_vector* x, double(*fun)(double)) {
    size_t i;
    for(i = 0; i < x->size; i ++)
        vset(x, i, fun(vget(x, i)));
    return;
}

void vct_log(gsl_vector* v) {
    int i, size = v->size;
    for (i = 0; i < size; i++)
        vset(v, i, safe_log(vget(v, i)));
    return;
}

void mtx_log(gsl_matrix* x) {
    size_t i, j;
    for (i = 0; i < x->size1; i++)
        for (j = 0; j < x->size2; j++)
            mset(x, i, j, safe_log(mget(x, i, j)));
    return;
}

double vnorm(const gsl_vector *v) {
    return gsl_blas_dnrm2(v);
}

double log_normalize(gsl_vector* x) {
    double v = vget(x, 0);
    size_t i;
    for (i = 1; i < x->size; i++)
        v = log_sum(v, vget(x, i));
    for (i = 0; i < x->size; i++)
        vset(x, i, vget(x,i)-v);
    return v;
}

double vnormalize(gsl_vector* x) {
    double v = vsum(x);
    if (v > 0 || v < 0)
        gsl_vector_scale(x, 1/v);
    return v;
}

void vnormalize_ex(gsl_vector* v, double total){
    if (total <= 0) {
        printf("vnormalize_ex error. total = %.5f\n", total);
        return;
    }
    gsl_vector_scale(v, 1/total);
    return;
}

void vct_exp(gsl_vector* x) {
    for (size_t i = 0; i < x->size; i++)
        vset(x, i, exp(vget(x, i)));
    return;
}

void mtx_exp(gsl_matrix* x) {
    size_t i, j;
    for (i = 0; i < x->size1; i++)
        for (j = 0; j < x->size2; j++)
            mset(x, i, j, exp(mget(x, i, j)));
    return;
}

double mahalanobis_distance(const gsl_matrix* m, const gsl_vector* u, const gsl_vector* v) {
    double val = 0;
    gsl_vector* x = gsl_vector_alloc(u->size);
    gsl_vector_memcpy(x, u);
    gsl_vector_sub(x, v);
    val = mahalanobis_prod(m, x, x);
    gsl_vector_free(x);
    return val;
}

double mahalanobis_prod(const gsl_matrix* m, const gsl_vector* u, const gsl_vector* v) {
    gsl_vector* x = gsl_vector_alloc(u->size);
    gsl_blas_dgemv(CblasNoTrans, 1.0, m, v, 0.0, x);
    double val = 0;
    gsl_blas_ddot(u, x, &val);
    gsl_vector_free(x);
    return val;
}

double matrix_dot_prod(const gsl_matrix* m1, const gsl_matrix* m2) {
    double val = 0, result;
    for (size_t i = 0; i < m1->size1; i ++) {
        gsl_vector_const_view v1 = gsl_matrix_const_row(m1, i);
        gsl_vector_const_view v2 = gsl_matrix_const_row(m2, i);
        gsl_blas_ddot(&v1.vector, &v2.vector, &result); 
        val += result;
    }
    return val;
}


bool file_exists(const char * filename) {
    if ( 0 == access(filename, R_OK))
        return true;
    return false;
}



int dir_exists(const char *dname) {
    struct stat st;
    int ret;
    if (stat(dname,&st) != 0) {
        return 0;
    }
    ret = S_ISDIR(st.st_mode);
    if(!ret) {
        errno = ENOTDIR;
    }
    return ret;
}

void make_directory(const char* name) {
    mkdir(name, S_IRUSR|S_IWUSR|S_IXUSR);
}

gsl_rng * new_random_number_generator(long seed) {
    gsl_rng * random_number_generator = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(random_number_generator, (long) seed);
    return random_number_generator;
}

void free_random_number_generator(gsl_rng * random_number_generator) {
    gsl_rng_free(random_number_generator);
}

void choose_k_from_n(int k, int n, int* result, int* src) {
    gsl_ran_choose(RANDOM_NUMBER, (void *) result,  k, (void *) src, n, sizeof(int));
}

void sample_k_from_n(int k, int n, int* result, int* src) {
    gsl_ran_sample(RANDOM_NUMBER, (void *) result,  k, (void *) src, n, sizeof(int));
}

double digamma(double x) {
    return gsl_sf_psi(x);
}

unsigned int rmultinomial(const gsl_vector* v) {
    size_t i;
    double sum = vsum(v);
    double u = runiform() * sum;
    double cum_sum = 0.0;
    for (i = 0; i < v->size; i ++) {
        cum_sum += vget(v, i);
        if (u < cum_sum) break;
    }
    return i;
}

double rgamma(double a, double b) {
  return gsl_ran_gamma_mt(RANDOM_NUMBER, a, b);
}

double rbeta(double a, double b) {
  return gsl_ran_beta(RANDOM_NUMBER, a, b);
}

unsigned int rbernoulli(double p) {
  return gsl_ran_bernoulli(RANDOM_NUMBER, p);
}

double runiform() {
  return gsl_rng_uniform_pos(RANDOM_NUMBER);
}

double runiform_ex(unsigned long int n) {
  return gsl_rng_uniform_int(RANDOM_NUMBER, n);
}

void rshuffle(void* base, size_t n, size_t size) {
  gsl_ran_shuffle(RANDOM_NUMBER, base, n, size);
}

unsigned long int runiform_int(unsigned long int n) {
  return  gsl_rng_uniform_int(RANDOM_NUMBER, n);
}

double my_trigamma(double x) {
    double p;
    int i;

    x=x+6;
    p=1/(x*x);
    p=(((((0.075757575757576*p-0.033333333333333)*p+0.0238095238095238)
         *p-0.033333333333333)*p+0.166666666666667)*p+1)/x+0.5*p;
    for (i=0; i<6 ;i++) {
        x=x-1;
        p=1/(x*x)+p;
    }
    return(p);
}


/*
 * taylor approximation of first derivative of the log gamma function - ψ(x)
 *
 */
double my_digamma(double x)
{
    double p;
    x=x+6;
    p=1/(x*x);
    p=(((0.004166666666667*p-0.003968253986254)*p+
	0.008333333333333)*p-0.083333333333333)*p;
    p=p+log(x)-0.5/x-1/(x-1)-1/(x-2)-1/(x-3)-1/(x-4)-1/(x-5)-1/(x-6);
    return p;
}

std::vector<int> FillUpIntSet(int begin, int end) {
    std::vector<int> v;
    for (int i = begin; i < end; ++i) 
        v.push_back(i);
    return v;
}

// random generator function:
int myrandom (int i) { 
  return std::rand()%i;
}

void GenerateRandomIntSet(std::vector<int>* v) {
    std::random_shuffle( v->begin(), v->end(), myrandom);
    return;
}

LONG get_long(double s){
    LONG l;
    LONG *ll = (LONG*)(&s);
    l = *ll;
    return l;
}

void mtx_fwrite(FILE* file, gsl_matrix* m) {
    const int size = sizeof(LONG);
    int end = -1;

    size_t size_1 = (size_t)htonl(m->size1);
    size_t size_2 = (size_t)htonl(m->size2);

    fwrite(&size_1, sizeof(size_t), 1, file);
    fwrite(&size_2, sizeof(size_t), 1, file);

    for (size_t i = 0; i < m->size1; ++i) {
        for (size_t j = 0; j < m->size2; ++j) {
            LONG l = get_long(mget(m, i, j));
            fwrite( &l, size, 1, file);
        }
    }

    fwrite(&end, sizeof(int), 1, file);
    return;
}

void read_data_through_pipe(FILE* file) {
    int val;
    fread(&val, sizeof(int), 1, file);
    printf("val = %d\n", val);
}
