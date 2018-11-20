#include "ldap.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <arpa/inet.h>
#include "ThreadPool.h"

extern gsl_rng * RANDOM_NUMBER;
int min_iter = 10;
double r = 1.0;
double CONVERGE = 1e-4; 
int NUMBER_ITERATE_UPDATE_VARIATION_PARAM = 1;
int NUMBER_ITERATE_OPE = 20;
int FACTOR_MUY = 1;


c_ldap::c_ldap() {
  	m_beta = NULL;
  	m_theta = NULL;
  	m_eta = NULL;
  	m_muy = NULL;
	
  	m_num_factors = 0; // m_num_topics
  	m_num_items = 0; // m_num_docs
  	m_num_users = 0; // num of users
}

c_ldap::~c_ldap() {
  	if (m_beta != NULL) gsl_matrix_free(m_beta);
  	if (m_theta != NULL) gsl_matrix_free(m_theta);
	if (m_muy != NULL) gsl_matrix_free(m_muy);
  	
	if (m_eta != NULL) gsl_matrix_free(m_eta);
	if (m_eta_shp != NULL) gsl_matrix_free(m_eta_shp);
	if (m_eta_rte != NULL) gsl_matrix_free(m_eta_rte);
}


void c_ldap::read_init_information(const char* theta_init_path, 
                                  const char* beta_init_path,
                                  const c_corpus* c) {
  	int num_topics = m_num_factors;
  	m_theta = gsl_matrix_alloc(c->m_num_docs, num_topics);
  	printf("\nreading theta initialization from %s\n", theta_init_path);
  	FILE * f = fopen(theta_init_path, "r");
  	mtx_fscanf(f, m_theta);
  	fclose(f);
	
  	for (size_t j = 0; j < m_theta->size1; j ++) {
    	gsl_vector_view theta_v = gsl_matrix_row(m_theta, j);
    	vnormalize(&theta_v.vector);
  	}

  	m_beta = gsl_matrix_alloc(num_topics, c->m_size_vocab);
  	printf("reading beta initialization from %s\n", beta_init_path);
  	f = fopen(beta_init_path, "r");
  	mtx_fscanf(f, m_beta);
  	fclose(f);
	
	if (mget(m_beta, 0, 0) < 0) {
    	mtx_exp(m_beta); // lay mu loai bo log do m_beta dang o ko gian log
  	}
  	else {
    	for (size_t j = 0; j < m_beta->size1; j ++) {
      		gsl_vector_view beta_v = gsl_matrix_row(m_beta, j);
      		vnormalize(&beta_v.vector); // chuan hoa theo k
    	}
  	}
}

void c_ldap::set_model_parameters(int num_factors, int num_users, int num_items) {
  	m_num_factors = num_factors;
  	m_num_users = num_users;
  	m_num_items = num_items;
}


void c_ldap::init_model(const c_data* users, const c_data* items, ldap_hyperparameter ldap_param) {
  	m_eta = gsl_matrix_calloc(m_num_users, m_num_factors); // eta[U][K]
	m_eta_shp = gsl_matrix_calloc(m_num_users, m_num_factors);
	m_eta_rte = gsl_matrix_calloc(m_num_users, m_num_factors);
	
  	m_muy = gsl_matrix_calloc(m_num_items, m_num_factors); // epsilon[D][K]
	gsl_matrix_memcpy(m_muy, m_theta);

  	// init value for VB
	// init for Gamma's variable	
	for (size_t i = 0; i < m_eta_shp->size1; i ++) {
      	for (size_t j = 0; j < m_eta_shp->size2; j ++) {	
			if (users->m_vec_len[i] == 0) {
				mset(m_eta_shp, i, j, 0);
			}
			mset(m_eta_shp, i, j, ldap_param.e + 0.01 * runiform());
	  	}	
	}
}


void c_ldap::learn_map_estimate(const c_data* users, const c_data* items, 
                    const c_corpus* c, const ldap_hyperparameter* param,
                    const char* directory) {
	// init model parameters
  	printf("\ninitializing the model ...\n");
  	init_model(users, items, *param);
	
	printf("begin learning use MAP estimate ...\n");
	
	// filename
  	char name[500];

  	// start time
  	time_t start, current;
  	time(&start);
  	int elapsed = 0;

  	int iter = 0;
  	double likelihood = -exp(50), likelihood_old;
  	double converge = 1.0;

  	/// create the state log file 
  	sprintf(name, "%s/state.log", directory);
  	FILE* file = fopen(name, "w");
  	fprintf(file, "iter time likelihood converge\n");
	
	int m;
	
	gsl_matrix* phi = gsl_matrix_calloc(m_num_items, m_num_factors);
	gsl_matrix* sum_phi = gsl_matrix_calloc(m_num_items, m_num_factors);
	gsl_matrix* eta_shp_temp = gsl_matrix_calloc(m_num_users, m_num_factors);
	gsl_matrix* delta = gsl_matrix_calloc(m_num_items, m_num_factors);
	gsl_vector* v_eta_sum = gsl_vector_calloc(m_num_factors);
	gsl_vector* v_sum = gsl_vector_calloc(m_num_factors);

	
	while ((iter < param->max_iter and converge > CONVERGE)) {		
		update_variational_param(eta_shp_temp, phi, sum_phi, v_sum, users, param);		
		get_eta();
		update_muy(sum_phi, delta, v_eta_sum, items, param);
		
		update_theta(items, c, param);
		update_beta(c->m_docs);

		iter++;	
		time(&current);
    	elapsed = (int)difftime(current, start);

    	fprintf(file, "%04d %06d\n", iter, elapsed);
    	fflush(file);
    	printf("iter=%04d, time=%06d\n", iter, elapsed);
		
		// save intermediate results
    	if (iter % param->save_lag == 0) {
			get_eta();
      		sprintf(name, "%s/%04d-eta.dat", directory, iter);
      		FILE * file_eta = fopen(name, "w");
      		mtx_fprintf(file_eta, m_eta);
      		fclose(file_eta);
			
			sprintf(name, "%s/%04d-muy.dat", directory, iter);
      		FILE * file_muy = fopen(name, "w");
      		mtx_fprintf(file_eta, m_muy);
      		fclose(file_muy);

        	sprintf(name, "%s/%04d-theta.dat", directory, iter);
        	FILE * file_theta = fopen(name, "w");
        	mtx_fprintf(file_theta, m_theta);
        	fclose(file_theta);

        	sprintf(name, "%s/%04d-beta.dat", directory, iter);
        	FILE * file_beta = fopen(name, "w");
        	mtx_fprintf(file_beta, m_beta);
        	fclose(file_beta);
      	}    
	}
	
	// save final results
	get_eta();

  	sprintf(name, "%s/final-eta.dat", directory);
  	FILE * file_eta = fopen(name, "w");
  	mtx_fprintf(file_eta, m_eta);
  	fclose(file_eta);
	
	sprintf(name, "%s/final-muy.dat", directory);
  	FILE * file_muy = fopen(name, "w");
  	mtx_fprintf(file_muy, m_muy);
  	fclose(file_muy);

    sprintf(name, "%s/final-theta.dat", directory);
    FILE * file_theta = fopen(name, "w");
    mtx_fprintf(file_theta, m_theta);
    fclose(file_theta);

    sprintf(name, "%s/final-beta.dat", directory);
    FILE * file_beta = fopen(name, "w");
    mtx_fprintf(file_beta, m_beta);
    fclose(file_beta);

	
	// free memory
	gsl_matrix_free(phi);
	gsl_matrix_free(sum_phi);
	gsl_matrix_free(eta_shp_temp);
	gsl_matrix_free(delta);
	gsl_vector_free(v_eta_sum);
	gsl_vector_free(v_sum);
}


void c_ldap::update_variational_param(gsl_matrix* eta_shp_temp, gsl_matrix* phi, gsl_matrix* sum_phi, gsl_vector* v_sum,
			const c_data* users, const ldap_hyperparameter* param) {
	double iter = 0;
	int n;
	double a;
	gsl_vector_view v_phi, v_sum_phi;
	gsl_vector_view v_eta_shp, v_eta_rte;
	int* item_ids = NULL;
	
	// update eta_rte
	col_sum(m_muy, v_sum);

	gsl_matrix_set_all(m_eta_rte, param->f);
	for (int u = 0; u < m_num_users; u++) {
		v_eta_rte = gsl_matrix_row(m_eta_rte, u);
		gsl_vector_add(&v_eta_rte.vector, v_sum);
	}
	
	while(iter < NUMBER_ITERATE_UPDATE_VARIATION_PARAM) {		
		gsl_matrix_set_zero(sum_phi);				
		gsl_matrix_set_all(eta_shp_temp, param->e);		
		
		for (int u = 0; u < m_num_users; u++) {		
			item_ids = users->m_vec_data[u];
			n = users->m_vec_len[u]; // n is number of docs user interested 
			if (n > 0) {
				v_eta_shp = gsl_matrix_row(eta_shp_temp, u);
				for (int d = 0; d < n; d++) {
					v_phi = gsl_matrix_row(phi, item_ids[d]);
					gsl_vector_set_zero(&v_phi.vector);
					v_sum_phi = gsl_matrix_row(sum_phi, item_ids[d]);
					for (int k = 0; k < m_num_factors; k++) {	
						a = exp(safe_log(mget(m_muy, item_ids[d], k)) + digamma(mget(m_eta_shp, u, k)) - safe_log(mget(m_eta_rte, u, k)));
						vset(&v_phi.vector, k, a);
					}		
					vnormalize(&v_phi.vector);
					gsl_vector_add(&v_sum_phi.vector, &v_phi.vector);
					
					// update eta shape					
					gsl_vector_add(&v_eta_shp.vector, &v_phi.vector);
				}
								
			}
		}
		// update eta shape
		gsl_matrix_memcpy(m_eta_shp, eta_shp_temp);
		iter++;
	}
    return;
}

void c_ldap::update_muy(gsl_matrix* sum_phi, gsl_matrix* delta, gsl_vector* v_eta_sum, 
        const c_data* items, const ldap_hyperparameter* param) {
	gsl_vector_view v_delta, v_muy, v_theta;
	double temp;
	int m = 0;
	
	// get delta
	col_sum(m_eta, v_eta_sum);
	
	gsl_matrix_memcpy(delta, m_theta);
	temp = 2 * param->ro;
	gsl_matrix_scale(delta, temp);
	
	for (int d = 0; d < m_num_items; d++) {
		v_delta = gsl_matrix_row(delta, d);
		gsl_vector_sub(&v_delta.vector, v_eta_sum);
	}
	
	gsl_matrix_memcpy(m_muy, delta);
	
	gsl_matrix_mul_elements(delta, delta);
	temp = 8 * param->ro;
	gsl_matrix_scale(sum_phi, temp);
	gsl_matrix_add(delta, sum_phi);
	
	// update muy
	mtx_sqrt(delta);
	gsl_matrix_add(m_muy, delta);
	temp = 0.25 / param->ro;
	gsl_matrix_scale(m_muy, temp);
	
	for (int d = 0; d < m_num_items; d++) {
		m = items->m_vec_len[d];
		if (m == 0) {
			v_theta = gsl_matrix_row(m_theta, d);
			v_muy   = gsl_matrix_row(m_muy, d);
			gsl_vector_memcpy(&v_muy.vector, &v_theta.vector);
			gsl_vector_scale(&v_muy.vector, FACTOR_MUY);
		}
	}

    return;
}

void c_ldap::update_theta(const c_data* items, const c_corpus* c, const ldap_hyperparameter* param) {
	
	ThreadPool pool(4);
	for (int d = 0; d < m_num_items; d++) {
		pool.enqueue(
			[&,d](){
				if (items->m_vec_len[d] > 0) {	
					ope_for_theta(c->m_docs[d], param, d);
				}		
			}
		);						
	}

}

void c_ldap::ope_for_theta(c_document* doc, const ldap_hyperparameter* param, int d) {
	
	gsl_vector* v_temp = gsl_vector_calloc(m_num_factors);
	gsl_vector* v_f1 = gsl_vector_calloc(m_num_factors);
	gsl_vector* v_f2 = gsl_vector_calloc(m_num_factors);


	int n_f1 = 0, n_f2 = 0;
  	double alpha = 0;
	double max, temp;

  	gsl_vector_view view_theta = gsl_matrix_row(m_theta, d);

  	n_f1 = 0; n_f2 = 0;
  	for (int iter = 1; iter <= NUMBER_ITERATE_OPE; iter++) {
    	if (rand() % 2 == 0) n_f1++;
    	else n_f2++;
		det_f1(v_f1, v_temp, doc, d, param->alpha);
		det_f2(v_f2, param, d);
		
		gsl_vector_scale(v_f1, n_f1);
		gsl_vector_scale(v_f2, n_f2);

		gsl_vector_sub(v_f1, v_f2);
    	alpha = 2.0f/((double)iter + 2.0f);

    	max = gsl_vector_max_index(v_f1);
    	gsl_vector_scale(&view_theta.vector, 1 - alpha);
    	vset(&view_theta.vector, max, vget(&view_theta.vector, max) + alpha);
  	}
  	
  	vnormalize(&view_theta.vector);	
	
	gsl_vector_free(v_temp);
	gsl_vector_free(v_f1);
	gsl_vector_free(v_f2);

    return;
}

void c_ldap::det_f1(gsl_vector* v_f1, gsl_vector* v_temp, c_document* doc, int d, double alpha) {
	double denominator;
	gsl_vector_set_zero(v_f1);
	gsl_vector_view view_beta, view_theta; 

	for (int i = 0; i < doc->m_length; ++i) {
		view_beta = gsl_matrix_column(m_beta, doc->m_words[i]);
		view_theta = gsl_matrix_row(m_theta, d);
		gsl_blas_ddot(&view_theta.vector, &view_beta.vector, &denominator);
		gsl_vector_memcpy(v_temp, &view_beta.vector);
		double temp = (double)doc->m_counts[i]/denominator;
		gsl_vector_scale(v_temp, temp);
		gsl_vector_add(v_f1, v_temp);
	}
	view_theta = gsl_matrix_row(m_theta, d);
	if (alpha != 1.0) {
		for (int k = 0; k < m_num_factors; ++k) {
			vset(v_temp, k, ((alpha-1.0)/vget(&view_theta.vector, k)));
		}
		gsl_vector_add(v_f1, v_temp);
	}
    return;
}

void c_ldap::det_f2(gsl_vector* v_f2, const ldap_hyperparameter* param, int d) {
	double temp = 2 * param->ro;

	gsl_vector_view view_theta = gsl_matrix_row(m_theta, d);
	gsl_vector_view view_muy   = gsl_matrix_row(m_muy, d);
	gsl_vector_memcpy(v_f2, &view_theta.vector);
	gsl_vector_sub(v_f2, &view_muy.vector);
	gsl_vector_scale(v_f2, temp);
    return;
}



void c_ldap::update_beta(vector<c_document*> m_docs) {
  	c_document* doc;
  	double temp = 0, temp2 = 0;
  	int count = 0;

  	gsl_matrix_set_zero(m_beta);
       
  	for (int k=0; k < m_num_factors; k++) {        
    	for (int d = 0; d < m_num_items; d++) {
      		temp2 = mget(m_theta, d, k);
      		doc = m_docs.at(d);          

      		for (int j = 0; j < doc->m_length; j++) {
        		if (doc->m_words[j] == 0) count++;
        		temp = doc->m_counts[j] * temp2 + mget(m_beta, k, doc->m_words[j]);
        		mset(m_beta, k, doc->m_words[j], temp);
      		}
    	}
    	gsl_vector_view row = gsl_matrix_row(m_beta, k);
    	vnormalize(&row.vector);
  	}
    return;
}

// get eta by expect
void c_ldap::get_eta() {
	gsl_matrix_memcpy(m_eta, m_eta_shp);
	gsl_matrix_div_elements(m_eta, m_eta_rte);
}
