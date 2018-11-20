// class for ldpa
//

#include "utils.h"
#include "corpus.h"
#include "data.h"

struct ldap_hyperparameter {
    double e;
    double f;
    double ro;
    double alpha;
    int    random_seed;
    int    max_iter;
    int    save_lag;
  
    void set(double _e, double _f, double _ro, double _alpha, int rs, int mi, int sl) {
        e = _e; f = _f; ro = _ro; alpha = _alpha;
        random_seed = rs; max_iter = mi;
        save_lag = sl;
    }

    void save(char* filename) {
        FILE * file = fopen(filename, "w");
        fprintf(file, "e = %.4f\n", e);
        fprintf(file, "f = %.4f\n", f);
        fprintf(file, "ro = %.4f\n", ro);
        fprintf(file, "alpha = %.4f\n", alpha);
        fprintf(file, "random seed = %d\n", (int)random_seed);
        fprintf(file, "max iter = %d\n", max_iter);
        fprintf(file, "save lag = %d\n", save_lag);
        fclose(file);
    }
};

class c_ldap {
public:
    c_ldap();
    ~c_ldap();
    void read_init_information(const char* theta_init_path, 
                             const char* beta_init_path, 
                             const c_corpus* c);
    void set_model_parameters(int num_factors, 
                            int num_users, 
                            int num_items);

    void learn_map_estimate(const c_data* users, const c_data* items,  
                          const c_corpus* c, const ldap_hyperparameter* param,
                          const char* directory);

    void init_model(const c_data* users, const c_data* items, ldap_hyperparameter ldap_param);

    double doc_inference(const c_document* doc, const gsl_vector* theta_v, 
                       const gsl_matrix* log_beta, gsl_matrix* phi, 
                       gsl_vector* gamma, gsl_matrix* word_ss, 
                       bool update_word_ss);
  
    void det_f2(gsl_vector* v_f2, const ldap_hyperparameter* param, int d);
  
    void det_f1(gsl_vector* v_f1, gsl_vector* v_temp, c_document* doc, int d, double alpha);
  
    void ope_for_theta(c_document* doc, const ldap_hyperparameter* param, int d);

	void update_theta(const c_data* items,const c_corpus* c, const ldap_hyperparameter* param);



    void update_beta(vector<c_document*> m_docs);
  
    void get_eta();
  
    double sum_f1(const c_data* items, const c_corpus* c);
  
    double update_r(gsl_matrix* U, gsl_matrix* V);
  
    void update_variational_param(gsl_matrix* eta_shp_temp, gsl_matrix* phi, gsl_matrix* sum_phi, gsl_vector* v_sum,
									              const c_data* users, const ldap_hyperparameter* param);
  
    void update_muy(gsl_matrix* sum_phi, gsl_matrix* delta, gsl_vector* v_eta_sum, const c_data* items, const ldap_hyperparameter* param);
  
  
public:
    gsl_matrix* m_beta;
    gsl_matrix* m_theta;
    gsl_matrix* m_muy;

    gsl_matrix* m_eta;
    gsl_matrix* m_eta_rte;
    gsl_matrix* m_eta_shp;
  

    int m_num_factors; // m_num_topics
    int m_num_items; // m_num_docs
    int m_num_users; // num of users
};
