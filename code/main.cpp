#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <ctime>
#include "ldap.h"

void print_usage_and_exit() {
	// print usage information
	printf("*********************************LDA-Poisson models for recommendations*********************************\n");
	printf("Authors: CongSon, congson1293@gmail.com, Infomation System Department, Hanoi University of Science and Technology.\n");
	printf("usage:\n");
	printf("      ldap [options]\n");
	printf("      --help:           print help information\n");

	printf("\n");
	printf("      --directory:      save directory, required\n");

	printf("\n");
	printf("      --user:           user file, required\n");
	printf("      --item:           item file, required\n");
	printf("      --e:              shape parameter of user vector regularizer, default 0.3\n");
	printf("      --f:              rate parameter of user vector regularizer, default 0.3\n");
	printf("      --ro:             variance of gaussian distribution, default 1.0\n");
	printf("      --alpha           hyperparameter of Dirichlet distribution, default 1.0\n");
	printf("      --p       		hyperparameter of Becnoulli distribution in OPE, default 0.5\n");
	printf("      --n_threads       number of threads to run Estep update theta, default 4\n");
	printf("\n");

	printf("      --random_seed:    the random seed, default from the current time\n");
	printf("      --save_lag:       the saving lag, default 20 (-1 means no savings for intermediate results)\n");
	printf("      --max_iter:       the max number of iterations, default 100\n");
	printf("\n");

	printf("      --num_factors:    the number of factors, default 100\n");
	printf("      --mult:           mult file, in lda-c format\n");
	printf("      --theta_init:     topic proportions file from lda\n");
	printf("      --beta_init:      topic distributions file from lda\n");

	printf("*******************************************************************************************************\n");

	exit(0);
}

int main(int argc, char* argv[]) {
	if (argc < 2) print_usage_and_exit();

	const char* const short_options = "h:d:x:i:a:b:u:v:r:s:m:k:t:e:y:w:";
	const struct option long_options[] = {
		{"help",          no_argument,       NULL, '?'},
		{"directory",     required_argument, NULL, 'd'},
		{"user",          required_argument, NULL, 'u'},
		{"item",          required_argument, NULL, 'i'},
		{"e",      		  required_argument, NULL, 'e'},
		{"f",      		  required_argument, NULL, 'f'},
		{"ro",      	  required_argument, NULL, 'x'},
		{"alpha",         required_argument, NULL, 'a'},
		{"p",         	  required_argument, NULL, 'p'},
		{"n_threads",     required_argument, NULL, 'n'},
		{"random_seed",   required_argument, NULL, 'r'},
		{"save_lag",      required_argument, NULL, 's'},
		{"max_iter",      required_argument, NULL, 'm'},
		{"num_factors",   required_argument, NULL, 'k'},
		{"mult",          required_argument, NULL, 'l'},
		{"theta_init",    required_argument, NULL, 't'},
		{"beta_init",     required_argument, NULL, 'b'},
		{NULL, 0, NULL, 0}
	};

	int temp = 0; 
	char filename[500];
	char*  directory = NULL;
	char*  user_path = NULL;
	char*  item_path = NULL;
	float e=0.3, f=0.3, ro=1.0, alpha=1.0;
	float p=0.5;
	int n_threads = 4;
	time_t t; time(&t);
	long   random_seed = (long) t;
	int    save_lag = 20;
	int    max_iter = 100;
	int    num_factors = 100;
	char*  mult_path = NULL;
	char*  theta_init_path = NULL;
	char*  beta_init_path = NULL;

	while(true) {
		temp = getopt_long(argc, argv, short_options, long_options, NULL);
		switch(temp) {
			case 'd':
				directory = optarg;
				break;
			case 'u':
				user_path = optarg;
				break;
			case 'i':
				item_path = optarg;
				break;
			case 'e':
				e = atof(optarg);
				break;
			case 'f':
				f = atof(optarg);
				break;
			case 'x':
				ro = atof(optarg);
				break;
			case 'a':
				alpha = atof(optarg);
				break;
			case 'p':
				p = atof(optarg);
				break;
			case 'n':
				n_threads = atof(optarg);
				break;
			case 'r':
				random_seed = atoi(optarg);
				break;
			case 's':
				save_lag = atoi(optarg);
				break;
			case 'm':
				max_iter =  atoi(optarg);
				break;    
			case 'k':
				num_factors = atoi(optarg);
				break;
			case 'l':
				mult_path = optarg;
				break;
			case 't':
				theta_init_path = optarg;
				break;
			case 'b':
				beta_init_path = optarg;
				break;
			case -1:
				break;
			case '?':
				print_usage_and_exit();
				break;
			default:
			break;
		}
		if (temp == -1)
			break;
	}
	
	if (!dir_exists(directory)) make_directory(directory);
	printf("result directory: %s\n", directory);

	if (!file_exists(user_path)) {
		printf("Missing user file %s doesn't exist! quit ...\n", user_path);
		exit(-1);
	}
	printf("user file: %s\n", user_path);

	if (!file_exists(item_path)) {
		printf("Missing item file %s doesn't exist! quit ...\n", item_path);
		exit(-1);
	}
	printf("item file: %s\n", item_path);
	

	printf("e: %.4f\n", e);
	printf("f: %.4f\n", f);
	printf("ro: %.4f\n", ro);
	printf("alpha: %.4f\n", alpha);
	printf("p: %.4f\n", p);
	printf("n_threads: %d\n", n_threads);
	printf("random seed: %d\n", (int)random_seed);
	printf("save lag: %d\n", save_lag);
	printf("max iter: %d\n", max_iter);
	printf("number of factors: %d\n", num_factors);

	if (!file_exists(item_path)) {
		printf("Missing mult file %s doesn't exist! quit ...\n", mult_path);
		exit(-1);
	}
	printf("mult file: %s\n", mult_path);
	
	if (theta_init_path == NULL) {
		printf("Missing topic proportions file must be provided ...\n");
		exit(-1);
	}
	if (!file_exists(theta_init_path)) {
		printf("Missing topic proportions file %s doesn't exist! quit ...\n", theta_init_path);
		exit(-1);
	}
	printf("topic proportions file: %s\n", theta_init_path);

	if (beta_init_path == NULL) {
		printf("Missing topic distributions file must be provided ...\n");
		exit(-1);
	}
	if (!file_exists(beta_init_path)) {
		printf("topic distributions file %s doesn't exist! quit ...\n", beta_init_path);
		exit(-1);
	}
	printf("topic distributions file: %s\n", beta_init_path);

	printf("\n");
	// save the settings
	ldap_hyperparameter ldap_param;
	ldap_param.set(e, f, ro, alpha, p, n_threads, random_seed, max_iter, save_lag);
	sprintf(filename, "%s/settings.txt", directory); 
	ldap_param.save(filename);

	srand(random_seed);

	printf("reading user matrix from %s ...\n", user_path);
	c_data* users = new c_data(); 
	users->read_data(user_path);
	int num_users = (int)users->m_vec_data.size();

	// read items
	printf("reading item matrix from %s ...\n", item_path);
	c_data* items = new c_data(); 
	items->read_data(item_path);
	int num_items = (int)items->m_vec_data.size();
	// create model instance
	c_ldap *ldap = new c_ldap();
	ldap->set_model_parameters(num_factors, num_users, num_items);

	c_corpus* c = NULL;
	// read word data
	c = new c_corpus();
	c->read_data(mult_path);
	ldap->read_init_information(theta_init_path, beta_init_path, c);

	ldap->learn_map_estimate(users, items, c, &ldap_param, directory);
	
	delete c;
	delete ldap;
	delete users;
	delete items;
	
	return 0;
}
