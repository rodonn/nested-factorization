#ifndef PEMB_INFERENCE_HPP
#define PEMB_INFERENCE_HPP

/*
__global__ void lookAhead_kern(double *d_eta_lookahead, double *d_prod_rho_alpha, 
							   double *d_eta_base, int i,
							   int *d_elem_context, int context_size, int Nitems) {
    int k = blockIdx.x*blockDim.x + threadIdx.x;

    // Each thread computes one element
    if(k<Nitems) {
	    bool found = false;
	    int j;
    	// Check if target element
    	if(k==i) {
    		d_eta_lookahead[k] = -myINFINITY;
    		found = true;
		}
		// Check if element in context
		if(!found) {
			for(int idx_j=0; idx_j<context_size; idx_j++) {
				j = d_elem_context[idx_j];
				if(k==j) {
					found = true;
    				d_eta_lookahead[k] = -myINFINITY;
				}
			}
		}
		// Compute eta_target
		if(!found) {
		    double aux = d_eta_base[k];
			for(int idx_j=0; idx_j<context_size; idx_j++) {
				j = d_elem_context[idx_j];
				aux += d_prod_rho_alpha[k*Nitems+j]/(context_size+1.0);
			}
			aux += d_prod_rho_alpha[k*Nitems+i]/(context_size+1.0);
			d_eta_lookahead[k] = aux;
		}
    }
}
*/

class my_infer {
public:

	static void inference_step(my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar, gsl_rng *semilla) {
		double logp;

		// 0. Subsample transactions at random
		int batchsize = my_min(param.batchsize,data.Ntrans);
		if(batchsize<=0) {
			batchsize = data.Ntrans;
		}
		std::vector<int> transaction_all(data.Ntrans);
		for(int t=0; t<data.Ntrans; t++) {
			transaction_all.at(t) = t;
		}
		std::vector<int> transaction_list(batchsize);
		gsl_ran_choose(semilla,transaction_list.data(),batchsize,transaction_all.data(),data.Ntrans,sizeof(int));

		// 0b. Sample from everything
		sample_all(semilla,data,hyper,param,pvar);

		// 1a. Compute the sum of the alpha's
		/*
		if(param.flag_shuffle==-1) {
			std::cout << "  computing sum of alphas..." << endl;
			compute_sum_alpha(data,hyper,param,pvar,transaction_list);
		}
		*/

		// 1b. Compute products rho*alpha, theta*alpha, etc.
		compute_prod_all(data,hyper,param,pvar,transaction_list);

		// 2. Initialize all gradients to prior
		std::cout << "  initializing gradients..." << endl;
		logp = set_grad_to_prior(data,hyper,param,pvar);

		// 3. Increase the gradients
		std::cout << "  increasing gradients..." << endl;
		logp += increase_gradients(data,hyper,param,pvar,transaction_list,semilla);

		// 3b. Compute logq
		double logq = compute_logq(data,hyper,param,pvar);

		// 4. Take gradient step
		std::cout << "  taking grad step..." << endl;
		take_grad_step(logp,data,hyper,param,pvar);

		// 5. Output the objective function to a file
		my_output::write_objective_function(param,logp,logq);
	}

	static void compute_prod_all(my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar, std::vector<int> &transaction_list) {
		// First, create active_users vector
		std::set<int> active_users_set;
		for(int &t : transaction_list) {
			int u = data.user_per_trans.get_object(t);
			active_users_set.insert(u);
		}
		std::vector<int> active_users(active_users_set.begin(), active_users_set.end());
		
		// Compute rho*alpha
		/*
		if(param.K>0) {
			compute_prod_one(pvar.rho, pvar.alpha, pvar.prod_rho_alpha, pvar.d_prod_rho_alpha);
		} else {
			set_to_zero(pvar.prod_rho_alpha);
			set_to_zero_device(pvar.d_prod_rho_alpha, data.Nitems*data.Nitems);
		}
		*/

		// Compute obsItems
		if(param.IC>0 && param.flag_obs2utility) {
			compute_prod_one(pvar.obsItems, data.attr_items, pvar.prod_obsItems, active_users);
		}

		// Compute obsUsers
		if(param.UC>0) {
			compute_prod_one(data.attr_users, pvar.obsUsers, pvar.prod_obsUsers, active_users);
		}

		// Compute theta*alpha
		if(param.flag_userVec==1) {
			/* compute_prod_one(pvar.theta, pvar.rho, pvar.prod_theta_alpha); */
		} else if(param.flag_userVec==3) {
			compute_prod_one(pvar.theta, pvar.alpha, pvar.prod_theta_alpha, active_users);
		} else if(param.flag_userVec!=0) {
			std::cerr << "[ERR] Not implemented function for userVec=" << std::to_string(param.flag_userVec) << endl;
			assert(0);
		}

		// Compute gamma*beta
		if(param.flag_price>0 && !param.noItemPriceLatents) {
			compute_prod_one(pvar.gamma, pvar.beta, pvar.prod_gamma_beta, active_users);
		}

		// Compute delta*mu
		if(param.flag_day>0) {
			compute_prod_one(pvar.delta, pvar.mu, pvar.prod_delta_mu);
		}
	}

	/*
	static void compute_prod_one(Matrix2D<var_pointmass> &M1, Matrix2D<var_pointmass> &M2, Matrix2D<double> &prod_result, 
								 double *d_ptr) {
		// Get the sizes
		int N = M1.get_size1();
		int M = M2.get_size1();
		int K = M1.get_size2();

		// Allocate memory (host)
		double *h_M1 = new double[N*K];
		double *h_M2 = new double[M*K];
		double *h_R = new double[N*M];

		// Allocate memory (device)
		double *d_M1;
		double *d_M2;
		double *d_R;
		d_allocate(&d_M1, N*K);
		d_allocate(&d_M2, M*K);
		if(d_ptr==nullptr) {
			d_allocate(&d_R, M*N);
		} else {
			d_R = d_ptr;
		}

		// Copy matrices to host pointers
		copy_matrix_to_host_pointer(M1, h_M1);
		copy_matrix_to_host_pointer(M2, h_M2);

		// Copy to device
		copy_h2d(d_M1, h_M1, N*K);
		copy_h2d(d_M2, h_M2, M*K);
		copy_h2d(d_R, h_R, M*N);

		// Launch kernel to compute matrix product
		dim3 n_blocks(ceil(N/2.0),ceil(M/256.0),1);
		dim3 n_threads_per_block(2,256,1);
		matMulKern_TransB<<<n_blocks, n_threads_per_block>>>(d_M1, d_M2, d_R, N, M, K);
		d_sync();

		// Copy to host
		copy_d2h(h_R, d_R, M*N);
		if(h_R[0]==0.0 && h_R[1]==0.0 && h_R[N*M-1]==0) {
			std::cerr << "[ERR] The cuda matmul function was not executed properly" << endl;
			assert(0);
		}

		// Copy host pointer to matrix
		copy_host_pointer_to_matrix(prod_result, h_R);

		// Free memory
		cudaFree(d_M1);
		cudaFree(d_M2);
		if(d_ptr==nullptr) {
			cudaFree(d_R);
		}
		delete [] h_M1;
		delete [] h_M2;
		delete [] h_R;
	}
	*/

	/*
	static void compute_prod_one(Matrix2D<var_gaussian> &M1, Matrix2D<var_gaussian> &M2, Matrix2D<double> &prod_result, 
								 double *d_ptr) {
		// Get the sizes
		int N = M1.get_size1();
		int M = M2.get_size1();
		int K = M1.get_size2();

		// Allocate memory (host)
		double *h_M1 = new double[N*K];
		double *h_M2 = new double[M*K];
		double *h_R = new double[N*M];

		// Allocate memory (device)
		double *d_M1;
		double *d_M2;
		double *d_R;
		d_allocate(&d_M1, N*K);
		d_allocate(&d_M2, M*K);
		if(d_ptr==nullptr) {
			d_allocate(&d_R, M*N);
		} else {
			d_R = d_ptr;
		}

		// Copy matrices to host pointers
		copy_matrix_to_host_pointer(M1, h_M1);
		copy_matrix_to_host_pointer(M2, h_M2);

		// Copy to device
		copy_h2d(d_M1, h_M1, N*K);
		copy_h2d(d_M2, h_M2, M*K);
		copy_h2d(d_R, h_R, M*N);

		// Launch kernel to compute matrix product
		dim3 n_blocks(ceil(N/2.0),ceil(M/256.0),1);
		dim3 n_threads_per_block(2,256,1);
		matMulKern_TransB<<<n_blocks, n_threads_per_block>>>(d_M1, d_M2, d_R, N, M, K);
		d_sync();

		// Copy to host
		copy_d2h(h_R, d_R, M*N);
		if(h_R[0]==0.0 && h_R[1]==0.0 && h_R[N*M-1]==0) {
			std::cerr << "[ERR] The cuda matmul function was not executed properly" << endl;
			assert(0);
		}

		// Copy host pointer to matrix
		copy_host_pointer_to_matrix(prod_result, h_R);

		// Free memory
		cudaFree(d_M1);
		cudaFree(d_M2);
		if(d_ptr==nullptr) {
			cudaFree(d_R);
		}
		delete [] h_M1;
		delete [] h_M2;
		delete [] h_R;
	}
	*/

	/*
	static void compute_prod_one(Matrix2D<var_gamma> &M1, Matrix2D<var_gamma> &M2, Matrix2D<double> &prod_result, 
								 double *d_ptr) {
		// Get the sizes
		int N = M1.get_size1();
		int M = M2.get_size1();
		int K = M1.get_size2();

		// Allocate memory (host)
		double *h_M1 = new double[N*K];
		double *h_M2 = new double[M*K];
		double *h_R = new double[N*M];

		// Allocate memory (device)
		double *d_M1;
		double *d_M2;
		double *d_R;
		d_allocate(&d_M1, N*K);
		d_allocate(&d_M2, M*K);
		if(d_ptr==nullptr) {
			d_allocate(&d_R, M*N);
		} else {
			d_R = d_ptr;
		}

		// Copy matrices to host pointers
		copy_matrix_to_host_pointer(M1, h_M1);
		copy_matrix_to_host_pointer(M2, h_M2);

		// Copy to device
		copy_h2d(d_M1, h_M1, N*K);
		copy_h2d(d_M2, h_M2, M*K);
		copy_h2d(d_R, h_R, M*N);

		// Launch kernel to compute matrix product
		dim3 n_blocks(ceil(N/2.0),ceil(M/256.0),1);
		dim3 n_threads_per_block(2,256,1);
		matMulKern_TransB<<<n_blocks, n_threads_per_block>>>(d_M1, d_M2, d_R, N, M, K);
		d_sync();

		// Copy to host
		copy_d2h(h_R, d_R, M*N);
		if(h_R[0]==0.0 && h_R[1]==0.0 && h_R[N*M-1]==0) {
			std::cerr << "[ERR] The cuda matmul function was not executed properly" << endl;
			assert(0);
		}

		// Copy host pointer to matrix
		copy_host_pointer_to_matrix(prod_result, h_R);

		// Free memory
		cudaFree(d_M1);
		cudaFree(d_M2);
		if(d_ptr==nullptr) {
			cudaFree(d_R);
		}
		delete [] h_M1;
		delete [] h_M2;
		delete [] h_R;
	}
	*/

	/*
	static void compute_prod_one(Matrix2D<var_gaussian> &M1, Matrix2D<double> &M2, Matrix2D<double> &prod_result, 
								 double *d_ptr) {
		// Get the sizes
		int N = M1.get_size1();
		int M = M2.get_size1();
		int K = M1.get_size2();

		// Allocate memory (host)
		double *h_M1 = new double[N*K];
		double *h_M2 = new double[M*K];
		double *h_R = new double[N*M];

		// Allocate memory (device)
		double *d_M1;
		double *d_M2;
		double *d_R;
		d_allocate(&d_M1, N*K);
		d_allocate(&d_M2, M*K);
		if(d_ptr==nullptr) {
			d_allocate(&d_R, M*N);
		} else {
			d_R = d_ptr;
		}

		// Copy matrices to host pointers
		copy_matrix_to_host_pointer(M1, h_M1);
		copy_matrix_to_host_pointer(M2, h_M2);

		// Copy to device
		copy_h2d(d_M1, h_M1, N*K);
		copy_h2d(d_M2, h_M2, M*K);
		copy_h2d(d_R, h_R, M*N);

		// Launch kernel to compute matrix product
		dim3 n_blocks(ceil(N/2.0),ceil(M/256.0),1);
		dim3 n_threads_per_block(2,256,1);
		matMulKern_TransB<<<n_blocks, n_threads_per_block>>>(d_M1, d_M2, d_R, N, M, K);
		d_sync();

		// Copy to host
		copy_d2h(h_R, d_R, M*N);
		if(h_R[0]==0.0 && h_R[1]==0.0 && h_R[N*M-1]==0) {
			std::cerr << "[ERR] The cuda matmul function was not executed properly" << endl;
			assert(0);
		}

		// Copy host pointer to matrix
		copy_host_pointer_to_matrix(prod_result, h_R);

		// Free memory
		cudaFree(d_M1);
		cudaFree(d_M2);
		if(d_ptr==nullptr) {
			cudaFree(d_R);
		}
		delete [] h_M1;
		delete [] h_M2;
		delete [] h_R;
	}
	*/

	/*	
	static void compute_prod_one(Matrix2D<var_pointmass> &M1, Matrix2D<var_pointmass> &M2, Matrix2D<double> &prod_result) {
		compute_prod_one(M1, M2, prod_result, nullptr);
	}

	static void compute_prod_one(Matrix2D<var_gaussian> &M1, Matrix2D<var_gaussian> &M2, Matrix2D<double> &prod_result) {
		compute_prod_one(M1, M2, prod_result, nullptr);
	}

	static void compute_prod_one(Matrix2D<var_gaussian> &M1, Matrix2D<double> &M2, Matrix2D<double> &prod_result) {
		compute_prod_one(M1, M2, prod_result, nullptr);
	}

	static void compute_prod_one(Matrix2D<var_gamma> &M1, Matrix2D<var_gamma> &M2, Matrix2D<double> &prod_result) {
		compute_prod_one(M1, M2, prod_result, nullptr);
	}
	*/

	static void compute_prod_one(Matrix2D<var_pointmass> &M1, Matrix2D<var_pointmass> &M2, Matrix2D<double> &prod_result) {
		int U = M1.get_size1();
		std::vector<int> active_users = std::vector<int>(U);
		for(int u=0; u<U; u++) {
			active_users.at(u) = u;
		}
		compute_prod_one(M1, M2, prod_result, active_users);
	}

	static void compute_prod_one(Matrix2D<var_gaussian> &M1, Matrix2D<var_gaussian> &M2, Matrix2D<double> &prod_result) {
		int U = M1.get_size1();
		std::vector<int> active_users = std::vector<int>(U);
		for(int u=0; u<U; u++) {
			active_users.at(u) = u;
		}
		compute_prod_one(M1, M2, prod_result, active_users);
	}

	static void compute_prod_one(Matrix2D<var_gaussian> &M1, Matrix2D<double> &M2, Matrix2D<double> &prod_result) {
		int U = M1.get_size1();
		std::vector<int> active_users = std::vector<int>(U);
		for(int u=0; u<U; u++) {
			active_users.at(u) = u;
		}
		compute_prod_one(M1, M2, prod_result, active_users);
	}

	static void compute_prod_one(Matrix2D<double> &M1, Matrix2D<var_gaussian> &M2, Matrix2D<double> &prod_result) {
		int U = M1.get_size1();
		std::vector<int> active_users = std::vector<int>(U);
		for(int u=0; u<U; u++) {
			active_users.at(u) = u;
		}
		compute_prod_one(M1, M2, prod_result, active_users);
	}

	static void compute_prod_one(Matrix2D<var_gamma> &M1, Matrix2D<var_gamma> &M2, Matrix2D<double> &prod_result) {
		int U = M1.get_size1();
		std::vector<int> active_users = std::vector<int>(U);
		for(int u=0; u<U; u++) {
			active_users.at(u) = u;
		}
		compute_prod_one(M1, M2, prod_result, active_users);
	}

	static void compute_prod_one(Matrix2D<var_pointmass> &M1, Matrix2D<var_pointmass> &M2, Matrix2D<double> &prod_result, std::vector<int> &active_users) {
		int M = M2.get_size1();
		int K = M1.get_size2();
		double aux;

		for(int &i: active_users) {
			for (int j=0; j<M; j++) {
				aux = 0.0;
				for(int k=0; k<K; k++) {
					aux += M1.get_object(i,k).z * M2.get_object(j,k).z;
				}
				prod_result.set_object(i,j,aux);
			}
		}
	}

	static void compute_prod_one(Matrix2D<var_gaussian> &M1, Matrix2D<var_gaussian> &M2, Matrix2D<double> &prod_result, std::vector<int> &active_users) {
		int M = M2.get_size1();
		int K = M1.get_size2();
		double aux;

		for(int &i: active_users) {
			for (int j=0; j<M; j++) {
				aux = 0.0;
				for(int k=0; k<K; k++) {
					aux += M1.get_object(i,k).z * M2.get_object(j,k).z;
				}
				prod_result.set_object(i,j,aux);
			}
		}
	}

	static void compute_prod_one(Matrix2D<var_gaussian> &M1, Matrix2D<double> &M2, Matrix2D<double> &prod_result, std::vector<int> &active_users) {
		int M = M2.get_size1();
		int K = M1.get_size2();
		double aux;

		for(int &i: active_users) {
			for (int j=0; j<M; j++) {
				aux = 0.0;
				for(int k=0; k<K; k++) {
					aux += M1.get_object(i,k).z * M2.get_object(j,k);
				}
				prod_result.set_object(i,j,aux);
			}
		}
	}

	static void compute_prod_one(Matrix2D<double> &M1, Matrix2D<var_gaussian> &M2, Matrix2D<double> &prod_result, std::vector<int> &active_users) {
		int M = M2.get_size1();
		int K = M1.get_size2();
		double aux;

		for(int &i: active_users) {
			for (int j=0; j<M; j++) {
				aux = 0.0;
				for(int k=0; k<K; k++) {
					aux += M1.get_object(i,k) * M2.get_object(j,k).z;
				}
				prod_result.set_object(i,j,aux);
			}
		}
	}

	static void compute_prod_one(Matrix2D<var_gamma> &M1, Matrix2D<var_gamma> &M2, Matrix2D<double> &prod_result, std::vector<int> &active_users) {
		int M = M2.get_size1();
		int K = M1.get_size2();
		double aux;

		for(int &i: active_users) {
			for (int j=0; j<M; j++) {
				aux = 0.0;
				for(int k=0; k<K; k++) {
					aux += M1.get_object(i,k).z * M2.get_object(j,k).z;
				}
				prod_result.set_object(i,j,aux);
			}
		}
	}

	/*
	static void copy_matrix_to_host_pointer(Matrix2D<var_pointmass> &M, double *h_p) {
		int K = M.get_size2();
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<K; j++) {
				h_p[i*K+j] = M.get_object(i,j).z;
			}
		}
	}

	static void copy_matrix_to_host_pointer(Matrix2D<var_gaussian> &M, double *h_p) {
		int K = M.get_size2();
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<K; j++) {
				h_p[i*K+j] = M.get_object(i,j).z;
			}
		}
	}

	static void copy_matrix_to_host_pointer(Matrix2D<var_gamma> &M, double *h_p) {
		int K = M.get_size2();
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<K; j++) {
				h_p[i*K+j] = M.get_object(i,j).z;
			}
		}
	}

	static void copy_matrix_to_host_pointer(Matrix2D<double> &M, double *h_p) {
		int K = M.get_size2();
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<K; j++) {
				h_p[i*K+j] = M.get_object(i,j);
			}
		}
	}

	static void copy_host_pointer_to_matrix(Matrix2D<double> &M, double *h_p) {
		int K = M.get_size2();
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<K; j++) {
				M.set_object(i, j, h_p[i*K+j]);
			}
		}
	}
	*/

	static double compute_logq(my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar) {
		double logq = 0.0;
		if(param.IC>0 && param.flag_obs2prior) {
			logq += compute_logq_mat(pvar.H_alpha, pvar.validH_alpha);
		}
		if(param.flag_price>0 && param.IC>0 && param.flag_obs2prior) {
			logq += compute_logq_mat(pvar.H_beta, pvar.validH_beta);
		}
		if(param.K>0) {
			/* logq += compute_logq_mat(pvar.rho); */
			logq += compute_logq_mat(pvar.alpha);
		}
		if(param.flag_itemIntercept) {
			logq += compute_logq_mat(pvar.lambda0);
		}
		if(param.IC>0 && param.flag_obs2utility) {
			logq += compute_logq_mat(pvar.obsItems);
		}
		if(param.UC>0) {
			logq += compute_logq_mat(pvar.obsUsers);
		}
		if(param.flag_userVec>0) {
			logq += compute_logq_mat(pvar.theta);
		}
		if(param.flag_price>0) {
			if (param.noItemPriceLatents) {
				logq += compute_logq_mat(pvar.gammaObsItem);
			} else {
				logq += compute_logq_mat(pvar.gamma);
				logq += compute_logq_mat(pvar.beta);
			}
		}
		if(param.flag_day>0) {
			logq += compute_logq_mat(pvar.delta);
			logq += compute_logq_mat(pvar.mu);
		}
		if(param.flag_weekdays) {
			logq += compute_logq_mat(pvar.weekdays);
		}
		return logq;
	}

	static double compute_logq_mat(Matrix1D<var_pointmass> &M) {
		double logq = 0.0;
		for(int i=0; i<M.get_size1(); i++) {
			logq += M.get_object(i).logq();
		}
		return logq;
	}

	static double compute_logq_mat(Matrix1D<var_gaussian> &M) {
		double logq = 0.0;
		for(int i=0; i<M.get_size1(); i++) {
			logq += M.get_object(i).logq();
		}
		return logq;
	}

	static double compute_logq_mat(Matrix1D<var_gamma> &M) {
		double logq = 0.0;
		for(int i=0; i<M.get_size1(); i++) {
			logq += M.get_object(i).logq();
		}
		return logq;
	}

	static double compute_logq_mat(Matrix2D<var_pointmass> &M) {
		double logq = 0.0;
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				logq += M.get_object(i,j).logq();
			}
		}
		return logq;
	}

	static double compute_logq_mat(Matrix2D<var_gaussian> &M) {
		double logq = 0.0;
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				logq += M.get_object(i,j).logq();
			}
		}
		return logq;
	}

	static double compute_logq_mat(Matrix2D<var_gamma> &M) {
		double logq = 0.0;
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				logq += M.get_object(i,j).logq();
			}
		}
		return logq;
	}

	static double compute_logq_mat(Matrix2D<var_gaussian> &M, Matrix2D<bool> &validM) {
		double logq = 0.0;
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				if(validM.get_object(i,j)) {
					logq += M.get_object(i,j).logq();
				}
			}
		}
		return logq;
	}

	static void sample_all(gsl_rng *semilla, my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar) {
		if(param.IC>0 && param.flag_obs2prior) {
			sample_mat(semilla, pvar.H_alpha, pvar.validH_alpha);
		}
		if(param.flag_price>0 && param.IC>0 && param.flag_obs2prior && !param.noItemPriceLatents) {
			sample_mat(semilla, pvar.H_beta, pvar.validH_beta);
		}
		if(param.K>0) {
			/* sample_mat(semilla, pvar.rho); */
			sample_mat(semilla, pvar.alpha);
		}
		if(param.flag_itemIntercept) {
			sample_mat(semilla, pvar.lambda0);
		}
		if(param.IC>0 && param.flag_obs2utility) {
			sample_mat(semilla, pvar.obsItems);
		}
		if(param.UC>0) {
			sample_mat(semilla, pvar.obsUsers);
		}
		if(param.flag_userVec>0) {
			sample_mat(semilla, pvar.theta);
		}
		if(param.flag_price>0) {
			if (param.noItemPriceLatents) {
				sample_mat(semilla, pvar.gammaObsItem);
			} else {
				sample_mat(semilla, pvar.gamma);
				sample_mat(semilla, pvar.beta);
			}
		}
		if(param.flag_day>0) {
			sample_mat(semilla, pvar.delta);
			sample_mat(semilla, pvar.mu);
		}
		if(param.flag_weekdays) {
			sample_mat(semilla, pvar.weekdays);
		}
	}

	static void set_to_mean_all(my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar) {
		if(param.IC>0 && param.flag_obs2prior) {
			set_to_mean_mat(pvar.H_alpha, pvar.validH_alpha);
		}
		if(param.flag_price>0 && param.IC>0 && param.flag_obs2prior && !param.noItemPriceLatents) {
			set_to_mean_mat(pvar.H_beta, pvar.validH_beta);
		}
		if(param.K>0) {
			/* set_to_mean_mat(pvar.rho); */
			set_to_mean_mat(pvar.alpha);
		}
		if(param.flag_itemIntercept) {
			set_to_mean_mat(pvar.lambda0);
		}
		if(param.IC>0 && param.flag_obs2utility) {
			set_to_mean_mat(pvar.obsItems);
		}
		if(param.UC>0) {
			set_to_mean_mat(pvar.obsUsers);
		}
		if(param.flag_userVec>0) {
			set_to_mean_mat(pvar.theta);
		}
		if(param.flag_price>0) {
			if (param.noItemPriceLatents) {
				set_to_mean_mat(pvar.gammaObsItem);
			} else {
				set_to_mean_mat(pvar.gamma);
				set_to_mean_mat(pvar.beta);
			}
		}
		if(param.flag_day>0) {
			set_to_mean_mat(pvar.delta);
			set_to_mean_mat(pvar.mu);
		}
		if(param.flag_weekdays) {
			set_to_mean_mat(pvar.weekdays);
		}
	}

	static double set_grad_to_prior(my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar) {
		double logp = 0.0;
		if(param.IC>0 && param.flag_obs2prior) {
			logp += set_grad_to_prior_mat(pvar.H_alpha,0.0,hyper.s2H,pvar.validH_alpha);
		}
		if(param.flag_price>0 && param.IC>0 && param.flag_obs2prior && !param.noItemPriceLatents) {
			logp += set_grad_to_prior_mat(pvar.H_beta,0.0,hyper.s2H,pvar.validH_beta);
		}
		if(param.K>0) {
			/* logp += set_grad_to_prior_mat(pvar.rho,0.0,hyper.s2rho); */
			if(param.IC>0 && param.flag_obs2prior) {
				logp += set_grad_to_prior_mat(pvar.alpha,pvar.H_alpha,pvar.validH_alpha,data.attr_items,hyper.s2obsPrior);
			} else {
				logp += set_grad_to_prior_mat(pvar.alpha,0.0,hyper.s2alpha);
			}
		}
		if(param.flag_itemIntercept) {
			logp += set_grad_to_prior_mat(pvar.lambda0,0.0,hyper.s2lambda);
		}
		if(param.IC>0 && param.flag_obs2utility) {
			logp += set_grad_to_prior_mat(pvar.obsItems,0.0,hyper.s2theta);
		}
		if(param.UC>0) {
			logp += set_grad_to_prior_mat(pvar.obsUsers,0.0,hyper.s2alpha);
		}
		if(param.flag_userVec>0) {
			logp += set_grad_to_prior_mat(pvar.theta,0.0,hyper.s2theta);
		}
		if(param.flag_price>0) {
			if (param.noItemPriceLatents) {
				logp += set_grad_to_prior_mat(pvar.gammaObsItem, hyper.mean_gammaObsItem, hyper.s2gammaObsItem);
			} else {
				logp += set_grad_to_prior_mat(pvar.gamma,hyper.mean_gamma,hyper.s2gamma);
				if(param.IC>0 && param.flag_obs2prior) {
					logp += set_grad_to_prior_mat(pvar.beta,pvar.H_beta,pvar.validH_beta,data.attr_items,hyper.s2obsPrior);
				} else {
					logp += set_grad_to_prior_mat(pvar.beta,hyper.mean_beta,hyper.s2beta);
				}
			}
		}
		if(param.flag_day>0) {
			logp += set_grad_to_prior_mat(pvar.delta,0.0,hyper.s2delta);
			logp += set_grad_to_prior_mat(pvar.mu,0.0,hyper.s2mu);
		}
		if(param.flag_weekdays) {
			logp += set_grad_to_prior_mat(pvar.weekdays,0.0,hyper.s2week);
		}
		return logp;
	}

	static void sample_mat(gsl_rng *semilla, Matrix1D<var_pointmass> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			M.get_object(i).sample(semilla);
		}
	}

	static void set_to_mean_mat(Matrix1D<var_pointmass> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			M.get_object(i).z = M.get_object(i).e_x;
		}
	}

	static void sample_mat(gsl_rng *semilla, Matrix1D<var_gaussian> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			M.get_object(i).sample(semilla);
		}
	}

	static void set_to_mean_mat(Matrix1D<var_gaussian> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			M.get_object(i).z = M.get_object(i).mean;
		}
	}

	static void sample_mat(gsl_rng *semilla, Matrix1D<var_gamma> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			M.get_object(i).sample(semilla);
		}
	}

	static void set_to_mean_mat(Matrix1D<var_gamma> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			M.get_object(i).z = M.get_object(i).mean;
		}
	}

	static void sample_mat(gsl_rng *semilla, Matrix2D<var_pointmass> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				M.get_object(i,j).sample(semilla);
			}
		}
	}

	static void set_to_mean_mat(Matrix2D<var_pointmass> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				M.get_object(i,j).z = M.get_object(i,j).e_x;
			}
		}
	}

	static void sample_mat(gsl_rng *semilla, Matrix2D<var_gaussian> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				M.get_object(i,j).sample(semilla);
			}
		}
	}

	static void set_to_mean_mat(Matrix2D<var_gaussian> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				M.get_object(i,j).z = M.get_object(i,j).mean;
			}
		}
	}

	static void sample_mat(gsl_rng *semilla, Matrix2D<var_gamma> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				M.get_object(i,j).sample(semilla);
			}
		}
	}

	static void set_to_mean_mat(Matrix2D<var_gamma> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				M.get_object(i,j).z = M.get_object(i,j).mean;
			}
		}
	}

	static void sample_mat(gsl_rng *semilla, Matrix2D<var_gaussian> &M, Matrix2D<bool> &validM) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				if(validM.get_object(i,j)) {
					M.get_object(i,j).sample(semilla);
				}
			}
		}
	}

	static void set_to_mean_mat(Matrix2D<var_gaussian> &M, Matrix2D<bool> &validM) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				if(validM.get_object(i,j)) {
					M.get_object(i,j).z = M.get_object(i,j).mean;
				}
			}
		}
	}

	static double set_grad_to_prior_mat(Matrix1D<var_pointmass> &M, double mm, double ss2) {
		double logp = 0.0;
		for(int i=0; i<M.get_size1(); i++) {
			logp += M.get_object(i).set_grad_to_prior(mm,ss2);
		}
		return logp;
	}

	static double set_grad_to_prior_mat(Matrix1D<var_gaussian> &M, double mm, double ss2) {
		double logp = 0.0;
		for(int i=0; i<M.get_size1(); i++) {
			logp += M.get_object(i).set_grad_to_prior(mm,sqrt(ss2));
		}
		return logp;
	}

	static double set_grad_to_prior_mat(Matrix1D<var_gamma> &M, double mm, double ss2) {
		double logp = 0.0;
		for(int i=0; i<M.get_size1(); i++) {
			logp += M.get_object(i).set_grad_to_prior(mm,ss2);
		}
		return logp;
	}

	static double set_grad_to_prior_mat(Matrix2D<var_pointmass> &M, double mm, double ss2) {
		double logp = 0.0;
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				logp += M.get_object(i,j).set_grad_to_prior(mm,ss2);
			}
		}
		return logp;
	}

	static double set_grad_to_prior_mat(Matrix2D<var_gaussian> &M, double mm, double ss2) {
		double logp = 0.0;
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				logp += M.get_object(i,j).set_grad_to_prior(mm,sqrt(ss2));
			}
		}
		return logp;
	}

	static double set_grad_to_prior_mat(Matrix2D<var_gamma> &M, double mm, double ss2) {
		double logp = 0.0;
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				logp += M.get_object(i,j).set_grad_to_prior(mm,ss2);
			}
		}
		return logp;
	}

	static double set_grad_to_prior_mat(Matrix2D<var_gaussian> &M, double mm, double ss2, Matrix2D<bool> &validM) {
		double logp = 0.0;
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				if(validM.get_object(i,j)) {
					logp += M.get_object(i,j).set_grad_to_prior(mm,sqrt(ss2));
				}
			}
		}
		return logp;
	}

	static double set_grad_to_prior_mat(Matrix2D<var_gaussian> &M, Matrix2D<var_gaussian> &Hprior, \
										Matrix2D<bool> &validHprior, Matrix2D<double> &attr, double ss2) {
		int Nitems = M.get_size1();
		int K = M.get_size2();
		int IC = Hprior.get_size1();

		double logp = 0.0;
		double mm;
		double aux;
		for(int i=0; i<Nitems; i++) {
			for(int k=0; k<K; k++) {
				mm = 0.0;
				for(int ic=0; ic<IC; ic++) {
					if(validHprior.get_object(ic,k)) {
						mm += Hprior.get_object(ic,k).z * attr.get_object(i,ic);
					}
				}
				// increase gradient of latent features
				logp += M.get_object(i,k).set_grad_to_prior(mm,sqrt(ss2));
				// increase gradient of H
				aux = (M.get_object(i,k).z-mm)/ss2;
				for(int ic=0; ic<IC; ic++) {
					if(validHprior.get_object(ic,k)) {
						Hprior.get_object(ic,k).increase_grad(attr.get_object(i,ic) * aux);
					}
				}
			}
		}
		return logp;
	}

	/*
	static void compute_sum_alpha(my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar) {
		std::vector<int> transaction_all(data.Ntrans);
		for(int t=0; t<data.Ntrans; t++) {
			transaction_all.at(t) = t;
		}
		compute_sum_alpha(data,hyper,param,pvar,transaction_all);
	}
	*/

	/*
	static void compute_sum_alpha(my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar, std::vector<int> &transaction_list) {
		// Loop over all transactions
		for(int &t : transaction_list) {
			double aux;
			int i;
			double y = 1.0;
			for(int k=0; k<param.K; k++) {
				aux = 0.0;
				// Loop over all lines of transaction t
				for(const int &ll : data.lines_per_trans.get_object(t)) {
					i = data.obs.y_item[ll];
					aux += y*pvar.alpha.get_object(i,k).z;
				}
				pvar.sum_alpha.set_object(t,k,aux);
			}
		}
	}
	*/

	static double increase_gradients(my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar,
									 std::vector<int> &transaction_list, gsl_rng *semilla) {
		double logp = 0.0;
		int u;
		int s;
		int i_prime;
		int argmax;
		int argmax_prime;
		double *suma = new double[param.K];
		double mm;
		double mm_prime;
		double price;
		double price_prime;
		double dL_deta;
		int batchsize = transaction_list.size();
		double sviFactor;
		double sigmoid;
		int count_i;
		std::vector<int> context_items;

		// For each transaction
		for(int &t : transaction_list) {
			// Get number of copies
			int Ncopies = 1;
			if(param.flag_shuffle>0) {
				Ncopies = param.flag_shuffle;
			}

			// Get user and session
			u = data.user_per_trans.get_object(t);
			s = data.session_per_trans.get_object(t);

			// Compute eta base for all items
			compute_eta_base(data,hyper,param,pvar,t,u,s);

			for(int copy=0; copy<Ncopies; copy++) {
				// Get the items in this transaction
				std::vector<int> items_sorted;
				for(int &idx_t : data.lines_per_trans.get_object(t)) {
					int i = data.obs.y_item[idx_t];
					items_sorted.push_back(i);
				}
				// Shuffle the items in this transaction
				if(param.flag_shuffle>0) {
					gsl_ran_shuffle(semilla,items_sorted.data(),data.lines_per_trans.get_object(t).size(),sizeof(int));
				}
				// If checkout, append the checkout item to the list of items
				if(param.flag_checkout && param.flag_shuffle>=0) {
					unsigned long long mid = ULLONG_MAX;
					int i = data.item_ids.find(mid)->second;
					items_sorted.push_back(i);
				}

				// Set suma to zero
				set_to_zero(suma, param.K);
				context_items.clear();

				// For each item
				for(int &i : items_sorted) {
					// Get the price
					price = data.get_price(i,s,param);

					// Auxiliary operations for conditionally specified model
					/*
					if(param.flag_shuffle==-1) {
						// Set suma accordingly
						substract_contribution_sum_alpha(t,i,1.0,data,param,pvar,suma);

						// Set context_items
						context_items.clear();
						for(const int &idx_j : data.items_per_trans.get_object(t)) {
							if(idx_j!=i) {
								context_items.push_back(idx_j);
							}
						}
					}
					*/

					// Bernoulli model
					if(param.flag_likelihood==0) {
						// ** POSITIVES **

						// Set sviFactor
						sviFactor = static_cast<double>(data.Ntrans)/static_cast<double>(batchsize*Ncopies);

						// Compute the sigmoid
						mm = my_sigmoid(compute_mean(data,param,pvar,t,i,u,s,context_items,argmax));
						dL_deta = 1.0-mm;
						dL_deta *= sviFactor;

						// Increase log_p
						logp += sviFactor*my_log(mm);

						// Increase gradients
						increase_gradients_t_i(data,hyper,param,pvar,suma,i,u,s,t,price,dL_deta,context_items,argmax);

						// ** NEGATIVES **

						// Set sviFactor
						sviFactor *= (data.Nitems-static_cast<double>(data.lines_per_trans.get_object(t).size()))/static_cast<double>(param.negsamples);
						sviFactor *= param.zeroFactor;

						// Sample negative items from the corresponding distribution
						count_i = 0;
						while(count_i<param.negsamples) {
							i_prime = take_negative_sample(i,semilla,data,param);
							// If this is indeed a negative sample
							if(i_prime!=i && !elem_in_vector(context_items,i_prime)) {
								// Increase the counts of processed negative samples
								count_i++;
								// Compute the sigmoid
								price_prime = data.get_price(i_prime,s,param);
								mm_prime = my_sigmoid(compute_mean(data,param,pvar,t,i_prime,u,s,context_items,argmax_prime));
								dL_deta = -mm_prime;
								dL_deta *= sviFactor;
								// Increase log_p
								logp += sviFactor*my_log(1.0-mm_prime);
								// Increase gradients
								increase_gradients_t_i(data,hyper,param,pvar,suma,i_prime,u,s,t,price_prime,dL_deta,context_items,argmax_prime);
							}
						}
					// One-vs-each bound
					} else if(param.flag_likelihood==1) {
						// Set sviFactor
						sviFactor = static_cast<double>(data.Ntrans)/static_cast<double>(batchsize*Ncopies);
						sviFactor *= (data.Nitems-static_cast<double>(context_items.size())-1.0)/static_cast<double>(param.negsamples);

						// ** POSITIVES **

						// Compute eta
						mm = compute_mean(data,param,pvar,t,i,u,s,context_items,argmax);

						// ** NEGATIVES **
						count_i = 0;
						while(count_i<param.negsamples) {
							i_prime = take_negative_sample(i,semilla,data,param);
							// If this is indeed a negative sample
							if(i_prime!=i && !elem_in_vector(context_items,i_prime)) {
								// Increase the counts of processed negative samples
								count_i++;
								// Compute eta for i_prime
								price_prime = data.get_price(i_prime,s,param);
								mm_prime = compute_mean(data,param,pvar,t,i_prime,u,s,context_items,argmax_prime);
								// Compute the sigmoid
								sigmoid = my_sigmoid(mm-mm_prime);
								dL_deta = 1.0-sigmoid;
								dL_deta *= sviFactor;
								// Increase log_p
								logp += sviFactor*my_log(sigmoid);
								// Increase gradients
								increase_gradients_t_i(data,hyper,param,pvar,suma,i,u,s,t,price,dL_deta,context_items,argmax);
								increase_gradients_t_i(data,hyper,param,pvar,suma,i_prime,u,s,t,price_prime,-dL_deta,context_items,argmax_prime);
							}
						}
					// Regularized softmax
					} else if(param.flag_likelihood==2) {
						// Set sviFactor
						sviFactor = static_cast<double>(data.Ntrans)/static_cast<double>(batchsize*Ncopies);
						double *probs = new double[param.negsamples+1];
						int *neg_items = new int[param.negsamples];
						int *neg_argmax = new int[param.negsamples];
						double *price_neg_items = new double[param.negsamples];
						double maximo = -myINFINITY;

						// ** POSITIVES **

						// Compute eta
						mm = compute_mean(data,param,pvar,t,i,u,s,context_items,argmax);
						probs[param.negsamples] = mm;
						maximo = my_max(mm,maximo);

						// ** NEGATIVES **
						count_i = 0;
						while(count_i<param.negsamples) {
							i_prime = take_negative_sample(i,semilla,data,param);
							// If this is indeed a negative sample
							if(i_prime!=i && !elem_in_vector(context_items,i_prime)) {
								// Compute eta for i_prime
								price_prime = data.get_price(i_prime,s,param);
								mm_prime = compute_mean(data,param,pvar,t,i_prime,u,s,context_items,argmax_prime);
								// Keep track of this item
								maximo = my_max(mm_prime,maximo);
								neg_items[count_i] = i_prime;
								neg_argmax[count_i] = argmax_prime;
								price_neg_items[count_i] = price_prime;
								probs[count_i] = mm_prime;
								// Increase the counts of processed negative samples
								count_i++;
							}
						}
						// Normalize
						double sum_exp = 0.0;
						for(int ns=0; ns<param.negsamples+1; ns++) {
							probs[ns] = my_exp(probs[ns]-maximo);
							sum_exp += probs[ns];
						}
						for(int ns=0; ns<param.negsamples+1; ns++) {
							probs[ns] /= sum_exp;
						}
						// Increase logp
						logp += sviFactor*my_log(probs[param.negsamples]);
						// Increase grads
						dL_deta = sviFactor*(1.0-probs[param.negsamples]);
						increase_gradients_t_i(data,hyper,param,pvar,suma,i,u,s,t,price,dL_deta,context_items,argmax);
						for(int ns=0; ns<param.negsamples; ns++) {
							dL_deta = -sviFactor*probs[ns];
							i_prime = neg_items[ns];
							argmax_prime = neg_argmax[ns];
							price_prime = price_neg_items[ns];
							increase_gradients_t_i(data,hyper,param,pvar,suma,i_prime,u,s,t,price_prime,dL_deta,context_items,argmax_prime);
						}
						// Free memory
						delete [] probs;
						delete [] price_neg_items;
						delete [] neg_items;
						delete [] neg_argmax;
					// Within-group softmax
					} else if(param.flag_likelihood==3) {
						// Set sviFactor
						sviFactor = static_cast<double>(data.Ntrans)/static_cast<double>(batchsize*Ncopies);
						std::vector<double> probs;
						std::vector<int> neg_items;
						std::vector<int> neg_argmax;
						std::vector<double> price_neg_items;
						double maximo = -myINFINITY;

						// ** POSITIVES **

						// Get itemgroup
						int g_i = data.group_per_item.get_object(i);

						// Compute eta
						mm = compute_mean(data,param,pvar,t,i,u,s,context_items,argmax);
						probs.insert(probs.begin(), mm);
						maximo = my_max(mm,maximo);

						// ** NEGATIVES **
						for(int &jj: data.items_per_group.get_object(g_i)) {
							i_prime = jj;
							// If this is indeed a negative sample
							if(i_prime!=i && !elem_in_vector(context_items,i_prime)) {
								// Compute eta for i_prime
								price_prime = data.get_price(i_prime,s,param);
								mm_prime = compute_mean(data,param,pvar,t,i_prime,u,s,context_items,argmax_prime);
								// Keep track of this item
								maximo = my_max(mm_prime,maximo);
								neg_items.insert(neg_items.begin(), i_prime);
								neg_argmax.insert(neg_argmax.begin(), argmax_prime);
								price_neg_items.insert(price_neg_items.begin(), price_prime);
								probs.insert(probs.begin(), mm_prime);
								// Increase the counts of processed negative samples
								count_i++;
							}
						}
						// Normalize
						double sum_exp = 0.0;
						for(unsigned int ns=0; ns<probs.size(); ns++) {
							probs.at(ns) = my_exp(probs.at(ns)-maximo);
							sum_exp += probs.at(ns);
						}
						for(unsigned int ns=0; ns<probs.size(); ns++) {
							probs.at(ns) /= sum_exp;
						}
						// Increase logp
						logp += sviFactor*my_log(probs.back());
						// Increase grads
						dL_deta = sviFactor*(1.0-probs.back());
						increase_gradients_t_i(data,hyper,param,pvar,suma,i,u,s,t,price,dL_deta,context_items,argmax);
						for(unsigned int ns=0; ns<neg_items.size(); ns++) {
							dL_deta = -sviFactor*probs.at(ns);
							i_prime = neg_items.at(ns);
							argmax_prime = neg_argmax.at(ns);
							price_prime = price_neg_items.at(ns);
							increase_gradients_t_i(data,hyper,param,pvar,suma,i_prime,u,s,t,price_prime,dL_deta,context_items,argmax_prime);
						}
					// Exact softmax
					} else if(param.flag_likelihood==4) {
						// Set sviFactor
						sviFactor = static_cast<double>(data.Ntrans)/static_cast<double>(batchsize*Ncopies);
						std::vector<double> probs;
						std::vector<int> neg_items;
						std::vector<int> neg_argmax;
						std::vector<double> price_neg_items;
						double maximo = -myINFINITY;

						// ** POSITIVES **

						// Compute eta
						mm = compute_mean(data,param,pvar,t,i,u,s,context_items,argmax);
						probs.insert(probs.begin(), mm);
						maximo = my_max(mm,maximo);

						// ** NEGATIVES **
						for(i_prime=0; i_prime<data.Nitems; i_prime++) {
							// If this is indeed a negative sample
							if(i_prime!=i && !elem_in_vector(context_items,i_prime)) {
								// Compute eta for i_prime
								price_prime = data.get_price(i_prime,s,param);
								mm_prime = compute_mean(data,param,pvar,t,i_prime,u,s,context_items,argmax_prime);
								// Keep track of this item
								maximo = my_max(mm_prime,maximo);
								neg_items.insert(neg_items.begin(), i_prime);
								neg_argmax.insert(neg_argmax.begin(), argmax_prime);
								price_neg_items.insert(price_neg_items.begin(), price_prime);
								probs.insert(probs.begin(), mm_prime);
								// Increase the counts of processed negative samples
								count_i++;
							}
						}
						// Normalize
						double sum_exp = 0.0;
						for(unsigned int ns=0; ns<probs.size(); ns++) {
							probs.at(ns) = my_exp(probs.at(ns)-maximo);
							sum_exp += probs.at(ns);
						}
						for(unsigned int ns=0; ns<probs.size(); ns++) {
							probs.at(ns) /= sum_exp;
						}
						// Increase logp
						logp += sviFactor*my_log(my_max(1.0e-12,probs.back()));
						// Increase grads
						dL_deta = sviFactor*(1.0-probs.back());
						increase_gradients_t_i(data,hyper,param,pvar,suma,i,u,s,t,price,dL_deta,context_items,argmax);
						for(unsigned int ns=0; ns<neg_items.size(); ns++) {
							dL_deta = -sviFactor*probs.at(ns);
							i_prime = neg_items.at(ns);
							argmax_prime = neg_argmax.at(ns);
							price_prime = price_neg_items.at(ns);
							increase_gradients_t_i(data,hyper,param,pvar,suma,i_prime,u,s,t,price_prime,dL_deta,context_items,argmax_prime);
						}
					/*
					// Softmax augmentation
					} else if(param.flag_likelihood==5) {
						// Set sviFactor
						double sviFactorClasses = (data.Nitems-static_cast<double>(context_items.size())-1.0)/static_cast<double>(param.negsamples);
						double sviFactorDatapoints = static_cast<double>(data.Ntrans)/static_cast<double>(batchsize*Ncopies);
						sviFactor = sviFactorClasses*sviFactorDatapoints;
						std::vector<double> neg_mm(param.negsamples);
						std::vector<double> neg_exp_mm(param.negsamples);
						std::vector<int> neg_items(param.negsamples);
						std::vector<int> neg_argmax(param.negsamples);
						std::vector<double> price_neg_items(param.negsamples);
						double maximo = -myINFINITY;
						double exp_maximo;

						// ** POSITIVES **

						// Compute eta
						mm = compute_mean(data,param,pvar,t,i,u,s,context_items,argmax);

						// ** NEGATIVES **
						count_i = 0;
						while(count_i<param.negsamples) {
							i_prime = take_negative_sample(i,semilla,data,param);
							// If this is indeed a negative sample
							if(i_prime!=i && !elem_in_vector(context_items,i_prime)) {
								// Compute eta for i_prime
								price_prime = data.get_price(i_prime,s,param);
								mm_prime = compute_mean(data,param,pvar,t,i_prime,u,s,context_items,argmax_prime);
								// Keep track of this item
								neg_items.at(count_i) = i_prime;
								neg_mm.at(count_i) = mm_prime;
								neg_argmax.at(count_i) = argmax_prime;
								price_neg_items.at(count_i) = price_prime;
								// Compute maximum
								maximo = my_max(maximo,mm_prime-mm);
								// Increase the counts of processed negative samples
								count_i++;
							}
						}
						// Compute local natural parameter
						double eta_augm = 0.0;
						for(count_i=0; count_i<param.negsamples; count_i++) {
							if(maximo>0.0) {
								neg_exp_mm.at(count_i) = my_exp(neg_mm.at(count_i)-mm-maximo);
							} else {
								neg_exp_mm.at(count_i) = my_exp(neg_mm.at(count_i)-mm);
							}
							eta_augm += neg_exp_mm.at(count_i);
						}
						eta_augm *= sviFactorClasses;
						// Increase log_p
						if(maximo>0.0) {
							exp_maximo = my_exp(-maximo);
						}
						if(maximo>0.0) {
							logp += sviFactorDatapoints*(1.0 - my_log(exp_maximo+eta_augm)-maximo - exp_maximo/(exp_maximo+eta_augm));
						} else {
							logp += sviFactorDatapoints*(1.0 - my_log(1.0+eta_augm) - 1.0/(1.0+eta_augm));
						}
						// Increase gradients
						for(count_i=0; count_i<param.negsamples; count_i++) {
							price_prime = price_neg_items.at(count_i);
							i_prime = neg_items.at(count_i);
							argmax_prime = neg_argmax.at(count_i);
							mm_prime = neg_exp_mm.at(count_i);
							// Increase gradients
							if(maximo>0.0) {
								dL_deta = sviFactor*mm_prime/(exp_maximo+eta_augm);
							} else {
								dL_deta = sviFactor*mm_prime/(1.0+eta_augm);
							}
							increase_gradients_t_i(data,hyper,param,pvar,suma,i,u,s,t,price,dL_deta,context_items,argmax);
							increase_gradients_t_i(data,hyper,param,pvar,suma,i_prime,u,s,t,price_prime,-dL_deta,context_items,argmax_prime);
							// Increase log_p
							logp -= dL_deta;
						}
					*/
					} else {
						std::cerr << "[ERR] The 'likelihood' parameter cannot take value " << std::to_string(param.flag_likelihood) << endl;
						assert(0);
					}

					/*
					// Increase suma
					if(param.flag_shuffle!=-1) {
						context_items.push_back(i);
						for(int k=0; k<param.K; k++) {
							suma[k] += pvar.alpha.get_object(i,k).z;
						}
					}
					*/
				}
			}
		}
		delete [] suma;
		return(logp);
	}

	static void increase_gradients_t_i(my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar,
									   double *suma_input, int i, int u, int s, int t, double price,
									   double dL_deta, std::vector<int> &context_items, int argmax) {
		double deta_dz;
		/* double y_j; */
		/* double y_j_prime; */
		/* int g_argmax = -1; */
		double *suma = new double[param.K];

		// Divide suma by the corresponding value
		double denomAvgContext = 0.0;
		if(context_items.size()>0) {
			denomAvgContext = static_cast<double>(context_items.size());
			for(int k=0; k<param.K; k++) {
				suma[k] = suma_input[k]/denomAvgContext;
			}
		} else {
			set_to_zero(suma, param.K);
		}

		// Increase gradient of rho
		/*
		for(int k=0; k<param.K; k++) {
			if(param.flag_userVec==0) {
				deta_dz = suma[k];
			} else if(param.flag_userVec==1) {
				deta_dz = suma[k]+pvar.theta.get_object(u,k).z;
			} else if(param.flag_userVec==2) {
				deta_dz = suma[k]*pvar.theta.get_object(u,k).z;
			} else if(param.flag_userVec==3) {
				deta_dz = suma[k];
			}
			pvar.rho.get_object(i,k).increase_grad(dL_deta*deta_dz);
			// (lookahead)
			*/ /*
			if(param.flag_lookahead && argmax!=-1) {
				double aux_suma_k = (suma_input[k]+pvar.alpha.get_object(i,k).z)/(1.0+denomAvgContext);
				if(param.flag_userVec==0) {
					deta_dz = aux_suma_k;
				} else if(param.flag_userVec==1) {
					deta_dz = aux_suma_k+pvar.theta.get_object(u,k).z;
				} else if(param.flag_userVec==2) {
					std::cerr << "[ERR] Not implemented" << endl;
					assert(0);
				} else if(param.flag_userVec==3) {
					deta_dz = aux_suma_k;
				}
				pvar.rho.get_object(argmax,k).increase_grad(dL_deta*deta_dz);
			}
			*/ /*
		}
		*/

		// Increase gradient of alpha for all items in the context
		/*
		for(const int &j : context_items) {
			// Scale down y_j to account for the averaging of elements in context
			y_j = 1.0/denomAvgContext;
			*/ /*
			if(param.flag_lookahead && argmax!=-1) {
				y_j_prime = 1.0/(denomAvgContext+1.0);
			}
			// Increase gradient
			for(int k=0; k<param.K; k++) {
				double aux_k = y_j*pvar.rho.get_object(i,k).z;
				if(param.flag_userVec==0) {
					deta_dz = aux_k;
				} else if(param.flag_userVec==1) {
					deta_dz = aux_k;
				} else if(param.flag_userVec==2) {
					deta_dz = aux_k*pvar.theta.get_object(u,k).z;
				} else if(param.flag_userVec==3) {
					deta_dz = aux_k;
				}
				// (lookahead)
				*/ /*
				if(param.flag_lookahead && argmax!=-1) {
					deta_dz += y_j_prime*pvar.rho.get_object(argmax,k).z;
				}
				*/ /*
				// increase grad
				pvar.alpha.get_object(j,k).increase_grad(dL_deta*deta_dz);
			}
		}
		*/
		// (lookahead, as alpha_i is also in the context)
		/*
		if(param.flag_lookahead && argmax!=-1) {
			y_j_prime = 1.0/(denomAvgContext+1.0);
			for(int k=0; k<param.K; k++) {
				deta_dz = y_j_prime*pvar.rho.get_object(argmax,k).z;
				pvar.alpha.get_object(i,k).increase_grad(dL_deta*deta_dz);
			}
		}
		*/

		// Increase gradient of alpha_i (only if param.flag_userVec==3)
		if(param.flag_userVec==3) {
			for(int k=0; k<param.K; k++) {
				deta_dz = pvar.theta.get_object(u,k).z;
				pvar.alpha.get_object(i,k).increase_grad(dL_deta*deta_dz);
			}
			// (lookahead)
			/*
			if(param.flag_lookahead && argmax!=-1) {
				for(int k=0; k<param.K; k++) {
					deta_dz = pvar.theta.get_object(u,k).z;
					pvar.alpha.get_object(argmax,k).increase_grad(dL_deta*deta_dz);
				}
			}
			*/
		}

		// Increase gradient of lambda0
		if(param.flag_itemIntercept) {
			pvar.lambda0.get_object(i).increase_grad(dL_deta);
			// (lookahead)
			/*
			if(param.flag_lookahead && argmax!=-1) {
				pvar.lambda0.get_object(argmax).increase_grad(dL_deta);
			}
			*/
		}

		// Increase gradient of obsItems
		if(param.IC>0 && param.flag_obs2utility) {
			for(int k=0; k<param.IC; k++) {
				deta_dz = data.attr_items.get_object(i,k);
				// (lookahead)
				/*
				if(param.flag_lookahead && argmax!=-1) {
					deta_dz += data.attr_items.get_object(argmax,k);
				}
				*/
				// increase grad
				pvar.obsItems.get_object(u,k).increase_grad(dL_deta*deta_dz);
			}
		}

		// Increase gradient of obsUsers
		if(param.UC>0) {
			for(int k=0; k<param.UC; k++) {
				deta_dz = data.attr_users.get_object(u,k);
				// increase grad
				pvar.obsUsers.get_object(i,k).increase_grad(dL_deta*deta_dz);
			}
		}

		// Increase gradient of theta
		if(param.flag_userVec>0) {
			for(int k=0; k<param.K; k++) {
				if(param.flag_userVec==1) {
					/* deta_dz = pvar.rho.get_object(i,k).z; */
				} else if(param.flag_userVec==2) {
					/* deta_dz = suma[k]*pvar.rho.get_object(i,k).z; */
				} else if(param.flag_userVec==3) {
					deta_dz = pvar.alpha.get_object(i,k).z;
				}
				// (lookahead)
				/*
				if(param.flag_lookahead && argmax!=-1) {
					if(param.flag_userVec==1) {
						deta_dz += pvar.rho.get_object(argmax,k).z;
					} else if(param.flag_userVec==3) {
						deta_dz += pvar.alpha.get_object(argmax,k).z;
					}
				}
				*/
				// increase grad
				pvar.theta.get_object(u,k).increase_grad(dL_deta*deta_dz);
			}
		}

		// Increase gradient of the price vectors
		if (param.noItemPriceLatents) {
			deta_dz = -price;
			pvar.gammaObsItem.get_object(u).increase_grad(dL_deta*deta_dz);
		} else {
			for(int k=0; k<param.flag_price; k++) {
				// gamma
				deta_dz = -price*pvar.beta.get_object(i,k).z;
				// (lookahead)
				/*
				if(param.flag_lookahead && argmax!=-1) {
					deta_dz += -price*pvar.beta.get_object(g_argmax,k).z;
				}
				*/
				// increase grad
				pvar.gamma.get_object(u,k).increase_grad(dL_deta*deta_dz);

				// beta
				deta_dz = -price*pvar.gamma.get_object(u,k).z;
				pvar.beta.get_object(i,k).increase_grad(dL_deta*deta_dz);
				// (lookahead)
				/*
				if(param.flag_lookahead && argmax!=-1) {
					deta_dz = -price*pvar.gamma.get_object(u,k).z;
					pvar.beta.get_object(g_argmax,k).increase_grad(dL_deta*deta_dz);
				}
				*/
			}
		}

		// Increase gradient of seasonal effect parameters
		for(int k=0; k<param.flag_day; k++) {
			int d = data.day_per_session.get_object(s);

			// delta
			deta_dz = pvar.mu.get_object(i,k).z;
			// (lookahead)
			/*
			if(param.flag_lookahead && argmax!=-1) {
				deta_dz += pvar.mu.get_object(g_argmax,k).z;
			}
			*/
			// increase grad
			pvar.delta.get_object(d,k).increase_grad(dL_deta*deta_dz);

			// mu
			deta_dz = pvar.delta.get_object(d,k).z;
			pvar.mu.get_object(i,k).increase_grad(dL_deta*deta_dz);
			// (lookahead)
			/*
			if(param.flag_lookahead && argmax!=-1) {
				deta_dz = pvar.delta.get_object(d,k).z;
				pvar.mu.get_object(g_argmax,k).increase_grad(dL_deta*deta_dz);
			}
			*/
		}

		// Increase gradient of weekdays effects
		if(param.flag_weekdays) {
			int w = data.weekday_per_session.get_object(s);
			pvar.weekdays.get_object(i,w).increase_grad(dL_deta);
		}

		// Free memory
		delete [] suma;
	}

	static int take_negative_sample(int i, gsl_rng *semilla, my_data &data, const my_param &param) {
		int i_prime;
		if(param.flag_nsFreq<=1) {
			i_prime = gsl_ran_discrete(semilla, data.negsampling_dis);
		} else if(param.flag_nsFreq>=2) {
			int g_i = data.group_per_item.get_object(i);
			i_prime = gsl_ran_discrete(semilla, data.negsampling_dis_per_group.at(g_i));
		} else {
			std::cerr << "[ERR] Wrong value for -nsFreq: " << std::to_string(param.flag_nsFreq) << endl;
			assert(0);
		}
		return i_prime;
	} 

	static bool elem_in_vector(std::vector<int> &vec, int val) {
		for(const int &i : vec) {
			if(i==val) {
				return true;
			}
		}
		return false;
	}

	static void compute_eta_base(my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar, int t, int u, int s) {
		compute_eta_base(data,hyper,param,pvar,t,u,s,pvar.eta_base);
	}

	static void compute_eta_base(my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar, int t, int u, int s, Matrix1D<double> &eta_base) {
		double eta;
		double price;
		int d;
		int w;
		if(param.flag_day>0) {
			d = data.day_per_session.get_object(s);
		}
		if(param.flag_weekdays) {
			w = data.weekday_per_session.get_object(s);
		}

		for(int i=0; i<data.Nitems; i++) {
			eta = 0.0;
			price = data.get_price(i,s,param);

			// Add intercept term
			if(param.flag_itemIntercept) {
				eta += pvar.lambda0.get_object(i).z;
			}
			// Add obsItems
			if(param.IC>0 && param.flag_obs2utility) {
				eta += pvar.prod_obsItems.get_object(u,i);
			}
			// Add obsUsers
			if(param.UC>0) {
				eta += pvar.prod_obsUsers.get_object(u,i);
			}
			// Add user vectors
			if(param.flag_userVec>0) {
				eta += pvar.prod_theta_alpha.get_object(u,i);
			}
			// Add price term
			if(param.flag_price>0) {
				if (param.noItemPriceLatents) {
					eta -= pvar.gammaObsItem.get_object(u).z*price;
				} else {
					eta -= pvar.prod_gamma_beta.get_object(u,i)*price;
				}
			}
			// Add seasonal effect
			if(param.flag_day>0) {
				eta += pvar.prod_delta_mu.get_object(d,i);
			}
			// Add weekdays effect
			if(param.flag_weekdays) {
				eta += pvar.weekdays.get_object(i,w).z;
			}

			eta_base.set_object(i,eta);
		}
	}

	static double compute_mean(my_data &data, const my_param &param, my_pvar &pvar, int t, int i, int u, int s,
							   std::vector<int> &elem_context, int &argmax) {
		return compute_mean(data,param,pvar,t,i,u,s,elem_context,argmax,pvar.eta_base);
	}

	static double compute_mean(my_data &data, const my_param &param, my_pvar &pvar, int t, int i, int u, int s,
							   std::vector<int> &elem_context, int &argmax, Matrix1D<double> &eta_base) {
		double eta = 0.0;
		/* double denomAvgContext = static_cast<double>(elem_context.size()); */
		argmax = -1;

		// Get checkout item
		/*
		int i_checkout = -1;
		if(param.flag_checkout) {
			unsigned long long mid = ULLONG_MAX;
			i_checkout = data.item_ids.find(mid)->second;
		}
		*/

		// Eta base
		eta = eta_base.get_object(i);
		// Add elements in context
		/*
		if(param.K>0) {
			for(int &j : elem_context) {
				eta += pvar.prod_rho_alpha.get_object(i,j)/denomAvgContext;
			}
		}
		*/

		// Look-ahead
		/*
		if(param.flag_lookahead>0 && i!=i_checkout) {
			double maximo = -myINFINITY;
			/////
			double aux;
			for(int k=0; k<data.Nitems; k++) {
				if(k!=i && !elem_in_vector(elem_context,k)) {
					aux = eta_base.get_object(k);
					for(int &j : elem_context) {
						aux += pvar.prod_rho_alpha.get_object(k,j)/(denomAvgContext+1.0);
					}
					aux += pvar.prod_rho_alpha.get_object(k,i)/(denomAvgContext+1.0);
					if(aux>maximo) {
						maximo = aux;
						argmax = k;
					}
				}
			}
			/////
			// Cuda: From here
			int *d_elem_context;
			d_allocate(&d_elem_context, elem_context.size());
			if(elem_context.size()>0) {
				copy_h2d(d_elem_context, elem_context.data(), elem_context.size());
			}
			double *h_eta_lookahead = new double[data.Nitems];
			double *d_eta_lookahead;
			d_allocate(&d_eta_lookahead, data.Nitems);
			double *d_eta_base;
			d_allocate(&d_eta_base, data.Nitems);
			copy_h2d(d_eta_base, eta_base.get_pointer(0), data.Nitems);
			int n_blocks = ceil(data.Nitems/512.0);
			int n_threads_per_block = 512;
			lookAhead_kern<<<n_blocks, n_threads_per_block>>>(d_eta_lookahead, pvar.d_prod_rho_alpha, d_eta_base, i, d_elem_context, elem_context.size(), data.Nitems);
			d_sync();
			copy_d2h(h_eta_lookahead, d_eta_lookahead, data.Nitems);
			thrust::device_ptr<double> t_eta_lookahead = thrust::device_pointer_cast(d_eta_lookahead);
			thrust::device_ptr<double> t_maximo = thrust::max_element(t_eta_lookahead, t_eta_lookahead+data.Nitems);
			argmax = t_maximo - t_eta_lookahead;
			maximo = h_eta_lookahead[argmax];
			cudaFree(d_elem_context);
			cudaFree(d_eta_base);
			cudaFree(d_eta_lookahead);
			delete [] h_eta_lookahead;
			// Cuda: Up to here

			eta += maximo;
		}
		*/

		// Return
		return eta;
	}

	/*
	static void substract_contribution_sum_alpha(int t, int i, double y, my_data &data, const my_param &param, my_pvar &pvar, double *vec) {
		y = 1.0;
		// Remove contribution from sum_lambdas
		for(int k=0; k<param.K; k++) {
			vec[k] = pvar.sum_alpha.get_object(t,k)-y*pvar.alpha.get_object(i,k).z;
		}
	}
	*/

	/*
	static void set_sum_alpha(int t, my_data &data, const my_param &param, my_pvar &pvar, double *vec) {
		for(int k=0; k<param.K; k++) {
			vec[k] = pvar.sum_alpha.get_object(t,k);
		}
	}
	*/

	static void compute_test_performance(int duration, my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar) {
		if(param.noTest) {
			return;
		}

		double llh = 0.0;
		int count = 0;
		int u;
		int i;
		int s;
		double *p_item = new double[data.Nitems];
		double sum_norm;
		double aux_max;
		int maxprob_j = -1;
		int argmax = -1;
		int val;
		double avgAcc;
		double avgPrecision;
		double avgRecall;
		double avgF1;
		int countValidPrecision = 0;
		int countValidRecall = 0;
		int g_i;
		Matrix1D<int> confMatrixDiag = Matrix1D<int>(data.Nitems);
		Matrix1D<int> allTrue_i = Matrix1D<int>(data.Nitems);
		Matrix1D<int> allPred_i = Matrix1D<int>(data.Nitems);
		Matrix1D<double> eta_base = Matrix1D<double>(data.Nitems);
		set_to_zero(confMatrixDiag);
		set_to_zero(allTrue_i);
		set_to_zero(allPred_i);

		// Create a vector specifying how many valid lines per transactions
		// there are in test.tsv. A line is valid if both the item and the user
		// appear in train.tsv (the user is necessary only when -userVec != 0)
		Matrix1D<int> test_valid_lines_per_trans = Matrix1D<int>(data.test_Ntrans);
		set_to_zero(test_valid_lines_per_trans);
		for(unsigned int ll=0; ll<data.obs_test.T; ll++) {
			u = data.obs_test.y_user[ll];
			i = data.obs_test.y_item[ll];
			int t = data.obs_test.y_trans[ll];
			if(data.lines_per_item.get_object(i).size()>0) {
				if(param.flag_userVec==0 || data.lines_per_user.get_object(u).size()>0) {
					int aux = 1+test_valid_lines_per_trans.get_object(t);
					test_valid_lines_per_trans.set_object(t,aux);
				}
			}
		}

		// Compute inner product for each line in test set
		count = 0;
		for(unsigned int ll=0; ll<data.obs_test.T; ll++) {
			u = data.obs_test.y_user[ll];
			i = data.obs_test.y_item[ll];
			s = data.obs_test.y_sess[ll];
			int t = data.obs_test.y_trans[ll];
			if(param.flag_likelihood==3) {
				g_i = data.group_per_item.get_object(i);
			}

			// Ignore "non-valid" transactions and items that are not present in train.tsv
			if(test_valid_lines_per_trans.get_object(t)>0 && data.lines_per_item.get_object(i).size()>0) {
				/*
				// Find elements in context
				std::vector<int> test_elem_context = data.test_items_per_trans.get_object(t);
				std::vector<int>::const_iterator elem_remove = std::find(test_elem_context.begin(), test_elem_context.end(), i);
				if(elem_remove==test_elem_context.end()) {
					std::cerr << "[ERR] This should not happen (compute_test_performance)" << endl;
					assert(0);
				}
				test_elem_context.erase(elem_remove);
				*/
				std::vector<int> test_elem_context;
				// Compute eta_base for all items
				compute_eta_base(data,hyper,param,pvar,-1,u,s,eta_base);
				// Compute the likelihood according to the softmax
				aux_max = -myINFINITY;
				maxprob_j = -1;
				for(int j=0; j<data.Nitems; j++) {
					if(param.flag_likelihood!=3 || g_i==data.group_per_item.get_object(j)) {
						p_item[j] = compute_mean(data,param,pvar,-1,j,u,s,test_elem_context,argmax,eta_base);
						if(p_item[j]>aux_max) {
							aux_max = p_item[j];
							maxprob_j = j;
						}
					}
				}
				sum_norm = 0.0;
				for(int j=0; j<data.Nitems; j++) {
					if(param.flag_likelihood!=3 || g_i==data.group_per_item.get_object(j)) {
						p_item[j] = my_exp(p_item[j]-aux_max);
						sum_norm += p_item[j];
					}
				}
				for(int j=0; j<data.Nitems; j++) {
					if(param.flag_likelihood!=3 || g_i==data.group_per_item.get_object(j)) {
						p_item[j] = p_item[j]/sum_norm;
					}
				}
				llh += my_log(p_item[i]);
				if(maxprob_j==i) {
					val = confMatrixDiag.get_object(i);
					confMatrixDiag.set_object(i, val+1);
				}
				val = allTrue_i.get_object(i);
				allTrue_i.set_object(i, val+1);
				val = allPred_i.get_object(maxprob_j);
				allPred_i.set_object(maxprob_j, val+1);
				// Increase count to compute average llh later
				count++;
			}
		}

		// Take the average
		llh /= static_cast<double>(count);
		avgAcc = 0.0;
		avgPrecision = 0.0;
		avgRecall = 0.0;
		for(int i=0; i<data.Nitems; i++) {
			avgAcc += confMatrixDiag.get_object(i);
			if(allPred_i.get_object(i)>0) {
				avgPrecision += static_cast<double>(confMatrixDiag.get_object(i))/static_cast<double>(allPred_i.get_object(i));
				countValidPrecision += 1;
			}
			if(allTrue_i.get_object(i)>0) {
				avgRecall += static_cast<double>(confMatrixDiag.get_object(i))/static_cast<double>(allTrue_i.get_object(i));
				countValidRecall += 1;
			}
		}
		avgAcc /= static_cast<double>(count);
		avgPrecision /= static_cast<double>(countValidPrecision);
		avgRecall /= static_cast<double>(countValidRecall);
		avgF1 = 2.0*avgPrecision*avgRecall/(avgPrecision+avgRecall);

		// Print to file
		string fname = param.outdir+"/test.tsv";
		char buffer[1000];
    	sprintf(buffer,"%d\t%d\t%.9f\t%.16f\t%.16f\t%.16f\t%.16f\t%d",param.it,duration,llh,avgAcc,avgPrecision,avgRecall,avgF1,count);
        my_output::write_line(fname,string(buffer));

		// Free memory
		delete [] p_item;
	}

	static void compute_train_performance(int duration, my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar) {
		if(param.noTrain) {
			return;
		}

		double llh = 0.0;
		int count = 0;
		int u;
		int i;
		int s;
		double *p_item = new double[data.Nitems];
		double sum_norm;
		double aux_max;
		int maxprob_j = -1;
		int argmax = -1;
		int val;
		double avgAcc;
		double avgPrecision;
		double avgRecall;
		double avgF1;
		int countValidPrecision = 0;
		int countValidRecall = 0;
		int g_i;
		Matrix1D<int> confMatrixDiag = Matrix1D<int>(data.Nitems);
		Matrix1D<int> allTrue_i = Matrix1D<int>(data.Nitems);
		Matrix1D<int> allPred_i = Matrix1D<int>(data.Nitems);
		Matrix1D<double> eta_base = Matrix1D<double>(data.Nitems);
		set_to_zero(confMatrixDiag);
		set_to_zero(allTrue_i);
		set_to_zero(allPred_i);

		// Compute inner product for each line in the training set
		count = 0;
		for(unsigned int ll=0; ll<data.obs.T; ll++) {
			u = data.obs.y_user[ll];
			i = data.obs.y_item[ll];
			s = data.obs.y_sess[ll];
			//int t = data.obs.y_trans[ll];
			if(param.flag_likelihood==3) {
				g_i = data.group_per_item.get_object(i);
			}

			/*
			// Find elements in context
			std::vector<int> elem_context = data.items_per_trans.get_object(t);
			elem_context.erase(std::remove(elem_context.begin(), elem_context.end(), i), elem_context.end());
			*/
			std::vector<int> elem_context;
			// Compute eta_base for all items
			compute_eta_base(data,hyper,param,pvar,-1,u,s,eta_base);
			// Compute the likelihood according to the softmax
			aux_max = -myINFINITY;
			maxprob_j = -1;
			for(int j=0; j<data.Nitems; j++) {
				if(param.flag_likelihood!=3 || g_i==data.group_per_item.get_object(j)) {
					p_item[j] = compute_mean(data,param,pvar,-1,j,u,s,elem_context,argmax,eta_base);
					if(p_item[j]>aux_max) {
						aux_max = p_item[j];
						maxprob_j = j;
					}
				}
			}
			sum_norm = 0.0;
			for(int j=0; j<data.Nitems; j++) {
				if(param.flag_likelihood!=3 || g_i==data.group_per_item.get_object(j)) {
					p_item[j] = my_exp(p_item[j]-aux_max);
					sum_norm += p_item[j];
				}
			}
			for(int j=0; j<data.Nitems; j++) {
				if(param.flag_likelihood!=3 || g_i==data.group_per_item.get_object(j)) {
					p_item[j] = p_item[j]/sum_norm;
				}
			}
			llh += my_log(p_item[i]);
			if(maxprob_j==i) {
				val = confMatrixDiag.get_object(i);
				confMatrixDiag.set_object(i, val+1);
			}
			val = allTrue_i.get_object(i);
			allTrue_i.set_object(i, val+1);
			val = allPred_i.get_object(maxprob_j);
			allPred_i.set_object(maxprob_j, val+1);
			// Increase count to compute average llh later
			count++;
		}

		// Take the average
		llh /= static_cast<double>(count);
		avgAcc = 0.0;
		avgPrecision = 0.0;
		avgRecall = 0.0;
		for(int i=0; i<data.Nitems; i++) {
			avgAcc += confMatrixDiag.get_object(i);
			if(allPred_i.get_object(i)>0) {
				avgPrecision += static_cast<double>(confMatrixDiag.get_object(i))/static_cast<double>(allPred_i.get_object(i));
				countValidPrecision += 1;
			}
			if(allTrue_i.get_object(i)>0) {
				avgRecall += static_cast<double>(confMatrixDiag.get_object(i))/static_cast<double>(allTrue_i.get_object(i));
				countValidRecall += 1;
			}
		}
		avgAcc /= static_cast<double>(count);
		avgPrecision /= static_cast<double>(countValidPrecision);
		avgRecall /= static_cast<double>(countValidRecall);
		avgF1 = 2.0*avgPrecision*avgRecall/(avgPrecision+avgRecall);

		// Print to file
		string fname = param.outdir+"/train.tsv";
		char buffer[1000];
    	sprintf(buffer,"%d\t%d\t%.9f\t%.16f\t%.16f\t%.16f\t%.16f\t%d",param.it,duration,llh,avgAcc,avgPrecision,avgRecall,avgF1,count);
        my_output::write_line(fname,string(buffer));

		// Free memory
		delete [] p_item;
	}

	static double compute_val_likelihood(int duration, my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar) {
		if(param.noVal) {
			return 0.0;
		}

		double llh = 0.0;
		int count = 0;
		int u;
		int i;
		int s;
		double *p_item = new double[data.Nitems];
		double sum_norm;
		double aux_max;
		int maxprob_j;
		int argmax = -1;
		int val;
		double avgAcc;
		double avgPrecision;
		double avgRecall;
		double avgF1;
		int countValidPrecision = 0;
		int countValidRecall = 0;
		int g_i;
		Matrix1D<int> confMatrixDiag = Matrix1D<int>(data.Nitems);
		Matrix1D<int> allTrue_i = Matrix1D<int>(data.Nitems);
		Matrix1D<int> allPred_i = Matrix1D<int>(data.Nitems);
		Matrix1D<double> eta_base = Matrix1D<double>(data.Nitems);
		set_to_zero(confMatrixDiag);
		set_to_zero(allTrue_i);
		set_to_zero(allPred_i);

		// Create a vector specifying how many valid lines per transactions
		// there are in valid.tsv. A line is valid if both the item and the user
		// appear in train.tsv (the user is necessary only when -userVec != 0)
		Matrix1D<int> val_valid_lines_per_trans = Matrix1D<int>(data.val_Ntrans);
		set_to_zero(val_valid_lines_per_trans);
		for(unsigned int ll=0; ll<data.obs_val.T; ll++) {
			u = data.obs_val.y_user[ll];
			i = data.obs_val.y_item[ll];
			int t = data.obs_val.y_trans[ll];
			if(data.lines_per_item.get_object(i).size()>0) {
				if(param.flag_userVec==0 || data.lines_per_user.get_object(u).size()>0) {
					int aux = 1+val_valid_lines_per_trans.get_object(t);
					val_valid_lines_per_trans.set_object(t,aux);
				}
			}
		}

		// Compute inner product for each line in val set
		count = 0;
		for(unsigned int ll=0; ll<data.obs_val.T; ll++) {
			u = data.obs_val.y_user[ll];
			i = data.obs_val.y_item[ll];
			s = data.obs_val.y_sess[ll];
			int t = data.obs_val.y_trans[ll];
			if(param.flag_likelihood==3) {
				g_i = data.group_per_item.get_object(i);
			}

			// Ignore "non-valid" transactions and items that are not present in train.tsv
			if(val_valid_lines_per_trans.get_object(t)>0 && data.lines_per_item.get_object(i).size()>0) {
				/*
				// Find elements in context
				std::vector<int> val_elem_context = data.val_items_per_trans.get_object(t);
				std::vector<int>::const_iterator elem_remove = std::find(val_elem_context.begin(), val_elem_context.end(), i);
				if(elem_remove==val_elem_context.end()) {
					std::cerr << "[ERR] This should not happen (compute_val_performance)" << endl;
					assert(0);
				}
				val_elem_context.erase(elem_remove);
				*/
				std::vector<int> val_elem_context;
				// Compute eta_base for all items
				compute_eta_base(data,hyper,param,pvar,-1,u,s,eta_base);
				// Compute the likelihood according to the softmax
				aux_max = -myINFINITY;
				maxprob_j = -1;
				for(int j=0; j<data.Nitems; j++) {
					if(param.flag_likelihood!=3 || g_i==data.group_per_item.get_object(j)) {
						p_item[j] = compute_mean(data,param,pvar,-1,j,u,s,val_elem_context,argmax,eta_base);
						if(p_item[j]>aux_max) {
							aux_max = p_item[j];
							maxprob_j = j;
						}
					}
				}
				sum_norm = 0.0;
				for(int j=0; j<data.Nitems; j++) {
					if(param.flag_likelihood!=3 || g_i==data.group_per_item.get_object(j)) {
						p_item[j] = my_exp(p_item[j]-aux_max);
						sum_norm += p_item[j];
					}
				}
				for(int j=0; j<data.Nitems; j++) {
					if(param.flag_likelihood!=3 || g_i==data.group_per_item.get_object(j)) {
						p_item[j] = p_item[j]/sum_norm;
					}
				}
				llh += my_log(p_item[i]);
				if(maxprob_j==i) {
					val = confMatrixDiag.get_object(i);
					confMatrixDiag.set_object(i, val+1);
				}
				val = allTrue_i.get_object(i);
				allTrue_i.set_object(i, val+1);
				val = allPred_i.get_object(maxprob_j);
				allPred_i.set_object(maxprob_j, val+1);
				// Increase count to compute average llh later
				count++;
			}
		}

		// Take the average
		llh /= static_cast<double>(count);
		avgAcc = 0.0;
		avgPrecision = 0.0;
		avgRecall = 0.0;
		for(int i=0; i<data.Nitems; i++) {
			avgAcc += confMatrixDiag.get_object(i);
			if(allPred_i.get_object(i)>0) {
				avgPrecision += static_cast<double>(confMatrixDiag.get_object(i))/static_cast<double>(allPred_i.get_object(i));
				countValidPrecision += 1;
			}
			if(allTrue_i.get_object(i)>0) {
				avgRecall += static_cast<double>(confMatrixDiag.get_object(i))/static_cast<double>(allTrue_i.get_object(i));
				countValidRecall += 1;
			}
		}
		avgAcc /= static_cast<double>(count);
		avgPrecision /= static_cast<double>(countValidPrecision);
		avgRecall /= static_cast<double>(countValidRecall);
		avgF1 = 2.0*avgPrecision*avgRecall/(avgPrecision+avgRecall);

		// Print to file
		string fname = param.outdir+"/valid.tsv";
		char buffer[1000];
    	sprintf(buffer,"%d\t%d\t%.9f\t%.16f\t%.16f\t%.16f\t%.16f\t%d",param.it,duration,llh,avgAcc,avgPrecision,avgRecall,avgF1,count);
        my_output::write_line(fname,string(buffer));

		// Free memory
		delete [] p_item;

		// Return
		return llh;
	}

	static void prepare_test_valid(gsl_rng *semilla, my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar) {
		// Sample from everything
		set_to_mean_all(data,hyper,param,pvar);

		// Create fake transaction_list
		std::vector<int> transaction_list = std::vector<int>(data.Ntrans);
		for(int t=0; t<data.Ntrans; t++){
			transaction_list.at(t) = t;
		} 

		// Compute products rho*alpha, theta*alpha, etc.
		compute_prod_all(data,hyper,param,pvar,transaction_list);		
	}

	static double compute_avg_norm(Matrix2D<var_pointmass> &m) {
		if(m.get_size1()==0 || m.get_size2()==0) {
			return 0.0;
		}
		double suma = 0.0;
		for(int i=0; i<m.get_size1(); i++) {
			for(int j=0; j<m.get_size2(); j++) {
				suma += my_pow2(m.get_object(i,j).z);
			}
		}
		return(suma/static_cast<double>(m.get_size1()*m.get_size2()));
	}

	static double compute_avg_norm(Matrix2D<var_gaussian> &m) {
		if(m.get_size1()==0 || m.get_size2()==0) {
			return 0.0;
		}
		double suma = 0.0;
		for(int i=0; i<m.get_size1(); i++) {
			for(int j=0; j<m.get_size2(); j++) {
				suma += my_pow2(m.get_object(i,j).z);
			}
		}
		return(suma/static_cast<double>(m.get_size1()*m.get_size2()));
	}

	static double compute_avg_norm(Matrix2D<var_gamma> &m) {
		if(m.get_size1()==0 || m.get_size2()==0) {
			return 0.0;
		}
		double suma = 0.0;
		for(int i=0; i<m.get_size1(); i++) {
			for(int j=0; j<m.get_size2(); j++) {
				suma += my_pow2(m.get_object(i,j).z);
			}
		}
		return(suma/static_cast<double>(m.get_size1()*m.get_size2()));
	}

	/*
	static void set_to_zero_device(double *d_vec, int K) {
		double *h_vec = new double[K];
		set_to_zero(h_vec, K);
		copy_h2d(d_vec, h_vec, K);
		delete [] h_vec;
	}
	*/

	static void set_to_zero(double *vec, int K) {
		for(int k=0; k<K; k++) {
			vec[k] = 0.0;
		}
	}

	static void set_to_zero(Matrix1D<int> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			M.set_object(i,0);
		}
	}

	static void set_to_zero(Matrix2D<double> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				M.set_object(i,j,0.0);
			}
		}
	}

	static void set_to_zero(Matrix3D<int> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				for(int k=0; k<M.get_size3(); k++) {
					M.set_object(i,j,k,0);
				}
			}
		}
	}

	static void take_grad_step(double logp, const my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar) {
		if(param.IC>0 && param.flag_obs2prior) {
			for(int ic=0; ic<param.IC; ic++) {
				for(int k=0; k<param.K; k++) {
					if(pvar.validH_alpha.get_object(ic,k)) {
						pvar.H_alpha.get_object(ic,k).update_param_var(logp, param.eta, param.flag_step_schedule);
					} else {
						pvar.H_alpha.get_object(ic,k).initialize(0.0, hyper.s2H);
					}
				}
			}
		}
		if(param.flag_price>0 && param.IC>0 && param.flag_obs2prior && !param.noItemPriceLatents) {
			for(int ic=0; ic<param.IC; ic++) {
				for(int k=0; k<param.flag_price; k++) {
					if(pvar.validH_beta.get_object(ic,k)) {
						pvar.H_beta.get_object(ic,k).update_param_var(logp, param.eta, param.flag_step_schedule);
					} else {
						pvar.H_beta.get_object(ic,k).initialize(0.0, hyper.s2H);
					}
				}
			}
		}
		for(int i=0; i<data.Nitems; i++) {
			for(int k=0; k<param.K; k++) {
				/* pvar.rho.get_object(i,k).update_param_var(logp, param.eta, param.flag_step_schedule); */
				pvar.alpha.get_object(i,k).update_param_var(logp, param.eta, param.flag_step_schedule);
			}
			if(param.flag_itemIntercept) {
				pvar.lambda0.get_object(i).update_param_var(logp, param.eta, param.flag_step_schedule);
			}
		}
		if(param.IC>0 && param.flag_obs2utility) {
			for(int u=0; u<data.Nusers; u++) {
				for(int k=0; k<param.IC; k++) {
					pvar.obsItems.get_object(u,k).update_param_var(logp, param.eta, param.flag_step_schedule);
				}
			}
		}
		if(param.UC>0) {
			for(int i=0; i<data.Nitems; i++) {
				for(int k=0; k<param.UC; k++) {
					pvar.obsUsers.get_object(i,k).update_param_var(logp, param.eta, param.flag_step_schedule);
				}
			}
		}
		if(param.flag_userVec>0) {
			for(int u=0; u<data.Nusers; u++) {
				for(int k=0; k<param.K; k++) {
					pvar.theta.get_object(u,k).update_param_var(logp, param.eta, param.flag_step_schedule);
				}
			}
		}
		if (param.noItemPriceLatents) {
			for(int u=0; u<data.Nusers; u++) {
				pvar.gammaObsItem.get_object(u).update_param_var(logp, param.eta, param.flag_step_schedule);
			}
		} else {
			for(int k=0; k<param.flag_price; k++) {
				for(int u=0; u<data.Nusers; u++) {
					pvar.gamma.get_object(u,k).update_param_var(logp, param.eta, param.flag_step_schedule);
				}
				for(int i=0; i<data.Nitems; i++) {
					pvar.beta.get_object(i,k).update_param_var(logp, param.eta, param.flag_step_schedule);
				}
			}
		}
		for(int k=0; k<param.flag_day; k++) {
			for(int d=0; d<data.Ndays; d++) {
				pvar.delta.get_object(d,k).update_param_var(logp, param.eta, param.flag_step_schedule);
			}
			for(int i=0; i<data.Nitems; i++) {
				pvar.mu.get_object(i,k).update_param_var(logp, param.eta, param.flag_step_schedule);
			}
		}
		if(param.flag_weekdays) {
			for(int i=0; i<data.Nitems; i++) {
				for(int w=0; w<data.Nweekdays; w++) {
					pvar.weekdays.get_object(i,w).update_param_var(logp, param.eta, param.flag_step_schedule);
				}
			}
		}
	}

};

#endif
