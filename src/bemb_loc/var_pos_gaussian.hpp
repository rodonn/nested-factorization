#ifndef REP_VAR_POS_GAUSSIAN
#define REP_VAR_POS_GAUSSIAN

/* Gaussian distribution */
class var_pos_gaussian {
private:
	double Gt_rhoP1;
	double Gt_rhoP2;
	double df_dz;
	double df_dz_neg;

public:
	double z;
	double ee;
	double mean;
	double sigma;
	double u_sigma;

	// Constructor
	var_pos_gaussian() {
		Gt_rhoP1 = 0.0;
		Gt_rhoP2 = 0.0;
		df_dz = 0.0;
		df_dz_neg = 0.0;
	}

	// Destructor: free memory
	~var_pos_gaussian() { }

	// Initialize randomly
	void initialize_random(gsl_rng *semilla, double val_mean, double val_sigma, double offset) {
		double mm;
		double ss;

		if(val_mean==0.0) {
			val_mean = 0.01;
		}

		mm = val_mean + gsl_ran_flat(semilla,-val_mean*offset,val_mean*offset);
		ss = val_sigma + gsl_ran_flat(semilla,-val_sigma*offset,val_sigma*offset);
		initialize(mm,ss);
	}

	// Initialize
	inline void initialize(double mm, double ss) {
		sigma = ss;
		u_sigma = my_log(my_exp(sigma)-1.0);
		mean = mm;
	}

	// Generate samples
	void sample(gsl_rng *semilla) {
		ee = gsl_ran_ugaussian(semilla);
		z = my_softplus(mean+sigma*ee);

		if(z<0.0001) {
			z = 0.0001;
			ee = (my_log(my_exp(z)-1.0)-mean)/sigma;
		}
	}

	// Set to mean
	void set_to_mean() {
		ee = 0.0;
		z = my_softplus(mean);
	}

	// Compute and follow gradient
	void update_param_var(double f, double eta, int step_schedule) {
		// Obtain gradients
        double grad_mean;
        double grad_sigma;
        double stepsize;
        double exp_z = my_exp(-z);
        double aux = 1.0-exp_z;

        // Gradient wrt mean (only g_rep)
        grad_mean = df_dz*aux;

        // Gradient wrt sigma (only g_rep)
        grad_sigma = df_dz*ee*aux;

        // Add esimate of gradient of entropy
        grad_sigma += 1.0/sigma + ee*exp_z;
        grad_mean += exp_z;

        // Convert to gradient w.r.t. sigma in unconstr. space
		grad_sigma *= (1.0-my_exp(-sigma));

        // Update Gt
        Gt_rhoP1 = var_stepsize::update_G(grad_mean, Gt_rhoP1, step_schedule);
        Gt_rhoP2 = var_stepsize::update_G(grad_sigma, Gt_rhoP2, step_schedule);

        // Follow gradient wrt mean
        stepsize = var_stepsize::get_stepsize(eta, Gt_rhoP1, step_schedule);
        mean += stepsize*grad_mean;
        mean = (mean<-9.21)?-9.21:mean;

        // Follow gradient wrt sigma
        stepsize = var_stepsize::get_stepsize(eta, Gt_rhoP2, step_schedule);
        u_sigma += stepsize*grad_sigma;
        u_sigma = (u_sigma<-9.2103)?-9.2103:u_sigma;
        sigma = my_log(1.0+my_exp(u_sigma));
	}

	// Set gradient to 0
	inline void set_grad_to_zero() {
		df_dz = 0.0;
		df_dz_neg = 0.0;
	}

	// Initialize for a new gradient descent algorithm
	inline void initialize_iterations() {
		Gt_rhoP1 = 0.0;
		Gt_rhoP2 = 0.0;
		df_dz = 0.0;
		df_dz_neg = 0.0;
	}

	// Set gradient to prior
	inline double set_grad_to_prior_gamma(double aa, double bb) {
		df_dz = (aa-1.0)/z - bb;
		df_dz_neg = 0.0;
		// Return
		return(aa*my_log(bb)-my_loggamma(aa)+(aa-1.0)*my_log(z)-bb*z);
	}

	// Increase gradient of the model
	inline void increase_grad(double val) {
		df_dz += val;
	}

	// Increase gradient for the negatives
	inline void increase_grad_neg(double val) {
		df_dz_neg += val;
	}

	// Scale gradient for the negatives and add to df_dz
	inline void scale_add_grad_neg(double val) {
		df_dz_neg *= val;
		df_dz += df_dz_neg;
	}

	// Return the gradient
	inline double get_grad() {
		return df_dz;
	}

	// Compute logq
	inline double logq() {
		double aux = -0.5*(M_LN2+M_LNPI-my_pow2(ee));
		aux -= my_log(sigma) + sigma*ee+mean - z;
		return aux;
	}
};

#endif
