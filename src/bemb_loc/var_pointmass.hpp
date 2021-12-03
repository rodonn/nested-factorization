#ifndef REP_VAR_POINTMASS
#define REP_VAR_POINTMASS

class var_pointmass {
public:
	double z;
	double e_x;
	double e_log;
	double grad;
	double grad_neg;
	double Gt;
	bool flag_positive;
	bool trainable;

	var_pointmass() {
		e_x = 0.0;
		e_log = 0.0;
		grad = 0.0;
		grad_neg = 0.0;
		Gt = 0.0;
		flag_positive = false;
		trainable = true;
	}

	void initialize(double vv, bool pp) {
		flag_positive = pp;
		if(flag_positive) {
			e_x = vv;
	        e_log = my_log(vv);
		} else {
			e_x = vv;
		}
	}
	
	void initialize_random(gsl_rng *semilla, double val_mean, bool pp, double offset) {
		double mm;

		if(val_mean==0.0) {
			val_mean = 0.01;
		}

		mm = val_mean + gsl_ran_flat(semilla, -val_mean*offset, val_mean*offset);
		initialize(mm,pp);
	}

	inline void set_grad_to_zero() {
		grad = 0.0;
		grad_neg = 0.0;
	}

	inline void set_to_mean() {
		z = e_x;
	}

	// Generate sample
	inline void sample() {
		z = e_x;
	}

	inline void sample(gsl_rng *semilla) {
		sample();
	}

	double set_grad_to_prior(double mm, double ss2) {
		double logp;

		if(flag_positive) {
			// Gamma prior (mm=shp; ss2=rte)
			grad = (mm-1.0)/e_x - ss2;
			logp = mm*my_log(ss2) - my_loggamma(mm) + (mm-1.0)*my_log(e_x) - ss2*e_x;
		} else {
			// Gaussian prior (mm=mean; ss2=var)
			grad = -(e_x-mm)/ss2;
			logp = -0.5*(M_LN2+M_LNPI+my_log(ss2)-grad*(e_x-mm));
		}
		grad_neg = 0.0;

		return logp;
	}

	inline void increase_grad(double val) {
		grad += val;
	}

	inline void increase_grad_neg(double val) {
		grad_neg += val;
	}

	inline void scale_add_grad_neg(double sviFactor) {
		grad_neg *= sviFactor;
		grad += grad_neg;
	}

	// step_schedule: 0=advi, 1=rmsprop, 2=adagrad
	void update_param_var(double logp, double eta, int flag_step_schedule) {
		if(!trainable) {
			return;
		}
		// Convert grad into gradient of the log
		if(flag_positive) {
			grad *= e_x;
		}
		// Increase Gt
		Gt = var_stepsize::update_G(grad, Gt, flag_step_schedule);
		double stepsize = var_stepsize::get_stepsize(eta, Gt, flag_step_schedule);
		// Take gradient step
		if(flag_positive) {
			if(Gt>0.0) {
				e_log += stepsize*grad;
				e_log = (e_log<-11.5129?-11.5129:e_log);
				e_log = (e_log>3.9120?3.9120:e_log);
				e_x = my_exp(e_log);
			}
		} else {
			if(Gt>0.0) {
				e_x += stepsize*grad;
				// e_log is not updated because it isn't used
			}
		}
	}

	// Return the gradient
	inline double get_grad() {
		return grad;
	}

	// Compute logq
	inline double logq() {
		return 0.0;
	}

	// Trainable
	inline void set_trainable(bool vv) {
		trainable = vv;
	}
};

#endif
