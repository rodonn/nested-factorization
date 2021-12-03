#ifndef HPF_PARAMS_HPP
#define HPF_PARAMS_HPP

class hpf_param {
public:
	string datadir;
	string outdir;
	int K;
	int UC;
	int IC;
	int ICV;
	int IChier;
	bool flag_session;
	bool flag_availability;
	unsigned long int seed;
	int rfreq;
	int Niter;
	bool factor_obs_std;
	bool factor_obs_ones;
	double factor_obs_extra;
	bool flag_itemIntercept;
	bool flag_lfirst;
	bool flag_ofirst;
    bool quiet;
	string label;
	double ini_offset;
	int L_phi;
	int it;
	double prev_val_llh;
	int n_val_decr;
	int nIterIni;
	bool flag_day;
	int flag_hourly;
	bool noVal;
	bool noTest;
	Matrix1D<double> gauss_mean;
	Matrix1D<double> gauss_var;

	hpf_param() {
		datadir = ".";
		outdir = ".";
		K = 50;
		UC = 0;
		IC = 0;
		ICV = 0;
		IChier = 0;
		flag_session = false;
		flag_availability = false;
		flag_itemIntercept = false;
		seed = 0;
		rfreq = 10;
		Niter = 1000;
		factor_obs_std = false;
		factor_obs_ones = false;
		factor_obs_extra = 1.0;
		flag_lfirst = false;
		flag_ofirst = false;
		quiet = false;
		label = "";
		ini_offset = 1.0;
		L_phi = K;
		it = 0;
		prev_val_llh = 0.0;
		n_val_decr = 0;
		nIterIni = 20;
		flag_day = false;
		flag_hourly = 0;
		noVal = false;
		noTest = false;
	}
};

class hpf_hyper {
public:
	double a;
	double ap;
	double bp;
	double c;
	double cp;
	double dp;
	double e;
	double f;
	double a_x;
	double b_x;
	double eta_k_shp;
	double eta_k_rte;
	double h_lk_shp;
	double h_lk_rte;

	hpf_hyper() {
		a = 0.3;
		bp = 0.3;
		c = 0.3;
		dp = 0.3;
		e = 0.3;
		f = 0.3;
		ap = 1.5;
		cp = 1.5;
		a_x = 0.05;
		b_x = 1.0;
		eta_k_shp = 0.3;
		eta_k_rte = 1.0;
		h_lk_shp = 1.0;
		h_lk_rte = 1.0;
	}
};

class hpf_data_aux {
public:
	unsigned int T;			// Number of (user,session,item) triplets
	unsigned int *y_user;	// User idx per (user,session,item) triplet
	unsigned int *y_item;	// Item idx per (user,session,item) triplet
	unsigned int *y_sess;	// Session idx per (user,session,item) triplet
	unsigned int *y_rating;	// Value (rating or #units) per (user,session,item) triplet

	hpf_data_aux() {
		T = 0;
		y_user = nullptr;
		y_item = nullptr;
		y_sess = nullptr;
		y_rating = nullptr;
	}

	hpf_data_aux(unsigned int T_) {
		allocate_all(T_);
	}

	~hpf_data_aux() {
		delete_all();
	}

	inline void allocate_all(unsigned int T_) {
		T = T_;
		y_user = new unsigned int[T];
		y_item = new unsigned int[T];
		y_sess = new unsigned int[T];
		y_rating = new unsigned int[T];
	}

	inline void delete_all() {
		delete [] y_user;
		delete [] y_item;
		delete [] y_sess;
		delete [] y_rating;
	}

	hpf_data_aux & operator=(const hpf_data_aux &rhs) {
    	// Check for self-assignment!
	    if (this!=&rhs) {
	    	// deallocate memory
			delete_all();
			// allocate memory for the contents of rhs
			allocate_all(rhs.T);
			// copy values from rhs
			for(unsigned int n=0; n<rhs.T; n++) {
				y_user[n] = rhs.y_user[n];
				y_item[n] = rhs.y_item[n];
				y_sess[n] = rhs.y_sess[n];
				y_rating[n] = rhs.y_rating[n];
			}
		}
		return *this;
	}
};

class hpf_avail_aux {
public:
	int index;
	int value;

	hpf_avail_aux & operator=(const hpf_avail_aux &rhs) {
    	// Check for self-assignment!
	    if (this!=&rhs) {
	    	index = rhs.index;
	    	value = rhs.value;
		}
		return *this;
	}
};

class hpf_data {
public:
	int Nitems;		// Number of items
	int m;			// Number of items (tmp)
	int Nusers;		// Number of users
	int n;			// Number of users (tmp)
	int Nsessions;	// Number of sessions
	int UC;			// Number of observed attributes per user
	int IC;			// Number of observed attributes per item
	int ICV;		// Number of observed attributes per item that vary with sessions
	int IChier;		// Number of observed attributes per item
	int Ndays;		// Number of calendar days
	int Nweekdays;  // Number of weekdays (typically 7)
	int NuserGroups;	// Number of user groups (only makes sense if flag_hourly>0 or flag_day is active)
	int NitemGroups;	// Number of item groups
	std::map<string,int> item_ids;			// Map containing the item id's
	std::map<string,int> user_ids;			// Map containing the user id's
	std::map<string,int> session_ids;		// Map containing the session id's
	std::map<string,int> day_ids;			// Map containing the day id's
	std::map<string,int> weekday_ids;		// Map containing the weekday id's
	std::map<string,int> group_ids;			// Map containing the usergroups id's
	std::map<string,int> itemgroup_ids;		// Map containing the itemgroups id's
	hpf_data_aux obs;			// Observations (train)
	hpf_data_aux obs_test;		// Observations (test)
	hpf_data_aux obs_val;		// Observations (validation)
	Matrix2D<double> attr_users;		// Attributes for users
	Matrix2D<double> attr_users_log;	// Log-attributes for users
	Matrix1D<double> attr_users_scaleF;	// Scale factor for each user attribute
	Matrix2D<double> attr_items;		// Attributes for items
	Matrix2D<bool> attr_items_hier;		// Attributes for items (only for IChier>0)
	Matrix2D<double> attr_items_log;	// Log-attributes for items
	Matrix1D<double> attr_items_scaleF;	// Scale factor for each item attribute
	Matrix3D<double> attr_items_isk;		// Attributes (that vary with sessions) for items
	Matrix3D<double> attr_items_isk_log;	// Log-attributes (that vary with sessions) for items
	Matrix1D<double> attr_items_isk_scaleF;	// Scale factor for each item attribute (that vary with sessions)
	Matrix1D<int> group_per_user;					// For each user, group to which she belongs
	Matrix1D<int> group_per_item;					// For each item, group to which it belongs
	Matrix1D<std::vector<int>> users_per_group;		// For each usergroup, list of the users that belong to that group
	Matrix1D<std::vector<int>> items_per_group;		// For each itemgroup, list of the items that belong to that group
	Matrix1D<std::vector<int>> sessions_per_item;	// For each item, list of the sessions in which that item is available
	Matrix1D<std::vector<int>> sessions_per_user;	// For each user, list of the sessions that she visits the supermarket
	Matrix1D<std::vector<int>> lines_per_user;		// For each user, list of "lines" in train.tsv in which that user appears
	Matrix1D<std::vector<int>> lines_per_item;		// For each item, list of "lines" in train.tsv in which that item appears
	Matrix3D<std::vector<int>> lines_per_xday;		// For each item/day/usergroup triplet, list of "lines" in train.tsv in which that triplet appears
	Matrix3D<std::vector<int>> lines_per_weekday;	// For each item/weekday/usergroup triplet, list of "lines" in train.tsv in which that triplet appears
	Matrix1D<int> day_per_session;					// For each session, day_id to which it corresponds
	Matrix1D<int> weekday_per_session;				// For each session, weekday_id to which it corresponds
	Matrix1D<double> hour_per_session;				// For each session, hour of the day
	Matrix1D<std::vector<int>> sessions_per_day;	// For each day, a list of sessions
	Matrix1D<std::vector<int>> sessions_per_weekday;// For each weekday, a list of sessions
	Matrix2D<double> auxsumrte_ic;					// Contains sum_t sum_i a_uit*attribute_i (size = Nusers x IC)
	Matrix2D<double> auxsumrte_icv;					// Contains sum_t sum_i a_uit*attribute_it (size = Nusers x IC)
	Matrix2D<double> auxsumrte_uc;					// Contains sum_t sum_u a_uit*attribute_u (size = Nitems x UC)
	Matrix3D<double> auxsumrte_xday;				// Contains sum_t sum_u a_uit (size = NuserGroups x NitemGroups x Ndays)
	Matrix4D<double> auxsumrte_xhourly;				// Contains sum_rte for the hourly variables (size = NuserGroups x NitemGroups x Nweekdays x Number of mixture components)
	Matrix1D<double> sum_alpha_p_per_user;			// same as above, but for each user (sums over those t in which user u appears). (Only used if sessions=1 and availability=0)
	Matrix2D<int> availabilityCounts;				// Counts how many times user u sees item i available (size = Nusers x Nitems)
	Matrix1D<std::vector<hpf_avail_aux>> availabilityNegCountsUser;		// Each entry = [#times user u goes to the store - availabilityCounts(u,i)] (length = Nusers; each vector contains a list of items)
	Matrix1D<std::vector<hpf_avail_aux>> availabilityNegCountsItem;		// Each entry = [#times user u goes to the store - availabilityCounts(u,i)] (length = Nitems; each vector contains a list of users)

	hpf_data() {
		m = -1;
		n = -1;
		UC = 0;
		IC = 0;
		ICV = 0;
		IChier = 0;
		Ndays = 0;
		Nweekdays = 0;
		NuserGroups = 0;
		NitemGroups = 0;
	}

	void create_other_data_structs(const hpf_param &param) {
		lines_per_user = Matrix1D<std::vector<int>>(Nusers);
		lines_per_item = Matrix1D<std::vector<int>>(Nitems);
		for(unsigned int t=0; t<obs.T; t++) {
			lines_per_user.get_object(obs.y_user[t]).push_back(t);
			lines_per_item.get_object(obs.y_item[t]).push_back(t);
		}
	}

	bool isavailable(int i, int s) {
		// returns true if item i is available at session s
		bool result = false;
		if(std::find(sessions_per_item.get_object(i).begin(),sessions_per_item.get_object(i).end(),s) != sessions_per_item.get_object(i).end()) {
			result = true;
		}
		return result;
	}
};

class hpf_gamma {
public:
	double shp;
	double rte;
	double e_x;
	double e_logx;

	void initialize_random(gsl_rng *semilla, double val_shp, double val_rte, double offset) {
		shp = val_shp+gsl_ran_flat(semilla,0.0,0.01*offset);
		rte = val_rte+gsl_ran_flat(semilla,0.0,0.01*offset);
		update_expectations();
	}

	void update_expectations() {
		e_x = shp/rte;
		e_logx = my_gsl_sf_psi(shp)-my_log(rte);
	}
};

class hpf_multinomial {
public:
	double phi;
	double rho;
};

class hpf_pvar {
public:
	Matrix2D<hpf_gamma> theta_uk;		// User latent preferences
	Matrix2D<hpf_gamma> beta_ik;		// Item latent attributes
	Matrix1D<hpf_gamma> beta_i0;		// Item latent intercepts
	Matrix2DNoncontiguous<hpf_multinomial> z_tk;		// Auxiliary variables for each observation
	Matrix1D<hpf_gamma> xi_u;			// Hierarchical prior over theta_uk 
	Matrix1D<hpf_gamma> eta_i;			// Hierarchical prior over beta_ik
	Matrix1D<hpf_gamma> eta_k;			// Hierarchical prior over beta_ik (only if IChier>0)
	Matrix2D<hpf_gamma> h_lk;			// Hierarchical prior over beta_ik (only if IChier>0)
	Matrix2D<hpf_gamma> sigma_uk;		// User's latent preferences corresponding to items' observables
	Matrix2D<hpf_gamma> sigma_uk_var;	// User's latent preferences corresponding to items' observables that vary over time
	Matrix2D<hpf_gamma> rho_ik;			// Item's latent attributes corresponding to users' observables
	Matrix3D<hpf_gamma> x_gid;			// Per-day per-itemgroup per-usergroup effects
	Matrix4D<hpf_gamma> x_giwn;			// Per-weekday per-itemgroup per-usergroup mixture weights

	hpf_pvar(const hpf_data &data, const hpf_param &param) {
 		theta_uk = Matrix2D<hpf_gamma>(data.Nusers,param.K);
		beta_ik = Matrix2D<hpf_gamma>(data.Nitems,param.K);
		z_tk = Matrix2DNoncontiguous<hpf_multinomial>(data.obs.T,param.L_phi);
		xi_u = Matrix1D<hpf_gamma>(data.Nusers);
		if(param.IChier==0) {
			eta_i = Matrix1D<hpf_gamma>(data.Nitems);
		}
		if(param.flag_itemIntercept) {
			beta_i0 = Matrix1D<hpf_gamma>(data.Nitems);
		}
		if(param.IC>0) {
			sigma_uk = Matrix2D<hpf_gamma>(data.Nusers,param.IC);
		}
		if(param.ICV>0) {
			sigma_uk_var = Matrix2D<hpf_gamma>(data.Nusers,param.ICV);
		}
		if(param.IChier>0) {
			eta_k = Matrix1D<hpf_gamma>(param.K);
			h_lk = Matrix2D<hpf_gamma>(param.IChier,param.K);
		}
		if(param.UC>0) {
			rho_ik = Matrix2D<hpf_gamma>(data.Nitems,param.UC);
		}
		if(param.flag_day) {
			x_gid = Matrix3D<hpf_gamma>(data.NuserGroups,data.NitemGroups,data.Ndays);
		}
		if(param.flag_hourly>0) {
			x_giwn = Matrix4D<hpf_gamma>(data.NuserGroups,data.NitemGroups,data.Nweekdays,param.flag_hourly);
		}
	}
	
	void initialize_random(gsl_rng *semilla, hpf_data &data, const hpf_param &param, const hpf_hyper &hyper) {
		// Initialize theta_uk
		for(int u=0; u<data.Nusers; u++) {
			for(int k=0; k<param.K; k++) {
				theta_uk.get_object(u,k).initialize_random(semilla,hyper.a,hyper.bp,param.ini_offset);
			}
			// Initialize xi_u
			xi_u.get_object(u).initialize_random(semilla,hyper.ap,hyper.ap/hyper.bp,param.ini_offset);
			// Initialize sigma_uk
			for(int k=0; k<param.IC; k++) {
				sigma_uk.get_object(u,k).initialize_random(semilla,hyper.e,hyper.bp*data.attr_items_scaleF.get_object(k),param.ini_offset);
			}
			// Initialize sigma_uk_var
			for(int k=0; k<param.ICV; k++) {
				sigma_uk_var.get_object(u,k).initialize_random(semilla,hyper.e,hyper.bp*data.attr_items_isk_scaleF.get_object(k),param.ini_offset);
			}
		}
		// Initialize beta_ik
		for(int i=0; i<data.Nitems; i++) {
			for(int k=0; k<param.K; k++) {
				beta_ik.get_object(i,k).initialize_random(semilla,hyper.c,hyper.dp,param.ini_offset);
			}
			// Initialize eta_i
			if(param.IChier==0) {
				eta_i.get_object(i).initialize_random(semilla,hyper.cp,hyper.cp/hyper.dp,param.ini_offset);
			}
			// Initialize rho_ik
			for(int k=0; k<param.UC; k++) {
				rho_ik.get_object(i,k).initialize_random(semilla,hyper.f,hyper.dp*data.attr_users_scaleF.get_object(k),param.ini_offset);
			}
		}
		// Initialize beta_i0
		if(param.flag_itemIntercept) {
			for(int i=0; i<data.Nitems; i++) {
				beta_i0.get_object(i).initialize_random(semilla,hyper.c,hyper.dp,param.ini_offset);
			}
		}
		// Initialize parameters if IChier>0
		if(param.IChier>0) {
			for(int k=0; k<param.K; k++) {
				eta_k.get_object(k).initialize_random(semilla,hyper.eta_k_shp,hyper.eta_k_rte,param.ini_offset);
				for(int ll=0; ll<param.IChier; ll++) {
					h_lk.get_object(ll,k).initialize_random(semilla,hyper.h_lk_shp,hyper.h_lk_rte,param.ini_offset);
				}
			}
		}
		// Initialize x_gid
		if(param.flag_day) {
			for(int g=0; g<data.NuserGroups; g++) {
				for(int i=0; i<data.NitemGroups; i++) {
					for(int d=0; d<data.Ndays; d++) {
						x_gid.get_object(g,i,d).initialize_random(semilla,hyper.a_x,hyper.b_x,param.ini_offset);
					}
				}
			}
		}
		// Initialize x_giwn
		if(param.flag_hourly>0) {
			for(int g=0; g<data.NuserGroups; g++) {
				for(int i=0; i<data.NitemGroups; i++) {
					for(int d=0; d<data.Nweekdays; d++) {
						for(int n=0; n<param.flag_hourly; n++) {
							x_giwn.get_object(g,i,d,n).initialize_random(semilla,hyper.a_x,hyper.b_x,param.ini_offset);
						}
					}
				}
			}
		}
		// Initialize z_tk
		for(unsigned int t=0; t<data.obs.T; t++) {
			for(int k=0; k<param.L_phi; k++) {
				z_tk.get_object(t,k).rho = my_log(my_gsl_ran_gamma(semilla,1.0,1.0));
			}
		}
		update_expectations_z();
	}

	void update_expectations_z() {
		double suma;
		double maximo;
		double aux;
		for(int t=0; t<z_tk.get_size1(); t++) {
			// Find the maximum
			maximo = -myINFINITY;
			for(int k=0; k<z_tk.get_size2(); k++) {
				aux = z_tk.get_object(t,k).rho;
				maximo = (maximo>aux)?maximo:aux;
			}
			// Exponentiate
			suma = 0.0;
			for(int k=0; k<z_tk.get_size2(); k++) {
				aux = my_exp(z_tk.get_object(t,k).rho-maximo);
				aux = (aux<1e-40)?1e-40:aux;
				z_tk.get_object(t,k).phi = aux;
				suma += aux;
			}
			// Normalize
			for(int k=0; k<z_tk.get_size2(); k++) {
				aux = z_tk.get_object(t,k).phi/suma;
				aux = (aux<1e-300)?1e-300:aux;
				z_tk.get_object(t,k).phi = aux;
			}
		}
	}
};

#endif
