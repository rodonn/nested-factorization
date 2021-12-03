#ifndef HPF_INFERENCE_HPP
#define HPF_INFERENCE_HPP

class hpf_infer {
public:
	static void run_inference_step(hpf_data &data, hpf_param &param, const hpf_hyper &hyper, hpf_pvar &pvar, bool warmup) {
		// Set the hierarchical rates (first iteration only)
		if(param.it==0) {
			update_shape_hier(data,param,hyper,pvar);
		}

		// Update (the rate of) xi_u
		update_xi_rte(data,param,hyper,pvar);

		// Update theta_uk
		if((!warmup) || (warmup && param.flag_lfirst)) {
			update_theta(data,param,hyper,pvar);
		}

		// Update sigma_uk
		if((!warmup) || (warmup && param.flag_ofirst)) {
			if(param.IC>0) {
				update_sigma(data,param,hyper,pvar);
			}
			if(param.ICV>0) {
				update_sigma_var(data,param,hyper,pvar);
			}
		}

		// Update (the rate of) eta_i
		if(param.IChier==0) {
			update_eta_rte(data,param,hyper,pvar);
		} else {
			update_eta_k(data,param,hyper,pvar);
			update_h_lk(data,param,hyper,pvar);
		}

		// Update beta_ik and beta_i0
		if((!warmup) || (warmup && param.flag_lfirst)) {
			if(param.flag_itemIntercept) {
				update_beta0(data,param,hyper,pvar);
			}
			update_beta(data,param,hyper,pvar);
		}

		// Update rho_ik
		if((!warmup) || (warmup && param.flag_ofirst)) {
			if(param.UC>0) {
				update_rho(data,param,hyper,pvar);
			}
		}

		// Update x_gid and x_giwn
		if((!warmup) || (warmup && param.flag_lfirst)) {
			if(param.flag_day) {
				update_xday(data,param,hyper,pvar);
			}
			if(param.flag_hourly>0) {
				update_xhourly(data,param,hyper,pvar);
			}
		}

		// Update z_tk
		update_z(data,param,hyper,pvar);
	}

	static void update_z(hpf_data &data, hpf_param &param, const hpf_hyper &hyper, hpf_pvar &pvar) {
		int i;
		int u;
		int g_u;
		int g_i;
		int d;
		int s;
		int auxD = 0;
		int auxI = 0;
		if(param.flag_day) {
			auxD = 1;
		}
		if(param.flag_itemIntercept) {
			auxI = 1;
		}
		for(unsigned int t=0; t<data.obs.T; t++) {
			u = data.obs.y_user[t];
			i = data.obs.y_item[t];
			if(param.flag_session) {
				s = data.obs.y_sess[t];
			}
			if(param.flag_itemIntercept) {
				pvar.z_tk.get_object(t,0).rho = pvar.beta_i0.get_object(i).e_logx;
			}
			for(int k=0; k<param.K; k++) {
				pvar.z_tk.get_object(t,auxI+k).rho = pvar.theta_uk.get_object(u,k).e_logx \
												     +pvar.beta_ik.get_object(i,k).e_logx;
			}
			for(int uc=0; uc<data.UC; uc++) {
				pvar.z_tk.get_object(t,auxI+param.K+uc).rho = data.attr_users_log.get_object(u,uc) \
														      +pvar.rho_ik.get_object(i,uc).e_logx;
			}
			for(int ic=0; ic<data.IC; ic++) {
				pvar.z_tk.get_object(t,auxI+param.K+param.UC+ic).rho = data.attr_items_log.get_object(i,ic) \
																	   +pvar.sigma_uk.get_object(u,ic).e_logx;
			}
			for(int ic=0; ic<data.ICV; ic++) {
				pvar.z_tk.get_object(t,auxI+param.K+param.UC+param.IC+ic).rho = data.attr_items_isk_log.get_object(i,s,ic) \
																 		        +pvar.sigma_uk_var.get_object(u,ic).e_logx;
			}
			if(param.flag_day) {
				g_u = data.group_per_user.get_object(u);
				g_i = data.group_per_item.get_object(i);
				d = data.day_per_session.get_object(s);
				pvar.z_tk.get_object(t,auxI+param.K+param.UC+param.IC+param.ICV).rho = pvar.x_gid.get_object(g_u,g_i,d).e_logx;
			}
			if(param.flag_hourly>0) {
				g_u = data.group_per_user.get_object(u);
				g_i = data.group_per_item.get_object(i);
				d = data.weekday_per_session.get_object(s);
				for(int n=0; n<param.flag_hourly; n++) {
					pvar.z_tk.get_object(t,auxI+param.K+param.UC+param.IC+param.ICV+auxD+n).rho = pvar.x_giwn.get_object(g_u,g_i,d,n).e_logx \
																								  -0.5*my_pow2(data.hour_per_session.get_object(s)-param.gauss_mean.get_object(n))/param.gauss_var.get_object(n) \
																								  -0.5*my_log(2.0*my_pi()*param.gauss_var.get_object(n));
				}
			}
		}
		pvar.update_expectations_z();
	}	

	static void update_shape_hier(hpf_data &data, const hpf_param &param, const hpf_hyper &hyper, hpf_pvar &pvar) {
		for(int u=0; u<data.Nusers; u++) {
			pvar.xi_u.get_object(u).shp = hyper.ap+param.K*hyper.a+(param.IC+param.ICV)*hyper.e;
		}
		update_expectations_gamma(pvar.xi_u);
		if(param.IChier==0) {
			int aux_intercept = 0;
			if(param.flag_itemIntercept) {
				aux_intercept = 1;
			}
			for(int i=0; i<data.Nitems; i++) {
				pvar.eta_i.get_object(i).shp = hyper.cp+(param.K+aux_intercept)*hyper.c+param.UC*hyper.f;
			}
			update_expectations_gamma(pvar.eta_i);
		}
	}

	static void update_xi_rte(hpf_data &data, const hpf_param &param, const hpf_hyper &hyper, hpf_pvar &pvar) {
		double suma;
		for(int u=0; u<data.Nusers; u++) {
			suma = hyper.ap/hyper.bp;
			for(int k=0; k<param.K; k++) {
				suma += pvar.theta_uk.get_object(u,k).e_x;
			}
			for(int ic=0; ic<param.IC; ic++) {
				suma += data.attr_items_scaleF.get_object(ic)*pvar.sigma_uk.get_object(u,ic).e_x;
			}
			for(int ic=0; ic<param.ICV; ic++) {
				suma += data.attr_items_isk_scaleF.get_object(ic)*pvar.sigma_uk_var.get_object(u,ic).e_x;
			}
			pvar.xi_u.get_object(u).rte = suma;
		}
		update_expectations_gamma(pvar.xi_u);
	}

	static void update_eta_rte(hpf_data &data, const hpf_param &param, const hpf_hyper &hyper, hpf_pvar &pvar) {
		double suma;
		for(int i=0; i<data.Nitems; i++) {
			suma = hyper.cp/hyper.dp;
			for(int k=0; k<param.K; k++) {
				suma += pvar.beta_ik.get_object(i,k).e_x;
			}
			if(param.flag_itemIntercept) {
				suma += pvar.beta_i0.get_object(i).e_x;
			}
			for(int uc=0; uc<param.UC; uc++) {
				suma += data.attr_users_scaleF.get_object(uc)*pvar.rho_ik.get_object(i,uc).e_x;
			}

			pvar.eta_i.get_object(i).rte = suma;
		}
		update_expectations_gamma(pvar.eta_i);
	}

	static void update_xday(hpf_data &data, const hpf_param &param, const hpf_hyper &hyper, hpf_pvar &pvar) {
		double new_shp;
		double new_rte;
		int t;
		double y;
		int auxI = 0;
		if(param.flag_itemIntercept) {
			auxI = 1;
		}

		// Update xday
		for(int g_u=0; g_u<data.NuserGroups; g_u++) {
			for(int g_i=0; g_i<data.NitemGroups; g_i++) {
				for(int d=0; d<data.Ndays; d++) {
					new_shp = hyper.a_x;
					for(std::vector<int>::iterator iter=data.lines_per_xday.get_object(g_u,g_i,d).begin(); iter!=data.lines_per_xday.get_object(g_u,g_i,d).end(); ++iter) {
						t = *iter;
						y = static_cast<double>(data.obs.y_rating[t]);
						new_shp += y*pvar.z_tk.get_object(t,auxI+param.K+param.UC+param.IC+param.ICV).phi;
					}
					new_rte = hyper.b_x+data.auxsumrte_xday.get_object(g_u,g_i,d);
					pvar.x_gid.get_object(g_u,g_i,d).rte = new_rte;
					pvar.x_gid.get_object(g_u,g_i,d).shp = new_shp;
				}
			}
		}

		// Update expectations
		update_expectations_gamma(pvar.x_gid);
	}
	
	static void update_xhourly(hpf_data &data, const hpf_param &param, const hpf_hyper &hyper, hpf_pvar &pvar) {
		double new_shp;
		double new_rte;
		int t;
		double y;
		int auxI = 0;
		int auxD = 0;
		if(param.flag_day) {
			auxD = 1;
		}
		if(param.flag_itemIntercept) {
			auxI = 1;
		}

		// Update xday
		for(int g_u=0; g_u<data.NuserGroups; g_u++) {
			for(int g_i=0; g_i<data.NitemGroups; g_i++) {
				for(int d=0; d<data.Nweekdays; d++) {
					for(int n=0; n<param.flag_hourly; n++) {
						new_shp = hyper.a_x;
						for(std::vector<int>::iterator iter=data.lines_per_weekday.get_object(g_u,g_i,d).begin(); iter!=data.lines_per_weekday.get_object(g_u,g_i,d).end(); ++iter) {
							t = *iter;
							y = static_cast<double>(data.obs.y_rating[t]);
							new_shp += y*pvar.z_tk.get_object(t,auxI+param.K+param.UC+param.IC+param.ICV+auxD+n).phi;
						}
						new_rte = hyper.b_x+data.auxsumrte_xhourly.get_object(g_u,g_i,d,n);
						pvar.x_giwn.get_object(g_u,g_i,d,n).rte = new_rte;
						pvar.x_giwn.get_object(g_u,g_i,d,n).shp = new_shp;
					}
				}
			}
		}

		// Update expectations
		update_expectations_gamma(pvar.x_giwn);
	}

	static void update_sigma(hpf_data &data, const hpf_param &param, const hpf_hyper &hyper, hpf_pvar &pvar) {
		double new_shp;
		double new_rte;
		int t;
		double y;
		int auxI = 0;
		if(param.flag_itemIntercept) {
			auxI = 1;
		}

		// Update sigma
		for(int u=0; u<data.Nusers; u++) {
			for(int ic=0; ic<param.IC; ic++) {
				new_shp = hyper.e;
				for(std::vector<int>::iterator iter=data.lines_per_user.get_object(u).begin(); iter!=data.lines_per_user.get_object(u).end(); ++iter) {
					t = *iter;
					y = static_cast<double>(data.obs.y_rating[t]);
					new_shp += y*pvar.z_tk.get_object(t,auxI+param.K+param.UC+ic).phi;
				}
				new_rte = data.auxsumrte_ic.get_object(u,ic);
				new_rte += data.attr_items_scaleF.get_object(ic)*pvar.xi_u.get_object(u).e_x;
				pvar.sigma_uk.get_object(u,ic).shp = new_shp;
				pvar.sigma_uk.get_object(u,ic).rte = new_rte;
			}
		}

		// Update expectations
		update_expectations_gamma(pvar.sigma_uk);
	}

	static void update_sigma_var(hpf_data &data, const hpf_param &param, const hpf_hyper &hyper, hpf_pvar &pvar) {
		double new_shp;
		double new_rte;
		int t;
		double y;
		int auxI = 0;
		if(param.flag_itemIntercept) {
			auxI = 1;
		}

		// Update sigma_var
		for(int u=0; u<data.Nusers; u++) {
			for(int ic=0; ic<param.ICV; ic++) {
				new_shp = hyper.e;
				for(std::vector<int>::iterator iter=data.lines_per_user.get_object(u).begin(); iter!=data.lines_per_user.get_object(u).end(); ++iter) {
					t = *iter;
					y = static_cast<double>(data.obs.y_rating[t]);
					new_shp += y*pvar.z_tk.get_object(t,auxI+param.K+param.UC+param.IC+ic).phi;
				}
				new_rte = data.auxsumrte_icv.get_object(u,ic);
				new_rte += data.attr_items_isk_scaleF.get_object(ic)*pvar.xi_u.get_object(u).e_x;
				pvar.sigma_uk_var.get_object(u,ic).shp = new_shp;
				pvar.sigma_uk_var.get_object(u,ic).rte = new_rte;
			}
		}

		// Update expectations
		update_expectations_gamma(pvar.sigma_uk_var);
	}

	static void update_theta(hpf_data &data, const hpf_param &param, const hpf_hyper &hyper, hpf_pvar &pvar) {
		double new_shp;
		double new_rte;
		int t;
		double y;
		double suma;
		hpf_avail_aux aAux;
		int auxI = 0;
		if(param.flag_itemIntercept) {
			auxI = 1;
		}

		// Auxiliary variables
		Matrix1D<double> sum_e_beta = Matrix1D<double>(param.K);
		for(int k=0; k<param.K; k++) {
			suma = 0.0;
			for(int i=0; i<data.Nitems; i++) {
				suma += pvar.beta_ik.get_object(i,k).e_x;
			}
			sum_e_beta.set_object(k,suma);
		}

		// Update theta
		for(int u=0; u<data.Nusers; u++) {
			for(int k=0; k<param.K; k++) {
				new_shp = hyper.a;
				for(std::vector<int>::iterator iter=data.lines_per_user.get_object(u).begin(); iter!=data.lines_per_user.get_object(u).end(); ++iter) {
					t = *iter;
					y = static_cast<double>(data.obs.y_rating[t]);
					new_shp += y*pvar.z_tk.get_object(t,auxI+k).phi;
				}
				new_rte = sum_e_beta.get_object(k);
				if(param.flag_session) {
					new_rte *= static_cast<double>(data.sessions_per_user.get_object(u).size());
				}
				if(param.flag_availability) {
					// run over all items that were UNavailable for at least one trip of user u
					for(std::vector<hpf_avail_aux>::iterator iter=data.availabilityNegCountsUser.get_object(u).begin(); iter!=data.availabilityNegCountsUser.get_object(u).end(); ++iter) {
						aAux = *iter;
						new_rte -= aAux.value*pvar.beta_ik.get_object(aAux.index,k).e_x;
					}
				}
				new_rte += pvar.xi_u.get_object(u).e_x;
				pvar.theta_uk.get_object(u,k).shp = new_shp;
				pvar.theta_uk.get_object(u,k).rte = new_rte;
			}
		}

		// Update expectations
		update_expectations_gamma(pvar.theta_uk);
	}

	static void update_rho(hpf_data &data, const hpf_param &param, const hpf_hyper &hyper, hpf_pvar &pvar) {
		double new_shp;
		double new_rte;
		int t;
		double y;
		int auxI = 0;
		if(param.flag_itemIntercept) {
			auxI = 1;
		}

		// Update rho
		for(int i=0; i<data.Nitems; i++) {
			for(int uc=0; uc<param.UC; uc++) {
				new_shp = hyper.f;
				for(std::vector<int>::iterator iter=data.lines_per_item.get_object(i).begin(); iter!=data.lines_per_item.get_object(i).end(); ++iter) {
					t = *iter;
					y = static_cast<double>(data.obs.y_rating[t]);
					new_shp += y*pvar.z_tk.get_object(t,auxI+param.K+uc).phi;
				}
				new_rte = data.auxsumrte_uc.get_object(i,uc);
				new_rte += data.attr_users_scaleF.get_object(uc)*pvar.eta_i.get_object(i).e_x;
				pvar.rho_ik.get_object(i,uc).shp = new_shp;
				pvar.rho_ik.get_object(i,uc).rte = new_rte;
			}
		}

		// Update expectations
		update_expectations_gamma(pvar.rho_ik);
	}

	static void update_eta_k(hpf_data &data, const hpf_param &param, const hpf_hyper &hyper, hpf_pvar &pvar) {
		double new_shp;
		double new_rte;
		double aux_rte;

		// Update eta_k
		for(int k=0; k<param.K; k++) {
			new_shp = hyper.eta_k_shp + static_cast<double>(data.Nitems)*hyper.c;
			new_rte = hyper.eta_k_rte;
			for(int i=0; i<data.Nitems; i++) {
				aux_rte = pvar.beta_ik.get_object(i,k).e_x;
				for(int ll=0; ll<param.IChier; ll++) {
					if(data.attr_items_hier.get_object(i,ll)) {
						aux_rte *= pvar.h_lk.get_object(ll,k).e_x;
					}
				}
				new_rte += aux_rte;
			}
			pvar.eta_k.get_object(k).shp = new_shp;
			pvar.eta_k.get_object(k).rte = new_rte;
		}

		// Update expectations
		update_expectations_gamma(pvar.eta_k);
	}

	static void update_h_lk(hpf_data &data, const hpf_param &param, const hpf_hyper &hyper, hpf_pvar &pvar) {
		double new_shp;
		double new_rte;
		double aux_rte;

		// Update h_lk
		for(int ll=0; ll<param.IChier; ll++) {
			for(int k=0; k<param.K; k++) {
				new_shp = hyper.h_lk_shp;
				new_rte = hyper.h_lk_rte;
				for(int i=0; i<data.Nitems; i++) {
					if(data.attr_items_hier.get_object(i,ll)) {
						aux_rte = pvar.beta_ik.get_object(i,k).e_x * pvar.eta_k.get_object(k).e_x;
						for(int ll_prime=0; ll_prime<param.IChier; ll_prime++) {
							if(ll!=ll_prime && data.attr_items_hier.get_object(i,ll_prime)) {
								aux_rte *= pvar.h_lk.get_object(ll_prime,k).e_x;
							}
						}
						new_shp += hyper.c;
						new_rte += aux_rte;
					}
				}
				pvar.h_lk.get_object(ll,k).shp = new_shp;
				pvar.h_lk.get_object(ll,k).rte = new_rte;

				// Update expectations
				update_expectations_gamma(pvar.h_lk,ll,k);
			}
		}
	}

	static void update_beta(hpf_data &data, const hpf_param &param, const hpf_hyper &hyper, hpf_pvar &pvar) {
		double new_shp;
		double new_rte;
		int t;
		double y;
		double suma;
		hpf_avail_aux aAux;
		int auxI = 0;
		if(param.flag_itemIntercept) {
			auxI = 1;
		}

		// Auxiliary variables
		Matrix1D<double> sum_e_theta = Matrix1D<double>(param.K);
		for(int k=0; k<param.K; k++) {
			suma = 0.0;
			for(int u=0; u<data.Nusers; u++) {
				if(param.flag_session) {
					suma += pvar.theta_uk.get_object(u,k).e_x*data.sessions_per_user.get_object(u).size();
				} else {
					suma += pvar.theta_uk.get_object(u,k).e_x;
				}
			}
			sum_e_theta.set_object(k,suma);
		}

		// Update beta
		for(int i=0; i<data.Nitems; i++) {
			for(int k=0; k<param.K; k++) {
				new_shp = hyper.c;
				for(std::vector<int>::iterator iter=data.lines_per_item.get_object(i).begin(); iter!=data.lines_per_item.get_object(i).end(); ++iter) {
					t = *iter;
					y = static_cast<double>(data.obs.y_rating[t]);
					new_shp += y*pvar.z_tk.get_object(t,auxI+k).phi;
				}
				new_rte = sum_e_theta.get_object(k);
				if(param.flag_availability) {
					// run over all users that found UNavailable item i at least once
					for(std::vector<hpf_avail_aux>::iterator iter=data.availabilityNegCountsItem.get_object(i).begin(); iter!=data.availabilityNegCountsItem.get_object(i).end(); ++iter) {
						aAux = *iter;
						new_rte -= aAux.value*pvar.theta_uk.get_object(aAux.index,k).e_x;
					}
				}
				if(param.IChier==0) {
					new_rte += pvar.eta_i.get_object(i).e_x;
				} else {
					double aux_rte = pvar.eta_k.get_object(k).e_x;
					for(int ll=0; ll<param.IChier; ll++) {
						if(data.attr_items_hier.get_object(i,ll)) {
							aux_rte *= pvar.h_lk.get_object(ll,k).e_x;
						}
					}
					new_rte += aux_rte;
				}
				pvar.beta_ik.get_object(i,k).shp = new_shp;
				pvar.beta_ik.get_object(i,k).rte = new_rte;
			}
		}

		// Update expectations
		update_expectations_gamma(pvar.beta_ik);
	}

	static void update_beta0(hpf_data &data, const hpf_param &param, const hpf_hyper &hyper, hpf_pvar &pvar) {
		double new_shp;
		double new_rte;
		int t;
		double y;
		hpf_avail_aux aAux;

		// Auxiliary variables
		double sum_e_theta = 0.0;
		for(int u=0; u<data.Nusers; u++) {
			if(param.flag_session) {
				sum_e_theta += data.sessions_per_user.get_object(u).size();
			} else {
				sum_e_theta += 1.0;
			}
		}

		// Update beta0
		for(int i=0; i<data.Nitems; i++) {
			new_shp = hyper.c;
			for(std::vector<int>::iterator iter=data.lines_per_item.get_object(i).begin(); iter!=data.lines_per_item.get_object(i).end(); ++iter) {
				t = *iter;
				y = static_cast<double>(data.obs.y_rating[t]);
				new_shp += y*pvar.z_tk.get_object(t,0).phi;
			}
			new_rte = sum_e_theta;
			if(param.flag_availability) {
				// run over all users that found UNavailable item i at least once
				for(std::vector<hpf_avail_aux>::iterator iter=data.availabilityNegCountsItem.get_object(i).begin(); iter!=data.availabilityNegCountsItem.get_object(i).end(); ++iter) {
					aAux = *iter;
					new_rte -= aAux.value;
				}
			}
			if(param.IChier==0) {
				new_rte += pvar.eta_i.get_object(i).e_x;
			} else {
				new_rte += hyper.c;
			}
	        pvar.beta_i0.get_object(i).shp = new_shp;
	        pvar.beta_i0.get_object(i).rte = new_rte;
		}

		// Update expectations
		update_expectations_gamma(pvar.beta_i0);
	}	

	static void update_expectations_gamma(Matrix4D<hpf_gamma> &m) {
		for(int u=0; u<m.get_size1(); u++) {
			for(int k=0; k<m.get_size2(); k++) {
				for(int v=0; v<m.get_size3(); v++) {
					for(int p=0; p<m.get_size4(); p++) {
						m.get_object(u,k,v,p).update_expectations();
					}
				}
			}
		}
	}

	static void update_expectations_gamma(Matrix3D<hpf_gamma> &m) {
		for(int u=0; u<m.get_size1(); u++) {
			for(int k=0; k<m.get_size2(); k++) {
				for(int v=0; v<m.get_size3(); v++) {
					m.get_object(u,k,v).update_expectations();
				}
			}
		}
	}

	static void update_expectations_gamma(Matrix2D<hpf_gamma> &m) {
		for(int u=0; u<m.get_size1(); u++) {
			for(int k=0; k<m.get_size2(); k++) {
				m.get_object(u,k).update_expectations();
			}
		}
	}

	static void update_expectations_gamma(Matrix2D<hpf_gamma> &m, int u, int k) {
		m.get_object(u,k).update_expectations();
	}

	static void update_expectations_gamma(Matrix1D<hpf_gamma> &m) {
		for(int u=0; u<m.get_size1(); u++) {
			m.get_object(u).update_expectations();
		}
	}

	static double compute_likelihood(const string &filename, double duration, hpf_data &data, const hpf_data_aux &data_aux, hpf_param &param, const hpf_hyper &hyper, hpf_pvar &pvar) {
	    if(param.noVal && filename == "validation") {
	      return 0.0;
	    }
	    if(param.noTest && filename == "test") {
	      return 0.0;
	    }

		int T = data_aux.T;
		double llh = 0.0;
		double mse = 0.0;
		double llh_fair = 0.0;
		double llh_fair_binary = 0.0;
		int count_nitems = 0;
		double poissmean;
		double y;
		int u;
		int i;
		int s;
		double *p_item = new double[data.Nitems];
		double sum_norm;

		string fname = param.outdir+"/"+filename+".txt";

		// For each line of the validation/test file
		for(int t=0; t<T; t++) {
			u = data_aux.y_user[t];
			i = data_aux.y_item[t];
			s = data_aux.y_sess[t];
			y = static_cast<double>(data_aux.y_rating[t]);
			// Compute the mean of the Poisson
			poissmean = compute_poisson_mean(data,param,hyper,pvar,u,i,s);
			// Truncate
			poissmean = (poissmean<1e-30)?1e-30:poissmean;
			// (A) Evaluate the Poisson likelihood
			llh += y*my_log(poissmean)-poissmean-my_gsl_sf_lngamma(y+1.0);
			mse += my_pow2(poissmean-y);
			// (B) Compute the "fair" llh (requires iterate over all items j)
			sum_norm = 0.0;
			for(int j=0; j<data.Nitems; j++) {
				p_item[j] = compute_poisson_mean(data,param,hyper,pvar,u,j,s);
				sum_norm += p_item[j];
			}
			// Normalize p_item
			for(int j=0; j<data.Nitems; j++) {
				p_item[j] /= sum_norm;
			}
			// Compute llh_fair
			llh_fair += y*my_log(p_item[i]);
			llh_fair_binary += my_log(p_item[i]);
			// Increase count to compute average llh later
			count_nitems += data_aux.y_rating[t];
		}

		// Obtain the average
		llh /= static_cast<double>(T);
		llh_fair /= static_cast<double>(count_nitems);
		llh_fair_binary /= static_cast<double>(T);
		mse /= static_cast<double>(T);
		// Print to file
		char buffer[300];
    	sprintf(buffer,"%d\t%d\t%.9f\t%.9f\t%.9f\t%.9f\t%d",param.it,static_cast<int>(duration),llh,llh_fair,llh_fair_binary,mse,T);
        hpf_output::write_line(fname,string(buffer));
        // Free memory
		delete [] p_item;
		// Return
		return llh;
	}

	static double compute_poisson_mean(hpf_data &data, hpf_param &param, const hpf_hyper &hyper, hpf_pvar &pvar, int u, int i, int s) {
		int g_u;
		int g_i;
		int d;
		int w;
		double hh;

		double poissmean = 0.0;
		if(param.flag_itemIntercept) {
			poissmean += pvar.beta_i0.get_object(i).e_x;
		}
		for(int k=0; k<param.K; k++) {
			poissmean += pvar.theta_uk.get_object(u,k).e_x\
						 *pvar.beta_ik.get_object(i,k).e_x;
		}
		for(int k=0; k<param.UC; k++) {
			poissmean += pvar.rho_ik.get_object(i,k).e_x\
						 *data.attr_users.get_object(u,k);
		}
		for(int k=0; k<param.IC; k++) {
			poissmean += pvar.sigma_uk.get_object(u,k).e_x\
						 *data.attr_items.get_object(i,k);
		}
		for(int k=0; k<param.ICV; k++) {
			poissmean += pvar.sigma_uk_var.get_object(u,k).e_x\
						 *data.attr_items_isk.get_object(i,k,s);
		}
		if(param.flag_day) {
			g_u = data.group_per_user.get_object(u);
			g_i = data.group_per_item.get_object(i);
			d = data.day_per_session.get_object(s);
			poissmean += pvar.x_gid.get_object(g_u,g_i,d).e_x;
		}
		if(param.flag_hourly>0) {
			g_u = data.group_per_user.get_object(u);
			g_i = data.group_per_item.get_object(i);
			w = data.weekday_per_session.get_object(s);
			hh = data.hour_per_session.get_object(s);
			for(int n=0; n<param.flag_hourly; n++) {
				poissmean += pvar.x_giwn.get_object(g_u,g_i,w,n).e_x \
				             *my_exp(-0.5*my_pow2(hh-param.gauss_mean.get_object(n))/param.gauss_var.get_object(n))/sqrt(2.0*my_pi()*param.gauss_var.get_object(n));
			}
		}
		// Return
		return poissmean;
	}
};

#endif
