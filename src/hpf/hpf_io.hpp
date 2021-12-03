#ifndef HPF_IO_HPP
#define HPF_IO_HPP

class hpf_input {
public:
	static void read_input_from_command_line(int argc, char **argv, hpf_data &data, hpf_param &param, hpf_hyper &hyper) {
	    int i = 0;
	    string val;
	    while(i<=argc-1) {
	        if(strcmp(argv[i], "-dir") == 0) {
	            val = string(argv[++i]);
	            val = remove_final_slash(val);
	            param.datadir = val;
	        } else if(strcmp(argv[i], "-outdir") == 0) {
	            val = string(argv[++i]);
	            val = remove_final_slash(val);
	            param.outdir = val;
	        } else if(strcmp(argv[i], "-m") == 0) {
	            val = string(argv[++i]);
	            data.m = std::stoi(val);
	        } else if(strcmp(argv[i], "-n") == 0) {
	            val = string(argv[++i]);
	            data.n = std::stoi(val);
	        } else if(strcmp(argv[i], "-k") == 0) {
	            val = string(argv[++i]);
	            param.K = std::stoi(val);
	            if(param.K>0) {
				    hyper.a = 1.0/sqrt(param.K);
				    hyper.c = 1.0/sqrt(param.K);
				    hyper.e = 1.0/sqrt(param.K);
				    hyper.f = 1.0/sqrt(param.K);
	            }
	        } else if(strcmp(argv[i], "-uc") == 0) {
	            val = string(argv[++i]);
	            data.UC = std::stoi(val);
	            param.UC = std::stoi(val);
	        } else if(strcmp(argv[i], "-ic") == 0) {
	            val = string(argv[++i]);
	            data.IC = std::stoi(val);
	            param.IC = std::stoi(val);
	        } else if(strcmp(argv[i], "-icVar") == 0) {
	            val = string(argv[++i]);
	            data.ICV = std::stoi(val);
	            param.ICV = std::stoi(val);
	            if(!param.flag_session && data.ICV>0) {
	            	param.flag_session = true;
	        		std::cerr << "[WARN] Implicityly using '-session' option because '-icVar " << param.ICV << "' was found" << endl;
	            }
	        } else if(strcmp(argv[i], "-icHier") == 0) {
	            val = string(argv[++i]);
	            data.IChier = std::stoi(val);
	            param.IChier = std::stoi(val);
	        } else if(strcmp(argv[i], "-nIterIni") == 0) {
	            val = string(argv[++i]);
	        	param.nIterIni = std::stoi(val);
	        } else if(strcmp(argv[i], "-session") == 0) {
	        	param.flag_session = true;
	        } else if(strcmp(argv[i], "-availability") == 0) {
	        	if(!param.flag_session) {
	        		std::cerr << "[WARN] Implicityly using '-session' option because '-availability' was found" << endl;
	        	}
	            param.flag_session = true;
	            param.flag_availability = true;
	        } else if(strcmp(argv[i], "-itemIntercept") == 0) {
	        	param.flag_itemIntercept = true;
	        } else if(strcmp(argv[i], "-days") == 0) {
	            param.flag_day = true;
	        	if(!param.flag_session) {
	        		std::cerr << "[WARN] Implicityly using '-session' option because '-days' was found" << endl;
		            param.flag_session = true;
	        	}
	        } else if(strcmp(argv[i], "-hourly") == 0) {
	            val = string(argv[++i]);
	            param.flag_hourly = std::stoi(val);
	        	if(!param.flag_session && param.flag_hourly>0) {
	        		std::cerr << "[WARN] Implicityly using '-session' option because '-hourly " << param.flag_hourly << "' was found" << endl;
		            param.flag_session = true;
	        	}
	        	// Also set the means and variances
	        	if(param.flag_hourly>0) {
	        		param.gauss_mean = Matrix1D<double>(param.flag_hourly);
	        		param.gauss_var = Matrix1D<double>(param.flag_hourly);
		        	for(int n=0; n<param.flag_hourly; n++) {
		        		param.gauss_mean.set_object(n,24.0*(1.0+n)/(1.0+param.flag_hourly));
		        		param.gauss_var.set_object(n,10.0);
		        	}
	        	}
	        } else if(strcmp(argv[i], "-gaussMeans") == 0) {
	        	for(int n=0; n<param.flag_hourly; n++) {
		            val = string(argv[++i]);
		            param.gauss_mean.set_object(n,std::stod(val));
	            }
	        } else if(strcmp(argv[i], "-gaussVars") == 0) {
	        	for(int n=0; n<param.flag_hourly; n++) {
		            val = string(argv[++i]);
		            param.gauss_var.set_object(n,std::stod(val));
	            }
	        } else if(strcmp(argv[i], "-seed") == 0) {
	            val = string(argv[++i]);
	        	param.seed = std::stoi(val);
	        } else if(strcmp(argv[i], "-rfreq") == 0) {
	            val = string(argv[++i]);
	        	param.rfreq = std::stoi(val);
	        } else if(strcmp(argv[i], "-max-iterations") == 0) {
	            val = string(argv[++i]);
	        	param.Niter = std::stoi(val);
	        } else if(strcmp(argv[i], "-nooffset") == 0) {
	            param.ini_offset = 0.0;
	        } else if(strcmp(argv[i], "-offset") == 0) {
	            val = string(argv[++i]);
	            param.ini_offset = std::stod(val);
	        } else if(strcmp(argv[i], "-std") == 0) {
	            param.factor_obs_std = true;
	            // (scales down observables by their standard deviation, instead of by their mean)
	        } else if(strcmp(argv[i], "-ones") == 0) {
	            param.factor_obs_ones = true;
	            // (does not scale down observables)
	        } else if(strcmp(argv[i], "-scfact") == 0) {
	            val = string(argv[++i]);
	            param.factor_obs_extra = std::stod(val);
	        } else if(strcmp(argv[i], "-lfirst") == 0) {
	            param.flag_lfirst = true;
	        } else if(strcmp(argv[i], "-ofirst") == 0) {
	            param.flag_ofirst = true;
			} else if(strcmp(argv[i], "-quiet") == 0) {
				param.quiet= true;
			} else if(strcmp(argv[i], "-noVal") == 0) {
				param.noVal = true;
			} else if(strcmp(argv[i], "-noTest") == 0) {
				param.noTest = true;
	        } else if(strcmp(argv[i], "-label") == 0) {
	            val = string(argv[++i]);
	            param.label = val;
	        } else if(strcmp(argv[i], "-a") == 0) {
	            val = string(argv[++i]);
	            hyper.a = std::stod(val);
	        } else if(strcmp(argv[i], "-bp") == 0) {
	            val = string(argv[++i]);
	            hyper.bp = std::stod(val);
	        } else if(strcmp(argv[i], "-c") == 0) {
	            val = string(argv[++i]);
	            hyper.c = std::stod(val);
	        } else if(strcmp(argv[i], "-dp") == 0) {
	            val = string(argv[++i]);
	            hyper.dp = std::stod(val);
	        } else if(strcmp(argv[i], "-e") == 0) {
	            val = string(argv[++i]);
	            hyper.e = std::stod(val);
	        } else if(strcmp(argv[i], "-f") == 0) {
	            val = string(argv[++i]);
	            hyper.f = std::stod(val);
	        } else if(strcmp(argv[i], "-ap") == 0) {
	            val = string(argv[++i]);
	            hyper.ap = std::stod(val);
	        } else if(strcmp(argv[i], "-cp") == 0) {
	            val = string(argv[++i]);
	            hyper.cp = std::stod(val);
	        } else if(strcmp(argv[i], "-ax") == 0) {
	            val = string(argv[++i]);
	            hyper.a_x = std::stod(val);
	        } else if(strcmp(argv[i], "-bx") == 0) {
	            val = string(argv[++i]);
	            hyper.b_x = std::stod(val);
	        } else if(i>0) {
	            fprintf(stdout,"[ERR] Unknown option %s\n", argv[i]);
	            assert(0);
	        } 
	        i++;
	    };
	}

    static string remove_final_slash(const string f) {
        string aux;
        if(f.back()=='/') {
            aux = f.substr(0,f.size()-1);
        } else {
            aux = f;
        }
        return aux;
    }

    static void read_data_file(hpf_data &data, const hpf_param &param) {
		string trainFname = param.datadir+"/train.tsv";
		string testFname = param.datadir+"/test.tsv";
		string validationFname = param.datadir+"/validation.tsv";
		unsigned int NiTr;
		unsigned int NuTr;
		unsigned int NsTr;
		unsigned int NdTr;

		// This is a 2-step process
		// Step 1: Count #users, #items (and #sessions, #days)
		std::cout << " +Stage 1/2..." << endl;
		int n_u = 0;	// Auxiliary index to last user that was read
		int n_i = 0;	// Auxiliary index to last item that was read
		int n_s = 0;	// Auxiliary index to last session that was read
		data.obs.T = count_input_tsv_file(trainFname,data,param,n_u,n_i,n_s);
		NuTr = data.user_ids.size();
		NiTr = data.item_ids.size();
		NsTr = data.session_ids.size();
		NdTr = data.day_ids.size();
		if(!param.noTest) {
			data.obs_test.T = count_input_tsv_file(testFname,data,param,n_u,n_i,n_s);
		}
		if(!param.noVal) {
			data.obs_val.T = count_input_tsv_file(validationFname,data,param,n_u,n_i,n_s);
		}
		if(data.user_ids.size()>NuTr) {
			std::cerr << "[WARN] There are users in test/validation that do not appear in train" << endl;
		}
		if(data.item_ids.size()>NiTr) {
			std::cerr << "[WARN] There are items in test/validation that do not appear in train" << endl;
		}
		if(data.session_ids.size()>NsTr) {
			std::cerr << "[WARN] There are sessions in test/validation that do not appear in train" << endl;
		}
		if(data.day_ids.size()>NdTr) {
			std::cerr << "[WARN] There are day identifiers in test/validation that do not appear in train" << endl;
		}
		data.Nusers = data.user_ids.size();
		data.Nitems = data.item_ids.size();
		data.Nsessions = data.session_ids.size();
		data.Ndays = data.day_ids.size();

		// Verify that the #users and items coincide with the provided by the user as input
		if(data.n>=0) {
			if(data.Nusers!=data.n) {
				std::cerr << "[ERR] Read " << data.Nusers << " users; expected n=" << data.n << endl;
				assert(0);
			}
		}
		if(data.m>=0) {
			if(data.Nitems!=data.m) {
				std::cerr << "[ERR] Read " << data.Nitems << " items; expected m=" << data.m << endl;
				assert(0);
			}
		}

		// Allocate memory to store data
		data.obs = hpf_data_aux(data.obs.T);
		data.obs_test = hpf_data_aux(data.obs_test.T);
		data.obs_val = hpf_data_aux(data.obs_val.T);

		// Step 2: Read the actual data
		std::cout << " +Stage 2/2..." << endl;
		read_input_tsv_file(trainFname,data,data.obs,param);
		if(!param.noTest) {
			read_input_tsv_file(testFname,data,data.obs_test,param);
		}
		if(!param.noVal) {
			read_input_tsv_file(validationFname,data,data.obs_val,param);
		}

		// Also create variable sessions_per_user
	  	if(param.flag_session) {
	  		data.sessions_per_user = Matrix1D<std::vector<int>>(data.Nusers);
	  		int u;
	  		int s;
	  		for(unsigned int t=0; t<data.obs.T; t++) {
	  			u = data.obs.y_user[t];
	  			s = data.obs.y_sess[t];
	  			if(std::find(data.sessions_per_user.get_object(u).begin(),data.sessions_per_user.get_object(u).end(),s) == data.sessions_per_user.get_object(u).end()) {
		  			// If this session isn't in the list, append
	  				data.sessions_per_user.get_object(u).push_back(s);
	  			}
	  		}
	  	}
	}

	static void compute_auxsumrte_xday(hpf_data &data, const hpf_param &param) {
		double mm;
		int u;
		int s;
		int i;

		data.auxsumrte_xday = Matrix3D<double>(data.NuserGroups,data.NitemGroups,data.Ndays);
		if(!param.flag_availability) {
			// everything is always available
			for(int g_u=0; g_u<data.NuserGroups; g_u++) {
				for(int d=0; d<data.Ndays; d++) {
					mm = 0.0;
					// accumulate how many times an user from group g_u goes to the store in day d
					for(std::vector<int>::iterator iter=data.users_per_group.get_object(g_u).begin(); iter!=data.users_per_group.get_object(g_u).end(); ++iter) {
						// for all users in group g_u
						u = *iter;
						for(std::vector<int>::iterator iterSess=data.sessions_per_user.get_object(u).begin(); iterSess!=data.sessions_per_user.get_object(u).end(); ++iterSess) {
							// for all sessions in which user u visits the supermarket
							s = *iterSess;
							if(data.day_per_session.get_object(s)==d) {
								mm += 1.0;
							}
						}
					}
					for(int g_i=0; g_i<data.NitemGroups; g_i++) {
						data.auxsumrte_xday.set_object(g_u,g_i,d,mm*data.items_per_group.get_object(g_i).size());
					}
				}
			}
		} else {
			// availability
			for(int g_u=0; g_u<data.NuserGroups; g_u++) {
				for(int d=0; d<data.Ndays; d++) {
					for(int g_i=0; g_i<data.NitemGroups; g_i++) {
						mm = 0.0;
						for(std::vector<int>::iterator iterItem=data.items_per_group.get_object(g_i).begin(); iterItem!=data.items_per_group.get_object(g_i).end(); ++iterItem) {
							// for all items in group g_i
							i = *iterItem;
							// accumulate how many times an user from group g_u goes to the store in day d and finds item i available
							for(std::vector<int>::iterator iterUser=data.users_per_group.get_object(g_u).begin(); iterUser!=data.users_per_group.get_object(g_u).end(); ++iterUser) {
								// for all users in group g_u
								u = *iterUser;
								for(std::vector<int>::iterator iterSess=data.sessions_per_user.get_object(u).begin(); iterSess!=data.sessions_per_user.get_object(u).end(); ++iterSess) {
									// for all sessions in which user u visits the supermarket
									s = *iterSess;
									if(data.day_per_session.get_object(s)==d) {
										if(data.isavailable(i,s)) {
											mm += 1.0;
										}
									}
								}
							}
						}
						data.auxsumrte_xday.set_object(g_u,g_i,d,mm);
					}
				}
			}
		}
	}

	static void compute_auxsumrte_xhourly(hpf_data &data, hpf_param &param) {
		double *mm = new double[param.flag_hourly];
		int u;
		int s;
		int i;

		data.auxsumrte_xhourly = Matrix4D<double>(data.NuserGroups,data.NitemGroups,data.Nweekdays,param.flag_hourly);
		if(!param.flag_availability) {
			// everything is always available
			for(int g_u=0; g_u<data.NuserGroups; g_u++) {
				for(int d=0; d<data.Nweekdays; d++) {
					// Initialize the auxiliary variable mm
					for(int n=0; n<param.flag_hourly; n++) {
						mm[n] = 0.0;
					}
					// accumulate how many times an user from group g_u goes to the store in weekday d
					for(std::vector<int>::iterator iter=data.users_per_group.get_object(g_u).begin(); iter!=data.users_per_group.get_object(g_u).end(); ++iter) {
						// for all users in group g_u
						u = *iter;
						for(std::vector<int>::iterator iterSess=data.sessions_per_user.get_object(u).begin(); iterSess!=data.sessions_per_user.get_object(u).end(); ++iterSess) {
							// for all sessions in which user u visits the supermarket
							s = *iterSess;
							if(data.weekday_per_session.get_object(s)==d) {
								for(int n=0; n<param.flag_hourly; n++) {
									mm[n] += my_exp(-0.5*my_pow2(data.hour_per_session.get_object(s)-param.gauss_mean.get_object(n))/param.gauss_var.get_object(n))/sqrt(2.0*my_pi()*param.gauss_var.get_object(n));
								}
							}
						}
					}
					for(int g_i=0; g_i<data.NitemGroups; g_i++) {
						for(int n=0; n<param.flag_hourly; n++) {
							data.auxsumrte_xhourly.set_object(g_u,g_i,d,n,mm[n]*data.items_per_group.get_object(g_i).size());
						}
					}
				}
			}
		} else {
			// availability
			for(int g_u=0; g_u<data.NuserGroups; g_u++) {
				for(int d=0; d<data.Nweekdays; d++) {
					for(int g_i=0; g_i<data.NitemGroups; g_i++) {
						// Initialize the auxiliary variable mm
						for(int n=0; n<param.flag_hourly; n++) {
							mm[n] = 0.0;
						}
						for(std::vector<int>::iterator iterItem=data.items_per_group.get_object(g_i).begin(); iterItem!=data.items_per_group.get_object(g_i).end(); ++iterItem) {
							// for all items in group g_i
							i = *iterItem;
							// accumulate how many times an user from group g_u goes to the store in day d and finds item i available
							for(std::vector<int>::iterator iterUser=data.users_per_group.get_object(g_u).begin(); iterUser!=data.users_per_group.get_object(g_u).end(); ++iterUser) {
								// for all users in group g_u
								u = *iterUser;
								for(std::vector<int>::iterator iterSess=data.sessions_per_user.get_object(u).begin(); iterSess!=data.sessions_per_user.get_object(u).end(); ++iterSess) {
									// for all sessions in which user u visits the supermarket
									s = *iterSess;
									if(data.weekday_per_session.get_object(s)==d) {
										if(data.isavailable(i,s)) {
											for(int n=0; n<param.flag_hourly; n++) {
												mm[n] += my_exp(-0.5*my_pow2(data.hour_per_session.get_object(s)-param.gauss_mean.get_object(n))/param.gauss_var.get_object(n))/sqrt(2.0*my_pi()*param.gauss_var.get_object(n));
											}
										}
									}
								}
							}
						}
						for(int n=0; n<param.flag_hourly; n++) {
							data.auxsumrte_xhourly.set_object(g_u,g_i,d,n,mm[n]);
						}
					}
				}
			}
		}
		delete [] mm;
	}

	static void compute_auxsumrte_obsIC(hpf_data &data, const hpf_param &param) {
		double mm;

		data.auxsumrte_ic = Matrix2D<double>(data.Nusers,data.IC);
		if(!param.flag_session) {
			// No sessions
			for(int ic=0; ic<param.IC; ic++) {
				mm = 0.0;
				for(int i=0; i<data.Nitems; i++) {
					mm += data.attr_items.get_object(i,ic);
				}
				for(int u=0; u<data.Nusers; u++) {
					data.auxsumrte_ic.set_object(u,ic,mm);
				}
			}
		} else if(param.flag_availability) {
			// Sessions + Availability
			for(int u=0; u<data.Nusers; u++) {
				for(int ic=0; ic<param.IC; ic++) {
					mm = 0.0;
					for(int i=0; i<data.Nitems; i++) {
						mm += data.attr_items.get_object(i,ic)*data.availabilityCounts.get_object(u,i);
					}
					data.auxsumrte_ic.set_object(u,ic,mm);
				}
			}
		} else {
			// Sessions but no availability
			for(int ic=0; ic<param.IC; ic++) {
				mm = 0.0;
				for(int i=0; i<data.Nitems; i++) {
					mm += data.attr_items.get_object(i,ic);
				}
				for(int u=0; u<data.Nusers; u++) {
					data.auxsumrte_ic.set_object(u,ic,mm*data.sessions_per_user.get_object(u).size());
				}
			}
		}
	}

	static void compute_auxsumrte_obsICV(hpf_data &data, const hpf_param &param) {
		double mm;
		int s;

		data.auxsumrte_icv = Matrix2D<double>(data.Nusers,data.ICV);
		if(!param.flag_session) {
			std::cerr << "[ERR] This should not happen ---" << \
					     "      Review The function 'hpf_input::compute_auxsumrte_obsICV'" << endl;
	     	assert(0);
		} else if(param.flag_availability) {
			// Sessions + Availability
			for(int u=0; u<data.Nusers; u++) {
				for(int ic=0; ic<param.ICV; ic++) {
					mm = 0.0;
					for(int i=0; i<data.Nitems; i++) {
						for(std::vector<int>::iterator iter=data.sessions_per_user.get_object(u).begin(); iter!=data.sessions_per_user.get_object(u).end(); iter++) {
							s = *iter;
							if(data.isavailable(i,s)) {
								mm += data.attr_items_isk.get_object(i,s,ic);
							}
						}
					}
					data.auxsumrte_icv.set_object(u,ic,mm);
				}
			}
		} else {
			// Sessions but no availability
			for(int u=0; u<data.Nusers; u++) {
				for(int ic=0; ic<param.ICV; ic++) {
					mm = 0.0;
					for(int i=0; i<data.Nitems; i++) {
						for(std::vector<int>::iterator iter=data.sessions_per_user.get_object(u).begin(); iter!=data.sessions_per_user.get_object(u).end(); iter++) {
							s = *iter;
							mm += data.attr_items_isk.get_object(i,s,ic);
						}
					}
					data.auxsumrte_icv.set_object(u,ic,mm);
				}
			}
		}
	}

	static void compute_auxsumrte_obsUC(hpf_data &data, const hpf_param &param) {
		double mm;

		data.auxsumrte_uc = Matrix2D<double>(data.Nitems,data.UC);
		if(!param.flag_session) {
			// No sessions
			for(int uc=0; uc<param.UC; uc++) {
				mm = 0.0;

				for(int u=0; u<data.Nusers; u++) {
					mm += data.attr_users.get_object(u,uc);
				}
				for(int i=0; i<data.Nitems; i++) {
					data.auxsumrte_uc.set_object(i,uc,mm);
				}
			}
		} else if(param.flag_availability) {
			// Sessions + Availability
			for(int i=0; i<data.Nitems; i++) {
				for(int uc=0; uc<param.UC; uc++) {
					mm = 0.0;
					for(int u=0; u<data.Nusers; u++) {
						mm += data.attr_users.get_object(u,uc)*data.availabilityCounts.get_object(u,i);
					}
					data.auxsumrte_uc.set_object(i,uc,mm);
				}
			}
		} else {
			// Sessions but no availability
			for(int uc=0; uc<param.UC; uc++) {
				mm = 0.0;

				for(int u=0; u<data.Nusers; u++) {
					mm += data.attr_users.get_object(u,uc)*data.sessions_per_user.get_object(u).size();
				}
				for(int i=0; i<data.Nitems; i++) {
					data.auxsumrte_uc.set_object(i,uc,mm);
				}
			}
		}
	}

	static void read_a_line(const hpf_param &param, FILE *fin, char *uid, char *mid, char *sid, unsigned int *rating) {
		if(param.flag_session) {
			// session
			fscanf(fin, "%[^\t]\t%[^\t]\t%[^\t]\t%u\n", uid, mid, sid, rating);
		} else if(!param.flag_session) {
			// !session
			fscanf(fin, "%[^\t]\t%[^\t]\t%u\n", uid, mid, rating);
		} else {
			std::cerr << "[ERR] This should not happen ---" << endl << \
					     "      Review function 'read_a_line' in 'hpf_io.cpp'" << endl;
			assert(0);
		}
	}

	static unsigned int count_input_tsv_file(string fname, hpf_data &data, const hpf_param &param, int &n_u, int &n_i, int &n_s) {
		char uid[1000];
		char mid[1000];
		char sid[1000];
		unsigned int rating;
		unsigned int nlines = 0;		// number of lines read

	  	FILE *fin = fopen(fname.c_str(),"r");
	  	if(!fin) {
	  		std::cerr << "[ERR] Unable to open " << fname << endl;
	  		assert(0);
	  	}
		while(!feof(fin)){
			// Read a line
			read_a_line(param,fin,uid,mid,sid,&rating);
			nlines++;
			// Append user id to the list of user id's
			if(data.user_ids.find(std::string(uid))==data.user_ids.end()) {
				data.user_ids.insert(std::pair<string,int>(std::string(uid),n_u));
				n_u++;
			}

			// Append item id to the list of item id's
			if(data.item_ids.find(std::string(mid))==data.item_ids.end()) {
				data.item_ids.insert(std::pair<string,int>(std::string(mid),n_i));
				n_i++;
			}

			// Append session id to the list of session id's
			if(param.flag_session) {
				if(data.session_ids.find(std::string(sid))==data.session_ids.end()) {
					data.session_ids.insert(std::pair<string,int>(std::string(sid),n_s));
					n_s++;
				}
			}
		}
		fclose(fin);
		return(nlines);
	}

	static void read_input_tsv_file(string fname, const hpf_data &data, hpf_data_aux &data_aux, const hpf_param &param) {
		char uid[1000];
		char mid[1000];
		char sid[1000];
		unsigned int rating;
		unsigned int count = 0;

	  	FILE *fin = fopen(fname.c_str(),"r");
	  	if(!fin) {
	  		std::cerr << "[ERR] Unable to open " << fname << endl;
	  		assert(0);
	  	}

		while(!feof(fin)){
			// Read a line
			read_a_line(param,fin,uid,mid,sid,&rating);
			// Store values
			data_aux.y_user[count] = data.user_ids.find(std::string(uid))->second;
			data_aux.y_item[count] = data.item_ids.find(std::string(mid))->second;
			data_aux.y_rating[count] = rating;
			if(param.flag_session) {
				data_aux.y_sess[count] = data.session_ids.find(std::string(sid))->second;
			}
			// Increase count
			count++;
		}
		fclose(fin);
	}

	static void read_obs_attributes(hpf_data &data, const hpf_param &param) {
		// Read obsUser.tsv
		if(param.UC>0) {
			string fname = param.datadir+"/obsUser.tsv";
			char uid[1000];
			int idx;
			double val;
		  	FILE *fin = fopen(fname.c_str(),"r");
		  	if(!fin) {
		  		std::cerr << "[ERR] Unable to open " << fname << endl;
		  		assert(0);
		  	}
			data.attr_users = Matrix2D<double>(data.Nusers,param.UC);
			data.attr_users_log = Matrix2D<double>(data.Nusers,param.UC);
			for(int count=0; count<data.Nusers; count++) {
				fscanf(fin,"%[^\t]\t",uid);
				std::map<string,int>::iterator iter_aux = data.user_ids.find(std::string(uid));
				if(iter_aux==data.user_ids.end()) {
					idx = data.Nusers;
					std::cerr << "[WARN] User " << std::string(uid) << " in 'obsUser.tsv' not found in train/test/validation" << endl;
				} else {
					idx = iter_aux->second;
				}
				for(int uc=0; uc<param.UC-1; uc++) {
					fscanf(fin,"%lf\t",&val);
					if(idx<data.Nusers) {
						data.attr_users.set_object(idx,uc,val);
						data.attr_users_log.set_object(idx,uc,my_log(val));
					}
				}
				fscanf(fin,"%lf\n",&val);
				if(idx<data.Nusers) {
					data.attr_users.set_object(idx,param.UC-1,val);
					data.attr_users_log.set_object(idx,param.UC-1,my_log(val));
				}
			}
			fclose(fin);
		}
		// Read obsItem.tsv
		if(param.IC>0) {
			string fname = param.datadir+"/obsItem.tsv";
			char mid[1000];
			int idx;
			double val;
		  	FILE *fin = fopen(fname.c_str(),"r");
		  	if(!fin) {
		  		std::cerr << "[ERR] Unable to open " << fname << endl;
		  		assert(0);
		  	}
			data.attr_items = Matrix2D<double>(data.Nitems,param.IC);
			data.attr_items_log = Matrix2D<double>(data.Nitems,param.IC);
			for(int count=0; count<data.Nitems; count++) {
				fscanf(fin,"%[^\t]\t",mid);
				std::map<string,int>::iterator iter_aux = data.item_ids.find(std::string(mid));
				if(iter_aux==data.item_ids.end()) {
					idx = data.Nitems;
					std::cerr << "[WARN] Item " << std::string(mid) << " in 'obsItem.tsv' not found in train/test/validation" << endl;
				} else {
					idx = iter_aux->second;
				}
				for(int ic=0; ic<param.IC-1; ic++) {
					fscanf(fin,"%lf\t",&val);
					if(idx<data.Nitems) {
						data.attr_items.set_object(idx,ic,val);
						data.attr_items_log.set_object(idx,ic,my_log(val));
					}
				}
				fscanf(fin,"%lf\n",&val);
				if(idx<data.Nitems) {
					data.attr_items.set_object(idx,param.IC-1,val);
					data.attr_items_log.set_object(idx,param.IC-1,my_log(val));
				}
			}
			fclose(fin);
		}
		// Read obsItemHier.tsv
		if(param.IChier>0) {
			string fname = param.datadir+"/obsItemHier.tsv";
			char mid[1000];
			int idx;
			int val;
			bool val_b;
		  	FILE *fin = fopen(fname.c_str(),"r");
		  	if(!fin) {
		  		std::cerr << "[ERR] Unable to open " << fname << endl;
		  		assert(0);
		  	}
			data.attr_items_hier = Matrix2D<bool>(data.Nitems,param.IChier);
			for(int count=0; count<data.Nitems; count++) {
				fscanf(fin,"%[^\t]\t",mid);
				std::map<string,int>::iterator iter_aux = data.item_ids.find(std::string(mid));
				if(iter_aux==data.item_ids.end()) {
					idx = data.Nitems;
					std::cerr << "[WARN] Item " << std::string(mid) << " in 'obsItemHier.tsv' not found in train/test/validation" << endl;
				} else {
					idx = iter_aux->second;
				}
				for(int ic=0; ic<param.IChier-1; ic++) {
					fscanf(fin,"%d\t",&val);
					val_b = (val>0);
					if(idx<data.Nitems) {
						data.attr_items_hier.set_object(idx,ic,val_b);
					}
				}
				fscanf(fin,"%d\n",&val);
				val_b = (val>0);
				if(idx<data.Nitems) {
					data.attr_items_hier.set_object(idx,param.IChier-1,val_b);
				}
			}
			fclose(fin);
		}
	}

	static void read_obsvar_attributes(hpf_data &data, const hpf_param &param) {
		// Read obsItemVar.tsv
		if(param.ICV>0) {
			string fname = param.datadir+"/obsItemVar.tsv";
			char mid[1000];
			char sid[1000];
			int idx_i;
			int idx_s;
			double val;
			int numWarnings = 0;

			FILE *fin = fopen(fname.c_str(),"r");
			if(!fin) {
				std::cerr << "[ERR] Unable to open " << fname << endl;
				assert(0);
			}

			data.attr_items_isk = Matrix3D<double>(data.Nitems,data.Nsessions,param.ICV);
			data.attr_items_isk_log = Matrix3D<double>(data.Nitems,data.Nsessions,param.ICV);
			while(!feof(fin)) {
				fscanf(fin,"%[^\t]\t%[^\t]\t",mid,sid);
				std::map<string,int>::iterator iter_i = data.item_ids.find(std::string(mid));
				std::map<string,int>::iterator iter_s = data.session_ids.find(std::string(sid));
				if(iter_i==data.item_ids.end()) {
					idx_i = data.Nitems;
					numWarnings++;
					if(numWarnings<=10) {
						std::cerr << "[WARN] Item " << std::string(mid) << " in 'obsItemVar.tsv' not found in train/test/validation" << endl;
					}
				} else {
					idx_i = iter_i->second;
				}
				if(iter_s==data.session_ids.end()) {
					idx_s = data.Nsessions;
					numWarnings++;
					if(numWarnings <= 10) {
						std::cerr << "[WARN] Session " << std::string(sid) << " in 'obsItemVar.tsv' not found in train/test/validation" << endl;
					}
				} else {
					idx_s = iter_s->second;
				}
				for(int ic=0; ic<param.ICV-1; ic++) {
					fscanf(fin,"%lf\t",&val);
					if(idx_i<data.Nitems && idx_s<data.Nsessions) {
						data.attr_items_isk.set_object(idx_i,idx_s,ic,val);
						data.attr_items_isk_log.set_object(idx_i,idx_s,ic,my_log(val));
					}
				}
				fscanf(fin,"%lf\n",&val);
				if(idx_i<data.Nitems && idx_s<data.Nsessions) {
					data.attr_items_isk.set_object(idx_i,idx_s,param.ICV-1,val);
					data.attr_items_isk_log.set_object(idx_i,idx_s,param.ICV-1,my_log(val));
				}
			}
			fclose(fin);

			if(numWarnings>0) {
				// Print out the total number of non matching sessions / items
				std::cerr << "[WARN] " << numWarnings << " rows of 'obsPrice.tsv' had sessions or items not found in the data files" << endl;
			}
		}
	}

	static void scale_obs_attributes(hpf_data &data, const hpf_param &param) {
		double mm;
		double ss;
		double ff;
		if(data.UC>0) {
			data.attr_users_scaleF = Matrix1D<double>(data.UC);
			for(int uc=0; uc<data.UC; uc++) {
				ff = param.factor_obs_extra;
				// Obtain the mean for that column
				mm = 0.0;
				for(int t=0; t<data.Nusers; t++) {
					mm += data.attr_users.get_object(t,uc);
				}
				mm /= static_cast<double>(data.Nusers);
				if(!param.factor_obs_std && !param.factor_obs_ones) {
					ff *= mm;
				} else if(param.factor_obs_std) {
					// Obtain the std
					ss = 0.0;
					for(int t=0; t<data.Nusers; t++) {
						ss += my_pow2(data.attr_users.get_object(t,uc)-mm);
					}
					ss /= (data.Nusers-1.0);
					ss = sqrt(ss);
					ff *= ss;
				}
				// Scale
				data.attr_users_scaleF.set_object(uc,ff);
			}
		}
		if(data.IC>0) {
			data.attr_items_scaleF = Matrix1D<double>(data.IC);
			for(int ic=0; ic<data.IC; ic++) {
				ff = param.factor_obs_extra;
				// Obtain the mean for that column
				mm = 0.0;
				for(int t=0; t<data.Nitems; t++) {
					mm += data.attr_items.get_object(t,ic);
				}
				mm /= static_cast<double>(data.Nitems);
				if(!param.factor_obs_std && !param.factor_obs_ones) {
					ff *= mm;
				} else if(param.factor_obs_std) {
					// Obtain the std
					ss = 0.0;
					for(int t=0; t<data.Nitems; t++) {
						ss += my_pow2(data.attr_items.get_object(t,ic)-mm);
					}
					ss /= (data.Nitems-1.0);
					ss = sqrt(ss);
					ff *= ss;
				}
				// Scale
				data.attr_items_scaleF.set_object(ic,ff);
			}
		}
	}

	static void scale_obsvar_attributes(hpf_data &data, const hpf_param &param) {
		double mm;
		double ss;
		double ff;

		data.attr_items_isk_scaleF = Matrix1D<double>(data.ICV);
		for(int ic=0; ic<data.ICV; ic++) {
			ff = param.factor_obs_extra;
			// Obtain the mean for that column
			mm = 0.0;
			for(int i=0; i<data.Nitems; i++) {
				for(int s=0; s<data.Nsessions; s++) {
					mm += data.attr_items_isk.get_object(i,s,ic);
				}
			}
			mm /= static_cast<double>(data.Nitems*data.Nsessions);
			if(!param.factor_obs_std && !param.factor_obs_ones) {
				ff *= mm;
			} else if(param.factor_obs_std) {
				// Obtain the std
				ss = 0.0;
				for(int i=0; i<data.Nitems; i++) {
					for(int s=0; s<data.Nsessions; s++) {
						ss += my_pow2(data.attr_items_isk.get_object(i,s,ic)-mm);
					}
				}
				ss /= (data.Nitems*data.Nsessions-1.0);
				ss = sqrt(ss);
				ff *= ss;
			}
			// Scale
			data.attr_items_isk_scaleF.set_object(ic,ff);
		}
	}

	static void read_availability(hpf_data &data, const hpf_param &param) {
		char mid[1000];
		char sid[1000];
		int idx;
		int idx_s;
		if(param.flag_availability) {
			data.sessions_per_item = Matrix1D<std::vector<int>>(data.Nitems);
			string fname = param.datadir+"/availability.tsv";
			FILE *fin = fopen(fname.c_str(),"r");
		  	if(!fin) {
		  		std::cerr << "[ERR] Unable to open " << fname << endl;
		  		assert(0);
		  	}
			while(!feof(fin)){
				fscanf(fin, "%[^\t]\t%[^\n]\n", mid, sid);
				std::map<string,int>::iterator iter_i = data.item_ids.find(std::string(mid));
				std::map<string,int>::iterator iter_s = data.session_ids.find(std::string(sid));
				if(iter_i==data.item_ids.end()) {
					idx = data.Nitems;
					std::cerr << "[WARN] Item " << std::string(mid) << " in 'availability.tsv' not found in train/test/validation" << endl;
				} else {
					idx = iter_i->second;
				}
				if(iter_s==data.session_ids.end()) {
					idx_s = data.Nsessions;
					std::cerr << "[WARN] Session " << std::string(sid) << " in 'availability.tsv' not found in train/test/validation" << endl;
				} else {
					idx_s = iter_s->second;
				}
				if(idx<data.Nitems && idx_s<data.Nsessions) {
					data.sessions_per_item.get_object(idx).push_back(idx_s);
				}
			}
			fclose(fin);
		}
	}

	static void create_availabilityCounts(hpf_data &data, const hpf_param &param) {
		data.availabilityCounts = Matrix2D<int>(data.Nusers,data.Nitems);
		int s;
		int aux;
		unsigned int u_aux;
		for(int u=0; u<data.Nusers; u++) {
			for(int i=0; i<data.Nitems; i++) {
				// run over all sessions of user u
				for(std::vector<int>::iterator iter=data.sessions_per_user.get_object(u).begin(); iter!=data.sessions_per_user.get_object(u).end(); iter++) {
					s = *iter;
					if(data.isavailable(i,s)) {
						aux = data.availabilityCounts.get_object(u,i);
						data.availabilityCounts.set_object(u,i,aux+1);
					}
				}
			}
		}

		// Compute "negative count" matrices
		data.availabilityNegCountsUser = Matrix1D<std::vector<hpf_avail_aux>>(data.Nusers);
		data.availabilityNegCountsItem = Matrix1D<std::vector<hpf_avail_aux>>(data.Nitems);
		hpf_avail_aux aAux;
		for(int u=0; u<data.Nusers; u++) {
			for(int i=0; i<data.Nitems; i++) {
				u_aux = data.availabilityCounts.get_object(u,i);
				if(u_aux < data.sessions_per_user.get_object(u).size()) {
					aAux.value = data.sessions_per_user.get_object(u).size()-u_aux;

					aAux.index = i;
					data.availabilityNegCountsUser.get_object(u).push_back(aAux);

					aAux.index = u;
					data.availabilityNegCountsItem.get_object(i).push_back(aAux);
				} else if(u_aux > data.sessions_per_user.get_object(u).size()) {
					std::cerr << "[ERR] This should not happen ---" << \
								 "      Review function 'create_availabilityCounts' in 'hpf_io.hpp'" << endl;
					assert(0);
				}
			}
		}
	}

	static void read_sess_days(hpf_data &data, const hpf_param &param) {
		char did[1000];
		char wid[1000];
		char sid[1000];
		int s;
		int d;
		int w;
		double hh;
		int n_d = 0;
		int n_w = 0;

		string fname = param.datadir+"/sess_days.tsv";
		// Read the file & create day_per_session
		data.day_per_session = Matrix1D<int>(data.Nsessions);
		data.weekday_per_session = Matrix1D<int>(data.Nsessions);
		data.hour_per_session = Matrix1D<double>(data.Nsessions);
		FILE *fin = fopen(fname.c_str(),"r");
	  	if(!fin) {
	  		std::cerr << "[ERR] Unable to open " << fname << endl;
	  		assert(0);
	  	}
		while(!feof(fin)){
			fscanf(fin, "%[^\t]\t%[^\t]\t%[^\t]\t%lf\n", sid, did, wid, &hh);
			std::map<string,int>::iterator iter_s = data.session_ids.find(std::string(sid));
			if(iter_s==data.session_ids.end()) {
				s = data.Nsessions;
				std::cerr << "[WARN] Session " << std::string(sid) << " in 'sess_days.tsv' not found in train/test/validation" << endl;
			} else {
				s = iter_s->second;
			}
			if(s<data.Nsessions) {
				std::map<string,int>::iterator iter_d = data.day_ids.find(std::string(did));
				if(iter_d==data.day_ids.end()) {
					data.day_ids.insert(std::pair<string,int>(std::string(did),n_d));
					n_d++;
					iter_d = data.day_ids.find(std::string(did));
				}
				d = iter_d->second;
				data.day_per_session.set_object(s,d);

				std::map<string,int>::iterator iter_w = data.weekday_ids.find(std::string(wid));
				if(iter_w==data.weekday_ids.end()) {
					data.weekday_ids.insert(std::pair<string,int>(std::string(wid),n_w));
					n_w++;
					iter_w = data.weekday_ids.find(std::string(wid));
				}
				w = iter_w->second;
				data.weekday_per_session.set_object(s,w);

				data.hour_per_session.set_object(s,hh);
			}
		}
		fclose(fin);

		// Count Ndays
		data.Ndays = data.day_ids.size();
		data.Nweekdays = data.weekday_ids.size();

		// Create sessions_per_day
		data.sessions_per_day = Matrix1D<std::vector<int>>(data.Ndays);
		for(int ss=0; ss<data.Nsessions; ss++) {
			d = data.day_per_session.get_object(ss);
			data.sessions_per_day.get_object(d).push_back(ss);
		}

		// Create sessions_per_weekday
		data.sessions_per_weekday = Matrix1D<std::vector<int>>(data.Nweekdays);
		for(int ss=0; ss<data.Nsessions; ss++) {
			w = data.weekday_per_session.get_object(ss);
			data.sessions_per_weekday.get_object(w).push_back(ss);
		}
	}

	static void read_itemgroups(hpf_data &data, const hpf_param &param) {
		char iid[1000];
		char gid[1000];
		int i;
		int g;
		int n_g = 0;

		data.group_per_item = Matrix1D<int>(data.Nitems);
		string fname = param.datadir+"/itemGroup.tsv";
		// Read the file
		FILE *fin = fopen(fname.c_str(),"r");
	  	if(!fin) {
	  		std::cerr << "[ERR] Unable to open " << fname << endl;
	  		assert(0);
	  	}
		while(!feof(fin)){
			fscanf(fin, "%[^\t]\t%[^\n]\n", iid, gid);
			std::map<string,int>::iterator iter_i = data.item_ids.find(std::string(iid));
			if(iter_i==data.item_ids.end()) {
				i = data.Nitems;
				std::cerr << "[WARN] Item " << std::string(iid) << " in 'itemGroup.tsv' not found in train/test/validation" << endl;
			} else {
				i = iter_i->second;
			}
			if(i<data.Nitems) {
				std::map<string,int>::iterator iter_g = data.itemgroup_ids.find(std::string(gid));
				if(iter_g==data.itemgroup_ids.end()) {
					data.itemgroup_ids.insert(std::pair<string,int>(std::string(gid),n_g));
					n_g++;
					iter_g = data.itemgroup_ids.find(std::string(gid));
				}
				g = iter_g->second;
				data.group_per_item.set_object(i,g);
			}
		}
		fclose(fin);
		// Other data variables
		data.NitemGroups = data.itemgroup_ids.size();
		data.items_per_group = Matrix1D<std::vector<int>>(data.NitemGroups);
		for(int ii=0; ii<data.Nitems; ii++) {
			data.items_per_group.get_object(data.group_per_item.get_object(ii)).push_back(ii);
		}
	}

	static void read_usergroups(hpf_data &data, const hpf_param &param) {
		char uid[1000];
		char gid[1000];
		int u;
		int g;
		int n_g = 0;

		data.group_per_user = Matrix1D<int>(data.Nusers);
		string fname = param.datadir+"/userGroup.tsv";
		// Read the file
		FILE *fin = fopen(fname.c_str(),"r");
		if(!fin) {
			std::cerr << "[ERR] Unable to open " << fname << endl;
			assert(0);
	  	}
		while(!feof(fin)){
			fscanf(fin, "%[^\t]\t%[^\n]\n", uid, gid);
			std::map<string,int>::iterator iter_u = data.user_ids.find(std::string(uid));
			if(iter_u==data.user_ids.end()) {
				u = data.Nusers;
				std::cerr << "[WARN] User " << std::string(uid) << " in 'userGroup.tsv' not found in train/test/validation" << endl;
			} else {
				u = iter_u->second;
			}
			if(u<data.Nusers) {
				std::map<string,int>::iterator iter_g = data.group_ids.find(std::string(gid));
				if(iter_g==data.group_ids.end()) {
					data.group_ids.insert(std::pair<string,int>(std::string(gid),n_g));
					n_g++;
					iter_g = data.group_ids.find(std::string(gid));
				}
				g = iter_g->second;
				data.group_per_user.set_object(u,g);
			}
		}
		fclose(fin);
		// Other data variables
		data.NuserGroups = data.group_ids.size();
		data.users_per_group = Matrix1D<std::vector<int>>(data.NuserGroups);
		for(int uu=0; uu<data.Nusers; uu++) {
			data.users_per_group.get_object(data.group_per_user.get_object(uu)).push_back(uu);
		}
	}

	static void create_lines_per_xday(hpf_data &data, const hpf_param &param) {
		int u;
		int i;
		int g_u;
  		int g_i;
  		int s;
  		int d;
  		int w;

  		data.lines_per_xday = Matrix3D<std::vector<int>>(data.NuserGroups,data.NitemGroups,data.Ndays);
  		data.lines_per_weekday = Matrix3D<std::vector<int>>(data.NuserGroups,data.NitemGroups,data.Nweekdays);
  		for(unsigned int t=0; t<data.obs.T; t++) {
  			u = data.obs.y_user[t];
  			g_u = data.group_per_user.get_object(u);
  			i = data.obs.y_item[t];
  			g_i = data.group_per_item.get_object(i);
  			s = data.obs.y_sess[t];
  			d = data.day_per_session.get_object(s);
  			w = data.weekday_per_session.get_object(s);
  			data.lines_per_xday.get_object(g_u,g_i,d).push_back(static_cast<int>(t));
  			data.lines_per_weekday.get_object(g_u,g_i,w).push_back(static_cast<int>(t));
	  	}
	}
};

class hpf_output {
public:
	static void create_output_folder(const hpf_data &data, hpf_param &param) {
		ostringstream sa;
    sa << "hpf-";
		sa << "k" << param.K << "-";
		sa << "uc" << param.UC << "-";
		sa << "ic" << param.IC << "-";
		sa << "icVar" << param.ICV;
		if(param.IChier>0) {
			sa << "-icHier" << param.IChier;
		}
		if(param.flag_itemIntercept) {
			sa << "-itemIntercept";
		}
		if(param.factor_obs_std) {
			sa << "-std";
		} else if(param.factor_obs_ones) {
			sa << "-ones";
		}
		if(param.factor_obs_extra != 1.0) {
			sa << "-scfact" << param.factor_obs_extra;
		}
		if(param.ini_offset!=1.0) {
			sa << "-offset" << param.ini_offset;
		}
		if(param.flag_lfirst) {
			sa << "-lfirst";
		} else if(param.flag_ofirst) {
			sa << "-ofirst";
		}
		if(param.flag_session) {
			sa << "-session";
		}
		if(param.flag_availability) {
			sa << "-availability";
		}
		if(param.flag_day) {
			sa << "-days";
		}
		if(param.flag_hourly>0) {
			sa << "-hourly" << param.flag_hourly;
		}
		if(param.label != "") {
			sa << "-" << param.label;
		}

		string prefix = sa.str();
		struct stat buffer;
		std::cout << "Output directory label: " << prefix << endl;
		int out = stat((param.outdir+"/"+prefix).c_str(), &buffer);
		if(out != 0) {
			std::cout << "Creating directory " << param.outdir << "/" << prefix << endl;
			if(mkdir((param.outdir+"/"+prefix).c_str(), S_IRUSR | S_IWUSR | S_IXUSR) != 0) {
				std::cerr << "[ERR] Cannot create output folder" << endl;
				assert(0);
			}
		}

		// Keep the new outdir
		param.outdir = param.outdir+"/"+prefix;
	}

    static void create_log_file(const hpf_data& data, hpf_param& param, const hpf_hyper& hyper) {
        // Write data
        write_log(param.outdir,"Data:");
        write_log(param.outdir," +datadir="+param.datadir);
        write_log(param.outdir," +Nusers="+std::to_string(data.Nusers));
        write_log(param.outdir," +Nitems="+std::to_string(data.Nitems));
        write_log(param.outdir," +Nsessions="+std::to_string(data.Nsessions));
        write_log(param.outdir," +Ndays="+std::to_string(data.Ndays));
        write_log(param.outdir," +Nweekdays="+std::to_string(data.Nweekdays));
        write_log(param.outdir," +NuserGroups="+std::to_string(data.NuserGroups));
        write_log(param.outdir," +NitemGroups="+std::to_string(data.NitemGroups));
        write_log(param.outdir," +uc="+std::to_string(data.UC));
        write_log(param.outdir," +ic="+std::to_string(data.IC));
        write_log(param.outdir," +icVar="+std::to_string(data.ICV));
        write_log(param.outdir," +icHier="+std::to_string(data.IChier));
        write_log(param.outdir," +Lines of train.tsv="+std::to_string(data.obs.T));
        write_log(param.outdir," +Lines of test.tsv="+std::to_string(data.obs_test.T));
        write_log(param.outdir," +Lines of validation.tsv="+std::to_string(data.obs_val.T));

        // Write parameters
        write_log(param.outdir,"Parameters:");
        write_log(param.outdir," +outdir="+param.outdir);
        write_log(param.outdir," +K="+std::to_string(param.K));
        write_log(param.outdir," +itemIntercept="+std::to_string(param.flag_itemIntercept));
        write_log(param.outdir," +session="+std::to_string(param.flag_session));
        write_log(param.outdir," +availability="+std::to_string(param.flag_availability));
        write_log(param.outdir," +days="+std::to_string(param.flag_day));
        write_log(param.outdir," +hourly="+std::to_string(param.flag_hourly));
        for(int n=0; n<param.flag_hourly; n++) {
	        write_log(param.outdir,"  +mean["+std::to_string(n)+"]="+std::to_string(param.gauss_mean.get_object(n)));
	        write_log(param.outdir,"  +var["+std::to_string(n)+"]="+std::to_string(param.gauss_var.get_object(n)));
        }
        write_log(param.outdir," +seed="+std::to_string(param.seed));
        write_log(param.outdir," +rfreq="+std::to_string(param.rfreq));
        write_log(param.outdir," +max-iterations="+std::to_string(param.Niter));
        write_log(param.outdir," +std="+std::to_string(param.factor_obs_std));
        write_log(param.outdir," +ones="+std::to_string(param.factor_obs_ones));
        write_log(param.outdir," +scaleFactor="+std::to_string(param.factor_obs_extra));
        write_log(param.outdir," +lfirst="+std::to_string(param.flag_lfirst));
        write_log(param.outdir," +ofirst="+std::to_string(param.flag_ofirst));
        write_log(param.outdir," +nIterIni="+std::to_string(param.nIterIni));
        write_log(param.outdir," +offset="+std::to_string(param.ini_offset));

        // Write hyperparameters
        write_log(param.outdir,"Hyperparameters:");
        write_log(param.outdir," +a="+std::to_string(hyper.a));
        write_log(param.outdir," +ap="+std::to_string(hyper.ap));
        write_log(param.outdir," +bp="+std::to_string(hyper.bp));
        write_log(param.outdir," +c="+std::to_string(hyper.c));
        write_log(param.outdir," +cp="+std::to_string(hyper.cp));
        write_log(param.outdir," +dp="+std::to_string(hyper.dp));
        write_log(param.outdir," +e="+std::to_string(hyper.e));
        write_log(param.outdir," +f="+std::to_string(hyper.f));
        write_log(param.outdir," +ax="+std::to_string(hyper.a_x));
        write_log(param.outdir," +bx="+std::to_string(hyper.b_x));
        write_log(param.outdir," +eta_k_shp="+std::to_string(hyper.eta_k_shp));
        write_log(param.outdir," +eta_k_rte="+std::to_string(hyper.eta_k_rte));
        write_log(param.outdir," +h_lk_shp="+std::to_string(hyper.h_lk_shp));
        write_log(param.outdir," +h_lk_rte="+std::to_string(hyper.h_lk_rte));
	}

    static void write_log(const string folder, const string str) {
        write_line(folder+"/log.txt",str);
    }

    static void write_line(const string file, const string str) {
        ofstream myfile;
        myfile.open(file,ios::out|ios::app);    // Append content to the end of existing file
        if(!myfile.is_open()) {
            std::cerr << "[ERR] file '" << file << "' could not be opened" << endl;
            assert(0);
        }
        myfile << str << endl;
        myfile.close();
    }

    static void write_telapsed(const string folder, int iter, clock_t t_ini, clock_t t_end) {
        double elapsed_secs = static_cast<double>(t_end-t_ini)/CLOCKS_PER_SEC;
        write_line(folder+"/telapsed.txt",std::to_string(iter)+"\t"+std::to_string(elapsed_secs));
    }

    static void write_max_file(const hpf_param &param, double duration, double val_llh, int why) {
    	char buffer[200];
    	sprintf(buffer,"%d\t%d\t%.9f\t%d",param.it,static_cast<int>(duration),val_llh,why);
        write_line(param.outdir+"/max.txt",string(buffer));
    }

    static void write_matrix(string filename, Matrix1D<hpf_gamma> &M) {
    	std::map<string,int> ids;
    	for(int k=0; k<M.get_size1(); k++) {
    		ids.insert(std::pair<string,int>(std::to_string(k+1),k));
    	}
    	write_matrix(filename,ids,M);
    }

    static void write_matrix(string filename, Matrix2D<hpf_gamma> &M) {
    	std::map<string,int> ids;
    	for(int k=0; k<M.get_size1(); k++) {
    		ids.insert(std::pair<string,int>(std::to_string(k+1),k));
    	}
    	write_matrix(filename,ids,M);
    }

    static void write_matrix(string filename, std::map<string,int> &ids, Matrix2D<hpf_gamma> &M) {
    	char buffer[100];
    	string aux;
    	std::map<string,int>::iterator iter_aux;
    	// Print shape
    	int count = 0;
    	for(iter_aux=ids.begin(); iter_aux!=ids.end(); ++iter_aux) {
    		string iName = iter_aux->first;
    		int i = iter_aux->second;
    		sprintf(buffer,"%d\t%s\t",count,iName.c_str());
    		count++;
    		aux = string(buffer);
    		for(int k=0; k<M.get_size2()-1; k++) {
	    		sprintf(buffer,"%.12f\t",M.get_object(i,k).shp);
    			aux += string(buffer);
    		}
    		sprintf(buffer,"%.12f",M.get_object(i,M.get_size2()-1).shp);
    		aux += string(buffer);
	    	write_line(filename+"_shape.tsv",aux);
    	}
    	// Print rate
    	count = 0;
    	for(iter_aux=ids.begin(); iter_aux!=ids.end(); ++iter_aux) {
    		string iName = iter_aux->first;
    		int i = iter_aux->second;
    		sprintf(buffer,"%d\t%s\t",count,iName.c_str());
    		count++;
    		aux = string(buffer);
    		for(int k=0; k<M.get_size2()-1; k++) {
	    		sprintf(buffer,"%.12f\t",M.get_object(i,k).rte);
    			aux += string(buffer);
    		}
    		sprintf(buffer,"%.12f",M.get_object(i,M.get_size2()-1).rte);
    		aux += string(buffer);
	    	write_line(filename+"_rate.tsv",aux);
    	}
    	// Print mean
    	count = 0;
    	for(iter_aux=ids.begin(); iter_aux!=ids.end(); ++iter_aux) {
    		string iName = iter_aux->first;
    		int i = iter_aux->second;
    		sprintf(buffer,"%d\t%s\t",count,iName.c_str());
    		count++;
    		aux = string(buffer);
    		for(int k=0; k<M.get_size2()-1; k++) {
	    		sprintf(buffer,"%.12f\t",M.get_object(i,k).e_x);
    			aux += string(buffer);
    		}
    		sprintf(buffer,"%.12f",M.get_object(i,M.get_size2()-1).e_x);
    		aux += string(buffer);
	    	write_line(filename+".tsv",aux);
    	}
    }

    static void write_matrix(string filename, std::map<string,int> &ids, Matrix1D<hpf_gamma> &M) {
    	char buffer[100];
    	string aux;
    	std::map<string,int>::iterator iter_aux;
    	// Print shape
    	int count = 0;
    	for(iter_aux=ids.begin(); iter_aux!=ids.end(); ++iter_aux) {
    		string iName = iter_aux->first;
    		int i = iter_aux->second;
    		sprintf(buffer,"%d\t%s\t%.12f",count,iName.c_str(),M.get_object(i).shp);
    		count++;
    		aux = string(buffer);
	    	write_line(filename+"_shape.tsv",aux);
    	}
    	// Print rate
    	count = 0;
    	for(iter_aux=ids.begin(); iter_aux!=ids.end(); ++iter_aux) {
    		string iName = iter_aux->first;
    		int i = iter_aux->second;
    		sprintf(buffer,"%d\t%s\t%.12f",count,iName.c_str(),M.get_object(i).rte);
    		count++;
    		aux = string(buffer);
	    	write_line(filename+"_rate.tsv",aux);
    	}
    	// Print mean
    	count = 0;
    	for(iter_aux=ids.begin(); iter_aux!=ids.end(); ++iter_aux) {
    		string iName = iter_aux->first;
    		int i = iter_aux->second;
    		sprintf(buffer,"%d\t%s\t%.12f",count,iName.c_str(),M.get_object(i).e_x);
    		count++;
    		aux = string(buffer);
	    	write_line(filename+".tsv",aux);
    	}
    }

    static void write_matrix_xday(string filename, std::map<string,int> &ids_g, std::map<string,int> &ids_i, std::map<string,int> &ids_d, Matrix3D<hpf_gamma> &M) {
    	string aux;
    	char buffer[300];

    	int g;
    	int i;
    	int d;

    	std::map<string,int>::iterator iter_g;
    	std::map<string,int>::iterator iter_i;
    	std::map<string,int>::iterator iter_d;

    	int count = 0;
    	for(iter_g=ids_g.begin(); iter_g!=ids_g.end(); ++iter_g) {
    		g = iter_g->second;
    		for(iter_i=ids_i.begin(); iter_i!=ids_i.end(); ++iter_i) {
    			i = iter_i->second;
    			for(iter_d=ids_d.begin(); iter_d!=ids_d.end(); ++iter_d) {
    				d = iter_d->second;

    				string gName = iter_g->first;
    				string iName = iter_i->first;
    				string dName = iter_d->first;

		    		// Print shape
		    		sprintf(buffer,"%d\t%s\t%s\t%s\t%.12f",count,gName.c_str(),iName.c_str(),dName.c_str(),M.get_object(g,i,d).shp);
		    		aux = string(buffer);
	    			write_line(filename+"_shape.tsv",aux);
	    			// Print rate
		    		sprintf(buffer,"%d\t%s\t%s\t%s\t%.12f",count,gName.c_str(),iName.c_str(),dName.c_str(),M.get_object(g,i,d).rte);
		    		aux = string(buffer);
	    			write_line(filename+"_rate.tsv",aux);
	    			// Print mean
		    		sprintf(buffer,"%d\t%s\t%s\t%s\t%.16f",count,gName.c_str(),iName.c_str(),dName.c_str(),M.get_object(g,i,d).e_x);
		    		aux = string(buffer);
	    			write_line(filename+".tsv",aux);
	    			// Increase line number
	    			count++;
	    		}
    		}
    	}
    }

    static void write_matrix_xhourly(string filename, std::map<string,int> &ids_g, std::map<string,int> &ids_i, std::map<string,int> &ids_d, Matrix4D<hpf_gamma> &M) {
    	string aux;
    	char buffer[300];

    	int g;
    	int i;
    	int d;

    	std::map<string,int>::iterator iter_g;
    	std::map<string,int>::iterator iter_i;
    	std::map<string,int>::iterator iter_d;

    	int count = 0;
    	for(iter_g=ids_g.begin(); iter_g!=ids_g.end(); ++iter_g) {
    		g = iter_g->second;
    		for(iter_i=ids_i.begin(); iter_i!=ids_i.end(); ++iter_i) {
    			i = iter_i->second;
    			for(iter_d=ids_d.begin(); iter_d!=ids_d.end(); ++iter_d) {
    				d = iter_d->second;

    				string gName = iter_g->first;
    				string iName = iter_i->first;
    				string dName = iter_d->first;

		    		// Print shape
		    		ostringstream sa;
		    		sprintf(buffer,"%d\t%s\t%s\t%s\t",count,gName.c_str(),iName.c_str(),dName.c_str());
		    		sa << buffer;
		    		for(int n=0; n<M.get_size4(); n++) {
		    			sprintf(buffer,"%.12f",M.get_object(g,i,d,n).shp);
		    			sa << buffer;
		    			if(n!=M.get_size4()-1) {
		    				sa << "\t";
		    			}
		    		}
		    		aux = sa.str();
	    			write_line(filename+"_shape.tsv",aux);
	    			// Print rate
		    		ostringstream saRte;
		    		sprintf(buffer,"%d\t%s\t%s\t%s\t",count,gName.c_str(),iName.c_str(),dName.c_str());
		    		saRte << buffer;
		    		for(int n=0; n<M.get_size4(); n++) {
		    			sprintf(buffer,"%.12f",M.get_object(g,i,d,n).rte);
		    			saRte << buffer;
		    			if(n!=M.get_size4()-1) {
		    				saRte << "\t";
		    			}
		    		}
		    		aux = saRte.str();
	    			write_line(filename+"_rate.tsv",aux);
	    			// Print mean
		    		ostringstream saMean;
		    		sprintf(buffer,"%d\t%s\t%s\t%s\t",count,gName.c_str(),iName.c_str(),dName.c_str());
		    		saMean << buffer;
		    		for(int n=0; n<M.get_size4(); n++) {
		    			sprintf(buffer,"%.16f",M.get_object(g,i,d,n).e_x);
		    			saMean << buffer;
		    			if(n!=M.get_size4()-1) {
		    				saMean << "\t";
		    			}
		    		}
		    		aux = saMean.str();
	    			write_line(filename+".tsv",aux);
	    			// Increase line number
	    			count++;
	    		}
    		}
    	}
    }

    static void write_matrix(string filename, Matrix2DNoncontiguous<hpf_multinomial> &M) {
    	char buffer[100];
    	string aux;
    	// Print values
    	for(int i=0; i<M.get_size1(); i++) {
    		sprintf(buffer,"%d\t",i);
    		aux = string(buffer);
    		for(int k=0; k<M.get_size2()-1; k++) {
	    		sprintf(buffer,"%.12f\t",M.get_object(i,k).phi);
    			aux += string(buffer);
    		}
    		sprintf(buffer,"%.12f",M.get_object(i,M.get_size2()-1).phi);
    		aux += string(buffer);
	    	write_line(filename+".tsv",aux);
    	}
    }
};

#endif
