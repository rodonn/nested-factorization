#include "my_headers_hpf.hpp"
#include "my_gsl_utilities.hpp"
#include "matrices.hpp"
#include "hpf_params.hpp"
#include "hpf_io.hpp"
#include "hpf_inference.hpp"

int main (int argc, char *argv[]) {

    hpf_param param;
    hpf_hyper hyper;
    hpf_data data;

    /******** Read options from input (argc, argv) ********/
    std::cout << "Initializing program..." << endl;
    hpf_input::read_input_from_command_line(argc,argv,data,param,hyper);

    /******** Read data from file ********/
    std::cout << "Reading data..." << endl;
    // Read train.tsv, validation.tsv and train.tsv
    hpf_input::read_data_file(data,param);
    // Read obsUser.tsv & obsItem.tsv
    if(param.UC>0 || param.IC>0 || param.IChier>0) {
        std::cout << "Reading observable attributes..." << endl;
        std::cout << " +Reading from file..." << endl;
        hpf_input::read_obs_attributes(data,param);
        std::cout << " +Computing scaling factor..." << endl;
        hpf_input::scale_obs_attributes(data,param);
    }
    if(param.ICV>0) {
        std::cout << "Reading observable item attributes that vary over time..." << endl;
        std::cout << " +Reading from file..." << endl;
        hpf_input::read_obsvar_attributes(data,param);
        std::cout << " +Computing scaling factor..." << endl;
        hpf_input::scale_obsvar_attributes(data,param);
    }
    // Read availability.tsv
    if(param.flag_availability) {
        std::cout << "Reading availability..." << endl;
        hpf_input::read_availability(data,param);
        std::cout << "Creating availability counts (this may take some time)..." << endl;
        hpf_input::create_availabilityCounts(data,param);
    }
    // Read userGroup.tsv
    if(param.flag_hourly>0 || param.flag_day) {
        std::cout << "Reading session/days mapping..." << endl;
        hpf_input::read_sess_days(data,param);
        if(data.NitemGroups==0) {
            std::cout << "Reading item groups..." << endl;
            hpf_input::read_itemgroups(data,param);
        }
        std::cout << "Reading user groups..." << endl;
        hpf_input::read_usergroups(data,param);
        std::cout << "Creating auxiliary data variables for day effects..." << endl;
        hpf_input::create_lines_per_xday(data,param);
        std::cout << "Computing auxiliary variables for per-day effects (this may take some time)..." << endl;
        if(param.flag_day) {
            hpf_input::compute_auxsumrte_xday(data,param);
        }
        if(param.flag_hourly) {
            hpf_input::compute_auxsumrte_xhourly(data,param);
        }
    }
    // Create auxiliary data structures
    std::cout << "Computing auxiliary data structures..." << endl;
    data.create_other_data_structs(param);
    // If observed attributes, also compute \sum_{t,i} \alpha_uit * attr_it (or equivalently)
    if(param.UC>0 || param.IC>0 || param.ICV>0) {
        std::cout << "Computing auxiliary variables for observable attributes (this may take some time)..." << endl;
        if(param.IC>0) {
            hpf_input::compute_auxsumrte_obsIC(data,param);
        }
        if(param.ICV>0) {
            hpf_input::compute_auxsumrte_obsICV(data,param);
        }
        if(param.UC>0) {
            hpf_input::compute_auxsumrte_obsUC(data,param);
        }
    }

    /******** Create output folder ********/
    hpf_output::create_output_folder(data,param);

    /******** Write first log ********/
    hpf_output::create_log_file(data,param,hyper);

    /******** Set the seed ********/
    gsl_rng *semilla = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(semilla,param.seed);

    /******** Allocate memory for variational parameters ********/
    std::cout << "Initializing latent parameters..." << endl;
    // Compute the length of the auxiliary multinomial variables z_tk
    param.L_phi = param.K+param.UC+param.IC+param.ICV;
    if(param.flag_itemIntercept) {
        param.L_phi = param.L_phi+1;
    }
    if(param.flag_day) {
        param.L_phi = param.L_phi+1;
    }
    if(param.flag_hourly>0) {
        param.L_phi = param.L_phi+param.flag_hourly;
    }
    hpf_pvar pvar(data,param);
    // Initialize variational parameters
    pvar.initialize_random(semilla,data,param,hyper);

    /******** Inference algorithm (initial iterations only) ********/
    if((param.flag_lfirst || param.flag_ofirst) && (param.UC>0 || param.IC>0 || param.ICV>0)) {
        std::cout << "Running initial iterations of inference..." << endl;
        for(param.it=0; param.it<param.nIterIni; (param.it)++) {
            std::cout << " +Warm-up iteration " << param.it << "/" << param.nIterIni << "..." << endl;
            hpf_infer::run_inference_step(data,param,hyper,pvar,true);
        }
    }

    /******* In Quiet Mode fewer lines of output are created ****/
    string lineEndChar;
    if (param.quiet) {
      lineEndChar = "\r";
    } else {
      lineEndChar = "\n";
    }
    /******** Inference algorithm ********/
    std::cout << "Running inference algorithm..." << endl;
    clock_t t_ini_abs = clock();
    clock_t t_ini;
    clock_t t_end;
    bool stop = false;
    param.it = 0;
    int why = -1;
    double val_llh;
    double duration;
    while(!stop) {
        std::cout << " +Iteration " << param.it << "..." << lineEndChar << flush;
        t_ini = clock();    // Measure time elapsed

        // Run an inference step
        hpf_infer::run_inference_step(data,param,hyper,pvar,false);

        // Write time elapsed
        t_end = clock();    // Measure time elapsed
        hpf_output::write_telapsed(param.outdir,param.it,t_ini,t_end);
        duration = static_cast<double>(t_end-t_ini_abs)/CLOCKS_PER_SEC;

        // Compute llh + Check stop criteria
        if((param.it%param.rfreq)==0) {
            //hpf_infer::compute_likelihood("train",duration,data,data.obs,param,hyper,pvar);
            hpf_infer::compute_likelihood("test",duration,data,data.obs_test,param,hyper,pvar);
            val_llh = hpf_infer::compute_likelihood("validation",duration,data,data.obs_val,param,hyper,pvar);
            if(param.it>30 && !param.noVal) {
                std::cout << " (validation llh relative change (abs value): " << fabs((val_llh-param.prev_val_llh)/param.prev_val_llh) << ")" << endl;
                // If log likelihood increased, is not zero, and it increased less than 0.000001 of the previous value, set why to zero
                if(val_llh>=param.prev_val_llh && param.prev_val_llh!=0 && fabs((val_llh-param.prev_val_llh)/param.prev_val_llh)<0.000001) {
                    stop = true;
                    why = 0;
                } else if(val_llh<param.prev_val_llh) {
                    // Count the number of times in a row that the likelihood decreased
                    (param.n_val_decr)++;
                } else if(val_llh>param.prev_val_llh) {
                    param.n_val_decr = 0;
                }
                if(param.n_val_decr>5) {
                    stop = true;
                    why = 1;
                }
            }
            param.prev_val_llh = val_llh;
        }
        if(param.it+1>=param.Niter) {
            stop = true;
            why = 2;
        }
        if(!stop) {
            (param.it)++;
        }
    }

    /******** Print output ********/
    hpf_output::write_max_file(param,duration,val_llh,why);
    if(param.K>0) {
        hpf_output::write_matrix(param.outdir+"/htheta",data.user_ids,pvar.theta_uk);
        hpf_output::write_matrix(param.outdir+"/hbeta",data.item_ids,pvar.beta_ik);
    }
    if(param.flag_itemIntercept) {
        hpf_output::write_matrix(param.outdir+"/hbeta0",data.item_ids,pvar.beta_i0);
    }
    hpf_output::write_matrix(param.outdir+"/thetarate",data.user_ids,pvar.xi_u);
    if(param.IChier==0) {
        hpf_output::write_matrix(param.outdir+"/betarate",data.item_ids,pvar.eta_i);
    }
    if(param.IC>0) {
        hpf_output::write_matrix(param.outdir+"/hsigma",data.user_ids,pvar.sigma_uk);
    }
    if(param.UC>0) {
        hpf_output::write_matrix(param.outdir+"/hrho",data.item_ids,pvar.rho_ik);
    }
    if(param.ICV>0) {
        hpf_output::write_matrix(param.outdir+"/hsigmaVar",data.user_ids,pvar.sigma_uk_var);
    }
    if(param.IChier>0) {
        hpf_output::write_matrix(param.outdir+"/hierprior_etak",pvar.eta_k);
        hpf_output::write_matrix(param.outdir+"/hierprior_hlk",pvar.h_lk);
    }
    if(param.flag_day) {
        hpf_output::write_matrix_xday(param.outdir+"/xday",data.group_ids,data.itemgroup_ids,data.day_ids,pvar.x_gid);
    }
    if(param.flag_hourly>0) {
        hpf_output::write_matrix_xhourly(param.outdir+"/xhourly",data.group_ids,data.itemgroup_ids,data.weekday_ids,pvar.x_giwn);
    }

    /******** Free memory ********/
    gsl_rng_free(semilla);

    /******** Return ********/
    return 0;
};
