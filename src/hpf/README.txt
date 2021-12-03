Hierarchical Poisson factorization with observed characteristics
----------------------------------------------------------------

USAGE:
	hpf [options]

OPTIONS:

-dir <string>		path to directory with all the data files
			[default: .]

-outdir <string>	path to directory where output files will be written
			[default: .]

-m <int>	  number of items
		  [default: automatically read from input files]
-n <int>	  number of users
		  [default: automatically read from input files]
-k <int>	  number of latent factors
		  [default: 50]
-itemIntercept   use per-item intercepts
		  [default: disabled]
-uc <int>         number of observed user attributes. If >0, requires an extra
		  input file (obsUser.tsv; see below)
		  [default: 0]
-ic <int>         number of observed item attributes. If >0, requires an extra
		  input file (obsItem.tsv; see below)
		  [default: 0]
-icVar <int>	  number of observed item attributes that vary over sessions
		  (such as price). If >0, requires an extra input file
		  (obsItemVar.tsv; see below)
		  [default: 0]

-session       indicates that the train, validation, and test sets include
		a session id (see format below) to handle multiple sessions
		[default: inactive]
-availability	implements the HPF code taking availability into account. It
		implicitly activates the '-session' option. If '-availability'
		is specified, a new input file (availability.tsv) is required
		indicating the items that are available in each session
		[default: inactive]
-days          use per-itemgroup per-usergroup per-day block effects. It
		requires three extra files (see format below): userGroup.tsv,
               containing the group to which each user belongs; itemGroup.tsv,
               containing the item to which each item belongs; and sess_days.tsv,
               containing the time information for each session.
		[default: disabled]
-hourly <int>  fit per-itemgroup per-usergroup per-weekday mixtures of N
               Gaussians, where N is the specified integer. If N<=0, this
               option is disabled and no Gaussians are fit. If N>0, it
               requires the same files as '-days'.
		[default: 0]
		NOTE: The '-days' option is *not* backwards-compatible, in the
		sense that it does *not* require 5 columns in test.tsv, train.tsv
		or validation.tsv (they should have 4 columns each). If any of
		these files contains 5 columns, the behavior of the code is
		undefined. Furthermore, sess_days.tsv should have 4 columns,
		even when '-hour 0' is used (in which case the two last columns
               will be ignored).
               NOTE 2: If Gaussian mixtures are used, the means and variances
               will be fixed to some values (see the corresponding options and
               defaults under '-gaussMeans' and '-gaussVar').

-noTest        ignore the test file; 'test.tsv' becomes is ignored if present
-noVal         ignore the validation file; 'validation.tsv' is ignored if
               present

-seed <int>		set random seed
			[default: 0]
-rfreq <int>		frequency (#iterations) for evaluating convergence
			[default: 10]
-max-iterations <int>	maximum number of iterations
			[default: 1000]

-nooffset		initialize the variational parameters without any random
			offset. It is equivalent to '-offset 0'
			[default: inactive]
-offset <double>	factor to multiply the random offset used for initialization
			[default: 1.0]

-std			the hyperparameters corresponding to the latent variables
			that multiply the observable attributes are set to the
			standard deviation of each observable attribute; instead
			of the mean
			[default: inactive]
-ones           	use instead a vector of ones (instead of the mean or the
			standard deviation)
			[default: inactive]
-scfact <double>	artificially scale these hyperparameters by an additional
			factor
			[default: 1.0]

-lfirst         run some iterations in which only the purely latent variables are
                updated, and the hidden variables corresponding to observed
                attributes are held fixed, before running inference
                [default: inactive]
-ofirst         run some iterations in which only the hidden variables corresponding
                to observed attributes are updated (and the rest are held fixed),
                before running inference
                [default: inactive]
-nIterIni <int> number of iterations with some parameters held fixed
                [default: 20]

-label <string>	label to be appended to the name of the output folder
		[default: none]
-quiet          avoid generating as many lines of output when run interactively.
                The iteration progress tracking all takes place on a single line.
                Does not work well when output is piped to a file.
                [default: disabled]

-gaussMeans <double> [<double> [<double> [...]]]   if '-hourly N' is used with N>0,
                                                   this allows specifying the means of
                                                   the Gaussian components (use N
                                                   <double> values)
                                                   [default: equally spaced, between 0
                                                    and 24]
-gaussVars <double> [<double> [<double> [...]]]    if '-hourly N' is used with N>0,
                                                   this allows specifying the variances
                                                   of the Gaussian components (use N
                                                   <double> values)
                                                   [default: 10.0 for all]
 NOTE: '-gaussMeans' and '-gaussVars' should always be specified *after* '-hourly'

-a
-ap
-bp         Hyperparameters
-c          [default] ap = cp = 1.5
-cp         	       bp = dp = 0.3
-dp		       a = c = e = f = 1.0/sqrt(k)
-e		       ax = 0.05, bx = 1.0
-f
-ax
-bx


The model
---------

The model is hierarchical Poisson factorization:
  y_uit ~ Poisson(beta_i0 + sum_k theta_uk*beta_ik + sum_k sigma_uk*obsItems_ik 
	           +sum_k rho_ik*obsUsers_uk + time_effect)

beta_i0: Unobserved item intercepts
theta_uk, beta_ik: Unobserved user/item attributes
sigma_uk, rho_ik: Latent variables that multiply the observable attributes

time_effect appears when '-days' or '-hourly N' (with N>0) is specified.
If '-days', the time effects have a constant term
     x_g(u)g(i)d
which depends on the usergroup g(u), itemgroup g(i), and day d.
If '-hourly N' (with N>0), the time effects include a mixture of Gaussians
     x_g(u)g(i)w{1}*G(h(t); mean_1,variance_1)+...+x_g(u)g(i)w{c}*G(h(t); mean_c,variance_c)
where c is the number of mixture components, w denotes the weekday, G(·) is
the Gaussian density function, h(t) is the hour of the day in which session t
occurs, and x_g(u)g(i)w{1:c} are the coefficients that the algorithm infers.

The priors are as follows:
	theta_uk ~ Gamma(a, xi_u)
	xi_u ~ Gamma(ap, ap/bp)
	beta_i0 ~ Gamma(c, eta_i)
	beta_ik ~ Gamma(c, eta_i)
	eta_i ~ Gamma(cp, cp/dp)
	sigma_uk ~ Gamma(e, factor_k*xi_u)
	rho_ik ~ Gamma(f, factor_k*eta_i)
	x_g(u)g(i)d ~ Gamma(a_x, b_x)
	x_g(u)g(i)w{n} ~ Gamma(a_x, b_x)
where the 'factor_k' variables are obtained as described above (see '—std',
'-ones', and '-scfact').


Compilation
-----------

You need to compile the code using the GSL library. An example of how to compile
the code is provided in the script compile.sh:

	g++ -Wall -std=c++11 -I/usr/local/include -lgsl -o hpf hpf.cpp


Input
-----

The model needs the following input files:

train.tsv	contains the training data. It has at least three columns, in
		tab-separated format. First column: user id (a non-negative
		integer). Second column: item id (a non-negative integer).
		Third column: number of units (a non-negative integer).
		If '-session' is specified, then the third column must be the
		session id (a non-negative integer), and the fourth column
		must be the number of units. Thus, the format of each line of
		train.tsv is as follows:
			user_id	item_id	[session_id]	units
test.tsv	contains the test data, in the same format as train.tsv
validation.tsv	contains the validation data (only used to assess convergence),
		in the same format as train.tsv

obsUser.tsv	if '-uc <val>' is specified, with val>0, this file is required.
		It contains the observable attributes for each user. It has
		tab-separated CSV format, where the first column corresponds to
		the user id and the rest of the columns correspond to the observed
		attributes. Thus, the format is as follows:
			user_id	attr_1	[attr_2 [attr_3 ...]]
		Each user in train.tsv, test.tsv or validation.tsv must appear
		exactly once in this file.
obsItem.tsv	if '-ic <val>' is specified, with val>0, this file is required.
		It contains the observable attributes for each item. It has
		tab-separated CSV format, where the first column corresponds to
		the item id and the rest of the columns correspond to the observed
		attributes. Thus, the format is as follows:
			item_id	attr_1	[attr_2 [attr_3 ...]]
		Each item in train.tsv, test.tsv or validation.tsv must appear
		exactly once in this file.
obsItemVar.tsv	if '-icVar <val>' is specified, with val>0, this file is required.
		It contains the observable attributes for each item. It has
		tab-separated CSV format, where the first column corresponds to
		the item id, the second column corresponds to the session id, and
		the rest of the columns correspond to the observed attributes.
		Thus, the format is as follows:
			item_id	session_id	attr_1	[attr_2 [attr_3 ...]]
		IMPORTANT: There should be as many lines in this file as the
		number of items multiplied by the number of sessions
		(taking into account all items and sessions in train.tsv,
		test.tsv, and validation.tsv).

availability.tsv	if '-availability' is specified, this file is required.
			It contains two columns (tab-separated) specifying which
			items are available at which sessions. The first column
			is the item id and the second column is the session id.
			Thus, the format for each line is:
				item_id	session_id
			(Note that an item may be available in several sessions.)

userGroup.tsv	        if either '-days' or '-hourly N' (with N>0), this file is
                       required. It contains the groups to which each user belongs
                       to. It has two columns (tab-separated). The first column
                       contains the user id and the second column contains the group
                       id (a non-negative integer). Thus, the format is:
				user_id	userGroup_id
                       Each user in train.tsv, test.tsv or validation.tsv must
                       appear exactly once in this file.
itemGroup.tsv          if either '-days' or '-hourly N' (with N>0), this file is
                       required. It contains the groups to which each item belongs
                       to. It has two columns (tab-separated). The first column
                       contains the item id and the second column contains the
                       group id (a non-negative integer). Thus, the format is:
				item_id	itemGroup_id
                       Each item in train.tsv, test.tsv or validation.tsv must
                       appear exactly once in this file.
                       NOTE: The itemGroup_ids have nothing to do with the
                       userGroup_ids since they represent different things.
sess_days.tsv          if either '-days' or '-hourly N' (with N>0) is specified,
                       this file is required. It contains the time information
                       for each session. This file has four columns (tab-separated).
                       The first column contains the session id, the second
                       column contains the calendar day, the third column contains 
                       the weekday id, and the third column contains the time t
                       (hour of the day), being the ids non-negative integers
                       (the time t is not necessarily an integer). Thus, the
                       format is: 
				session_id	day_id	weekday_id	time_hours
                       Each session in train.tsv, test.tsv or validation.tsv must
                       appear exactly once in this file.
                       NOTE: This file is not backwards-compatible. Be careful
                       if you have used previous versions of hpf.


Output
------

The main output of the model are the variational parameters for all the
latent variables. The following list has the names according to the latex
document, followed by the name in the code: beta (hbeta), rho (hrho),
eta (betarate), theta (htheta), sigma (hsigma), and xi (thetarate).
Additionally, three other files are produced: xday (for per-day effects),
xhourly (for per-weekday mixtures of Gaussians), and hsigmaVar (same as
hsigma, but for the latent variables that multiply the varying observable
attributes). For each of these variables, the output includes one file
with the shape parameters, one with the rate parameters, and one with the
means.

All these files have the general format:
	line_number	id	value_1	[value_2 [value_3 [...]]]
where id is the user or item id.

If '-days' is specified, the files for xday have the format:
	line_number	userGroup_id	itemGroup_id	day_id	value
If '-hourly N' (with N>0) is specified, the files for xhourly have the format:
	line_number	userGroup_id	itemGroup_id	weekday_id	value_0	...	value_N

The code also writes two files, validation.txt and test.txt, which give a
summary of the evolution of the log-likelihood and mean squared error of both
sets. Their format is:
	iteration	duration	log-lik	log-lik_normalized	log-lik_normalized_bin	mse	number_lines


It also produces logjoint.txt, log.txt, t_elapsed.txt, and max.txt.
