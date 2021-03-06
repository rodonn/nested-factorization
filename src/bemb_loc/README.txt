Softmax model for retail (supermarket) data

This is the version that assumes the 'prices' are the same across all users
but vary across sessions.

A related model from the TTFM paper is similar but allows distances to vary across users 
(Whereas in the current model all users shopping in same session see the same prices).
https://www.openicpsr.org/openicpsr/project/114442/version/V1/view

*********
* MODEL *
*********

The model is given by
```
prob(product i | user u, week w, weekday d) \propto exp{ lambda0_i + theta_u*alpha_i + obsItem_u*obsItem_i + obsUser_u*obsUser_i + mu_i*delta_w + weekday_id - gamma_u*beta_i*log(price)}
```

If desired, the parameters alpha_i and beta_i can be placed a prior that depends on the item observables:
```
p(alpha_ik | H_k, obsItem_i) = Gaussian( mean=H_k*obsItem_i, variance=s2obsPrior )
p(beta_ik | H'_k, obsItem_i) = Gaussian( mean=H'_k*obsItem_i, variance=s2obsPrior )
```


*********************
* COMPILATION & RUN *
*********************

For compilation, use:
g++ -std=c++11 -Wall -o emb emb.cpp `gsl-config --cflags --libs`

You should now be able to run the code ./emb with the arguments below.


*************
* ARGUMENTS *
*************

-dir <string>
	path to the data folder
-outdir <string>
	path to the output folder
-K <int>
	number of latent factors
-max-iterations <int>
	maximum number of iterations
-rfreq <int>
	the test log-lik will be computed every 'rfreq' iterations
-eta <double>
	stepsize parameter
-step_schedule <int>
	stepsize schedule (0=advi, 1=rmsprop, 2=adagrad)
-saveCycle <int>
	save intermediate output files every 'saveCycle' iterations
-printInitMatrixVal
    save initial matrix values
-batchsize <int>
	number of datapoints per batch
-userVec <int>
	incorporate per-user vectors (theta_u)? (0=no, 3=yes)
-price <int>
	incorporate price? (0=no, integer=number of latent price components)
	if yes, the input file 'item_sess_price.tsv' is required
-priceThreshold <double>
	specify the minimum value of the prices (after normalization, if required)
	default: 0
-days <int>
	incorporate per-week effects? (0=no, integer=number of latent day components)
	if yes, the input file 'sess_days.tsv' is required
-weekdays
	include weekday_id in the model
	if active, the input files 'sess_days.tsv' and 'itemGroup.tsv' are required
	if active, -itemIntercept gets deactivated
-itemIntercept
	incorporate item intercepts (lambda0)
-UC <int>
	incorporate user observables? (0=no, integer=number of observables)
	if yes, the input file 'obsUser.tsv' is required
-IC <int>
	incorporate item observables? (0=no, integer=number of observables)
	if yes, the input file 'obsItem.tsv' is required
-obs2prior
	the item observables affect the prior on the item latent features
-obs2utility
	the item observables directly affect the utility
-ICgroups <int> <int>-<int> ... <int>-<int>
	defines groups of item observables
	first integer indicates number of groups
	the rest of parameters specify the range of each group
	[example: '-ICgroups 2 1-10 11-30' creates two groups, where group 1 has attributes 1 through 10 and group 2 has 11-30 (including the extremes)]
	[warning: ranges are 1-indexed, i.e., the first observed attribute corresponds to index 1]
	[only used if '-IC' and '-obs2prior' are specified]
-ICeffects <int> <int>-<int> ... <int>-<int>
	defines which latent features are influenced by each group of item observables
	first integer indicates number of groups
	the rest of parameters specify the range of latent factors for each group
	[example: '-ICeffects 2 1-5 6-10' indicates that group 1 affects latent features 1 through 5 and group 2 affects features 6-10]
	[example: '-ICeffects 2 1-5 -' indicates that group 1 affects latent features 1 through 5 and group 2 does not affect the prior over the latent features]
	[warning: keep all the ranges below K]
	[only used if '-IC' and '-obs2prior' are specified]
-ICeffectsPrice <int> <int>-<int> ... <int>-<int>
	defines which latent price features are influence by each group of item observables
	first integer indicates number of groups
	the rest of parameters specify the range of latent price factors for each group
	[example: '-ICeffects 2 - 1-5' indicates that group 1 does not affect the prior over the latent price vectors, and group 2 affects latent price factors 1 through 5]
	[warning: keep all the ranges below the value of '-price']
	[only used if '-IC', '-obs2prior', and '-price' are specified]
-s2obsPrior <double>
	variance of the prior over the latent features given the observables
	[only used if '-obs2prior' is specified]
-shuffle <int>
	shuffle the products in each visit? (-1=conditionally specified model, 0=don't shuffle, 1+=shuffle)
	[this is really not needed in the model we are considering for products, so simply set shuffle=0]
-likelihood <int>
	type of likelihood (0=bernoulli, 1=one-vs-each, 2=regularized softmax, 3=within-group softmax, 4=softmax)
	[use either 4 for exact softmax or 1 for one-vs-each (fast approximation to softmax), or 3 (within-group softmax)]
	[if 1, you need to specify 'negsamples' (see below)]
-negsamples <int>
	number of negative samples
	[this is ignored for likelihoods 3 and 4]
-valTolerance <double>
	stop when the change in validation log-likelihood is below this threshold
	[for now set this value to 0 so that it stops after 'max-iterations' iterations]
	[I still need to implement a better way to stop the algorithm]
-valConsecutive <double>
	stop when the validation log-likelihood increases for 'valConsecutive' times in a row
	[for now set this value larger than 'max-iterations' so that it stops after 'max-iterations' iterations]
	[I still need to implement a better way to stop the algorithm]
-keepAbove <int>
	keep only items with more than 'keepAbove' occurrences in the training set
-skipheader
	skip the first line of all input files
-label <string>
	string to be appended to the output folder name
-disableAutoLabel
  Normally runs are labeled based on the set of parameters used. disableAutoLabel causes the output folder to depend only
  on whatever is passed in as -label ( it gets output to outdir/emb-{label} ).
  This makes it easier to know exactly where the outputs will be created.

************
* EXAMPLES *
************

./emb -dir /proj/sml/usr/franrruiz/safegraph/dat -outdir /proj/sml/usr/franrruiz/safegraph/out -K 20 -max-iterations 20000 -rfreq 100 -eta 0.01 -step_schedule 0 -saveCycle 30000 -batchsize 2000 -userVec 3 -price 10 -days 5 -itemIntercept -shuffle 0 -likelihood 4 -negsamples 100 -valTolerance 0 -valConsecutive 100000 -skipheader -keepAbove 3 -label keepAbove3

./emb -dir /proj/sml/usr/franrruiz/safegraph/dat -outdir /proj/sml/usr/franrruiz/safegraph/out -K 20 -IC 179 -obs2prior -ICgroups 1 1-179 -ICeffects 1-20 -ICeffectsPrice 1-10 -max-iterations 20000 -rfreq 100 -eta 0.01 -step_schedule 0 -saveCycle 30000 -batchsize 2000 -userVec 3 -price 10 -days 5 -itemIntercept -shuffle 0 -likelihood 4 -negsamples 100 -valTolerance 0 -valConsecutive 100000 -s2obsPrior 0.01 -skipheader -keepAbove 3 -label keepAbove3-ICgroups1

[note: the parameters '-ICgroups', '-ICeffects', and '-ICeffectsPrice' are not reflected in the output file name. Use '-label' to assign a friendly name to each configuration]


*******************
* OTHER ARGUMENTS *
*******************

-seed <int>
	set the random seed
-nsFreq <int>
	choose the sampling scheme for "negative samples" (-1=uniform; 0=unigram; 1=unigram^(3/4); 2+=biased to item group)
-zeroFactor <double>
	downweight factor for the zeroes
	[ignored if likelihood takes values 1 or 4]
-checkout
	include a checkout item
-normPrice
	normalize prices by their mean values
-normPriceMin
	normalize prices by their minimum values
-normPriceVal <double>
	normalize prices by the provided value
-noVal
	don't use validation set
-noTest
	don't use test set
-noTrain
	don't report performance (log-likelihod, etc.) on training set
-gamma <double>
	coefficient for rmsprop
-stdIni <double>
	standard deviation used for initialization of the latent variables
-keepOnly <int>
	keep only the 'keepOnly' most frequent items
-noItemPriceLatents
    Treat item log prices as item observables with a single dimensional per user latent vector for this observable. Beta parameters (which are per item latent variables corresponding to price) are eliminated in this model, and gamma is called gammaObsItem in the code which indicates that it is a latent coefficient of an observed item characteristic (log price). Log price is different from other item attributes in that its value differs over shopping trips.

Hyperparameters
---------------
prior variance over the latent variables
-s2alpha <double>		(only if obs2prior is not used)
	[default: 1.0]
-s2beta <double>		(only if obs2prior is not used)
	[default: 1.0]
-s2theta <double>
	[default: 1.0]
-s2gamma <double>
	[default: 1.0]
-s2lambda <double>
	[default: 1.0]
-s2mu <double>
-s2delta <double>
	[default: 0.01]
-s2H <double>
	[default: 1.0]
-s2week <double>
	[default: 1.0]

prior mean over the latent variables
-meangamma <double>		shifts the mean of the normal distribution for gamma
	[default: 0.0]
-meanbeta <double>		shifts the mean of the normal distribution for gamma (only if obs2prior is not used)
	[default: 0.0]


**********
* OUTPUT *
**********

* Columns of valid.tsv, test.tsv, train.tsv:
	iteration   duration_seconds   log_likelihood   accuracy   precision   recall   f1score   total_instances

* Columns of param_innerProducts.tsv:
	user_id   item_id   val1   val2   val3

	where
	  val1 = lambda0_i+theta_u*alpha_i+obsItem_u*obsItem_i+obsUser_u*obsUser_i
	  val2 = lambda0_i+theta_u*alpha_i+obsItem_u*obsItem_i+obsUser_u*obsUser_i+mu_i*average_delta_w+average_weekday_i
	  val3 = gamma_u*beta_i


Accuracy is the fraction of correctly classified instances. Precision_i is the fraction of instances where we correctly declared `i` out of all instances where the algorithm declared `i` (then I take the average across all `i`). Recall is the fraction of instances where we correctly declared `i` out of all of the cases where the true of choice was `i` (also averaged across all `i`). F1-score is defined according to the formula `f1_score = 2*precision*recall / (precision+recall)`
