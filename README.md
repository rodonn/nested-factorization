# nested-factorization
Implementation of "Counterfactual Inference for Consumer Choice Across Many Product Categories"


The directory src/bemb\_loc contains the primary Nested Factorization model used in the paper.
As described in the text, the model needs to be used twice in order to estimate the nesting structure.
First to estimate product level preferences and then a second run to estimate category level preferences.

The directory src/hpf contains a modified version of the original HPF model that allows for observed characteristics of products
and also allows for products to be unavailable during some of a user's shopping sessions.

Related package implemented in Python:
[hpfrec](https://github.com/david-cortes/hpfrec)
