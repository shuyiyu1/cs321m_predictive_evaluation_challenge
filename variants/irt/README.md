# IRT-Style Submission

This submission predicts with a psychometric decomposition:

`logit P(correct) = global + subject ability + predicted item easiness + benchmark/condition effects`.

The item easiness model is trained on aggregated public item pass rates, rather
than raw subject-item response rows, so it is designed for the challenge's
item-cold-start setting. Runtime imports use only the Python standard library.

