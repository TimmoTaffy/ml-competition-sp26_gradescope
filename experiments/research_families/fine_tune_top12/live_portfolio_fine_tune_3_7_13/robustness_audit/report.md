# Portfolio Parameter Robustness Audit

## Summary

| candidate | robustness_class | exact_rank | exact_final_score | rounded_final_score | rounded_delta_final_score | exact_raw_score | rounded_delta_raw_score | median_delta_raw_score | median_final_score | median_delta_final_score | p25_final_score | pct_within_0_02 | pct_within_0_05 | pct_within_raw_0_001 | pct_within_raw_0_0025 | neighbor_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced_extra_wide_capacity | stable | 407 | 0.562877 | 0.485736 | -0.077141 | 0.005980 | -0.000642 | -0.000862 | 0.481553 | -0.081323 | 0.432530 | 0.061224 | 0.244898 | 0.673469 | 1.000000 | 49 |
| recent_compact_current | acceptable | 483 | 0.526911 | 0.460101 | -0.066810 | 0.007060 | -0.000914 | -0.001746 | 0.408269 | -0.118642 | 0.336632 | 0.061224 | 0.183673 | 0.326531 | 0.571429 | 49 |
| stable_medium_capacity | acceptable | 804 | 0.398070 | 0.321755 | -0.076315 | 0.005226 | -0.001251 | -0.001137 | 0.332066 | -0.066004 | 0.318391 | 0.062500 | 0.281250 | 0.421875 | 1.000000 | 64 |
| stable_compact_current | fragile | 1 | 0.860241 | 0.855393 | -0.004848 | 0.010422 | -0.000067 | -0.002442 | 0.651210 | -0.209031 | 0.589098 | 0.060714 | 0.085714 | 0.082143 | 0.517857 | 280 |
| recent_compact_capacity | fragile | 77 | 0.732464 | 0.678454 | -0.054010 | 0.010033 | -0.000877 | -0.002701 | 0.520131 | -0.212333 | 0.452135 | 0.025000 | 0.070000 | 0.100000 | 0.430000 | 200 |
| balanced_medium_current | fragile | 107 | 0.706388 | 0.693524 | -0.012864 | 0.008753 | -0.000078 | -0.002891 | 0.499537 | -0.206852 | 0.409682 | 0.071429 | 0.071429 | 0.071429 | 0.392857 | 28 |
| short_aggressive_wide_capacity | fragile | 111 | 0.704716 | 0.644700 | -0.060016 | 0.008748 | -0.000534 | -0.002444 | 0.462762 | -0.241954 | 0.404286 | 0.013393 | 0.044643 | 0.156250 | 0.517857 | 224 |
| stable_extra_wide_capacity | fragile | 145 | 0.687195 | 0.589626 | -0.097570 | 0.008113 | -0.001891 | -0.001894 | 0.581445 | -0.105750 | 0.497668 | 0.035714 | 0.125000 | 0.196429 | 0.714286 | 56 |
| short_aggressive_compact_capacity | fragile | 551 | 0.500269 | 0.461729 | -0.038540 | 0.005560 | -0.000239 | -0.001463 | 0.375481 | -0.124788 | 0.324579 | 0.024390 | 0.097561 | 0.195122 | 0.780488 | 41 |
| short_aggressive_medium_capacity | fragile | 552 | 0.500269 | 0.461729 | -0.038540 | 0.005560 | -0.000239 | -0.001463 | 0.375481 | -0.124788 | 0.324579 | 0.024390 | 0.097561 | 0.195122 | 0.780488 | 41 |
| economic_medium_capacity | fragile | 1009 | 0.321434 | 0.282561 | -0.038873 | 0.003326 | -0.001058 | -0.002465 | 0.173098 | -0.148336 | 0.083628 | 0.064286 | 0.121429 | 0.107143 | 0.500000 | 140 |
| stable_compact_capacity | spiky | 76 | 0.732702 | 0.363665 | -0.369037 | 0.008562 | -0.004776 | -0.002180 | 0.526606 | -0.206096 | 0.373837 | 0.069767 | 0.139535 | 0.302326 | 0.627907 | 43 |

## Exact And Rounded Rows

| candidate | audit_kind | final_score | mean_3w | mean_7w | mean_13w | hit_7w | worst_13w | model0_weight | top_k | internal_max_weight | risk_filter_enabled | weighting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stable_compact_current | exact | 0.860241 | 0.025358 | 0.012828 | 0.009603 | 0.571429 | -0.016028 | 0.546500 | 31 | 0.031758 | True | rank |
| stable_compact_current | rounded | 0.855393 | 0.025273 | 0.012791 | 0.009508 | 0.571429 | -0.016033 | 0.550000 | 31 | 0.032000 | True | rank |
| stable_compact_capacity | exact | 0.732702 | 0.020242 | 0.011442 | 0.006138 | 0.714286 | -0.019857 | 0.978000 | 33 | 0.030303 | True | equal |
| recent_compact_capacity | exact | 0.732464 | 0.019930 | 0.013559 | 0.004208 | 1.000000 | -0.036608 | 0.055000 | 31 | 0.030758 | True | rank |
| balanced_medium_current | exact | 0.706388 | 0.017938 | 0.010486 | 0.008989 | 0.857143 | -0.026803 | 0.000000 | 49 | 0.030000 | True | equal |
| short_aggressive_wide_capacity | exact | 0.704716 | 0.023493 | 0.009775 | 0.005904 | 0.571429 | -0.018214 | 0.593500 | 30 | 0.036000 | True | rank |
| balanced_medium_current | rounded | 0.693524 | 0.017779 | 0.010381 | 0.008730 | 0.857143 | -0.027064 | 0.000000 | 50 | 0.030000 | True | equal |
| stable_extra_wide_capacity | exact | 0.687195 | 0.017963 | 0.014538 | 0.007822 | 0.857143 | -0.023449 | 0.285000 | 46 | 0.030000 | False | equal |
| recent_compact_capacity | rounded | 0.678454 | 0.018883 | 0.013079 | 0.003196 | 1.000000 | -0.038397 | 0.050000 | 31 | 0.030000 | True | rank |
| short_aggressive_wide_capacity | rounded | 0.644700 | 0.022882 | 0.009003 | 0.005301 | 0.571429 | -0.018031 | 0.600000 | 30 | 0.035000 | True | rank |
| stable_extra_wide_capacity | rounded | 0.589626 | 0.016781 | 0.011454 | 0.006811 | 0.857143 | -0.022983 | 0.300000 | 45 | 0.030000 | False | equal |
| balanced_extra_wide_capacity | exact | 0.562877 | 0.013275 | 0.010546 | 0.005973 | 0.857143 | -0.021252 | 0.725000 | 64 | 0.030000 | False | equal |
| recent_compact_current | exact | 0.526911 | 0.009355 | 0.010408 | 0.006329 | 1.000000 | -0.016834 | 0.825000 | 37 | 0.030000 | True | equal |
| short_aggressive_compact_capacity | exact | 0.500269 | 0.012536 | 0.011256 | 0.006868 | 0.857143 | -0.024507 | 0.945500 | 31 | 0.032258 | True | equal |
| short_aggressive_medium_capacity | exact | 0.500269 | 0.012536 | 0.011256 | 0.006868 | 0.857143 | -0.024507 | 0.945500 | 31 | 0.032258 | True | equal |
| balanced_extra_wide_capacity | rounded | 0.485736 | 0.012876 | 0.009715 | 0.005010 | 0.857143 | -0.022712 | 0.700000 | 65 | 0.030000 | False | equal |
| short_aggressive_compact_capacity | rounded | 0.461729 | 0.012536 | 0.010420 | 0.006581 | 0.857143 | -0.024507 | 0.950000 | 31 | 0.032000 | True | equal |
| short_aggressive_medium_capacity | rounded | 0.461729 | 0.012536 | 0.010420 | 0.006581 | 0.857143 | -0.024507 | 0.950000 | 31 | 0.032000 | True | equal |
| recent_compact_current | rounded | 0.460101 | 0.007904 | 0.009328 | 0.005957 | 1.000000 | -0.016834 | 0.800000 | 37 | 0.030000 | True | equal |
| stable_medium_capacity | exact | 0.398070 | 0.011591 | 0.005089 | 0.003124 | 0.857143 | -0.013304 | 0.772500 | 93 | 0.030000 | True | equal |
| stable_compact_capacity | rounded | 0.363665 | 0.011676 | 0.008590 | 0.002987 | 0.714286 | -0.025884 | 1.000000 | 33 | 0.030000 | True | equal |
| stable_medium_capacity | rounded | 0.321755 | 0.010329 | 0.003980 | 0.002244 | 0.714286 | -0.014298 | 0.750000 | 90 | 0.030000 | True | equal |
| economic_medium_capacity | exact | 0.321434 | 0.020039 | 0.003098 | 0.004267 | 0.571429 | -0.018529 | 0.998000 | 30 | 0.036000 | True | rank |
| economic_medium_capacity | rounded | 0.282561 | 0.017095 | 0.005113 | 0.003688 | 0.571429 | -0.015148 | 1.000000 | 30 | 0.035000 | True | rank |

## Interpretation

- `stable`: rounded and nearby settings remain close to the exact tuned score.
- `acceptable`: some degradation, but not a single-parameter spike.
- `spiky`: exact setting is strong but rounded/nearby settings degrade sharply.
- `fragile`: weak or unstable neighborhood.
