# Statistical Comparison Report: PINN vs Baseline Models

**Generated**: 2026-01-29 18:51:56.653518

---

## Overall Performance

```
                   rmse       mae       mape  r2_score  sharpe_ratio  sortino_ratio  calmar_ratio  max_drawdown  total_return  win_rate  directional_accuracy
model                                                                                                                                                        
lstm           1.048011  0.476430  10.096059  0.801837           NaN            NaN           NaN           NaN           NaN       NaN             51.398829
gru            1.176313  0.433225   8.002935  0.750346           NaN            NaN           NaN           NaN           NaN       NaN             51.431360
transformer    1.295408  0.554728  11.925508  0.697235           NaN            NaN           NaN           NaN           NaN       NaN             51.106051
pinn_baseline  1.021955  0.355332   6.583410  0.811568      0.435011       1.936467           NaN   -332.752620   -546.550688  0.283722              0.512687
pinn_gbm       1.074543  0.390600   7.284619  0.791676      0.392836       1.753924           NaN   -113.349476    -96.252630  0.277886              0.512036
pinn_ou        0.979064  0.348500   6.661631  0.827053      0.424091       1.880010           NaN   -345.992593   -351.542414  0.286965              0.517892
pinn_global    0.855631  0.355048   6.991471  0.867911      0.406288       1.822422           NaN   -273.507407   -165.023545  0.275292              0.509759
pinn_gbm_ou    0.930748  0.350859   7.123376  0.843701      0.409517       1.825200           NaN   -202.762848   -183.473143  0.282750              0.512036
```

---

## pinn_global vs lstm

```
              metric  pinn_global      lstm  difference  improvement_%      winner     t_pvalue  wilcoxon_pvalue   cohens_d significant
                rmse     0.855631  1.048011   -0.192380      18.356649 pinn_global 5.154159e-07     1.303852e-07  -1.801570           ✓
                 mae     0.355048  0.476430   -0.121382      25.477372 pinn_global 2.004473e-15     1.862645e-09  -3.315101           ✓
                mape     6.991471 10.096059   -3.104588      30.750495 pinn_global 7.641142e-18     1.862645e-09  -4.031241           ✓
            r2_score     0.867911  0.801837    0.066075       8.240446 pinn_global 1.887523e-01     4.048972e-02   0.531218           ✗
directional_accuracy     0.509759 51.398829  -50.889070     -99.008228        lstm 2.892180e-32     1.862645e-09 -13.005603           ✓
```

---

## pinn_gbm vs lstm

```
              metric  pinn_gbm      lstm  difference  improvement_%   winner     t_pvalue  wilcoxon_pvalue   cohens_d significant
                rmse  1.074543  1.048011    0.026532      -2.531613     lstm 8.554017e-01     2.285528e-01   0.536352           ✗
                 mae  0.390600  0.476430   -0.085830      18.015144 pinn_gbm 1.947919e-08     5.718321e-07  -2.112981           ✓
                mape  7.284619 10.096059   -2.811440      27.846906 pinn_gbm 5.580713e-12     1.862645e-09  -2.976777           ✓
            r2_score  0.791676  0.801837   -0.010160      -1.267147     lstm 7.564308e-01     5.698576e-01   0.546922           ✗
directional_accuracy  0.512036 51.398829  -50.886792     -99.003797     lstm 2.744609e-29     1.862645e-09 -13.737706           ✓
```

---

## pinn_ou vs lstm

```
              metric  pinn_ou      lstm  difference  improvement_%  winner     t_pvalue  wilcoxon_pvalue   cohens_d significant
                rmse 0.979064  1.048011   -0.068947       6.578856 pinn_ou 6.923558e-02     4.279546e-01  -0.955168           ✗
                 mae 0.348500  0.476430   -0.127930      26.851701 pinn_ou 6.030370e-13     9.313226e-09  -3.773871           ✓
                mape 6.661631 10.096059   -3.434428      34.017509 pinn_ou 6.776572e-14     1.862645e-09  -4.683786           ✓
            r2_score 0.827053  0.801837    0.025216       3.144793 pinn_ou 5.475718e-01     4.044945e-01   0.466845           ✗
directional_accuracy 0.517892 51.398829  -50.880937     -98.992405    lstm 2.823736e-31     1.862645e-09 -16.993917           ✓
```

---

## pinn_global vs gru

```
              metric  pinn_global       gru  difference  improvement_%      winner     t_pvalue  wilcoxon_pvalue   cohens_d significant
                rmse     0.855631  1.176313   -0.320682      27.261598 pinn_global 1.165898e-11     3.725290e-09  -3.279630           ✓
                 mae     0.355048  0.433225   -0.078177      18.045272 pinn_global 3.085673e-09     8.326024e-07  -1.819247           ✓
                mape     6.991471  8.002935   -1.011465      12.638670 pinn_global 5.228557e-05     1.528710e-04  -0.834783           ✓
            r2_score     0.867911  0.750346    0.117565      15.668092 pinn_global 2.248412e-05     5.548261e-04   1.477489           ✓
directional_accuracy     0.509759 51.431360  -50.921601     -99.008855         gru 5.474028e-32     1.862645e-09 -15.456639           ✓
```

---

## pinn_global vs transformer

```
              metric  pinn_global  transformer  difference  improvement_%      winner     t_pvalue  wilcoxon_pvalue   cohens_d significant
                rmse     0.855631     1.295408   -0.439777      33.948902 pinn_global 8.917636e-18     1.862645e-09  -3.461992           ✓
                 mae     0.355048     0.554728   -0.199679      35.995936 pinn_global 1.240541e-15     1.862645e-09  -4.319148           ✓
                mape     6.991471    11.925508   -4.934037      41.373809 pinn_global 2.968329e-22     1.862645e-09  -4.348073           ✓
            r2_score     0.867911     0.697235    0.170676      24.478971 pinn_global 5.487223e-12     3.539026e-08   2.421134           ✓
directional_accuracy     0.509759    51.106051  -50.596291     -99.002546 transformer 4.475477e-28     1.862645e-09 -11.928109           ✓
```

---

