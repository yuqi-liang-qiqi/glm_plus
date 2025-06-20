# glmplus

`glmplus` is a Python package that extends traditional Generalized Linear Models (GLMs) to support statistically robust and practically useful variants that are not currently available in mainstream Python libraries.

Key features:
- 🟣 Rare Events Logistic Regression (`Relogit`) with finite-sample bias correction
- 🔵 Firth Penalized Logistic Regression (for small samples / separation)
- 🟢 Zero-Inflated Models (ZIP, ZINB)
- 🟠 Case-control correction for sampling imbalance
- 🟡 Survey-weighted GLMs (planned)
- 🧩 Compatible with scikit-learn-style APIs

Whether you're conducting applied research in social science, epidemiology, or any domain involving class imbalance or nonstandard count data, **glmplus** helps you obtain reliable and interpretable models beyond the defaults.

> Inspired by the [Zelig](http://docs.zeligproject.org/articles/zelig_relogit.html) framework in R and the methodological work by King & Zeng (2001).
