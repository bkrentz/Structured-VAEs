# Forecasting

This is the main project.

The code simulates a Poisson LDS, fits a vanilla structured VAE and a Bayesian/ARD version, then forecasts the held-out end of each sequence (the suffix) from the observed start (the prefix).

Forecasting works by inferring latents from the observed prefix, and then rolling learned dynamics forward.

## Files

- `data.py` makes the synthetic data.
- `model.py` holds the parameter classes and the recognition/decoder networks.
- `inference.py` does Gaussian smoothing and sufficient statistics.
- `free_energy.py` computes the free energy terms.
- `train.py` fits the point and Bayesian models.
- `forecast.py` does prefix inference and suffix forecasting.
- `diagnostics.py` and `viz.py` contain metrics and plots.
- `svae_forecasting.py` re-exports the public functions for the notebook.


From this directory, run checks with:

```bash
python -c "import svae_forecasting as sf; print(sf.run_self_checks())"
```
