# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-dependent-fs: Dependent Frequency-Severity Neural Model
# MAGIC
# MAGIC This notebook demonstrates the full workflow:
# MAGIC 1. Generate synthetic motor claims data with known frequency-severity dependence (γ = -0.15)
# MAGIC 2. Fit the shared-trunk neural model
# MAGIC 3. Check that the model recovers γ
# MAGIC 4. Compare pure premium predictions against the independence assumption
# MAGIC 5. Run diagnostics: Lorenz curves, calibration, latent correlation

# COMMAND ----------

# MAGIC %pip install insurance-dependent-fs matplotlib

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from insurance_dependent_fs import (
    DependentFSModel,
    DependentFSDiagnostics,
    SharedTrunkConfig,
    make_dependent_claims,
    make_independent_claims,
)
from insurance_dependent_fs.benchmarks import feature_cols
from insurance_dependent_fs.training import TrainingConfig

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic data with known dependence

# COMMAND ----------

TRUE_GAMMA = -0.15
N_POLICIES = 30_000

df_train, df_test = make_dependent_claims(
    n_policies=N_POLICIES,
    gamma=TRUE_GAMMA,
    base_freq=0.08,
    base_sev=3_000.0,
    phi=1.5,
    n_features=6,
    seed=42,
)

fc = feature_cols(df_train)

print(f"Training set:  {len(df_train):,} policies")
print(f"Test set:      {len(df_test):,} policies")
print(f"Claim rate:    {df_train['n_claims'].mean():.3f}")
print(f"Mean severity: £{df_train.loc[df_train['n_claims']>0, 'avg_severity'].mean():.0f}")
print(f"Mean pure prem:£{(df_train['n_claims'] * df_train['avg_severity'] / df_train['exposure']).mean():.0f}")

# COMMAND ----------

# Verify the induced negative correlation
pos = df_train["n_claims"] > 0
corr = np.corrcoef(df_train.loc[pos, "n_claims"], df_train.loc[pos, "avg_severity"])[0, 1]
print(f"Correlation(N, avg_severity | N>0) = {corr:.3f}  (expected negative)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fit the dependent model

# COMMAND ----------

model = DependentFSModel(
    trunk_config=SharedTrunkConfig(
        hidden_dims=[128, 64],
        latent_dim=32,
        dropout=0.1,
        activation="elu",
        use_batch_norm=True,
    ),
    training_config=TrainingConfig(
        max_epochs=80,
        batch_size=512,
        lr=1e-3,
        auto_balance=True,
        patience=12,
        verbose=True,
    ),
    use_explicit_gamma=True,
    val_fraction=0.1,
    random_state=42,
)

model.fit(
    df_train[fc].values.astype(np.float32),
    df_train["n_claims"].values,
    df_train["avg_severity"].values,
    df_train["exposure"].values,
)

print(f"\nRecovered γ = {model.gamma_:.4f}  (true γ = {TRUE_GAMMA})")
print(f"Epochs run:  {len(model.training_history()['train_loss'])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Training history

# COMMAND ----------

hist = model.training_history()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(hist["train_loss"], label="Train")
axes[0].plot(hist["val_loss"], label="Val")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Joint loss")
axes[0].set_title("Training history")
axes[0].legend()

axes[1].plot(hist["gamma"])
axes[1].axhline(TRUE_GAMMA, linestyle="--", color="red", label=f"True γ={TRUE_GAMMA}")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("γ")
axes[1].set_title("Dependence parameter convergence")
axes[1].legend()

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Pure premium predictions

# COMMAND ----------

X_test = df_test[fc].values.astype(np.float32)
exp_test = df_test["exposure"].values

pp_analytical = model.predict_pure_premium(X_test, exp_test, method="analytical")
pp_mc = model.predict_pure_premium(X_test, exp_test, method="mc", n_mc=2000)

actual_pp = df_test["n_claims"].values * df_test["avg_severity"].values / df_test["exposure"].values

print(f"Mean actual pure premium:     £{actual_pp.mean():.2f}")
print(f"Mean predicted (analytical):  £{pp_analytical.mean():.2f}")
print(f"Mean predicted (MC):          £{pp_mc.mean():.2f}")
print(f"\nAnalytical vs MC correlation:  {np.corrcoef(pp_analytical, pp_mc)[0,1]:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Diagnostics

# COMMAND ----------

diag = DependentFSDiagnostics(
    model,
    X_test,
    df_test["n_claims"].values,
    df_test["avg_severity"].values,
    df_test["exposure"].values,
)

# COMMAND ----------

# Gini coefficients
gini = diag.gini_summary()
print("Gini coefficients (model lift):")
for k, v in gini.items():
    print(f"  {k}: {v:.4f}")

# COMMAND ----------

# Lorenz curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
diag.plot_lorenz(target="frequency", ax=axes[0])
diag.plot_lorenz(target="pure_premium", ax=axes[1])
plt.tight_layout()
plt.show()

# COMMAND ----------

# Calibration plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
diag.plot_calibration(target="frequency", ax=axes[0])
diag.plot_calibration(target="pure_premium", ax=axes[1])
plt.tight_layout()
plt.show()

# COMMAND ----------

# Latent correlation analysis
lc = diag.latent_correlation()
print(f"Latent dimensions active for frequency: {lc['n_freq_active']}")
print(f"Latent dimensions active for severity:  {lc['n_sev_active']}")

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(lc["latent_corr"], cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_title("Latent dimension correlation matrix")
ax.set_xlabel("Latent dimension")
ax.set_ylabel("Latent dimension")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Comparison vs independence assumption

# COMMAND ----------

comparison = diag.vs_independent(n_mc=1000)
print("Dependent vs independent model comparison:")
for k, v in comparison.items():
    print(f"  {k}: {v:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Null check: model on independent data

# COMMAND ----------

df_null_train, df_null_test = make_independent_claims(
    n_policies=10_000, n_features=6, seed=10
)
fc_null = feature_cols(df_null_train)

model_null = DependentFSModel(
    trunk_config=SharedTrunkConfig(hidden_dims=[64, 32], latent_dim=16, use_batch_norm=False),
    training_config=TrainingConfig(max_epochs=30, verbose=False, patience=5, auto_balance=True),
    use_explicit_gamma=True,
    val_fraction=0.1,
)
model_null.fit(
    df_null_train[fc_null].values.astype(np.float32),
    df_null_train["n_claims"].values,
    df_null_train["avg_severity"].values,
    df_null_train["exposure"].values,
)
print(f"Null model (γ=0 data) recovered γ = {model_null.gamma_:.4f}  (should be near 0)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary
# MAGIC
# MAGIC The model:
# MAGIC - Correctly recovers the sign and approximate magnitude of γ from synthetic data
# MAGIC - Trains stably with joint Poisson-Gamma loss via auto-balancing
# MAGIC - Analytical and MC pure premium methods agree closely
# MAGIC - Gini coefficients confirm the model has meaningful lift
# MAGIC - On independent data, γ converges near zero (no spurious dependence)
