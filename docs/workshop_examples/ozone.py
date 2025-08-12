import matplotlib.pyplot as plt
import pandas as pd

from pymgcv.families import Gamma, Gaussian
from pymgcv.gam import GAM
from pymgcv.plot import plot_gam, plot_qq, plot_residuals_vs_linear_predictor
from pymgcv.terms import S
from pymgcv.utils import load_rdata_dataframe_from_url

data = load_rdata_dataframe_from_url(
    "https://github.com/mfasiolo/testGam/raw/master/data/ozone.rda",
)
pd.plotting.scatter_matrix(data)
plt.show()


data.columns = [c.lower().replace(".", "_") for c in data.columns]
data.head()

# We'll reuse the smooth terms across a few models,
# o3 as a function of the other variables
predictor = {
    "o3": S("vh")
    + S("wind")
    + S("humidity")
    + S("temp")
    + S("ibh")
    + S("dpg")
    + S("ibt")
    + S("vis")
    + S("doy"),
}

# Fit a guassian gam and plot fitted effects
gam_gaussian_log_link = GAM(predictor, family=Gaussian(link="log"))
gam_gaussian_log_link.fit(data)

plot_gam(gam_gaussian_log_link, ncols=3)
plt.show()

# Diagnostic plots
fig, axes = plt.subplots(ncols=2, layout="constrained")
plot_qq(gam_gaussian_log_link, ax=axes[0])
plot_residuals_vs_linear_predictor(gam_gaussian_log_link, ax=axes[1])
fig.suptitle("Diagnostic plots: Gaussian GAM with log link")
fig.show()

# We can see the residuals get larger with the linear predictor
# so we can try the gamma response distribution
gam_gamma_log_link = GAM(predictor, family=Gamma(link="log"))
gam_gamma_log_link.fit(data)
fig, axes = plt.subplots(ncols=2, layout="constrained")
plot_qq(gam_gamma_log_link, ax=axes[0])
plot_residuals_vs_linear_predictor(gam_gamma_log_link, ax=axes[1])
fig.suptitle("Diagnostic plots: Gamma GAM with log link")
fig.show()


# Residuals look better.
# Let's test against identity link
gam_gamma_id_link = GAM(predictor, family=Gamma(link="identity"))
gam_gamma_id_link.fit(data)

models = {
    "gamma log link": gam_gamma_log_link,
    "gamma identity link": gam_gamma_id_link,
}

for name, model in models.items():
    print(f"{name}: AIC = {model.aic():.2f}, total EDF = {sum(model.edf()):.2f}")

# Multiplicative model (log link) is better (lower AIC)
print(gam_gamma_log_link.summary())

# We can drop ibt which has the highest p value,
# and consider dropping humidity

models = {
    "no_ibt": GAM(
        {
            "o3": S("vh")
            + S("wind")
            + S("humidity")
            + S("temp")
            + S("ibh")
            + S("dpg")
            + S("vis")
            + S("doy"),
        },
    ),
    "no_ibt_or_humidity": GAM(
        {
            "o3": S("vh")
            + S("wind")
            + S("temp")
            + S("ibh")
            + S("dpg")
            + S("vis")
            + S("doy"),
        },
    ),
}
for name, model in models.items():
    model.fit(data)
    print(f"{name}: AIC = {model.aic():.2f}, total EDF = {sum(model.edf()):.2f}")
