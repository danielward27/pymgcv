# # # """Diagnostics and model comparison."""

# # # import pandas as pd
# # # from rpy2.robjects.packages import importr

# # # from pymgcv.converters import to_py
# # # from pymgcv.gam import AbstractGAM

# # # rmgcv = importr("mgcv")
# # # rstats = importr("stats")


# # def aic(*models: AbstractGAM, k: float = 2) -> pd.DataFrame:
# #     rgams = []
# #     for m in models:
# #         if m.fit_state is None:
# #             raise ValueError("Cannot compute AIC before fitting.")
# #         rgams.append(m.fit_state.rgam)
# #     res = rstats.AIC(*rgams, k=k)
# #     return pd.DataFrame(
# #         {"AIC": to_py(res.rx2["AIC"]), "df": to_py(res.rx2["df"])},
# #     )


# # # pd.DataFrame(

# # # )

# # # def degrees_of_freedom(gam: AbstractGAM) -> pd.DataFrame:
# # #     """Calculate degrees of freedom for fitted GAM models.


# # def log_likelihood(gam: AbstractGAM) -> float:
# #     if gam.fit_state is None:
# #         raise ValueError("Cannot compute log-likelihood before fitting.")
# #     return rstats.logLik(gam.fit_state.rgam)
