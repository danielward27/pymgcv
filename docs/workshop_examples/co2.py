if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    from pymgcv.basis_functions import CubicSpline
    from pymgcv.gam import GAM, AbstractGAM
    from pymgcv.terms import S
    from pymgcv.utils import load_rdata_dataframe_from_url

    data = load_rdata_dataframe_from_url(
        "https://github.com/cran/gamair/raw/master/data/co2s.rda",
    )

    plt.plot(data["c.month"], data["co2"])
    plt.show()

    gam = GAM({"co2": S("c.month", bs=CubicSpline(), k=100)})
    gam.fit(data)

    # Use it for prediction
    predict_data = {
        "c.month": np.arange(1, 544),
        "month": (np.arange(0, 543) % 12) + 1,
    }

    def plot_prediction(gam: AbstractGAM, data, predict_data):
        predictions = gam.predict(data=predict_data, compute_se=True)
        plt.plot(data["c.month"], data["co2"], label="data", color="black")
        plt.plot(
            predict_data["c.month"],
            predictions["co2"].fit,
            label="predictions",
            color="tab:orange",
        )
        plt.fill_between(
            predict_data["c.month"],
            predictions["co2"].fit - 2 * predictions["co2"].se,
            predictions["co2"].fit + 2 * predictions["co2"].se,
            alpha=0.2,
            color="tab:orange",
        )
        plt.legend()
        plt.show()

    plot_prediction(gam, data, predict_data)

    # Nonsense extrapolation! Fit a better model
    # Specify cyclical month component and overall time component

    gam = GAM(
        {
            "co2": S("c.month", bs=CubicSpline(), k=50)
            + S("month", bs=CubicSpline(cyclic=True), k=12),
        },
    )
    gam.fit(data, knots={"month": np.arange(1, 13)})
    plot_prediction(gam, data, predict_data)

    # More reasonable extrapolation!
