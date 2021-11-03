import datetime
from pathlib import Path

from fbprophet import Prophet
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

########################################################################################

COMPLETLE_COLUMN_LIST = [
    "ds",
    "trend",
    "yhat_lower",
    "yhat_upper",
    "trend_lower",
    "trend_upper",
    "additive_terms",
    "additive_terms_lower",
    "additive_terms_upper",
    "daily",
    "daily_lower",
    "daily_upper",
    "weekly",
    "weekly_lower",
    "weekly_upper",
    "yearly",
    "yearly_lower",
    "yearly_upper",
    "multiplicative_terms",
    "multiplicative_terms_lower",
    "multiplicative_terms_upper",
    "yhat",
    "y",
    "trend_changepoint",
]


def create_prophet_dataframe(
    agg_temperatur_df: pd.DataFrame,
    ds: str,
    y: str,
    yearly=True,
    weekly=True,
    daily=True,
    freq="H",
    number_freq_into_future=100,
) -> pd.DataFrame:

    prophet_df = agg_temperatur_df.loc[:, [ds, y]].rename(columns={ds: "ds", y: "y"})

    model = Prophet(
        growth="linear",
        changepoints=None,
        n_changepoints=25,
        changepoint_range=0.8,
        yearly_seasonality=yearly,
        weekly_seasonality=weekly,
        daily_seasonality=daily,
        holidays=None,
        seasonality_mode="additive",
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        changepoint_prior_scale=0.05,
        mcmc_samples=0,
        interval_width=0.8,
        uncertainty_samples=1000,
        stan_backend=None,
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=number_freq_into_future, freq=freq)

    forecast = model.predict(future)

    forecast = pd.merge(forecast, prophet_df, left_on="ds", right_on="ds", how="left")

    change_points_df = forecast.loc[:, ["ds", "trend"]]
    change_points_df.loc[:, "tangent"] = (
        forecast.trend - forecast.trend.shift(1)
    ).round(5)

    _grp_ = change_points_df.groupby("tangent")

    change_points_df = pd.DataFrame(
        dict(
            ds_changepoint=_grp_.ds.last(), trend_changepoint=_grp_.trend.last()
        )  # test
    ).reset_index()

    forecast = pd.merge(
        forecast,
        change_points_df.drop(columns="tangent"),
        left_on="ds",
        right_on="ds_changepoint",
        how="left",
    ).drop(columns="ds_changepoint")

    for x in COMPLETLE_COLUMN_LIST:
        if x not in forecast.columns:
            forecast[x] = 0

    return forecast


########################################################################################


def plot_forecast(forecast: pd.DataFrame, analysis_dir: Path) -> None:
    fig = make_subplots(
        rows=3,
        cols=3,
        specs=[
            [{"rowspan": 1, "colspan": 2}, None, {}],
            [{"rowspan": 1, "colspan": 2}, None, {}],
            [{}, {}, {}],
        ],
        subplot_titles=(
            "Empirical Data + Trendline",
            "Component: Daily Seasonality",
            "Y_hat",
            "Component: Weekly Saisonality",
            "Residuals (y - y_hat)",
            "Violin Plot: Residuals",
            "Component: Yearly Saisonality",
        ),
        print_grid=True,
        horizontal_spacing=0.02,
        vertical_spacing=0.06,
    )

    # big one: #############################################################################

    fig.add_trace(
        go.Scatter(
            x=forecast.ds,
            y=forecast.trend_lower,
            mode="lines",
            showlegend=False,
            line=dict(width=1.0, color="rgb(176,196,222,0)"),  # rgb(176,196,222)
            fillcolor="rgba(176,196,222, 0.0)",
            stackgroup="one",  # define stack group
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.ds,
            y=forecast.trend_upper,
            mode="lines",
            showlegend=False,
            line=dict(width=1.0, color="rgb(176,196,222,0.75)"),
            fillcolor="rgba(176,196,222, 0.5)",
            stackgroup="one",  # define stack group
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=forecast.ds,
            y=forecast.trend,
            showlegend=False,
            line_color="blue",
            name="trend",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=forecast.ds,
            y=forecast.trend_changepoint,
            mode="markers",
            marker=dict(
                color="LightSkyBlue", size=10, line=dict(color="MediumPurple", width=2)
            ),
            name="change point",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=forecast.ds,
            y=forecast["y"],
            name="y",
            mode="lines",
            line_color="red",
            opacity=0.75,
        ),
        row=1,
        col=1,
    )

    # big one ##############################################################################
    row_yhat = 2
    col_yhat = 1

    fig.add_trace(
        go.Scatter(
            x=forecast.ds,
            y=forecast.yhat_lower,
            mode="lines",
            showlegend=False,
            line=dict(width=1.0, color="rgb(176,196,222,0.75)"),
            fillcolor="rgba(176,196,222, 0.0)",
            stackgroup="one",  # define stack group
        ),
        row=row_yhat,
        col=col_yhat,
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.ds,
            y=forecast.yhat_upper,
            mode="lines",
            line=dict(width=1.0, color="rgb(176,196,222,0.75)"),
            fillcolor="rgba(176,196,222, 0.5)",
            showlegend=False,
            name="y_hat Margin of ",
            stackgroup="one",  # define stack group
        ),
        row=row_yhat,
        col=col_yhat,
    )

    fig.add_trace(
        go.Scatter(
            x=forecast.ds,
            y=forecast.yhat,
            name="yhat",
            showlegend=False,
            line_color="blue",
            opacity=0.5,
        ),
        row=row_yhat,
        col=col_yhat,
    )

    # samll one: ###########################################################################

    fig.add_trace(
        go.Scatter(
            x=forecast.ds,
            y=forecast.apply(lambda row: row.y - row.yhat, axis=1),
            name="Residual (y-yhat)",
            line_color="blue",
            opacity=0.75,
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    # small one: ###########################################################################
    fig.add_trace(
        go.Violin(
            x=forecast.apply(lambda row: row.y - row.yhat, axis=1),
            name="",
            box_visible=True,
            meanline_visible=True,
            # points="all",
            showlegend=False,
            line_color="blue",
            fillcolor="rgba(176,196,222, 0.5)",
        ),
        row=3,
        col=2,
    )

    # small one: ###########################################################################
    row_weekly = 2
    col_weekly = 3

    fig.add_trace(
        go.Scatter(
            x=forecast.ds,
            y=forecast.weekly_lower,
            mode="lines",
            showlegend=False,
            line=dict(width=1.0, color="rgb(176,196,222,0.75)"),
            fillcolor="rgba(176,196,222, 0.0)",
            stackgroup="one",  # define stack group
        ),
        row=row_weekly,
        col=col_weekly,
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.ds,
            y=forecast.weekly_upper,
            mode="lines",
            showlegend=False,
            line=dict(width=1.0, color="rgb(176,196,222,0.75)"),
            fillcolor="rgba(176,196,222, 0.5)",
            stackgroup="one",  # define stack group
        ),
        row=row_weekly,
        col=col_weekly,
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.ds,
            y=forecast.weekly,
            name="weekly",
            mode="lines",
            showlegend=False,
            line_color="blue",
            opacity=0.5,
        ),
        row=row_weekly,
        col=col_weekly,
    )

    # small one: ###########################################################################
    row_daily = 1
    col_daily = 3

    fig.add_trace(
        go.Scatter(
            x=forecast.ds,
            y=forecast.daily_lower,
            mode="lines",
            showlegend=False,
            line=dict(width=1.0, color="rgb(176,196,222,0.75)"),
            fillcolor="rgba(176,196,222, 0.0)",
            stackgroup="one",  # define stack group
        ),
        row=row_daily,
        col=col_daily,
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.ds,
            y=forecast.daily_upper,
            mode="lines",
            line=dict(width=1.0, color="rgb(176,196,222,0.75)"),
            fillcolor="rgba(176,196,222, 0.5)",
            showlegend=False,
            stackgroup="one",  # define stack group
        ),
        row=row_daily,
        col=col_daily,
    )

    fig.add_trace(
        go.Scatter(
            x=forecast.ds,
            y=forecast.daily,
            name="daily",
            showlegend=False,
            line_color="blue",
            opacity=0.5,
        ),
        row=row_daily,
        col=col_daily,
    )

    # small one: ###########################################################################
    row_yearly = 3
    col_yearly = 3

    fig.add_trace(
        go.Scatter(
            x=forecast.ds,
            y=forecast.yearly_lower,
            mode="lines",
            showlegend=False,
            line=dict(width=1.0, color="rgb(176,196,222,0.75)"),
            fillcolor="rgba(176,196,222, 0.0)",
            stackgroup="one",  # define stack group
        ),
        row=row_yearly,
        col=col_yearly,
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.ds,
            y=forecast.yearly_upper,
            mode="lines",
            line=dict(width=1.0, color="rgb(176,196,222,0.75)"),
            fillcolor="rgba(176,196,222, 0.5)",
            showlegend=False,
            stackgroup="one",  # define stack group
        ),
        row=row_yearly,
        col=col_yearly,
    )

    fig.add_trace(
        go.Scatter(
            x=forecast.ds,
            y=forecast.yearly,
            name="yearly",
            showlegend=False,
            line_color="blue",
            opacity=0.5,
        ),
        row=row_yearly,
        col=col_yearly,
    )

    # Update xaxis properties  ###############################################
    upper_date = pd.Timestamp(
        forecast[~forecast["y"].isna()].ds.values[-1]
    )  # - datetime.timedelta(days=1)
    fig.update_xaxes(  # title_text="xaxis 2 title",
        range=[upper_date - datetime.timedelta(days=14), upper_date],
        row=row_weekly,
        col=col_weekly,
    )
    fig.update_xaxes(  # title_text="xaxis 2 title",
        range=[upper_date - datetime.timedelta(days=2), upper_date],
        row=row_daily,
        col=col_daily,
    )
    fig.update_xaxes(  # title_text="xaxis 2 title",
        range=[upper_date - datetime.timedelta(days=2 * 365), upper_date],
        row=row_yearly,
        col=col_yearly,
    )

    fig.update_xaxes(  # title_text="xaxis 2 title",
        range=[upper_date - datetime.timedelta(days=14), upper_date],
        row=1,
        col=1,
    )

    fig.update_xaxes(  # title_text="xaxis 2 title",
        range=[upper_date - datetime.timedelta(days=14), upper_date],
        row=row_yhat,
        col=col_yhat,
    )

    # Layout #################################################################
    fig.update_layout(
        margin=dict(l=15, r=15, t=20, b=15),
        paper_bgcolor="LightSteelBlue",
    )

    # fig.show(renderer="browser")
    drop_it_here = str(analysis_dir / "fft_deconstruction.html")
    fig.write_html(drop_it_here)

    print(drop_it_here)
