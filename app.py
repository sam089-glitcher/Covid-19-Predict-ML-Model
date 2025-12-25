import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(
    page_title="COVID-19 Advanced Analytics",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 COVID-19 Advanced Visualization & Forecasting")
st.markdown("Using multiple datasets, interactive maps, timelines & AI forecasting")

# =======================
# LOAD DATA (CACHED)
# =======================
@st.cache_data
def load_data():
    df0 = pd.read_csv("country_wise_latest.csv")
    df1 = pd.read_csv("covid_19_clean_complete.csv")
    df2 = pd.read_csv("day_wise.csv")
    df3 = pd.read_csv("full_grouped.csv")
    df4 = pd.read_csv("usa_county_wise.csv")
    df5 = pd.read_csv("worldometer_data.csv")
    return df0, df1, df2, df3, df4, df5

df0, df1, df2, df3, df4, df5 = load_data()

# =======================
# SIDEBAR NAVIGATION
# =======================
section = st.sidebar.radio(
    "Navigate",
    [
        "🌍 Country-wise World Map",
        "📈 Global Timeline",
        "🗺️ Worldometer Severity Map",
        "🇺🇸 USA Heatmap",
        "🔥 Top Countries Analysis",
        "⏳ Animated World Spread",
        "🔮 AI Forecast"
    ]
)

# =======================
# 1️⃣ COUNTRY-WISE MAP
# =======================
if section == "🌍 Country-wise World Map":
    world = df0[["Country/Region", "Confirmed"]].copy()
    world.columns = ["Country", "Cases"]

    world["Cases Range"] = pd.cut(
        world["Cases"],
        [-1, 5e4, 2e5, 8e5, 1.5e6, 1e9],
        labels=["U50K", "50K–200K", "200K–800K", "800K–1.5M", "1.5M+"]
    )

    fig = px.choropleth(
        world,
        locations="Country",
        locationmode="country names",
        color="Cases Range",
        title="🌍 COVID-19 Cases (Country Wise)",
        template="plotly_dark"
    )

    fig.update_geos(visible=False)
    st.plotly_chart(fig, use_container_width=True)

# =======================
# 2️⃣ GLOBAL TIMELINE
# =======================
elif section == "📈 Global Timeline":
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df2["Date"], y=df2["Confirmed"], name="Confirmed"))
    fig.add_trace(go.Scatter(x=df2["Date"], y=df2["Deaths"], name="Deaths"))
    fig.add_trace(go.Scatter(x=df2["Date"], y=df2["Recovered"], name="Recovered"))

    fig.update_layout(
        title="📈 Global COVID-19 Timeline",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Count"
    )

    st.plotly_chart(fig, use_container_width=True)

# =======================
# 3️⃣ WORLDOMETER MAP
# =======================
elif section == "🗺️ Worldometer Severity Map":
    world = df5[["Country/Region", "TotalCases"]].copy()
    world.columns = ["Country", "Cases"]

    world["Cases"] = (
        world["Cases"].astype(str)
        .str.replace(",", "")
        .replace("nan", np.nan)
        .astype(float)
    )

    world["Severity"] = pd.cut(
        world["Cases"],
        [-1, 5e4, 2e5, 8e5, 1.5e6, 1e9],
        labels=["Low", "Moderate", "High", "Very High", "Extreme"]
    )

    fig = px.choropleth(
        world,
        locations="Country",
        locationmode="country names",
        color="Severity",
        color_discrete_sequence=px.colors.sequential.Reds,
        title="🌍 Global COVID-19 Severity (Worldometer)",
        template="plotly_dark"
    )

    fig.update_geos(visible=False)
    st.plotly_chart(fig, use_container_width=True)

# =======================
# 4️⃣ USA HEATMAP
# =======================
elif section == "🇺🇸 USA Heatmap":
    fig = px.choropleth(
        df4,
        locations="Province_State",
        locationmode="USA-states",
        color="Confirmed",
        scope="usa",
        color_continuous_scale="Inferno",
        title="🇺🇸 COVID-19 Cases Across USA",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

# =======================
# 5️⃣ TOP COUNTRIES
# =======================
elif section == "🔥 Top Countries Analysis":
    df0["Death Rate"] = df0["Deaths"] / df0["Confirmed"]
    top = df0.sort_values("Confirmed", ascending=False).head(15)

    fig = px.bar(
        top,
        x="Country/Region",
        y="Confirmed",
        color="Death Rate",
        color_continuous_scale="Reds",
        title="🔥 Top 15 Countries: Cases vs Death Rate",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

# =======================
# 6️⃣ ANIMATED SPREAD
# =======================
elif section == "⏳ Animated World Spread":
    df3["Date"] = pd.to_datetime(df3["Date"])

    fig = px.choropleth(
        df3,
        locations="Country/Region",
        locationmode="country names",
        color="Confirmed",
        animation_frame=df3["Date"].dt.strftime("%Y-%m-%d"),
        color_continuous_scale="Reds",
        title="⏳ COVID-19 Spread Over Time",
        template="plotly_dark"
    )

    fig.update_geos(visible=False)
    st.plotly_chart(fig, use_container_width=True)

# =======================
# 7️⃣ AI FORECAST
# =======================
elif section == "🔮 AI Forecast":
    cases = df1.groupby("Date")["Confirmed"].sum().reset_index()
    cases["Date"] = pd.to_datetime(cases["Date"])

    df_fb = pd.DataFrame({
        "ds": cases["Date"],
        "y": cases["Confirmed"]
    })

    model = Prophet(weekly_seasonality=True)
    model.fit(df_fb)

    future = model.make_future_dataframe(periods=60)
    forecast = model.predict(future)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        name="Prediction",
        line=dict(color="cyan")
    ))

    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_upper"],
        fill=None,
        mode="lines",
        line=dict(color="gray"),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_lower"],
        fill="tonexty",
        mode="lines",
        line=dict(color="gray"),
        name="Confidence Interval"
    ))

    fig.update_layout(
        title="🔮 AI Forecast: Next 60 Days COVID-19 Cases",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)
