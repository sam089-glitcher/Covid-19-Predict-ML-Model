# 🌍 COVID-19 Predict & Visualize — ML Web App

🔗 **Live Application:**  
https://covid-19-predict-ml-model-sammy.streamlit.app/

An interactive **COVID-19 data visualization and forecasting web application** built using **Python, Streamlit, Plotly, and Machine Learning**.  
The app combines multiple real-world datasets to deliver **advanced analytics, maps, animations, and AI-based predictions**.

---

## ✨ Features

- 🌍 Country-wise Global COVID-19 Map  
- 📈 Worldwide Timeline (Confirmed, Deaths, Recovered)  
- 🗺️ Worldometer Severity Heatmap  
- 🇺🇸 USA State-wise COVID-19 Heatmap  
- 🔥 Top 15 Countries Analysis (Cases vs Death Rate)  
- ⏳ Animated Global Spread Over Time  
- 🔮 AI-based Forecasting (Next 60 Days)  
- 🌑 Dark-themed interactive UI  
- ⚡ Fast & responsive Streamlit web app  

---

## 🧠 Technology Stack

- **Programming Language:** Python  
- **Web Framework:** Streamlit  
- **Data Analysis:** Pandas, NumPy  
- **Visualization:** Plotly, Plotly Graph Objects  
- **Machine Learning:** Facebook Prophet  
- **Mapping:** Choropleth Maps (ISO-3 Standard)  
- **Deployment:** Streamlit Cloud  

---

## 📂 Datasets Used

This project integrates **six different COVID-19 datasets**:

1. `country_wise_latest.csv` – Latest country-level statistics  
2. `covid_19_clean_complete.csv` – Clean historical time-series data  
3. `day_wise.csv` – Global daily case progression  
4. `full_grouped.csv` – Date-wise country-level data  
5. `usa_county_wise.csv` – USA state-wise data  
6. `worldometer_data.csv` – Worldometer global statistics  

Using multiple datasets enables **richer spatial, temporal, and predictive analysis**.

---

## 🔮 Machine Learning Forecasting

- Model Used: **Facebook Prophet**
- Trained on: **Global confirmed cases**
- Forecast Horizon: **Next 60 days**
- Includes **confidence intervals**
- Automatically handles **trend & seasonality**

---

## 🚀 How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/covid-19-predict-ml-model.git
cd covid-19-predict-ml-model
