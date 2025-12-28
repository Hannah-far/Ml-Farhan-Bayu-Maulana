import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# --- 1. Persiapan Data (Load Data) ---
@st.cache_data
def load_data():
    
    df = pd.read_csv('day.csv')
    
   
    df['dteday'] = pd.to_datetime(df['dteday'])
    
    df['season_label'] = df['season'].map({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'})
    return df


df = load_data()

#-- 2. Membuat Dashboard dengan Streamlit ---
st.title("Dashboard Analisis Bike Sharing 🚲")
st.write("Nama: **Farhan Bayu Maulana** (NIM: 09011282429105)")
st.markdown("---") 

# Filter Data
st.sidebar.header("Filter Data")


pilihan_musim = df['season_label'].unique()
selected_season = st.sidebar.multiselect("Pilih Musim:", pilihan_musim, default=pilihan_musim)


filtered_df = df[df['season_label'].isin(selected_season)]

# Summary Metrics
st.subheader("Ringkasan Data")


col1, col2, col3 = st.columns(3)

total_sewa = filtered_df['cnt'].sum()
rata_suhu = filtered_df['temp'].mean()
jumlah_hari = len(filtered_df)


col1.metric("Total Penyewaan", f"{total_sewa:,}")
col2.metric("Rata-rata Suhu", f"{rata_suhu:.2f}")
col3.metric("Jumlah Hari Data", jumlah_hari)

st.markdown("---")

# Visualisasi Data
st.subheader("1. Bagaimana Tren Penyewaan Sepeda?")

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(filtered_df['dteday'], filtered_df['cnt'], color='tab:blue')
ax.set_xlabel("Tanggal")
ax.set_ylabel("Jumlah Sewa")
ax.set_title("Grafik Penyewaan Harian")
# labels and title 
st.pyplot(fig)

st.subheader("2. Pengaruh Musim Terhadap Penyewaan")

fig2, ax2 = plt.subplots(figsize=(8, 5))

sns.barplot(
    x='season_label', 
    y='cnt', 
    data=df, 
    palette='viridis', 
    ax=ax2
)
ax2.set_xlabel("Musim")
ax2.set_ylabel("Rata-rata Sewa")
ax2.set_title("Rata-rata Penyewaan per Musim")

st.pyplot(fig2)

# Prediksi dengan Machine Learning
st.markdown("---")
st.subheader("3. Prediksi Jumlah Sewa (Machine Learning)")
st.write("Masukkan kondisi cuaca di bawah ini untuk memprediksi jumlah sewa.")


suhu = st.slider("Suhu (0.0 - 1.0)", 0.0, 1.0, 0.5)
lembab = st.slider("Kelembaban (0.0 - 1.0)", 0.0, 1.0, 0.5)
angin = st.slider("Kecepatan Angin (0.0 - 1.0)", 0.0, 1.0, 0.2)


X = df[['temp', 'hum', 'windspeed']]
y = df['cnt']

model = LinearRegression()
model.fit(X, y)


if st.button("Prediksi Sekarang"):
    hasil = model.predict([[suhu, lembab, angin]])
    st.success(f"Kira-kira jumlah penyewaan adalah: **{int(hasil[0])} sepeda**")