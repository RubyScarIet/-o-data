import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# Cấu hình đường dẫn
BASE_DIR = r"c:\Users\tinca\OneDrive\Desktop\data\code"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_k4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = r"c:\Users\tinca\OneDrive\Desktop\data\online_gaming_behavior_dataset.csv"
print(f"--- [1/5] Nạp dữ liệu từ: {CSV_PATH} ---")
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

# -------------------------------------------------------------------------
# 1. KỸ NGHỆ ĐẶC TRƯNG (FEATURE ENGINEERING) CHO 4 PERSONAS
# -------------------------------------------------------------------------
print("--- [2/5] Tạo các chỉ số hành vi chuyên sâu ---")

# Thời lượng chơi thực tế trong tuần (giờ)
df['playtime'] = (df['SessionsPerWeek'] * df['AvgSessionDurationMinutes']) / 60.0

# Điểm kỹ năng (Skill Score)
diff_weight = {'Easy': 1.0, 'Medium': 1.5, 'Hard': 2.0}
df['skill_score'] = df['PlayerLevel'] * df['GameDifficulty'].str.strip().map(diff_weight).fillna(1.0)

# Tốc độ thu thập thành tựu/vật phẩm
df['item_collection_rate'] = df['AchievementsUnlocked'] / (df['playtime'] + 1.0)

# Tỷ lệ thắng ước tính (Win/Loss proxy)
df['win_loss_ratio'] = (df['AchievementsUnlocked'] * 1.5 + df['PlayerLevel']) / (df['playtime'] + 5.0)

# Tỷ lệ chi tiêu nạp tiền (InGamePurchases)
df['spending_metric'] = df['InGamePurchases'].astype(float)

features_k4 = [
    'playtime',
    'SessionsPerWeek',
    'win_loss_ratio',
    'item_collection_rate',
    'skill_score',
    'spending_metric'
]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features_k4])

# -------------------------------------------------------------------------
# 2. HUẤN LUYỆN K-MEANS VỚI K = 4
# -------------------------------------------------------------------------
print("--- [3/5] Huấn luyện K-Means với K = 4 ---")
kmeans_k4 = KMeans(n_clusters=4, random_state=42, n_init=15)
df['Cluster'] = kmeans_k4.fit_predict(X_scaled)

# Tính toán định lượng
eval_sample_idx = np.random.RandomState(42).choice(len(X_scaled), size=10000, replace=False)
sil_k4 = silhouette_score(X_scaled[eval_sample_idx], df['Cluster'].iloc[eval_sample_idx])
db_k4 = davies_bouldin_score(X_scaled, df['Cluster'])
print(f"K = 4 Results: Silhouette Score = {sil_k4:.4f} | Davies-Bouldin Index = {db_k4:.4f}")

# -------------------------------------------------------------------------
# 3. LẬP BẢNG CHÂN DUNG (MEAN & MEDIAN) VÀ ĐỊNH DANH 4 PERSONA
# -------------------------------------------------------------------------
profile_mean = df.groupby('Cluster')[features_k4].mean().round(2)
profile_median = df.groupby('Cluster')[features_k4].median().round(2)
counts = df.groupby('Cluster')['PlayerID'].count()
pcts = (counts / len(df) * 100).round(2)

print("\n--- BẢNG GIÁ TRỊ TRUNG BÌNH THEO CỤM (MEAN) ---")
print(profile_mean)

# Định danh thông minh theo 4 Persona trong Plan:
# 1. Cluster có spending_metric cao nhất -> Competitive Whales / High-Spenders
# 2. Trong các cụm F2P:
#    - Cụm có playtime & SessionsPerWeek cao nhất -> Hardcore Grinders
#    - Cụm có playtime thấp nhất, win_loss thấp -> Churn Risk / Attrition
#    - Cụm còn lại -> Casual Players

spend_c = profile_mean['spending_metric'].idxmax()
other_cs = [c for c in range(4) if c != spend_c]

hardcore_c = profile_mean.loc[other_cs, 'SessionsPerWeek'].idxmax()
remaining_cs = [c for c in other_cs if c != hardcore_c]

# Sắp xếp theo SessionsPerWeek để phân biệt Casual vs Churn Risk
sorted_rem = profile_mean.loc[remaining_cs, 'SessionsPerWeek'].sort_values()
churn_c = sorted_rem.index[0]      # Thấp nhất
casual_c = sorted_rem.index[1]     # Trung bình

persona_mapping = {
    casual_c: "Cluster 0: Casual Players",
    hardcore_c: "Cluster 1: Hardcore Grinders",
    spend_c: "Cluster 2: Competitive Whales / High-Spenders",
    churn_c: "Cluster 3: Churn Risk / Attrition"
}

df['Persona'] = df['Cluster'].map(persona_mapping)
profile_mean['Persona'] = profile_mean.index.map(persona_mapping)
profile_mean['PlayerCount'] = counts
profile_mean['Percentage (%)'] = pcts

# Bảng Median
profile_median['Persona'] = profile_median.index.map(persona_mapping)
profile_median['PlayerCount'] = counts
profile_median['Percentage (%)'] = pcts

print("\n=== KẾT QUẢ ĐỊNH DANH 4 PERSONA (MEAN) ===")
print(profile_mean[['Persona', 'playtime', 'SessionsPerWeek', 'skill_score', 'spending_metric', 'PlayerCount', 'Percentage (%)']])

# Lưu bảng chân dung ra CSV
profile_mean.to_csv(os.path.join(OUTPUT_DIR, "player_personas_k4_mean.csv"), index=False)
profile_median.to_csv(os.path.join(OUTPUT_DIR, "player_personas_k4_median.csv"), index=False)

# -------------------------------------------------------------------------
# 4. TRỰC QUAN HÓA: RADAR CHART 4 PERSONAS & PCA 2D
# -------------------------------------------------------------------------
print("\n--- [4/5] Trực quan hóa Radar Chart 4 Personas & PCA 2D ---")

# Radar Chart
radar_feats = ['playtime', 'SessionsPerWeek', 'win_loss_ratio', 'item_collection_rate', 'skill_score', 'spending_metric']
radar_labels = [
    'Playtime\n(Thời gian)', 
    'Sessions/Wk\n(Tần suất)', 
    'Win/Loss\n(Tỷ lệ thắng)', 
    'Item Coll.\n(Thu thập)', 
    'Skill\n(Kỹ năng)', 
    'Spending\n(Chi tiêu)'
]

minmax = MinMaxScaler()
radar_norm = pd.DataFrame(minmax.fit_transform(profile_mean[radar_feats]), columns=radar_feats)
radar_norm['Persona'] = profile_mean['Persona'].values

num_vars = len(radar_labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
persona_colors = {
    "Cluster 0: Casual Players": "#3498db",                         # Xanh dương
    "Cluster 1: Hardcore Grinders": "#2ecc71",                       # Xanh lá
    "Cluster 2: Competitive Whales / High-Spenders": "#e74c3c",       # Đỏ
    "Cluster 3: Churn Risk / Attrition": "#f39c12"                   # Cam vàng
}

for _, row in radar_norm.iterrows():
    p_name = row['Persona']
    vals = row[radar_feats].tolist()
    vals += vals[:1]
    ax.plot(angles, vals, color=persona_colors[p_name], linewidth=2.5, label=p_name)
    ax.fill(angles, vals, color=persona_colors[p_name], alpha=0.2)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.05)
plt.title("Player Personas Radar Chart (4 Personas - K=4)", size=14, y=1.08, fontweight='bold')
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=10)
plt.tight_layout()
radar_k4_path = os.path.join(OUTPUT_DIR, "player_personas_k4_radar.png")
plt.savefig(radar_k4_path, dpi=300, bbox_inches='tight')
plt.close()

# PCA 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

sample_df = df.sample(n=3000, random_state=42)
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=sample_df,
    x='PCA1',
    y='PCA2',
    hue='Persona',
    palette=persona_colors,
    alpha=0.7,
    s=40
)
plt.title('PCA 2D Projection of 4 Player Personas', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
pca_k4_path = os.path.join(OUTPUT_DIR, "player_personas_k4_pca.png")
plt.tight_layout()
plt.savefig(pca_k4_path, dpi=300)
plt.close()

# -------------------------------------------------------------------------
# 5. ĐỐI CHIẾU ENGAGEMENT LEVEL
# -------------------------------------------------------------------------
print("--- [5/5] Đối chiếu với nhãn EngagementLevel ---")
ctab_pct = pd.crosstab(df['Persona'], df['EngagementLevel'].str.strip(), normalize='index') * 100
plt.figure(figsize=(8, 5))
sns.heatmap(ctab_pct, annot=True, fmt='.1f', cmap='Blues', cbar=True)
plt.title('Tỷ lệ phân bổ EngagementLevel theo 4 Persona (%)', fontsize=12)
plt.xlabel('Engagement Level gốc')
plt.ylabel('Persona phát hiện bởi K-Means (K=4)')
plt.tight_layout()
heatmap_k4_path = os.path.join(OUTPUT_DIR, "personas_k4_engagement_heatmap.png")
plt.savefig(heatmap_k4_path, dpi=300)
plt.close()

# Lưu dataset kết quả
df.to_csv(os.path.join(OUTPUT_DIR, "segmented_gaming_behavior_k4.csv"), index=False)
print("Hoàn tất phân chia 4 Personas thành công!")

