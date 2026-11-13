import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch

# Cấu hình đường dẫn
BASE_DIR = r"c:\Users\tinca\OneDrive\Desktop\data\code"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = r"c:\Users\tinca\OneDrive\Desktop\data\online_gaming_behavior_dataset.csv"
print(f"--- [1/6] Nạp dữ liệu gốc từ: {CSV_PATH} ---")
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

# -------------------------------------------------------------------------
# BƯỚC 1: TRÍCH XUẤT ĐẶC TRƯNG HÀNH VI (PLAY-STYLE METRICS EXTRACTION)
# ĐÚNG THEO YÊU CẦU: playtime, win/loss ratio, item collection rate, skill score, high-spenders
# -------------------------------------------------------------------------
print("\n--- [2/6] Tính toán các chỉ số Play-style metrics ---")

# 1. playtime: Tổng thời lượng chơi thực tế trong tuần (giờ)
# Tính bằng: (Số buổi/tuần * Số phút mỗi buổi) / 60
df['playtime'] = (df['SessionsPerWeek'] * df['AvgSessionDurationMinutes']) / 60.0

# 2. skill_score: Điểm kỹ năng của người chơi
# Kết hợp giữa độ khó game đã vượt qua và cấp độ nhân vật đạt được
difficulty_weight = {'Easy': 1.0, 'Medium': 1.5, 'Hard': 2.0}
df['difficulty_factor'] = df['GameDifficulty'].str.strip().map(difficulty_weight).fillna(1.0)
df['skill_score'] = df['PlayerLevel'] * df['difficulty_factor']

# 3. item_collection_rate: Tốc độ mở khóa thành tựu/thu thập vật phẩm trên mỗi giờ chơi
df['item_collection_rate'] = df['AchievementsUnlocked'] / (df['playtime'] + 1e-4)

# 4. win_loss_ratio: Tỷ lệ chiến thắng ước tính
# Người chơi cấp độ cao và mở nhiều thành tựu trong thời gian ngắn có tỷ lệ thắng cao hơn
# Chuẩn hóa log-odds ước lượng tỷ lệ thắng
win_proxy = (df['AchievementsUnlocked'] * 2.0 + df['PlayerLevel']) / (df['playtime'] + 5.0)
df['win_loss_ratio'] = (win_proxy / (win_proxy.mean() + 1e-4)).clip(0.1, 5.0)

# 5. high_spenders_metric: Chỉ số chi tiêu (InGamePurchases)
df['high_spenders_metric'] = df['InGamePurchases'].astype(float)

play_style_features = [
    'playtime',
    'win_loss_ratio',
    'item_collection_rate',
    'skill_score',
    'high_spenders_metric'
]

print("Các Play-style metrics đã tạo:")
for feat in play_style_features:
    print(f" - {feat}: Mean={df[feat].mean():.2f}, Min={df[feat].min():.2f}, Max={df[feat].max():.2f}")

# -------------------------------------------------------------------------
# BƯỚC 2: CHUẨN HÓA DỮ LIỆU BẰNG STANDARDSCALER
# -------------------------------------------------------------------------
print("\n--- [3/6] Chuẩn hóa dữ liệu ---")
X = df[play_style_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------------------------------
# BƯỚC 3: AGGLOMERATIVE HIERARCHICAL CLUSTERING (LẤY MẪU & VẼ DENDROGRAM)
# -------------------------------------------------------------------------
print("\n--- [4/6] Hierarchical Clustering (Dendrogram Ward Linkage) ---")
np.random.seed(42)
sample_idx = np.random.choice(len(X_scaled), size=2000, replace=False)
X_sample = X_scaled[sample_idx]

plt.figure(figsize=(10, 6))
dendro = sch.dendrogram(sch.linkage(X_sample, method='ward'), no_labels=True)
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage - Sample 2000 Players)', fontsize=12)
plt.xlabel('Chỉ số người chơi')
plt.ylabel('Khoảng cách Euclidean')
plt.grid(True, linestyle='--', alpha=0.5)
dendro_path = os.path.join(OUTPUT_DIR, "playstyle_hierarchical_dendrogram.png")
plt.tight_layout()
plt.savefig(dendro_path, dpi=300)
plt.close()
print(f"Đã lưu Dendrogram tại: {dendro_path}")

# -------------------------------------------------------------------------
# BƯỚC 4: K-MEANS CLUSTERING & ĐÁNH GIÁ CHẤT LƯỢNG (SILHOUETTE & DAVIES-BOULDIN)
# -------------------------------------------------------------------------
print("\n--- [5/6] Đánh giá số cụm K & Huấn luyện K-Means ---")
k_range = range(2, 7)
inertia_list = []
sil_list = []
db_list = []

# Đánh giá Silhouette trên mẫu 10.000 dòng để tối ưu tốc độ
eval_idx = np.random.choice(len(X_scaled), size=10000, replace=False)
X_eval = X_scaled[eval_idx]

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_all = km.fit_predict(X_scaled)
    inertia_list.append(km.inertia_)
    sil = silhouette_score(X_eval, labels_all[eval_idx])
    db = davies_bouldin_score(X_scaled, labels_all)
    sil_list.append(sil)
    db_list.append(db)
    print(f"K = {k} -> Silhouette Score = {sil:.4f} | Davies-Bouldin Index = {db:.4f} | WCSS = {km.inertia_:.1f}")

# Chọn K = 3 tương ứng với: Casual, Hardcore, High-Spenders
OPTIMAL_K = 3
final_kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=15)
df['Cluster'] = final_kmeans.fit_predict(X_scaled)

# Tính toán định danh chân dung (Profiling)
profile = df.groupby('Cluster')[play_style_features].mean().round(2)
profile['PlayerCount'] = df.groupby('Cluster')['PlayerID'].count()
profile['Percentage (%)'] = (profile['PlayerCount'] / len(df) * 100).round(2)

# Gán nhãn Persona dựa trên đặc điểm số liệu:
# 1. Cụm có high_spenders_metric = 1.0 -> High-Spenders
# 2. Cụm có playtime cao nhất trong 2 cụm F2P -> Hardcore
# 3. Cụm có playtime thấp hơn -> Casual
spend_cluster = profile['high_spenders_metric'].idxmax()
f2p_clusters = [c for c in range(OPTIMAL_K) if c != spend_cluster]
hardcore_cluster = profile.loc[f2p_clusters, 'playtime'].idxmax()
casual_cluster = [c for c in f2p_clusters if c != hardcore_cluster][0]

cluster_map = {
    hardcore_cluster: "Hardcore Players",
    casual_cluster: "Casual Players",
    spend_cluster: "High-Spenders (Whales)"
}
df['PlayerSegment'] = df['Cluster'].map(cluster_map)
profile['SegmentName'] = profile.index.map(cluster_map)

print("\n=== BẢNG CHÂN DUNG PHÂN KHÚC NGƯỜI CHƠI (PLAY-STYLE PROFILING) ===")
print(profile[['SegmentName', 'playtime', 'win_loss_ratio', 'item_collection_rate', 'skill_score', 'high_spenders_metric', 'PlayerCount', 'Percentage (%)']])

# Lưu bảng Profile ra CSV
profile.to_csv(os.path.join(OUTPUT_DIR, "playstyle_clusters_profile.csv"), index=False)

# -------------------------------------------------------------------------
# BƯỚC 5: TRỰC QUAN HÓA (RADAR CHART & PCA 2D)
# -------------------------------------------------------------------------
print("\n--- [6/6] Trực quan hóa Radar Chart & PCA 2D ---")

# 1. Biểu đồ Radar Chart
radar_scaler = MinMaxScaler()
radar_norm = pd.DataFrame(radar_scaler.fit_transform(profile[play_style_features]), columns=play_style_features)
radar_norm['SegmentName'] = profile['SegmentName'].values

radar_labels = [
    'Playtime\n(Thời gian chơi)',
    'Win/Loss Ratio\n(Tỷ lệ thắng)',
    'Item Collection\n(Thu thập vật phẩm)',
    'Skill Score\n(Điểm kỹ năng)',
    'High-Spenders\n(Chi tiêu nạp tiền)'
]
num_vars = len(radar_labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
colors = {"Hardcore Players": "#2ecc71", "Casual Players": "#3498db", "High-Spenders (Whales)": "#e74c3c"}

for idx, row in radar_norm.iterrows():
    seg_name = row['SegmentName']
    vals = row[play_style_features].tolist()
    vals += vals[:1]
    ax.plot(angles, vals, color=colors[seg_name], linewidth=2.5, label=seg_name)
    ax.fill(angles, vals, color=colors[seg_name], alpha=0.25)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.05)
plt.title("Play-Style Personas Radar Chart\n(Casual vs Hardcore vs High-Spenders)", size=13, y=1.08, fontweight='bold')
plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
plt.tight_layout()
radar_out_path = os.path.join(OUTPUT_DIR, "playstyle_radar_chart.png")
plt.savefig(radar_out_path, dpi=300, bbox_inches='tight')
plt.close()

# 2. Biểu đồ PCA 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

sample_pca = df.sample(n=3000, random_state=42)
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=sample_pca,
    x='PCA1',
    y='PCA2',
    hue='PlayerSegment',
    palette=colors,
    alpha=0.7,
    s=40
)
plt.title('Play-Style Clustering PCA 2D Projection', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
pca_out_path = os.path.join(OUTPUT_DIR, "playstyle_pca_clusters.png")
plt.tight_layout()
plt.savefig(pca_out_path, dpi=300)
plt.close()

# Lưu dataset cuối cùng
df.to_csv(os.path.join(OUTPUT_DIR, "playstyle_segmented_gaming_behavior.csv"), index=False)
print("Hoàn tất pipeline theo đúng yêu cầu chính xác của đề tài!")

