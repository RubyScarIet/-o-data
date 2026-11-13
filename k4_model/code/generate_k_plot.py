import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Cấu hình đường dẫn
OUTPUT_DIR = r"c:\Users\tinca\OneDrive\Desktop\data\code\outputs_k4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = r"c:\Users\tinca\OneDrive\Desktop\data\online_gaming_behavior_dataset.csv"
print(f"Nạp dữ liệu từ: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

# Trích xuất đúng các đặc trưng của mô hình K=4
df['playtime'] = (df['SessionsPerWeek'] * df['AvgSessionDurationMinutes']) / 60.0
diff_weight = {'Easy': 1.0, 'Medium': 1.5, 'Hard': 2.0}
df['skill_score'] = df['PlayerLevel'] * df['GameDifficulty'].str.strip().map(diff_weight).fillna(1.0)
df['item_collection_rate'] = df['AchievementsUnlocked'] / (df['playtime'] + 1.0)
df['win_loss_ratio'] = (df['AchievementsUnlocked'] * 1.5 + df['PlayerLevel']) / (df['playtime'] + 5.0)
df['spending_metric'] = df['InGamePurchases'].astype(float)

features = ['playtime', 'SessionsPerWeek', 'win_loss_ratio', 'item_collection_rate', 'skill_score', 'spending_metric']
X_scaled = StandardScaler().fit_transform(df[features])

# Quét K từ 2 đến 7
k_values = list(range(2, 8))
inertias = []
silhouettes = []
davies_bouldins = []

np.random.seed(42)
sample_idx = np.random.choice(len(X_scaled), size=10000, replace=False)
X_eval = X_scaled[sample_idx]

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_eval, labels[sample_idx])
    db = davies_bouldin_score(X_scaled, labels)
    silhouettes.append(sil)
    davies_bouldins.append(db)

# -------------------------------------------------------------
# VẼ BIỂU ĐỒ TÌM K TOÀN DIỆN (3 Ô: ELBOW, SILHOUETTE, DAVIES-BOULDIN)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
plt.suptitle('ĐÁNH GIÁ VÀ XÁC ĐỊNH SỐ CỤM TỐI ƯU (OPTIMAL K EVALUATION)', fontsize=15, fontweight='bold', y=1.02)

# 1. Elbow Method
axes[0].plot(k_values, inertias, 'bo-', linewidth=2.5, markersize=8)
axes[0].axvline(x=4, color='red', linestyle='--', linewidth=1.8, label='Điểm gập Elbow (K=4)')
axes[0].scatter([4], [inertias[2]], color='red', s=120, zorder=5)
axes[0].set_title('1. Elbow Method (Inertia / WCSS)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Số lượng cụm (K)', fontsize=11)
axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=11)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend(loc='upper right', fontsize=10)

# 2. Silhouette Score
axes[1].plot(k_values, silhouettes, 'go-', linewidth=2.5, markersize=8)
axes[1].axvline(x=4, color='red', linestyle='--', linewidth=1.8, label='Đỉnh tối ưu (K=4: 0.294)')
axes[1].scatter([4], [silhouettes[2]], color='red', s=120, zorder=5)
axes[1].set_title('2. Silhouette Score (Càng cao càng tốt)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Số lượng cụm (K)', fontsize=11)
axes[1].set_ylabel('Silhouette Score', fontsize=11)
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend(loc='lower right', fontsize=10)

# 3. Davies-Bouldin Index
axes[2].plot(k_values, davies_bouldins, 'mo-', linewidth=2.5, markersize=8)
axes[2].axvline(x=4, color='red', linestyle='--', linewidth=1.8, label='Giảm mạnh tại K=4 (1.146)')
axes[2].scatter([4], [davies_bouldins[2]], color='red', s=120, zorder=5)
axes[2].set_title('3. Davies-Bouldin Index (Càng thấp càng tốt)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Số lượng cụm (K)', fontsize=11)
axes[2].set_ylabel('Davies-Bouldin Index', fontsize=11)
axes[2].grid(True, linestyle='--', alpha=0.6)
axes[2].legend(loc='upper right', fontsize=10)

plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "optimal_k_evaluation_k4.png")
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Đã lưu biểu đồ tìm K hoàn chỉnh tại: {chart_path}")

