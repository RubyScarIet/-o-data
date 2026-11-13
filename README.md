# 🎮 Topic 12: Gamer Behavior Clustering & Player Profiling
**Khoa học Dữ liệu & Khai phá Dữ liệu (Data Mining Project)**

Dự án ứng dụng các kỹ thuật học máy không giám sát (**Agglomerative Hierarchical Clustering** kết hợp **K-Means**) trên tập dữ liệu hơn 40.000 người chơi game (`online_gaming_behavior_dataset.csv`) để phân khúc hành vi, xây dựng chân dung người chơi (Player Profiling) và đề xuất các chiến lược kinh doanh/giữ chân người chơi (Monetization & Retention Strategies).

---

## 📁 Cấu trúc Thư mục Dự án (Repository Structure)

```text
topic_12_gamer_clustering/
├── data/
│   └── online_gaming_behavior_dataset.csv     # Dữ liệu gốc 40.034 bản ghi
├── k3_model/                                  # PHÂN CỤM MÔ HÌNH K = 3
│   ├── code/
│   │   └── playstyle_pipeline.py              # Script chạy trọn vẹn pipeline K=3
│   ├── figures/                               # Biểu đồ trực quan hóa K=3
│   │   ├── playstyle_hierarchical_dendrogram.png
│   │   ├── playstyle_radar_chart.png
│   │   ├── playstyle_pca_clusters.png
│   │   ├── cluster_evaluation_kmeans.png
│   │   └── cluster_engagement_heatmap.png
│   ├── results/                               # Bảng số liệu & dataset phân cụm
│   │   ├── playstyle_clusters_profile.csv
│   │   └── playstyle_segmented_gaming_behavior.csv
│   └── REPORT_K3.md                           # Báo cáo chi tiết mô hình K=3
├── k4_model/                                  # PHÂN CỤM MÔ HÌNH TỐI ƯU K = 4
│   ├── code/
│   │   ├── k4_personas_pipeline.py            # Pipeline trích xuất đặc trưng & chạy K=4
│   │   └── generate_k_plot.py                 # Script vẽ đồ thị 3 ô chứng minh K=4
│   ├── figures/                               # Biểu đồ trực quan hóa K=4
│   │   ├── optimal_k_evaluation_k4.png        # Biểu đồ 3 ô (Elbow, Silhouette, DB)
│   │   ├── player_personas_k4_radar.png       # Radar chart 4 Personas
│   │   ├── player_personas_k4_pca.png         # Chiếu không gian 2D PCA 4 cụm
│   │   └── personas_k4_engagement_heatmap.png # Heatmap đối chiếu Engagement
│   ├── results/                               # Bảng số liệu Mean, Median & dataset K=4
│   │   ├── player_personas_k4_mean.csv
│   │   ├── player_personas_k4_median.csv
│   │   └── segmented_gaming_behavior_k4.csv
│   └── REPORT_K4.md                           # Báo cáo toàn diện & giải thích thuật ngữ K=4
├── .gitignore                                 # Bỏ qua file rác / cache
└── README.md                                  # Hướng dẫn tổng quan dự án
```

---

## ⚖️ So sánh Tổng quan: Mô hình $K = 3$ vs Mô hình $K = 4$

| Tiêu chí đánh giá | Mô hình $K = 3$ | Mô hình $K = 4$ *(Được đề xuất)* |
| :--- | :---: | :---: |
| **Silhouette Score** *(Càng cao càng tốt)* | `0.2603` | **`0.2938`** *(Đạt đỉnh cực đại)* |
| **Davies-Bouldin Index** *(Càng thấp càng tốt)*| `1.2538` | **`1.1457`** *(Cụm cô đặc hơn hẳn)* |
| **Elbow Method (WCSS)** | Nằm trên sườn dốc giảm | **Nằm ngay điểm gập khuỷu tay** |
| **Các Persona nhận diện** | 3 nhóm: Casual, Hardcore, Whales | 4 nhóm: Casual, Hardcore, Whales, **Churn Risk** |
| **Giá trị ứng dụng kinh doanh** | Khai thác doanh thu F2P vs P2P | Nhận diện thêm **nguy cơ rời bỏ game** (cứu 21% khách trả phí cũ) |

---

## 🚀 Hướng dẫn Cài đặt & Chạy lại Thực nghiệm (Reproducibility)

### 1. Cài đặt các thư viện yêu cầu:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

### 2. Chạy mô hình $K = 3$:
```bash
cd k3_model/code
python playstyle_pipeline.py
```

### 3. Chạy mô hình $K = 4$ (Khuyên dùng):
```bash
cd k4_model/code
python k4_personas_pipeline.py
python generate_k_plot.py
```

---

## 📌 Tóm tắt các Chân dung Người chơi (4 Personas - K=4)
1. **Cluster 0: Casual Players (39.56%):** Chơi nhẹ nhàng (7.4h/tuần, 6.5 buổi), 0% nạp tiền $\rightarrow$ Áp dụng gói tân thủ $0.99 và nhiệm vụ ngắn 5 phút.
2. **Cluster 1: Hardcore Grinders (33.58%):** Dân cày chay nhiệt huyết (26.8h/tuần, ~15 buổi/tuần), 0% nạp tiền $\rightarrow$ Áp dụng vé Battle Pass cày cấp và xem quảng cáo nhận thưởng (Rewarded Ads).
3. **Cluster 2: Competitive Whales (18.18%):** Đại gia nạp tiền (100% nạp tiền, chơi 16.3h/tuần) $\rightarrow$ Cung cấp gói VIP giới hạn, sự kiện Gacha có bảo hiểm.
4. **Cluster 3: Churn Risk / Attrition (8.68%):** Cựu chiến binh chán game (Playtime tụt về 0.8h/tuần, kỹ năng cao 79.0, có 21% từng nạp tiền) $\rightarrow$ Mở rộng phó bản End-game cấp 120 và gửi chiến dịch tri ân cựu binh (Win-Back).

