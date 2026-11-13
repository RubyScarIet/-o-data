# 🎮 BÁO CÁO PHÂN CỤM NGƯỜI CHƠI MÔ HÌNH K = 3
## Topic 12: Gamer Behavior Clustering & Player Profiling

---

## 1. TỔNG QUAN MÔ HÌNH K = 3
* **Phương pháp:** Trích xuất các chỉ số phong cách chơi (*playtime, win/loss ratio, item collection rate, skill score, spending*), chuẩn hóa bằng `StandardScaler`, sử dụng cây phân cấp **Hierarchical Dendrogram (Ward Linkage)** lấy mẫu 2.000 dòng để định hình cấu trúc, sau đó huấn luyện **K-Means ($K = 3$)** trên toàn bộ 40.034 người chơi.
* **Chỉ số đánh giá định lượng:**
  * **Silhouette Score:** `0.2603`
  * **Davies-Bouldin Index:** `1.2538`

---

## 2. KẾT QUẢ PHÂN CỤM & CHÂN DUNG 3 PERSONAS

| Chỉ số hành vi | 🔴 High-Spenders (Whales) | 🟢 Hardcore Players | 🔵 Casual Players |
| :--- | :---: | :---: | :---: |
| **Số lượng & Tỷ lệ (%)** | **7,778 (19.43%)** | **30,893 (77.17%)** | **1,363 (3.40%)** |
| **Playtime (giờ/tuần)** | **15.55** | **15.47** | **0.00** |
| **Skill Score (Điểm kỹ năng)** | **67.52** | **66.91** | **69.17** |
| **Win/Loss Ratio (Tỷ lệ thắng)** | 0.93 | 0.93 | 3.09 |
| **Item Collection Rate** | 1,457.51 | 1,126.90 | 324,240.65 |
| **Chỉ số nạp tiền (Spending)** | **1.00 (100% nạp)** | **0.00 (0% nạp)** | **0.19 (19% nạp)** |

### Đặc điểm từng Persona:
1. **High-Spenders (Whales - 19.4%):** 100% đã nạp tiền, thời gian chơi và kỹ năng cao, đóng góp toàn bộ doanh thu giao dịch trực tiếp (IAP).
2. **Hardcore Players (77.2%):** Dân cày chay F2P, thời lượng chơi ngang ngửa Whales nhưng 0% nạp tiền. Lực lượng nòng cốt giữ nhiệt cộng đồng.
3. **Casual Players (3.4%):** Nhóm vãng lai, số giờ chơi gần như bằng 0.

---

## 3. CHIẾN LƯỢC MONETIZATION MỤC TIÊU
* **High-Spenders:** Trang phục/vật phẩm giới hạn (Limited Skins), sự kiện Gacha có bảo hiểm, đặc quyền VIP.
* **Hardcore Players:** Vé Battle Pass theo mùa, gói nạp đầu siêu rẻ $0.99, tính năng xem video quảng cáo nhận quà (Rewarded Ads).
* **Casual Players:** Quà tặng tái xuất (Comeback bundle), nhiệm vụ 5 phút.

---

## 4. DANH MỤC TÀI NGUYÊN (K = 3)
* **Code:** `code/playstyle_pipeline.py`
* **Biểu đồ (figures/):**
  * `playstyle_hierarchical_dendrogram.png`: Cây Dendrogram Ward Linkage.
  * `playstyle_radar_chart.png`: Biểu đồ Radar 3 Personas.
  * `playstyle_pca_clusters.png`: Không gian chiếu 2D PCA.
  * `cluster_evaluation_kmeans.png`: Đánh giá Elbow & Silhouette $K=3$.
* **Dữ liệu xuất (results/):**
  * `playstyle_clusters_profile.csv`: Bảng thống kê giá trị trung bình.
  * `playstyle_segmented_gaming_behavior.csv`: Dữ liệu 40.034 dòng gắn nhãn cụm.

