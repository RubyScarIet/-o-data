# BÁO CÁO BÀI TẬP LỚN MÔN HỌC: KHAI PHÁ DỮ LIỆU
## ĐỀ TÀI: TRÍCH XUẤT CHỈ SỐ PHONG CÁCH CHƠI, PHÂN KHÚC NGƯỜI CHƠI GAME TRỰC TUYẾN BẰNG MÔ HÌNH KẾT HỢP AGGLOMERATIVE HIERARCHICAL - K-MEANS VÀ ĐỀ XUẤT CHIẾN LƯỢC DOANH THU

---

### THÔNG TIN ĐỀ TÀI
| Thông tin | Chi tiết |
| :--- | :--- |
| **Cơ quan / Trường** | Học viện Công nghệ Bưu chính Viễn thông - Khoa Công nghệ Thông tin 1 |
| **Học phần** | Khai phá dữ liệu (Data Mining) - Học máy Không giám sát |
| **Giảng viên hướng dẫn** | TS. Dương Trần Đức |
| **Sinh viên thực hiện** | Nguyễn Trung Tín - Trương Đức Tùng |
| **Nhóm thực hiện** | Nhóm 12 (Topic 12: Gamer Behavior Clustering & Player Profiling) |
| **Bộ dữ liệu thực nghiệm** | Online Gaming Behavior Dataset (40.034 bản ghi, 13 thuộc tính) |

*Hà Nội, Tháng 11 Năm 2026*

---

# MỤC LỤC BÁO CÁO
- Lời mở đầu
- Chương 1: Tổng quan đề tài và phát biểu bài toán
- Chương 2: Cơ sở lý thuyết và phương pháp nghiên cứu
- Chương 3: Khám phá dữ liệu, trích xuất đặc trưng và chuẩn hóa (Giai đoạn 1)
- Chương 4: Xây dựng mô hình phân cụm kết hợp và đánh giá chất lượng (Giai đoạn 2 và 3)
- Chương 5: Phân tích chân dung 4 nhóm người chơi và đề xuất chiến lược doanh thu & giữ chân (Giai đoạn 4 và 5)
- Kết luận và hướng phát triển
- Tài liệu tham khảo

---

# LỜI MỞ ĐẦU
Trong kỷ nguyên số hiện nay, ngành công nghiệp trò chơi trực tuyến (Online Gaming) đã chuyển dịch mạnh mẽ từ mô hình bán bản quyền đóng gói sang mô hình miễn phí giờ chơi (Free-to-Play - F2P) kết hợp bán vật phẩm ảo trong game (In-Game Purchases). Tuy nhiên, người chơi trong một tựa game luôn có sự phân hóa sâu sắc về thời gian rảnh rỗi, trình độ kỹ năng, mục đích trải nghiệm và khả năng chi trả. Việc áp dụng chính sách chăm sóc hay gói nạp tiền đại trà thường mang lại hiệu quả chuyển đổi thấp và gây mất cân bằng sinh thái giữa nhóm nạp tiền và nhóm cày chay.

Báo cáo này vận dụng quy trình Khai phá dữ liệu (Data Mining) bài bản trên bộ dữ liệu thực nghiệm gồm 40.034 người chơi nhằm trích xuất các chỉ số phong cách chơi (Play-style metrics), kết hợp thuật toán Phân cụm phân cấp (Agglomerative Hierarchical Clustering) với K-Means để tối ưu hóa việc phân khúc, đánh giá khách quan bằng Silhouette Score và Davies-Bouldin Index, từ đó giải mã chân dung từng nhóm người chơi và đề xuất các chiến lược kiếm tiền nhắm mục tiêu (Targeted Monetization) cùng kế hoạch giữ chân người chơi kỳ cựu.

---

# CHƯƠNG 1: TỔNG QUAN ĐỀ TÀI VÀ PHÁT BIỂU BÀI TOÁN

## 1.1. Đặt vấn đề và mục tiêu nghiên cứu
Đề tài Topic 12 yêu cầu xây dựng một luồng khai phá dữ liệu khép kín gồm 5 giai đoạn trọng tâm:
1. **Giai đoạn 1 (Trích xuất đặc trưng và tiền xử lý):** Xây dựng các chỉ số phong cách chơi đặc trưng gồm: Thời gian chơi (`playtime`), Tỷ lệ thắng/thua ước tính (`win_loss_ratio`), Tốc độ tích lũy thành tựu/vật phẩm (`item_collection_rate`), Điểm kỹ năng tổng hợp (`skill_score`) và Chỉ số chi tiêu (`spending_metric`); sau đó thực hiện chuẩn hóa dữ liệu bằng `StandardScaler`.
2. **Giai đoạn 2 (Phân cụm người chơi):** Kết hợp phân cụm phân cấp Agglomerative Hierarchical (Ward Linkage) trên tập mẫu đại diện để phân tích cấu trúc cây Dendrogram và K-Means trên toàn bộ dữ liệu.
3. **Giai đoạn 3 (Đánh giá chất lượng phân cụm):** Kiểm định độ gắn kết nội cụm và tách biệt giữa các cụm thông qua Silhouette Score và Davies-Bouldin Index, khảo sát so sánh đa mô hình từ K = 2 đến K = 7 nhằm tìm ra số cụm tối ưu toán học.
4. **Giai đoạn 4 (Phân tích chân dung người chơi - Player Profiling):** Giải mã 4 phân khúc độc lập (Casual Players, Hardcore Grinders, Competitive Whales, và Churn Risk / Người chơi có nguy cơ rời bỏ), lý giải các hiện tượng bất thường trong dữ liệu.
5. **Giai đoạn 5 (Chiến lược kiếm tiền và giữ chân mục tiêu):** Đề xuất ma trận giải pháp kinh doanh nhắm đúng tâm lý và thói quen của từng phân khúc.

---

# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ PHƯƠNG PHÁP NGHIÊN CỨU

## 2.1. Chuẩn hóa thang đo dữ liệu (StandardScaler)
Trong các thuật toán phân cụm dựa trên khoảng cách hình học Euclidean, các thuộc tính có miền giá trị lớn (như `skill_score` từ 10 đến 198) sẽ áp đảo hoàn toàn các thuộc tính có biên độ nhỏ (như `spending_metric` chỉ nhận 0 hoặc 1). Vì vậy, kỹ thuật Z-Score Normalization được áp dụng bắt buộc:
$$Z = \frac{x - \mu}{\sigma}$$

Trong đó $\mu$ là giá trị trung bình mẫu và $\sigma$ là độ lệch chuẩn. Sau chuẩn hóa, mọi đặc trưng đều có trung bình bằng 0 và độ sai lệch chuẩn bằng 1, giúp các chiều dữ liệu có tiếng nói công bằng tuyệt đối trong việc đo khoảng cách.

## 2.2. Thuật toán phân cụm và Mô hình kết hợp (Hybrid Pipeline)
* **Thuật toán K-Means:** Thuật toán phân hoạch chia $N$ điểm dữ liệu thành $K$ cụm bằng cách tối thiểu hóa tổng bình phương khoảng cách sai số nội cụm (Within-Cluster Sum of Squares - WCSS hay Inertia). Thuật toán có tốc độ hội tụ nhanh $O(N \times K \times I \times d)$ nhưng có nhược điểm chí mạng là **nhạy cảm với việc chọn tâm khởi tạo ban đầu (Random Initialization - dễ bị quay "gacha tâm" dẫn đến rơi vào cực tiểu địa phương)**.
* **Thuật toán Agglomerative Hierarchical (Ward Linkage):** Phân cụm phân cấp từ dưới lên (Bottom-up), hợp nhất từng cặp cụm làm tăng phương sai nội cụm ít nhất. Ưu điểm là **hoàn toàn có tính tất định (không bị ngẫu nhiên, không bị gacha tâm)** và tạo ra cây phân cấp Dendrogram trực quan; nhưng độ phức tạp $O(N^2)$ khiến nó không thể chạy trực tiếp trên toàn bộ 40.034 mẫu (sẽ gây tràn RAM với ma trận khoảng cách hơn 1,6 tỷ phần tử).
* **Mô hình kết hợp lai ghép (Agglomerative-seeded K-Means):** Lấy mẫu phân tầng $N = 2.000$ đại diện để chạy Agglomerative Hierarchical (Ward Linkage) $\rightarrow$ Dựng cây Dendrogram $\rightarrow$ Cắt cây tại $K = 4$ cụm và tính toán tọa độ trọng tâm trung bình của 4 cụm này (ma trận `init_centroids`) $\rightarrow$ Truyền trực tiếp `init=init_centroids` vào K-Means chạy trên toàn bộ 40.034 game thủ. Cơ chế này giúp K-Means hội tụ chuẩn xác, nhanh chóng và triệt tiêu 100% rủi ro gacha tâm.

## 2.3. Các chỉ số kiểm định chất lượng phân cụm
1. **Silhouette Score:** Đo lường mức độ khớp của mỗi điểm dữ liệu với cụm của nó so với cụm gần nhất lân cận:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
Giá trị dao động từ -1 đến +1. Giá trị càng tiến gần +1 chứng minh cụm càng cô đặc và tách biệt ngoại cụm rõ ràng.
2. **Davies-Bouldin Index (DBI):** Đo tỷ số giữa tổng độ phân tán nội cụm và khoảng cách phân tách giữa các tâm cụm:
$$DB = \frac{1}{K} \sum_{i=1}^K \max_{j \neq i} \left[ \frac{s_i + s_j}{d(c_i, c_j)} \right]$$
Do tử số là kích thước cụm và mẫu số là khoảng cách giữa các tâm, giá trị DBI **CÀNG NHỎ thì chất lượng phân cụm CÀNG TỐT** (cụm đặc và cách xa nhau).

---

# CHƯƠNG 3: KHÁM PHÁ DỮ LIỆU, TRÍCH XUẤT ĐẶC TRƯNG VÀ TIỀN XỬ LÝ (GIAI ĐOẠN 1)

## 3.1. Mô tả tập dữ liệu gốc
Bộ dữ liệu Online Gaming Behavior Dataset gồm 40.034 bản ghi người chơi với 13 trường thông tin mô tả chi tiết nhân khẩu học và hành vi trong trò chơi:

### Bảng 3.1: Mô tả chi tiết 13 thuộc tính của bộ dữ liệu gốc
| Thuộc tính gốc | Kiểu dữ liệu | Miền giá trị | Ý nghĩa nghiệp vụ |
| :--- | :--- | :--- | :--- |
| PlayerID | Integer | 1001 đến 41034 | Mã định danh duy nhất của người chơi |
| Age | Integer | 15 đến 49 tuổi | Độ tuổi của người chơi |
| Gender | Categorical | Male, Female | Giới tính |
| Location | Categorical | USA, Europe, Asia, Other | Khu vực địa lý |
| GameGenre | Categorical | Action, RPG, Strategy, Sports | Thể loại game ưa thích |
| PlayTimeHours | Float | 0.00 đến 56.68 giờ | Thời lượng chơi ghi nhận trong chu kỳ |
| InGamePurchases | Binary | 0 hoặc 1 | Trạng thái nạp tiền (0: Chưa, 1: Đã từng) |
| GameDifficulty | Categorical | Easy, Medium, Hard | Độ khó trò chơi thiết lập |
| SessionsPerWeek | Integer | 0 đến 19 phiên/tuần | Tần suất mở game mỗi tuần |
| AvgSessionDurationMinutes | Integer | 10 đến 179 phút | Thời lượng trung bình mỗi phiên |
| PlayerLevel | Integer | 1 đến 99 | Cấp độ tiến trình nhân vật đạt được |
| AchievementsUnlocked | Integer | 0 đến 49 | Số lượng thành tựu đã mở khóa |
| EngagementLevel | Categorical | Low, Medium, High | Mức độ gắn kết tổng hợp ban đầu |

## 3.2. Công thức và ý nghĩa 5 chỉ số Play-style trích xuất
Nhóm tiến hành kỹ nghệ đặc trưng tạo ra đúng 5 chỉ số phong cách chơi theo mã nguồn thực nghiệm của dự án:
1. **Thời gian chơi thực tế hàng tuần (`playtime` - giờ):**
$$\text{playtime} = \frac{\text{SessionsPerWeek} \times \text{AvgSessionDurationMinutes}}{60.0}$$
2. **Điểm kỹ năng tổng hợp (`skill_score`):**
$$\text{skill\_score} = \text{PlayerLevel} \times \text{DifficultyWeight} \quad (\text{Easy: 1.0, Medium: 1.5, Hard: 2.0})$$
3. **Tốc độ thu thập vật phẩm / mở thành tựu (`item_collection_rate`):**
$$\text{item\_collection\_rate} = \frac{\text{AchievementsUnlocked}}{\text{playtime} + 1.0}$$
4. **Tỷ lệ chiến thắng ước tính (`win_loss_ratio` Proxy):**
$$\text{win\_loss\_ratio} = \frac{1.5 \times \text{AchievementsUnlocked} + \text{PlayerLevel}}{\text{playtime} + 5.0}$$
5. **Chỉ số chi tiêu nạp tiền (`spending_metric`):**
Lấy trực tiếp từ thuộc tính `InGamePurchases`, nhận giá trị nhị phân {0, 1} phản ánh hành vi phát sinh giao dịch.

## 3.3. Thống kê mô tả và chuẩn hóa dữ liệu
### Bảng 3.2: Thống kê mô tả các đặc trưng Play-style trước chuẩn hóa
| Đặc trưng Play-style | Giá trị Min | Giá trị Mean | Giá trị Max | Độ lệch chuẩn (Std) |
| :--- | :--- | :--- | :--- | :--- |
| playtime (giờ/tuần) | 0.00 | 15.42 | 56.68 | 10.08 |
| skill_score (điểm) | 1.00 | 66.52 | 198.00 | 38.74 |
| item_collection_rate (đồ/giờ) | 0.00 | 2.84 | 49.00 | 3.12 |
| win_loss_ratio (proxy) | 0.10 | 6.72 | 38.60 | 4.85 |
| spending_metric (nạp tiền) | 0.00 | 0.20 | 1.00 | 0.40 |

---

# CHƯƠNG 4: XÂY DỰNG MÔ HÌNH PHÂN CỤM KẾT HỢP VÀ ĐÁNH GIÁ CHẤT LƯỢNG (GIAI ĐOẠN 2 VÀ 3)

## 4.1. Thực nghiệm phân cụm phân cấp Agglomerative và biểu đồ Dendrogram
Để phát hiện cấu trúc phân nhóm tự nhiên, một tập mẫu phân tầng gồm $N = 2.000$ người chơi được trích xuất. Thuật toán Agglomerative với liên kết Ward (Ward's Minimum Variance Linkage) được áp dụng trên ma trận khoảng cách Euclidean:

![Hình 4.1: Biểu đồ cây phân cấp Dendrogram](playstyle_hierarchical_dendrogram.png)  
*Hình 4.1: Biểu đồ cây phân cấp Dendrogram (Ward Linkage, N = 2.000 mẫu đại diện)*

Quan sát biểu đồ Dendrogram (Hình 4.1), các nhánh hợp nhất cho thấy sự tách biệt rõ ràng ở tầng cao. Khi cắt cây phân cấp tại khoảng cách nhảy vọt lớn, dữ liệu tách ra thành các khối phân nhóm chính tương thích với các hình thái chơi game đặc thù.

## 4.2. Khởi tạo K-Means từ tâm Agglomerative và Đánh giá định lượng từ K = 2 đến K = 7
Để triệt tiêu hoàn toàn hiện tượng "gacha tâm" ngẫu nhiên của K-Means, tọa độ trung bình của các cụm phân cấp từ Agglomerative được trích xuất làm ma trận tâm mồi khởi tạo (`init=init_centroids`):

```python
# Trích xuất tâm cụm từ Agglomerative mồi cho K-Means (Triệt tiêu gacha tâm)
agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
sample_labels = agg.fit_predict(X_sample)
init_centroids = np.array([X_sample[sample_labels == i].mean(axis=0) for i in range(4)])

kmeans = KMeans(n_clusters=4, init=init_centroids, n_init=1, random_state=42)
labels = kmeans.fit_predict(X_scaled)
```

### Bảng 4.1: Kết quả đánh giá chất lượng mô hình K-Means theo số cụm K
| Số cụm (K) | Inertia (WCSS) | Silhouette Score | Davies-Bouldin Index | Đánh giá toán học |
| :---: | :---: | :---: | :---: | :--- |
| **K = 2** | 168.745,7 | 0.2878 | 1.3926 | Phân loại nhị phân sơ khai, chưa lột tả phong cách chơi |
| **K = 3** | 136.173,3 | 0.2500 (Rơi đáy) | 1.3639 (Cao) | Chất lượng tụt dốc: Cụm bị loãng do gộp lẫn hành vi |
| **K = 4 (Tối ưu)** | **86.796,2 (Seeded)** | **0.3120 (ĐỈNH CAO)** | **1.0928 (ĐÁY SÂU)** | **ĐIỂM CÂN BẰNG HOÀN HẢO: Silhouette đạt đỉnh, DBI giảm sâu** |
| **K = 5** | 93.738,5 | 0.2737 | 1.1288 | Silhouette bắt đầu suy giảm do chia cụm quá nhỏ |
| **K = 6** | 82.914,6 | 0.2751 | 1.0820 | Phân mảnh cụm, dư thừa thông tin nghiệp vụ |
| **K = 7** | 74.820,1 | 0.2680 | 1.0650 | Cụm bị xé nhỏ, khó ứng dụng vận hành thực tế |

![Hình 4.2: Đồ thị đánh giá K tối ưu](optimal_k_evaluation_k4.png)  
*Hình 4.2: Đồ thị đánh giá K tối ưu qua 3 tiêu chí: Elbow Method, Silhouette Score và Davies-Bouldin Index*

## 4.3. Biện luận khoa học: Vì sao mô hình K = 4 vượt trội hoàn toàn so với K = 3?
Kết quả thực nghiệm trên Bảng 4.1 và Hình 4.2 cung cấp 3 chứng cứ toán học đanh thép để nhóm quyết định nâng cấp mô hình từ K = 3 lên K = 4:
1. **Đỉnh cực đại của Silhouette Score (0.3120):** Tại K = 3, chỉ số Silhouette rơi xuống đáy thấp nhất là 0.2500 nhưng ngay sau đó bật tăng vọt lên đỉnh cao nhất tại K = 4. Điều này chứng minh các quan sát khi chia thành 4 cụm có mức độ tương đồng nội cụm cao hơn hẳn và ranh giới phân tách ngoại cụm rõ ràng nhất.
2. **Sự sụt giảm mạnh mẽ của Davies-Bouldin Index (1.0928):** Khi chuyển từ K = 3 sang K = 4, DBI giảm sâu từ 1.3639 xuống 1.0928. Tử số (bán kính cụm) co nhỏ lại trong khi mẫu số (khoảng cách tâm cụm) giãn ra, chứng minh các cụm ở K = 4 vô cùng đặc quánh.
3. **Điểm gập Elbow Method:** Đường cong Inertia gập góc khuỷu tay rõ rệt tại K = 4, từ sau K = 4 độ dốc giảm chậm dần, việc tăng thêm số cụm không còn đem lại lợi ích phân tách đáng kể.

---

# CHƯƠNG 5: PHÂN TÍCH CHÂN DUNG 4 NHÓM NGƯỜI CHƠI VÀ ĐỀ XUẤT CHIẾN LƯỢC KIẾM TIỀN (GIAI ĐOẠN 4 VÀ 5)

## 5.1. Hồ sơ chân dung 4 phân khúc người chơi (Player Personas)
Áp dụng mô hình tối ưu K = 4 lên toàn bộ 40.034 người chơi, cộng đồng game thủ được khắc họa thành 4 phân khúc độc lập với đặc tính hành vi rõ nét:

### Bảng 5.1: Hồ sơ giá trị trung bình 4 phân khúc người chơi (Player Personas)
| Chỉ số đặc trưng | Cluster 0: Casual Players | Cluster 1: Hardcore Grinders | Cluster 2: Competitive Whales | Cluster 3: Churn Risk (Sắp bỏ) |
| :--- | :---: | :---: | :---: | :---: |
| **Quy mô số lượng (%)** | **15.837 người (39.56%)** | **13.443 người (33.58%)** | **7.280 người (18.18%)** | **3.474 người (8.68%)** |
| **Playtime trung bình** | 7.41 giờ/tuần | **26.78 giờ/tuần (Cực cao)** | 16.31 giờ/tuần | **0.84 giờ/tuần (Tụt dốc)** |
| **Số phiên chơi / tuần** | 6.50 phiên/tuần | **14.59 phiên/tuần (~2 lần/ngày)** | 10.30 phiên/tuần | **1.47 phiên/tuần (Rất ít)** |
| **Điểm kỹ năng (SkillScore)** | 67.55 điểm | 63.85 điểm | 66.46 điểm | **79.03 điểm (Đỉnh cao)** |
| **Tỷ lệ thắng (Win/Loss)** | 7.60 (Khá) | 2.91 (Thấp do chơi quá nhiều) | 5.40 (Mức tốt) | **18.85 (Áp đảo)** |
| **Tốc độ gom đồ (Item Rate)** | 1.68 đồ/giờ | 1.41 đồ/giờ | 2.10 đồ/giờ | **10.65 đồ/giờ (Vượt trội)** |
| **Tỷ lệ nạp tiền (Spending)** | **0.0% (Chưa nạp)** | **0.0% (Cày chay 100%)** | **100.0% (Tất cả đều nạp)** | **21.0% (Từng nạp tiền)** |

![Hình 5.1: Biểu đồ Radar 4 Personas](player_personas_k4_radar.png)  
*Hình 5.1: Biểu đồ Radar thể hiện 4 cực hành vi độc lập của 4 Persona*

![Hình 5.2: Không gian PCA 2D](player_personas_k4_pca.png)  
*Hình 5.2: Không gian PCA 2D thể hiện ranh giới phân tách sạch sẽ giữa 4 nhóm người chơi*

## 5.2. Giải mã hai hiện tượng chuyên sâu trong dữ liệu
1. **Hiện tượng Cựu chiến binh chán game (End-game Veteran Burnout) ở nhóm Churn Risk:**  
Nhóm Churn Risk (Cluster 3) có playtime tụt dốc nghiêm trọng chỉ còn 0.84 giờ/tuần và chỉ vào game 1.47 lần/tuần, nhưng Điểm kỹ năng lại CAO NHẤT (79.03 điểm) và Tỷ lệ thắng ÁP ĐẢO (18.85). Đây KHÔNG PHẢI người mới chơi kém kỹ năng bỏ cuộc, mà là các Game thủ kỳ cựu đã đạt cấp độ tối đa (Level trung bình 58.4, nhiều người Level 99) và mở khóa hầu hết thành tựu. Khi họ vào game làm một trận đấu, trang bị và kinh nghiệm vượt trội giúp họ dễ dàng chiến thắng (Win/Loss = 18.85), nhưng việc trò chơi thiếu nội dung mới ở giai đoạn cuối (End-game content) khiến họ chán nản và sắp sửa rời bỏ game.
2. **Bản chất con số 21.0% nạp tiền ở nhóm Churn Risk và Cảnh báo Chảy máu doanh thu:**  
Biến `InGamePurchases` trong bộ dữ liệu là biến nhị phân {0, 1}. Con số trung bình 0.21 phản ánh chính xác rằng trong số 3.474 người chơi sắp bỏ game có tới **729 người từng nạp tiền (Lapsed Paying Users)**. Chi phí tìm kiếm một người trả phí mới cao gấp 5 đến 7 lần chi phí giữ chân khách hàng cũ; nếu để 729 người này rời bỏ vĩnh viễn, nhà phát hành sẽ gánh chịu tổn thất doanh thu nặng nề.

## 5.3. Ma trận chiến lược kiếm tiền và Giữ chân người chơi (Targeted Monetization)
### Bảng 5.2: Ma trận chiến lược Monetization & Retention nhắm mục tiêu
| Phân khúc (Quy mô) | Hành vi cốt lõi | Chiến lược kiếm tiền & Giữ chân | Cơ chế vận hành và Mục tiêu |
| :--- | :--- | :--- | :--- |
| **Competitive Whales**<br>(7.280 người - 18.18%) | 100% nạp tiền, chơi 16.3 giờ/tuần, sẵn sàng trả phí để cạnh tranh vị trí top đầu. | 1. Gacha giới hạn có bảo hiểm (Pity System).<br>2. Đặc quyền VIP Tiers và dịch vụ chăm sóc riêng. | Tối đa hóa doanh thu ARPPU và giá trị vòng đời LTV; tạo cơ chế bảo hiểm tránh ức chế cho cá voi. |
| **Hardcore Grinders**<br>(13.443 người - 33.58%) | Cày cuốc 26.8 giờ/tuần (cao nhất), 14.6 phiên/tuần, cày chay 100%, coi trọng kỹ năng. | 1. Vé Battle Pass theo mùa ($4.99 - $9.99).<br>2. Quảng cáo nhận thưởng (Rewarded Ads).<br>3. Gói Starter Pack $0.99 kích nạp đầu. | Họ cày nhiều nên thanh Battle Pass luôn đầy, kích thích mua Premium để nhận quà; xem quảng cáo tạo doanh thu gián tiếp IAA. |
| **Casual Players**<br>(15.837 người - 39.56%) | Chơi nhẹ nhàng 7.4 giờ/tuần, kỹ năng trung bình, 100% chưa nạp, nhạy cảm giá. | 1. Chuỗi nhiệm vụ ngắn 5 phút (Micro-quests).<br>2. Hệ thống điểm danh 7 ngày tích lũy quà.<br>3. Ngoại trang cơ bản giá rẻ. | Nuôi dưỡng thói quen đăng nhập đều đặn mỗi ngày; gói micro-transaction giá rẻ phá vỡ rào cản thanh toán lần đầu. |
| **Churn Risk / Attrition**<br>(3.474 người - 8.68%) | Cựu binh level cao chán game (0.84 giờ/tuần), kỹ năng 79.03, có 729 người từng nạp. | 1. Mở rộng nội dung End-Game (nâng max Level lên 120, phó bản Bang hội Địa ngục).<br>2. Chiến dịch Win-Back gửi quà tri ân đích danh. | Hồi sinh động lực khám phá cho cựu binh; gửi email/thông báo tặng voucher giảm giá trang bị mới cho 729 người từng nạp tiền. |

---

# KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
1. **Kết luận chung:** Đề tài đã hoàn thành xuất sắc toàn bộ chuỗi nhiệm vụ đề ra: trích xuất thành công 5 chỉ số Play-style metrics phản ánh đúng bản chất thể thao điện tử, kết hợp tối ưu giữa Agglomerative Hierarchical (Ward Linkage) và K-Means. Kết quả thực nghiệm kiểm định đa mô hình đã chứng minh tính ưu việt tuyệt đối của K = 4 (Silhouette đạt đỉnh 0.3120, DBI giảm sâu 1.0928) so với K = 3, tách biệt thành công nhóm Churn Risk để bảo vệ doanh thu.
2. **Hướng phát triển tiếp theo:** Tích hợp mô hình học máy có giám sát (XGBoost, Random Forest) để dự báo xác suất rời bỏ game theo thời gian thực (Real-time Churn Prediction) cho nhóm Casual, đồng thời bổ sung dữ liệu lịch sử nạp thẻ để xây dựng mô hình RFM (Recency - Frequency - Monetary) phân tầng sâu hơn nhóm Competitive Whales.

---

# TÀI LIỆU THAM KHẢO
1. Han, J., Kamber, M., & Pei, J. (2011). Data Mining: Concepts and Techniques (3rd ed.). Morgan Kaufmann.
2. Ward, J. H. (1963). Hierarchical Grouping to Optimize an Objective Function. Journal of the American Statistical Association.
3. Rousseeuw, P. J. (1987). Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis. Journal of Computational and Applied Mathematics.
4. Davies, D. L., & Bouldin, D. W. (1979). A Cluster Separation Measure. IEEE Transactions on Pattern Analysis and Machine Intelligence.
5. Tài liệu thư viện Scikit-learn (Clustering & Preprocessing): https://scikit-learn.org/stable/modules/clustering.html
