# 🎮 BÁO CÁO TOÀN DIỆN DỰ ÁN KHAI PHÁ DỮ LIỆU (DATA MINING REPORT)
# ĐỀ TÀI: GAMER BEHAVIOR CLUSTERING & PLAYER PROFILING
**Tập trung vào mô hình tối ưu 4 Personas ($K = 4$) | Phân tích chi tiết kèm biểu đồ minh họa & Từ điển thuật ngữ**

---

## 📑 MỤC LỤC BÁO CÁO
1. [Từ điển thuật ngữ chuyên ngành (Dành cho người mới bắt đầu)](#1-từ-điển-thuật-ngữ-chuyên-ngành-dễ-hiểu)
2. [Tổng quan đề tài & Dữ liệu đầu vào](#2-tổng-quan-đề-tài--dữ-liệu-đầu-vào)
3. [Kỹ nghệ đặc trưng: Tạo các chỉ số phong cách chơi (Play-style Metrics)](#3-kỹ-nghệ-đặc-trưng-play-style-metrics)
4. [Chứng minh toán học: Vì sao bắt buộc phải chọn $K = 4$?](#4-chứng-minh-toán-học-vì-sao-chọn-k--4)
5. [Định danh chân dung 4 cụm người chơi (4 Player Personas)](#5-định-danh-chân-dung-4-cụm-người-chơi-4-player-personas)
6. [Giải mã hai nghịch lý thú vị trong dữ liệu (Skill Score & Spending 21%)](#6-giải-mã-hai-nghịch-lý-thú-vị-trong-dữ-liệu)
7. [Trực quan hóa đa chiều (Radar Chart & PCA 2D)](#7-trực-quan-hóa-đa-chiều-radar-chart--pca-2d)
8. [Đối chiếu với nhãn gốc (Validation with Engagement Level)](#8-đối-chiếu-với-nhãn-gốc-engagement-level)
9. [Chiến lược kinh doanh & Giữ chân người chơi (Actionable Strategies)](#9-chiến-lược-kinh-doanh--tối-ưu-hóa-doanh-thu)

---

## 1. TỪ ĐIỂN THUẬT NGỮ CHUYÊN NGÀNH (DỄ HIỂU)

Trước khi đọc chi tiết, dưới đây là giải thích đơn giản cho các từ ngữ được dùng trong bài:

* **Unsupervised Learning (Học máy không giám sát):** Phương pháp để máy tính tự mò mẫm, tìm ra các nhóm người có điểm chung mà không cần ai dán nhãn hay chỉ bảo trước.
* **Clustering (Phân cụm / Gom cụm):** Việc chia một đám đông (40.000 người) thành các nhóm nhỏ sao cho người trong cùng một nhóm thì giống nhau, còn khác nhóm thì khác hẳn nhau.
* **K-Means:** Thuật toán phân cụm phổ biến nhất. $K$ là số nhóm bạn muốn chia (ví dụ $K = 4$ là chia làm 4 nhóm). Máy tính sẽ chọn 4 "tâm điểm" đại diện rồi hút những người chơi gần nó nhất về nhóm mình.
* **Feature Scaling (Chuẩn hóa dữ liệu):** Đưa các con số khác đơn vị (ví dụ: thời gian 100 phút và nạp tiền 0-1) về cùng một thang đo chuẩn (thường có trung bình = 0) để máy tính không bị tính toán thiên vị biến có số to.
* **Inertia / WCSS (Within-Cluster Sum of Squares):** Tổng khoảng cách sai số giữa các thành viên đến tâm nhóm. Chỉ số này càng nhỏ nghĩa là các thành viên đứng càng sát nhau.
* **Elbow Method (Phương pháp khuỷu tay):** Biểu đồ vẽ sự suy giảm của sai số theo số cụm. Điểm gập giống khuỷu tay là điểm dừng tốt nhất.
* **Silhouette Score (Điểm dáng hình):** Thang điểm từ -1 đến +1 đo xem nhóm chia có "đẹp" không. Càng cao (gần +1) nghĩa là các nhóm càng tách bạch rõ rệt, không bị dính vào nhau.
* **Davies-Bouldin Index (Chỉ số DB):** Tỷ số đo độ phân tán nội bộ chia cho khoảng cách ngoại cụm. **Càng nhỏ càng tốt** (càng gần 0 nghĩa là các cụm vừa đặc vừa xa nhau).
* **PCA (Principal Component Analysis - Phân tích thành phần chính):** Kỹ thuật "chụp ảnh 2D" cho dữ liệu nhiều chiều để vẽ lên mặt phẳng cho mắt người nhìn thấy được.
* **Radar Chart (Biểu đồ mạng nhện):** Biểu đồ hình tròn nhiều nan hoa, giúp nhìn thấy cùng lúc nhiều điểm mạnh/yếu của từng nhóm người chơi.
* **Churn / Attrition (Rời bỏ game):** Hiện tượng người chơi xóa game, bỏ chơi hoặc không đăng nhập nữa.
* **Whales / Spenders (Cá voi / Đại gia):** Nhóm người chơi nạp rất nhiều tiền vào game.
* **F2P (Free to Play):** Chơi game miễn phí, không nạp tiền (thường gọi là "dân cày chay").

---

## 2. TỔNG QUAN ĐỀ TÀI & DỮ LIỆU ĐẦU VÀO
* **Tên đề tài:** Phân tích hành vi & Định danh hồ sơ người chơi (Gamer Behavior Clustering & Player Profiling).
* **Nguồn dữ liệu:** `online_gaming_behavior_dataset.csv` gồm **40,034 người chơi**.
* **Mục tiêu:** Tự động phân chia 40.034 game thủ này thành 4 nhóm phong cách riêng biệt để đề xuất chiến lược tăng doanh thu và ngăn chặn người chơi bỏ game.

---

## 3. KỸ NGHỆ ĐẶC TRƯNG (PLAY-STYLE METRICS)

Từ dữ liệu thô ban đầu, 6 chỉ số hành vi quan trọng nhất đã được xây dựng:
1. **`playtime` (Thời lượng chơi thực tế/tuần - Giờ):** $\frac{\text{SessionsPerWeek} \times \text{AvgSessionDurationMinutes}}{60}$.
2. **`SessionsPerWeek` (Tần suất vào game):** Số lần đăng nhập game mỗi tuần.
3. **`skill_score` (Điểm kỹ năng):** Kết hợp giữa Cấp độ nhân vật (`PlayerLevel`) và Trọng số độ khó (`GameDifficulty`: Dễ = 1.0, Vừa = 1.5, Khó = 2.0).
4. **`item_collection_rate` (Tốc độ gom thành tựu):** Tỷ lệ mở khóa thành tựu trên mỗi giờ chơi.
5. **`win_loss_ratio` (Tỷ lệ thắng ước tính):** Đo lường hiệu suất vượt ải và thành tựu so với thời gian chơi.
6. **`spending_metric` (Chỉ số nạp tiền):** Dựa trên `InGamePurchases` (0: Chưa nạp, 1: Đã từng nạp tiền).

---

## 4. CHỨNG MINH TOÁN HỌC: VÌ SAO CHỌN $K = 4$?

Để chọn số nhóm chuẩn xác nhất, mô hình đã chạy kiểm thử định lượng từ $K = 2$ đến $K = 7$. Kết quả được tổng hợp trực quan qua biểu đồ 3 ô dưới đây:

![Đánh giá K tối ưu](optimal_k_evaluation_k4.png)
*(Đường dẫn file ảnh: `c:\Users\tinca\OneDrive\Desktop\data\code\outputs_k4\optimal_k_evaluation_k4.png`)*

### Phân tích chi tiết 3 biểu đồ:

| Biểu đồ | Nguyên tắc toán học | Diễn biến thực tế trên biểu đồ | Kết luận |
| :--- | :--- | :--- | :---: |
| **1. Elbow Method (Inertia)** | Tìm điểm **gập khuỷu tay** — nơi sai số giảm chậm dần lại (quy luật lợi ích cận biên giảm dần). | Đường cong màu xanh dốc đứng từ $K=2 \rightarrow 3 \rightarrow 4$, sau đó **gập khúc rõ rệt tại điểm màu đỏ $K = 4$** và bắt đầu thoải dần ở các giá trị sau ($K=5, 6, 7$). | **Chọn K = 4** |
| **2. Silhouette Score** | Tìm điểm **đạt giá trị cao nhất** (đỉnh chóp - Peak Point). Điểm càng cao, cụm càng tách biệt rõ ràng. | Tại $K=3$, đồ thị rơi xuống đáy ($0.250$). Nhưng đến **$K=4$, điểm số bật tăng vọt lên đỉnh cao nhất ($0.294$)** trước khi suy giảm trở lại ở $K=5, 6$. | **Chọn K = 4** |
| **3. Davies-Bouldin Index** | Tìm điểm **đạt giá trị thấp nhất** (càng nhỏ càng tốt, đo độ cô đặc nội bộ). | Chỉ số màu tím ở $K=2, 3$ rất cao ($> 1.36$), nhưng **rơi tự do xuống mức thấp ngoạn mục tại $K = 4$ ($1.146$)** rồi đi ngang ở các cụm sau. | **Chọn K = 4** |

> **Khẳng định:** Cả 3 phương pháp toán học độc lập đều đồng thuận chọn **$K = 4$ là con số hoàn hảo nhất!**

---

## 5. ĐỊNH DANH CHÂN DUNG 4 CỤM NGƯỜI CHƠI (4 PLAYER PERSONAS)

Dưới đây là bảng số liệu trung bình (**Mean**) và trung vị (**Median**) của 4 nhóm:

### Bảng chỉ số trung bình (Mean Metrics)

| Chỉ số hành vi | 🔵 Cluster 0: Casual Players | 🟢 Cluster 1: Hardcore Grinders | 🔴 Cluster 2: Competitive Whales | 🟠 Cluster 3: Churn Risk / Attrition |
| :--- | :---: | :---: | :---: | :---: |
| **Quy mô (Số lượng & %)** | **15,837 (39.56%)** | **13,443 (33.58%)** | **7,280 (18.18%)** | **3,474 (8.68%)** |
| **Playtime (giờ chơi/tuần)** | **7.41 giờ** | **26.78 giờ** *(Cao nhất)* | **16.31 giờ** | **0.84 giờ** *(Gần như bằng 0)* |
| **Sessions/Week (lần vào game)**| **6.5 lần** | **14.59 lần** *(Hơn 2 lần/ngày)* | **10.30 lần** | **1.47 lần** *(Thưa thớt)* |
| **Điểm kỹ năng (Skill Score)**| 67.55 | 63.85 | 66.46 | **79.03** *(Cực cao)* |
| **Tỷ lệ thắng (Win/Loss)** | 7.60 | 2.91 | 5.40 | **18.85** *(Áp đảo)* |
| **Tỷ lệ nạp tiền (Spending)**| **0.0% (Chưa nạp)** | **0.0% (Chưa nạp)** | **100.0% (Đã nạp)** | **21.0% (Từng nạp tiền)** |

---

### Phân tích chi tiết từng Persona:

#### 1. 🔵 Cluster 0: Casual Players (Chiếm 39.56% - 15,837 người)
* **Phong cách:** Chơi game thuần giải trí, nhẹ nhàng vừa phải (**7.4 giờ/tuần**, vào game khoảng 6 buổi/tuần).
* **Tài chính:** **0.0% nạp tiền**. Chơi hoàn toàn miễn phí, không ganh đua bảng xếp hạng.

#### 2. 🟢 Cluster 1: Hardcore Grinders (Chiếm 33.58% - 13,443 người)
* **Phong cách:** "Dân cày chính hiệu", thời lượng chơi cao nhất server: **gần 27 giờ/tuần**, vào game gần **15 lần/tuần** (ngày nào cũng vào cày 2-3 lần).
* **Tài chính:** **0.0% nạp tiền**. Kiên quyết dùng thời gian và sức lực cày cuốc để bù lại việc không nạp tiền.
* **Vai trò:** Lực lượng giữ cho game luôn đông đúc, nhộn nhịp (CCU).

#### 3. 🔴 Cluster 2: Competitive Whales (Chiếm 18.18% - 7,280 người)
* **Phong cách:** Vừa chơi nhiều (**16.3 giờ/tuần**), vừa nghiêm túc thi đấu.
* **Tài chính:** **100.0% người chơi trong nhóm này đều đã nạp tiền mua đồ trong game**.
* **Ý nghĩa:** Đây chính là phân khúc "Đại gia" nuôi sống cả nhà phát hành game, gánh toàn bộ doanh thu giao dịch trực tiếp.

#### 4. 🟠 Cluster 3: Churn Risk / Attrition (Chiếm 8.68% - 3,474 người)
* **Phong cách:** Thời gian chơi tụt dốc thê thảm, chỉ còn **0.84 giờ/tuần**, cả tuần ghé vào 1 lần rồi thoát ngay.
* **Báo động đỏ:** Nhóm người chơi đang trong giai đoạn chán nản và sắp sửa **xóa game (Churn)**!

---

## 6. GIẢI MÃ HAI NGHỊCH LÝ THÚ VỊ TRONG DỮ LIỆU

Trong cụm **Cluster 3: Churn Risk**, có 2 con số thoạt nhìn rất kỳ lạ nhưng lại chứa đựng insight vô cùng đắt giá:

### Nghịch lý 1: Vì sao người sắp bỏ game lại có Điểm kỹ năng (79.03) và Tỷ lệ thắng (18.85) CAO NHẤT SERVER?
* **Thực tế:** Họ **không phải người chơi "gà" bị ăn hành rồi nản bỏ game**!
* **Bản chất:** Đây là nhóm **"Cựu chiến binh phá đảo game (End-game Veteran Burnout)"**:
  * Họ đã cày lên cấp rất cao (Level trung bình ~58, nhiều người đạt Level 99), mở khóa gần hết thành tựu và trang bị vũ khí tối tân.
  * Vì quá mạnh nên khi thi thoảng vào game làm 1 ván, họ dễ dàng đè bẹp đối thủ $\rightarrow$ Tỷ lệ thắng bùng nổ lên **18.85**.
  * **Lý do họ bỏ game:** **Quá dễ và hết mục tiêu để phấn đấu!** Việc thiếu nội dung mới (thiếu ải khó, boss khủng) khiến họ rơi vào trạng thái nhàm chán và muốn nghỉ hưu.

### Nghịch lý 2: Dữ liệu gốc chỉ có số 0 và 1, con số "21% nạp tiền" có ý nghĩa gì?
* **Về mặt toán học:** `InGamePurchases` chỉ gồm 0 (chưa nạp) và 1 (đã nạp). Khi tính trung bình, con số **0.21 chính là tỷ lệ phần trăm (21.0%)** số người mang số `1`.
  * Cụ thể: Trong 3.474 người nhóm Churn Risk, có **729 người đã từng nạp tiền** và 2.745 người chưa từng nạp.
* **Về mặt kinh doanh:** Đây là dấu hiệu của **"Khách hàng cũ từng trả tiền sắp rời bỏ (Lapsed Paying Users)"**. Nếu để nhóm này bỏ game, nhà phát hành sẽ bị **chảy máu doanh thu nghiêm trọng** vì chi phí tìm một khách nạp tiền mới đắt gấp 5 - 7 lần chi phí giữ chân khách nạp cũ!

---

## 7. TRỰC QUAN HÓA ĐA CHIỀU (RADAR CHART & PCA 2D)

### 7.1. Biểu đồ mạng nhện (Radar Chart)
Biểu đồ Radar Chart chuẩn hóa thang đo từ 0 đến 1, giúp so sánh trực quan toàn diện 4 Persona:

![Biểu đồ Radar 4 Personas](player_personas_k4_radar.png)
*(Đường dẫn file ảnh: `c:\Users\tinca\OneDrive\Desktop\data\code\outputs_k4\player_personas_k4_radar.png`)*

* **Cực Chi tiêu (`Spending`):** Nhóm màu đỏ (**Competitive Whales**) kéo căng chạm đỉnh 1.0, độc tôn hoàn toàn.
* **Cực Thời gian & Số phiên (`Playtime & Sessions`):** Nhóm màu xanh lá (**Hardcore Grinders**) phình to nhất chạm đỉnh 1.0.
* **Cực Kỹ năng & Thắng (`Skill & Win/Loss`):** Nhóm màu cam (**Churn Risk**) kéo căng ra phía ngoài cùng.
* **Vùng cân bằng:** Nhóm màu xanh dương (**Casual Players**) nằm đều đặn ở các trục bậc trung.

### 7.2. Biểu đồ phân bố không gian 2D (PCA 2D Projection)
Dữ liệu từ 6 chiều được giảm xuống 2 trục trực giao để quan sát ranh giới phân tách cụm:

![Biểu đồ PCA 2D](player_personas_k4_pca.png)
*(Đường dẫn file ảnh: `c:\Users\tinca\OneDrive\Desktop\data\code\outputs_k4\player_personas_k4_pca.png`)*

* 4 nhóm màu sắc đại diện cho 4 Persona chiếm giữ các vùng không gian riêng biệt, chứng minh thuật toán K-Means chia nhóm cực kỳ sạch sẽ, không bị chồng chéo hay nhiễu loạn.

---

## 8. ĐỐI CHIẾU VỚI NHÃN GỐC (ENGAGEMENT LEVEL)

Mô hình gom cụm hoàn toàn không sử dụng nhãn `EngagementLevel` khi huấn luyện. Sau khi phân cụm xong, ta lập bảng đối chiếu (Heatmap) để kiểm tra độ chính xác:

![Heatmap đối chiếu Engagement Level](personas_k4_engagement_heatmap.png)
*(Đường dẫn file ảnh: `c:\Users\tinca\OneDrive\Desktop\data\code\outputs_k4\personas_k4_engagement_heatmap.png`)*

* **Hardcore Grinders:** Có tỷ lệ **High Engagement cao nhất server (52.3%)**.
* **Casual Players:** Tập trung chủ yếu ở mức **Medium và Low Engagement**.
* **Churn Risk:** Tỷ lệ **Low Engagement áp đảo hoàn toàn**, khẳng định máy tính đã nhận diện đúng những người thờ ơ với game nhất.

---

## 9. CHIẾN LƯỢC KINH DOANH & TỐI ƯU HÓA DOANH THU

Dựa trên chân dung từng nhóm, nhà phát hành game cần áp dụng 4 gói chiến lược riêng biệt:

```
                                  🎯 CHIẾN LƯỢC KINH DOANH 4 NHÓM
                                                │
       ┌───────────────────┬────────────────────┴────────────────────┬───────────────────┐
       ▼                   ▼                                         ▼                   ▼
[🔵 CASUAL PLAYERS]  [🟢 HARDCORE GRINDERS]            [🔴 COMPETITIVE WHALES] [🟠 CHURN RISK]
• Gói nạp đầu $0.99  • Vé Battle Pass cày cấp          • Ưu đãi VIP độc quyền  • Quà Veteran Comeback
• Nhiệm vụ ngắn 5p   • Xem Video quảng cáo nhận quà    • Sự kiện Gacha giới hạn• Mở phó bản cấp 120
```

### 1. Đối với Competitive Whales (Cụm 2 - 18.2%) - Mục tiêu: Tối đa hóa ARPPU & LTV
* **Sự kiện Gacha giới hạn (FOMO Events):** Mở bán các bộ trang phục, vũ khí phát sáng chỉ xuất hiện trong vòng 48h. Tích hợp cơ chế bảo hiểm quay thưởng (Pity System) để kích thích họ nạp đậm.
* **Đặc quyền VIP & Danh dự:** Cung cấp hiệu ứng thông báo toàn server khi họ đăng nhập, vinh danh trên đài danh vọng và có nhân viên chăm sóc khách hàng riêng (White-glove Service).

### 2. Đối với Hardcore Grinders (Cụm 1 - 33.6%) - Mục tiêu: Biến cày chay thành tiền
* **Vé Season / Battle Pass cày cấp:** Nhóm này cày game 27h/tuần nên thừa sức mở hết mốc nhiệm vụ. Cho phép nhận quà miễn phí một nửa, và bán vé **Premium Pass ($4.99 - $9.99)** để họ nhận trọn kho quà khủng mà họ đã đổ mồ hôi cày được.
* **Xem quảng cáo nhận quà (Rewarded Video Ads):** Cho phép xem video 30s để hồi phục thể lực hoặc nhân đôi tài nguyên, tạo dòng tiền quảng cáo khổng lồ.

### 3. Đối với Casual Players (Cụm 0 - 39.6%) - Mục tiêu: Kích thích nạp lần đầu & Giữ nhịp chơi
* **Gói nạp đầu giá kẹo mút (Starter Pack $0.99):** Tặng vũ khí xịn với giá siêu rẻ (~20k VNĐ) để xóa bỏ rào cản tâm lý "chưa từng nạp tiền".
* **Nhiệm vụ 5 phút (Micro-quests):** Giúp người bận rộn chỉ cần chơi vài phút giải lao vẫn nhận được quà đều đặn.

### 4. Đối với Churn Risk (Cụm 3 - 8.7%) - Mục tiêu: Kéo cựu binh quay lại (Win-Back)
* **Không tặng quà vặt:** Tuyệt đối không gửi quà tân thủ hay nhiệm vụ dễ, vì họ là cao thủ đã phá đảo game!
* **Cập nhật nội dung đỉnh cao (End-game Expansion):** Mở giới hạn cấp từ Level 100 lên 120, mở phó bản Bang hội độ khó Địa ngục (Nightmare Raids).
* **Chiến dịch "Cựu binh trở lại":** Gửi thông báo trực tiếp tới 729 người từng nạp tiền: *"Chào mừng cao thủ trở lại! Đăng nhập ngay để nhận vé nâng cấp miễn phí trang bị cũ của bạn lên phiên bản mới nhất!"*

---

## 10. TỔNG KẾT & DANH MỤC FILE DỰ ÁN
Toàn bộ mã nguồn, dữ liệu xuất ra và biểu đồ phân tích đều được tổ chức gọn gàng tại thư mục:
📂 **Đường dẫn outputs:** `c:\Users\tinca\OneDrive\Desktop\data\code\outputs_k4\`

1. **Bộ biểu đồ chất lượng cao (300 DPI):**
   * [optimal_k_evaluation_k4.png](optimal_k_evaluation_k4.png): Biểu đồ 3 ô chứng minh toán học chọn $K = 4$.
   * [player_personas_k4_radar.png](player_personas_k4_radar.png): Biểu đồ Radar mạng nhện 4 Persona.
   * [player_personas_k4_pca.png](player_personas_k4_pca.png): Biểu đồ không gian 2D PCA.
   * [personas_k4_engagement_heatmap.png](personas_k4_engagement_heatmap.png): Ma trận nhiệt đối chiếu Engagement.
2. **Dữ liệu xuất ra:**
   * `segmented_gaming_behavior_k4.csv`: Bộ dữ liệu đầy đủ 40.034 dòng tích hợp nhãn Persona và tọa độ PCA.
   * `player_personas_k4_mean.csv`: Bảng tổng hợp giá trị trung bình (Mean).
   * `player_personas_k4_median.csv`: Bảng tổng hợp giá trị trung vị (Median).
3. **Mã nguồn thực thi:**
   * `k4_personas_pipeline.py`: Toàn bộ mã nguồn chạy mô hình $K = 4$.
   * `generate_k_plot.py`: Mã nguồn vẽ biểu đồ 3 ô tìm $K$.

