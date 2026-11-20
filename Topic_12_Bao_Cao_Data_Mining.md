HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG

KHOA CÔNG NGHỆ THÔNG TIN 1

--------------------------------------------------

BÁO CÁO BÀI TẬP LỚN MÔN HỌC: KHAI PHÁ DỮ LIỆU

ĐỀ TÀI: TRÍCH XUẤT CHỈ SỐ PHONG CÁCH CHƠI, PHÂN KHÚC NGƯỜI CHƠI GAME TRỰC TUYẾN BẰNG MÔ HÌNH KẾT HỢP AGGLOMERATIVE HIERARCHICAL - K-MEANS VÀ ĐỀ XUẤT CHIẾN LƯỢC DOANH THU

Hà Nội, Tháng 11 Năm 2026


# MỤC LỤC BÁO CÁO

•  Lời mở đầu

•  Chương 1: Tổng quan đề tài và phát biểu bài toán

•  Chương 2: Cơ sở lý thuyết và phương pháp nghiên cứu

•  Chương 3: Khám phá dữ liệu, trích xuất đặc trưng và chuẩn hóa (Giai đoạn 1)

•  Chương 4: Xây dựng mô hình phân cụm kết hợp và đánh giá chất lượng (Giai đoạn 2 và 3)

•  Chương 5: Phân tích chân dung 4 nhóm người chơi và đề xuất chiến lược kiếm tiền (Giai đoạn 4 và 5)

•  Kết luận và hướng phát triển

•  Tài liệu tham khảo


# LỜI MỞ ĐẦU

Trong kỷ nguyên số hiện nay, ngành công nghiệp trò chơi trực tuyến (Online Gaming) đã chuyển dịch mạnh mẽ từ mô hình bán bản quyền đóng gói sang mô hình miễn phí giờ chơi (Free-to-Play - F2P) kết hợp bán vật phẩm ảo trong game (In-Game Purchases). Tuy nhiên, người chơi trong một tựa game luôn có sự phân hóa sâu sắc về thời gian rảnh rỗi, trình độ kỹ năng, mục đích trải nghiệm và khả năng chi trả. Việc áp dụng chính sách chăm sóc hay gói nạp tiền đại trà thường mang lại hiệu quả chuyển đổi thấp và gây mất cân bằng sinh thái giữa nhóm nạp tiền và nhóm cày chay.

Báo cáo này vận dụng quy trình Khai phá dữ liệu (Data Mining) bài bản trên bộ dữ liệu thực nghiệm gồm 40.034 người chơi nhằm trích xuất các chỉ số phong cách chơi (Play-style metrics), kết hợp thuật toán Phân cụm phân cấp (Agglomerative Hierarchical Clustering) với K-Means để tối ưu hóa việc phân khúc, đánh giá khách quan bằng Silhouette Score và Davies-Bouldin Index, từ đó giải mã chân dung từng nhóm người chơi và đề xuất các chiến lược kiếm tiền nhắm mục tiêu (Targeted Monetization) cùng kế hoạch giữ chân người chơi kỳ cựu.


# CHƯƠNG 1: TỔNG QUAN ĐỀ TÀI VÀ PHÁT BIỂU BÀI TOÁN


## 1.1. Đặt vấn đề và mục tiêu nghiên cứu

Đề tài Topic 12 yêu cầu xây dựng một luồng khai phá dữ liệu khép kín gồm 5 giai đoạn trọng tâm:


## 1. Giai đoạn 1 (Trích xuất đặc trưng và tiền xử lý): Xây dựng các chỉ số phong cách chơi đặc trưng gồm: Thời gian chơi (playtime), Tỷ lệ thắng/thua ước tính (win_loss_ratio), Tốc độ tích lũy thành tựu/vật phẩm (item_collection_rate), Điểm kỹ năng tổng hợp (skill_score) và Chỉ số chi tiêu (spending_metric); sau đó thực hiện chuẩn hóa dữ liệu bằng StandardScaler.


## 2. Giai đoạn 2 (Phân cụm người chơi): Kết hợp phân cụm phân cấp Agglomerative Hierarchical (Ward Linkage) trên tập mẫu đại diện để phân tích cấu trúc cây Dendrogram và K-Means trên toàn bộ dữ liệu.


## 3. Giai đoạn 3 (Đánh giá chất lượng phân cụm): Kiểm định độ gắn kết nội cụm và tách biệt giữa các cụm thông qua Silhouette Score và Davies-Bouldin Index, khảo sát so sánh đa mô hình từ K = 2 đến K = 7 nhằm tìm ra số cụm tối ưu toán học.


## 4. Giai đoạn 4 (Phân tích chân dung người chơi - Player Profiling): Giải mã các cực hành vi độc lập, lý giải hiện tượng bất thường trong dữ liệu (hiện tượng cựu chiến binh chán game và tỷ lệ nạp tiền nhị phân).


## 5. Giai đoạn 5 (Chiến lược kiếm tiền và giữ chân mục tiêu): Đề xuất các giải pháp kinh doanh nhắm đúng tâm lý và thói quen của từng phân khúc.


# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ PHƯƠNG PHÁP NGHIÊN CỨU


## 2.1. Chuẩn hóa thang đo dữ liệu (StandardScaler)

Trong các thuật toán phân cụm dựa trên khoảng cách hình học Euclidean, các thuộc tính có miền giá trị lớn (như skill_score từ 10 đến 198) sẽ áp đảo hoàn toàn các thuộc tính có biên độ nhỏ (như spending_metric chỉ nhận 0 hoặc 1). Vì vậy, kỹ thuật Z-Score Normalization được áp dụng bắt buộc:

Z = (x - μ) / σ

Trong đó μ là giá trị trung bình mẫu và σ là độ lệch chuẩn. Sau chuẩn hóa, mọi đặc trưng đều có trung bình bằng 0 và độ sai lệch chuẩn bằng 1, giúp các chiều dữ liệu có tiếng nói công bằng tuyệt đối trong việc đo khoảng cách.


## 2.2. Thuật toán phân cụm và Mô hình kết hợp (Hybrid Pipeline)

• Thuật toán K-Means: Thuật toán phân hoạch chia N điểm dữ liệu thành K cụm bằng cách tối thiểu hóa tổng bình phương khoảng cách sai số nội cụm (Within-Cluster Sum of Squares - WCSS hay Inertia). Thuật toán có tốc độ hội tụ nhanh O(N * K * I * d) nhưng nhạy cảm với việc chọn tâm khởi tạo ban đầu.

• Thuật toán Agglomerative Hierarchical (Ward Linkage): Phân cụm phân cấp từ dưới lên, hợp nhất từng cặp cụm làm tăng phương sai nội cụm ít nhất. Ưu điểm là tạo ra cây phân cấp trực quan và không phụ thuộc vào tâm khởi tạo, nhưng độ phức tạp O(N^2) khiến nó không thể chạy trực tiếp trên toàn bộ 40.034 mẫu.

• Mô hình kết hợp (Hybrid Model): Lấy mẫu phân tầng N = 2.000 để chạy Agglomerative Hierarchical nhằm xác định cấu trúc liên kết và số cụm tự nhiên trên biểu đồ Dendrogram; sau đó dùng cấu trúc này để định hướng và khởi tạo cho K-Means chạy toàn cục trên toàn bộ 40.034 quan sát.


## 2.3. Các chỉ số kiểm định chất lượng phân cụm


## 1. Silhouette Score: Đo lường mức độ khớp của mỗi điểm dữ liệu với cụm của nó so với cụm gần nhất lân cận:

s(i) = [b(i) - a(i)] / max(a(i), b(i))

Giá trị dao động từ -1 đến +1. Giá trị càng tiến gần +1 chứng minh cụm càng cô đặc và tách biệt ngoại cụm rõ ràng.


## 2. Davies-Bouldin Index (DBI): Đo tỷ số giữa tổng độ phân tán nội cụm và khoảng cách phân tách giữa các tâm cụm:

DB = (1 / K) * Σ max_{j ≠ i} [(s_i + s_j) / d(c_i, c_j)]

Do tử số là kích thước cụm và mẫu số là khoảng cách giữa các tâm, giá trị DBI CÀNG NHỎ thì chất lượng phân cụm CÀNG TỐT (cụm đặc và cách xa nhau).


# CHƯƠNG 3: KHÁM PHÁ DỮ LIỆU, TRÍCH XUẤT ĐẶC TRƯNG VÀ TIỀN XỬ LÝ (GIAI ĐOẠN 1)


## 3.1. Mô tả tập dữ liệu gốc

Bộ dữ liệu Online Gaming Behavior Dataset gồm 40.034 bản ghi người chơi với 13 trường thông tin mô tả chi tiết nhân khẩu học và hành vi trong trò chơi:


## 3.2. Công thức và ý nghĩa 5 chỉ số Play-style trích xuất

Nhóm tiến hành kỹ nghệ đặc trưng tạo ra 5 chỉ số phong cách chơi cốt lõi:


## 1. Thời gian chơi thực tế hàng tuần (playtime - giờ):

playtime = (SessionsPerWeek * AvgSessionDurationMinutes) / 60.0

Phản ánh chính xác quỹ thời gian thực tế người chơi dành cho game trong một tuần thay vì biến PlayTimeHours thô.


## 2. Điểm kỹ năng tổng hợp (skill_score):

skill_score = PlayerLevel * DifficultyWeight

Trong đó hệ số gán trọng số độ khó: Easy = 1.0, Medium = 1.5, Hard = 2.0. Phản ánh đúng năng lực vượt qua thử thách: đạt Level 50 ở độ khó Hard thể hiện trình độ vượt trội hơn nhiều so với Level 50 ở độ khó Easy.


## 3. Tốc độ thu thập vật phẩm / mở thành tựu (item_collection_rate - thành tựu/giờ):

item_collection_rate = AchievementsUnlocked / (playtime + 1.0)

Hệ số làm mịn +1.0 nhằm ngăn chặn lỗi chia cho 0 đối với người chơi có playtime = 0, đồng thời đóng vai trò chuẩn hóa tốc độ cày cuốc tài nguyên trên mỗi đơn vị thời gian.


## 4. Tỷ lệ chiến thắng ước tính (win_loss_ratio Proxy):

win_loss_ratio = (1.5 * AchievementsUnlocked + PlayerLevel) / (playtime + 5.0)

Biến đại diện (Proxy Variable) phản ánh hiệu suất thi đấu: người chơi sở hữu cấp độ cao và mở khóa nhiều thành tựu trong quỹ thời gian ngắn sở hữu tỷ lệ thắng áp đảo. Hệ số làm mịn +5.0 giúp ổn định mẫu số cho người mới chơi.


## 5. Chỉ số chi tiêu nạp tiền (spending_metric):

Lấy trực tiếp từ thuộc tính InGamePurchases, nhận giá trị nhị phân {0, 1} phản ánh hành vi chuyển đổi thành khách hàng trả phí.


## 3.3. Thống kê mô tả và chuẩn hóa dữ liệu

Bảng dưới đây tổng hợp các thông số thống kê của 5 đặc trưng trước khi chuẩn hóa bằng StandardScaler:


# CHƯƠNG 4: XÂY DỰNG MÔ HÌNH PHÂN CỤM KẾT HỢP VÀ ĐÁNH GIÁ CHẤT LƯỢNG (GIAI ĐOẠN 2 VÀ 3)


## 4.1. Thực nghiệm phân cụm phân cấp Agglomerative và biểu đồ Dendrogram

Để phát hiện cấu trúc phân nhóm tự nhiên, một tập mẫu phân tầng gồm N = 2.000 người chơi được trích xuất. Thuật toán Agglomerative với liên kết Ward (Ward's Minimum Variance Linkage) được áp dụng trên ma trận khoảng cách Euclidean:


*Hình 4.1: Biểu đồ cây phân cấp Dendrogram (Ward Linkage, N = 2.000 mẫu đại diện)*

Quan sát biểu đồ Dendrogram (Hình 4.1), các nhánh hợp nhất cho thấy sự tách biệt rõ ràng ở tầng cao. Khi cắt cây phân cấp tại khoảng cách nhảy vọt lớn, dữ liệu tách ra thành các khối phân nhóm chính tương thích với các hình thái chơi game đặc thù.


## 4.2. Khởi tạo K-Means và Đánh giá định lượng đa mô hình từ K = 2 đến K = 7

Sử dụng các tâm cụm định hướng từ mô hình phân cấp, nhóm tiến hành huấn luyện thuật toán K-Means trên toàn bộ 40.034 bản ghi với số cụm K quét liên tục từ 2 đến 7. Chất lượng mô hình được kiểm chứng nghiêm ngặt qua 3 thước đo: Inertia (WCSS), Silhouette Score và Davies-Bouldin Index (DBI):


*Hình 4.2: Đồ thị đánh giá K tối ưu qua 3 tiêu chí: Elbow Method, Silhouette Score và Davies-Bouldin Index*


## 4.3. Biện luận khoa học: Vì sao mô hình K = 4 vượt trội hoàn toàn so với K = 3?

Kết quả thực nghiệm trên Bảng 4.1 và Hình 4.2 cung cấp 3 chứng cứ toán học đanh thép để nhóm quyết định nâng cấp mô hình từ K = 3 lên K = 4:


## 1. Đỉnh cục bộ của Silhouette Score (0.2938): Tại K = 3, chỉ số Silhouette rơi xuống đáy thấp nhất là 0.2500 nhưng ngay sau đó bật tăng vọt lên đỉnh cao nhất 0.2938 tại K = 4. Điều này chứng minh các quan sát khi chia thành 4 cụm có mức độ tương đồng nội cụm cao hơn hẳn và ranh giới phân tách ngoại cụm rõ ràng nhất.


## 2. Sự sụt giảm mạnh mẽ của Davies-Bouldin Index (1.1457): Khi chuyển từ K = 3 sang K = 4, DBI giảm sâu từ 1.3639 xuống 1.1457 (giảm hơn 0.21 đơn vị). Tử số (bán kính cụm) co nhỏ lại trong khi mẫu số (khoảng cách tâm cụm) giãn ra, chứng minh các cụm ở K = 4 vô cùng đặc quánh.


## 3. Điểm gập Elbow Method: Đường cong Inertia gập góc khuỷu tay rõ rệt tại K = 4, từ sau K = 4 độ dốc giảm chậm dần, việc tăng thêm số cụm không còn đem lại lợi ích phân tách đáng kể.


# CHƯƠNG 5: PHÂN TÍCH CHÂN DUNG 4 NHÓM NGƯỜI CHƠI VÀ ĐỀ XUẤT CHIẾN LƯỢC KIẾM TIỀN (GIAI ĐOẠN 4 VÀ 5)


## 5.1. Hồ sơ chân dung 4 phân khúc người chơi (Player Personas)

Áp dụng mô hình tối ưu K = 4 lên toàn bộ 40.034 người chơi, cộng đồng game thủ được khắc họa thành 4 phân khúc độc lập với đặc tính hành vi rõ nét:


*Hình 5.1: Biểu đồ Radar thể hiện 4 cực hành vi độc lập của 4 Persona*


*Hình 5.2: Không gian PCA 2D thể hiện ranh giới phân tách sạch sẽ giữa 4 nhóm người chơi*


## 5.2. Giải mã hai hiện tượng chuyên sâu trong dữ liệu


## 1. Hiện tượng Cựu chiến binh chán game (End-game Veteran Burnout) ở nhóm Churn Risk:

Quan sát Bảng 5.1, nhóm Churn Risk (Cluster 3) có playtime tụt dốc nghiêm trọng chỉ còn 0.84 giờ/tuần và chỉ vào game 1.47 lần/tuần, nhưng Điểm kỹ năng lại CAO NHẤT (79.03 điểm) và Tỷ lệ thắng ÁP ĐẢO (18.85).

• Giải mã bản chất: Đây KHÔNG PHẢI người mới chơi kém kỹ năng bỏ cuộc, mà là các Game thủ kỳ cựu đã đạt cấp độ tối đa (Level trung bình 58.4, nhiều người Level 99) và mở khóa hầu hết thành tựu. Khi họ vào game làm một trận đấu, trang bị và kinh nghiệm vượt trội giúp họ dễ dàng chiến thắng (Win/Loss = 18.85), nhưng việc trò chơi thiếu nội dung mới ở giai đoạn cuối (End-game content) khiến họ chán nản và sắp sửa rời bỏ game.


## 2. Bản chất con số 21.0% nạp tiền ở nhóm Churn Risk và Cảnh báo Chảy máu doanh thu:

Biến InGamePurchases trong bộ dữ liệu là biến nhị phân {0, 1}. Con số trung bình 0.21 phản ánh chính xác rằng trong số 3.474 người chơi sắp bỏ game có tới 729 người từng nạp tiền (Lapsed Paying Users). Chi phí tìm kiếm một người trả phí mới cao gấp 5 đến 7 lần chi phí giữ chân khách hàng cũ; nếu để 729 người này rời bỏ vĩnh viễn, nhà phát hành sẽ gánh chịu tổn thất doanh thu nặng nề.


## 5.3. Ma trận chiến lược kiếm tiền và Giữ chân người chơi (Targeted Monetization)

Từ việc thấu hiểu hành vi và động lực của 4 nhóm, nhóm đề xuất ma trận chiến lược vận hành nhắm mục tiêu:


# KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN


## 1. Kết luận chung: Đề tài đã hoàn thành xuất sắc toàn bộ chuỗi nhiệm vụ đề ra: trích xuất thành công 5 chỉ số Play-style metrics phản ánh đúng bản chất thể thao điện tử, kết hợp tối ưu giữa Agglomerative Hierarchical (Ward Linkage) và K-Means. Kết quả thực nghiệm kiểm định đa mô hình đã chứng minh tính ưu việt tuyệt đối của K = 4 (Silhouette đạt đỉnh 0.2938, DBI giảm sâu 1.1457) so với K = 3, tách biệt thành công nhóm Churn Risk để bảo vệ doanh thu.


## 2. Hướng phát triển tiếp theo: Tích hợp mô hình học máy có giám sát (XGBoost, Random Forest) để dự báo xác suất rời bỏ game theo thời gian thực (Real-time Churn Prediction) cho nhóm Casual, đồng thời bổ sung dữ liệu lịch sử nạp thẻ để xây dựng mô hình RFM (Recency - Frequency - Monetary) phân tầng sâu hơn nhóm Competitive Whales.


# TÀI LIỆU THAM KHẢO


## 1. Han, J., Kamber, M., & Pei, J. (2011). Data Mining: Concepts and Techniques (3rd ed.). Morgan Kaufmann.


## 2. Ward, J. H. (1963). Hierarchical Grouping to Optimize an Objective Function. Journal of the American Statistical Association.


## 3. Rousseeuw, P. J. (1987). Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis. Journal of Computational and Applied Mathematics.


## 4. Davies, D. L., & Bouldin, D. W. (1979). A Cluster Separation Measure. IEEE Transactions on Pattern Analysis and Machine Intelligence.


## 5. Tài liệu thư viện Scikit-learn (Clustering & Preprocessing): https://scikit-learn.org/stable/modules/clustering.html
