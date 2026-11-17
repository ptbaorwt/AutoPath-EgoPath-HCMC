# Hướng dẫn quay video và gán nhãn tập kiểm thử TP.HCM

Mục tiêu: **3 video công khai trên YouTube quay tại TP.HCM, 250–300 khung hình có nhãn** (khoảng 85–90 khung mỗi video),
mỗi khung gắn thẻ điều kiện. Tập này chỉ dùng để **kiểm thử** (không huấn luyện), nên chất lượng nhãn quan trọng hơn số lượng.

## 1. Video nguồn

| Tệp | Điều kiện | Nguồn (YouTube) |
|---|---|---|
| vid01.mp4 | Ban ngày, giao thông trung tâm | Saigon City Center 4K – World Driving Tours, https://www.youtube.com/watch?v=gZWiLrQnb10 |
| vid03.mp4 | Ban đêm, Nguyễn Huệ và trung tâm | Ho Chi Minh City Night Drive – Road and Sky, https://www.youtube.com/watch?v=uzlCWwkYihA |
| vid04.mp4 | Mưa giờ cao điểm, trong ô tô | 4K Rainy Drive in Saigon – Vietnam Naturally, https://www.youtube.com/watch?v=4C5CEwULTSc |

- Tải bản **1080p** (không cần 4K, khung hình sẽ được thu về 1280×720), đổi tên như cột "Tệp", chép vào `AutoPath\hcmc\raw\`.
- Ghi lại cho bảng thống kê: tên kênh, tiêu đề chính xác, thời lượng, ngày đăng.
- Video thuộc bản quyền của tác giả kênh: chỉ dùng nội bộ để đánh giá, **không đăng lại** video gốc hay video demo công khai;
  link demo gửi cô để ở chế độ riêng tư/không công khai. Trích dẫn nguồn trong báo cáo (đã có trong Tài liệu tham khảo).

## 2. Khung hình (ĐÃ LÀM XONG 17/11)

- Đã trích 324 khung ứng viên, lọc bỏ 62 khung (xe dừng đèn đỏ trùng nhau, xe đang rẽ hoặc vào giao lộ,
  cảnh quay ra vỉa hè, cần gạt mưa che, quá tối không thấy làn) -> chuyển sang `hcmc\frames_loai\`.
- Còn **262 khung** trong `hcmc\frames\`: vid01 (ngày) 88, vid03 (đêm) 87, vid04 (mưa) 87.
- Đã chia đều cho 4 người, mỗi người khoảng 65 khung trộn cả 3 video: `hcmc\phan_cong\<tên>.txt`.

## 3. Gán nhãn

Mỗi người bấm đúp file của mình: `4_gan_nhan_Bao.bat`, `4_gan_nhan_Danh.bat`, `4_gan_nhan_Nguyen.bat`, `4_gan_nhan_Phuc.bat`.
Nhãn lưu vào `hcmc\labels_<tên>.json`. Có thể thoát (`q`) bất kỳ lúc nào; chạy lại sẽ tiếp tục từ khung chưa gán.
Thẻ `night` (vid03) và `rain` (vid04) đã bật sẵn; chỉ cần thêm thẻ khác nếu có (bấm 2, 3, 4).
Nếu mỗi người làm trên máy riêng: chép cả thư mục AutoPath, gán xong gửi lại tệp `hcmc\labels_<tên>.json`.

| Phím | Tác dụng |
|---|---|
| Click trái | Thêm điểm cho vạch đang chọn, **click từ gần xe ra xa**, 4–8 điểm mỗi vạch |
| `n` | Chuyển vạch TRÁI (xanh dương) / PHẢI (đỏ) |
| `u` | Xóa điểm vừa click |
| `1`–`5` | Bật/tắt thẻ: 1 = night, 2 = no_marking, 3 = intersection, 4 = occlusion, 5 = rain |
| `s` hoặc Enter | Lưu và sang khung tiếp |
| `k` | Bỏ qua khung (không xác định được làn) |
| `q` | Thoát |

### Quy tắc thống nhất (cả nhóm phải làm giống nhau)

1. **Vạch trái và phải là hai ranh giới của làn xe mình đang chạy**, không phải hai mép đường.
   Đường đi = trung điểm hai vạch, do code tự tính.
2. **Đường có vạch:** click đúng tâm vạch sơn. Vạch đứt thì click qua cả khoảng trống, coi như vạch liền.
3. **Đường không có vạch (`2`):** tưởng tượng một làn rộng khoảng 3–3,5 m mà người lái đang bám theo
   (thường cách mép phải hoặc dải phân cách khoảng một làn). Click hai ranh giới của làn tưởng tượng đó.
   Không lấy hai mép của cả mặt đường rộng.
4. **Bị xe che (`4`):** chỉ click phần vạch nhìn thấy hoặc suy ra chắc chắn. Nếu xe phía trước che đoạn xa,
   dừng vạch ở chỗ bị che; **không vẽ xuyên qua xe**. Nếu che gần hết, nhấn `k`.
5. **Giao lộ (`3`):** nếu xe đi thẳng, click theo hướng đi thẳng qua giao lộ đến chỗ còn nhìn thấy làn phía bên kia.
   Nếu xe đang rẽ hoặc không rõ hướng đi, nhấn `k`.
6. **Chuyển làn:** xe đang đè vạch thì nhấn `k` (đề tài không xử lý chuyển làn).
7. **Điểm xa nhất** dừng ở chỗ còn phân biệt được vạch, không kéo đến tận đường chân trời nếu không thấy rõ.
8. Một khung có thể có nhiều thẻ (ví dụ night + rain). Đường bình thường ban ngày thì không cần thẻ.

### Kiểm tra chéo

Mỗi người xem lại ngẫu nhiên 20 khung của người khác bằng ô "Xem thử vài nhãn" trong notebook
(đổi đường dẫn sang `hcmc/labels.json`). Khung nào hai người không đồng ý về vị trí làn thì bỏ.

## 4. Đánh giá

Chạy `5_danh_gia_TPHCM.bat`: đánh giá AutoPath-ResNet18 (có/không tăng cường), MobileNetV3, AutoSteer 2.0,
Canny + Hough trên `hcmc/labels.json`; quét α trên từng video; dựng video demo có cảnh báo vật cản; vẽ hình và bảng.
Kết quả nằm ở `results/*_hcmc/`, `results/alpha_*.csv`, `figures_hcmc/tables.md`.
