# AutoPath: Dự đoán đường đi (ego-path) cho xe tự lái bằng học sâu end-to-end

Đồ án môn **Thị giác máy tính và ứng dụng** (IUH, Khoa CNTT). Nhóm 1, lớp DHKHMT19BTT. GVHD: TS. Lê Thị Vĩnh Thanh.

Từ một khung hình camera trước xe, mô hình dự đoán trực tiếp đường đi của xe dưới dạng **64 hàng neo**.
Mỗi hàng gồm một hoành độ và một cờ hợp lệ, không cần bước phát hiện vạch làn riêng.
Có ba phương pháp được so sánh trên cùng nhãn và cùng metric:

| Phương pháp | Mô tả |
|---|---|
| AutoPath-ResNet18 / AutoPath-MobileNetV3 | Mô hình của nhóm: backbone pretrained ImageNet, đầu hồi quy 64 hàng neo |
| AutoSteer 2.0 | Mô hình mã nguồn mở của Autoware Foundation (baseline học sâu) |
| Canny + Hough | Baseline cổ điển, không học |

## Kết quả chính

**Tập test TuSimple (2.770 ảnh, GPU Tesla T4)**

| Phương pháp | mAP@50 | Sai số ngang TB (px) | FPS (T4) |
|---|---|---|---|
| AutoPath-ResNet18 | 0,958 | 6,93 | 190,9 |
| AutoPath-MobileNetV3-Small | 0,946 | 8,47 | 185,9 |
| AutoSteer 2.0 (*) | 0,969 | 4,62 | 178,0 |
| Canny + Hough | 0,002 | 16,52 | 518,5 (CPU) |

(*) Mã nguồn AutoSteer 2.0 đưa khoảng 75% ảnh test TuSimple vào tập huấn luyện, nên số liệu này chỉ để tham khảo.

**Tập kiểm thử TP.HCM tự gán nhãn (187 khung: ngày, đêm, mưa; giao thông hỗn hợp)**

| Phương pháp | mAP@50 (hàng có nhãn) | Sai số ngang TB (px) |
|---|---|---|
| Canny + Hough | 0,462 | 60,5 |
| AutoSteer 2.0 | 0,739 | 35,1 |
| AutoPath-ResNet18 | 0,160 | 53,9 |
| AutoPath-ResNet18 + căn góc camera (`autopath/align.py`) | 0,619 | 45,9 |

**Tốc độ trên laptop RTX 3050:** mô hình 252,7 FPS; toàn hệ thống (960×540) 52,6 FPS, 27,3 FPS khi bật cảnh báo vật cản.

## Cấu trúc

```
autopath/geometry.py    biểu diễn 64 hàng neo, chuyển nhãn TuSimple -> ego-path
autopath/data.py        đọc TuSimple, chia train/val theo clip, Dataset + augmentation
autopath/model.py       AutoPathNet (resnet18 / mobilenet_v3_small / mobilenet_v3_large)
autopath/metrics.py     loss, mAP@50 đúng cách tính của AutoSteer, sai số ngang (px)
autopath/predictors.py  giao diện dự đoán chung cho AutoPath / AutoSteer / Canny+Hough
autopath/smoothing.py   làm mượt EMA, đo jitter/lag, vẽ overlay
autopath/hazard.py      module phụ: cảnh báo người/xe nằm trên đường đi (YOLOv8n, demo --warn)
train.py                huấn luyện
evaluate.py             đánh giá + ảnh lỗi + thống kê theo độ cong
benchmark_fps.py        FPS, số tham số, dung lượng
demo_video.py           video demo + quét alpha (Thí nghiệm 4) + FPS toàn hệ thống (--stats-json)
tools/                  trích frame, gán nhãn tập TP.HCM, vẽ hình/bảng cho báo cáo, dữ liệu giả, kiểm thử chức năng
tools/run_hcmc_pipeline.py  chạy toàn bộ thực nghiệm TP.HCM có điểm lưu (notebooks/AutoPath_Kaggle_HCMC.ipynb)
HUONG_DAN_GAN_NHAN.md   quy tắc quay video và gán nhãn tập TP.HCM
notebooks/AutoPath_Kaggle.ipynb   chạy toàn bộ trên Kaggle GPU
```

## Cài đặt

```bash
pip install -r requirements.txt
```

Dữ liệu: TuSimple trên Kaggle (`manideep1108/tusimple`). Code tự tìm `label_data_*.json` và `test_label.json`
trong thư mục bạn chỉ định, nên không phụ thuộc chính xác cấu trúc thư mục.

## Chạy các thí nghiệm

```bash
TUS=/kaggle/input/tusimple/TUSimple          # sửa theo máy của bạn

# Thí nghiệm 1 (RQ1): so sánh backbone
python train.py --data-root $TUS --backbone resnet18           --name res18_aug
python train.py --data-root $TUS --backbone mobilenet_v3_small --name mbv3s_aug
python train.py --data-root $TUS --backbone mobilenet_v3_large --name mbv3l_aug   # tùy chọn

# Thí nghiệm 2 (RQ2): ảnh hưởng của augmentation
python train.py --data-root $TUS --backbone resnet18 --no-aug  --name res18_noaug

# (Tùy chọn) đầu GAP như đề cương ban đầu, để chứng minh lý do chọn đầu "spatial"
python train.py --data-root $TUS --backbone resnet18 --head gap --name res18_gap

# Đánh giá trên tập test TuSimple
for r in res18_aug mbv3s_aug res18_noaug; do
  python evaluate.py --data-root $TUS --method autopath --ckpt runs/$r/best.pt --out results/$r
done
python evaluate.py --data-root $TUS --method classical --out results/classical

# Thí nghiệm 3 (RQ3): AutoSteer 2.0
git clone https://github.com/autowarefoundation/auto_steer.git
gdown https://drive.google.com/uc?id=1-wfPyu7HId7YDSh_T_DkB3Ma1zPc7Fu_ -O weights/autosteer.pth
python evaluate.py --data-root $TUS --method autosteer --autosteer-repo auto_steer \
    --autosteer-weights weights/autosteer.pth --out results/autosteer
```

> **Lưu ý về so sánh với AutoSteer 2.0.** Script `data_parsing/TuSimple/converter.py` của AutoSteer
> đưa `test_label.json` vào tập val, rồi hàm `expand_training_set(fract=0.25)` chuyển khoảng 75% số ảnh
> này sang tập train. Như vậy AutoSteer 2.0 **đã thấy phần lớn tập test TuSimple khi huấn luyện**,
> nên kết quả của nó trên tập test này bị thiên lệch có lợi. Phép so sánh công bằng là trên **tập TP.HCM
> tự gán nhãn**, tập mà cả hai mô hình đều chưa từng thấy.

```bash
# Tập TP.HCM: quay video -> trích frame -> gán nhãn -> đánh giá cả 3 phương pháp
python tools/extract_frames.py --video raw/*.mp4 --out hcmc/frames --every-sec 1
python tools/label_egopath.py --frames hcmc/frames --out hcmc/labels.json
python evaluate.py --custom-json hcmc/labels.json --method autopath --ckpt runs/res18_aug/best.pt --out results/res18_hcmc
python evaluate.py --custom-json hcmc/labels.json --method autosteer --autosteer-repo auto_steer --autosteer-weights weights/autosteer.pth --out results/autosteer_hcmc
python evaluate.py --custom-json hcmc/labels.json --method classical --out results/classical_hcmc

# Tốc độ (chạy trên RTX 3050 của nhóm để ghi vào báo cáo)
python benchmark_fps.py --ckpts runs/res18_aug/best.pt runs/mbv3s_aug/best.pt \
    --autosteer-repo auto_steer --autosteer-weights weights/autosteer.pth --classical

# Thí nghiệm 4 (RQ4): chọn alpha cho EMA + video demo
python demo_video.py --video raw/vid05.mp4 --ckpt runs/res18_aug/best.pt --alphas 0.1 0.2 0.35 0.5 0.7 1.0 --no-write --out results/demo.mp4
python demo_video.py --video raw/vid05.mp4 --ckpt runs/res18_aug/best.pt --alpha 0.35 --out results/demo.mp4

# Hình và bảng cho Chương 4
python tools/make_figures.py --runs "runs/*" --results "results/*/results.json" \
    --benchmark results/benchmark.json --alpha-csv results/demo_alpha_sweep.csv --out figures
```

## Định nghĩa metric (ghi trong báo cáo, mục 2.6)

- **mAP@50**: tính đúng như AutoSteer 2.0. Với mỗi ảnh, lấy khoảng cách d là trung bình trên 64 hàng của
  |x_dự đoán − x_nhãn|, đơn vị là tỉ lệ bề rộng ảnh. Ảnh được tính là đúng nếu d < 0.05.
  AP được tính bằng nội suy 101 điểm theo độ tin cậy. "mAP" là trung bình AP tại các ngưỡng 0.02, 0.05 và 0.10.
- **Sai số ngang**: |Δx| trung bình (px, ảnh gốc 1280 px) trên các hàng mà cả dự đoán và nhãn đều hợp lệ.
- **IoU hàng**: độ trùng khớp giữa đoạn đường đi dự đoán và đoạn đường đi trong nhãn.

## Kiểm thử nhanh không cần dữ liệu thật

```bash
python tools/make_fake_tusimple.py --out /tmp/fake --video
python train.py --data-root /tmp/fake --epochs 2 --batch 8 --no-pretrained --workers 0
```

## Dữ liệu và trọng số không có trong repo

- **TuSimple**: tải từ Kaggle (`manideep1108/tusimple`), trỏ `--data-root` tới thư mục giải nén.
- **Video TP.HCM**: ba video công khai trên YouTube (World Driving Tours, Road and Sky, Vietnam Naturally), thuộc bản quyền chủ kênh nên không phân phối lại. Nhãn của nhóm có trong `hcmc/labels.json` (187 khung, định dạng TuSimple), thời điểm từng khung trong `hcmc/frames/frames_index.csv`.
- **Trọng số AutoPath** (`runs/*/best.pt`) và **AutoSteer 2.0** (`weights/autosteer.pth`): không lưu trong git do dung lượng; liên hệ nhóm hoặc huấn luyện lại bằng `train.py`.

## Thành viên

| Thành viên | MSSV | Vai trò |
|---|---|---|
| Phạm Thái Bảo (nhóm trưởng) | 23731181 | Điều phối, dữ liệu |
| Nguyễn Thành Danh | 23673331 | Mô hình, demo |
| Nguyễn Hà Phúc Nguyên | 23670141 | Huấn luyện |
| Tạ Nguyên Phúc | 23663831 | Đánh giá, baseline |
