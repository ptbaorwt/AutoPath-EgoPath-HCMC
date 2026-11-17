| Phương pháp | Tập | n | mAP@50 | mAP | Acc@0.05 | Sai số ngang TB (px) | Trung vị (px) | IoU hàng |
|---|---|---|---|---|---|---|---|---|
| AutoSteer-2.0 | test | 2770 | 0.9685 | 0.8659 | 0.9736 | 4.62 | 3.60 | 0.955 |
| Canny+Hough [classical] | test | 2770 | 0.0016 | 0.0673 | 0.0032 | 16.52 | 8.16 | 0.422 |
| AutoPath-mobilenet_v3_small [mbv3s_aug] | test | 2770 | 0.9458 | 0.8162 | 0.9430 | 8.47 | 6.09 | 0.948 |
| AutoPath-resnet18 [res18_aug] | test | 2770 | 0.9576 | 0.8380 | 0.9563 | 6.93 | 5.15 | 0.951 |
| AutoPath-resnet18 [res18_noaug] | test | 2770 | 0.9517 | 0.8308 | 0.9516 | 7.22 | 4.96 | 0.950 |

| Phương pháp | Nhóm độ cong | n | mAP@50 | Sai số ngang TB (px) |
|---|---|---|---|---|
| AutoSteer-2.0 (test) | thẳng (<20px) | 1714 | 0.9682 | 4.42 |
| AutoSteer-2.0 (test) | cong vừa (20-60px) | 544 | 0.9884 | 4.92 |
| AutoSteer-2.0 (test) | cong gắt (>60px) | 512 | 0.9591 | 4.99 |
| Canny+Hough [classical] (test) | thẳng (<20px) | 1714 | 0.0027 | 15.18 |
| Canny+Hough [classical] (test) | cong vừa (20-60px) | 544 | 0.0000 | 20.97 |
| Canny+Hough [classical] (test) | cong gắt (>60px) | 512 | 0.0000 | 16.33 |
| AutoPath-mobilenet_v3_small [mbv3s_aug] (test) | thẳng (<20px) | 1714 | 0.9504 | 8.18 |
| AutoPath-mobilenet_v3_small [mbv3s_aug] (test) | cong vừa (20-60px) | 544 | 0.9493 | 8.72 |
| AutoPath-mobilenet_v3_small [mbv3s_aug] (test) | cong gắt (>60px) | 512 | 0.9334 | 9.18 |
| AutoPath-resnet18 [res18_aug] (test) | thẳng (<20px) | 1714 | 0.9576 | 6.66 |
| AutoPath-resnet18 [res18_aug] (test) | cong vừa (20-60px) | 544 | 0.9718 | 7.21 |
| AutoPath-resnet18 [res18_aug] (test) | cong gắt (>60px) | 512 | 0.9511 | 7.52 |
| AutoPath-resnet18 [res18_noaug] (test) | thẳng (<20px) | 1714 | 0.9508 | 6.67 |
| AutoPath-resnet18 [res18_noaug] (test) | cong vừa (20-60px) | 544 | 0.9567 | 7.49 |
| AutoPath-resnet18 [res18_noaug] (test) | cong gắt (>60px) | 512 | 0.9644 | 8.75 |

Thiết bị đo: NVIDIA GeForce RTX 3050 Laptop GPU

| Phương pháp | Đầu vào | Tham số (M) | Dung lượng (MB) | FPS | Độ trễ TB (ms) |
|---|---|---|---|---|---|
| AutoPath-resnet18 | 512x256 | 11.48 | 43.8 | 252.7 | 3.96 |
| AutoPath-mobilenet_v3_small | 512x256 | 1.23 | 4.7 | 120.4 | 8.31 |
| AutoSteer-2.0 | 1024x512 | 8.22 | 31.4 | 98.2 | 10.18 |
| Canny+Hough (CPU) | 640x320 | 0.00 | 0.0 | 586.5 | 1.71 |
