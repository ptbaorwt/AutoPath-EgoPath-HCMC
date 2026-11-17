| Phương pháp | Tập | n | mAP@50 | mAP | Acc@0.05 | Sai số ngang TB (px) | Trung vị (px) | IoU hàng |
|---|---|---|---|---|---|---|---|---|
| AutoSteer-2.0 [autosteer_hcmc] | custom | 187 | 0.4119 | 0.4167 | 0.5348 | 35.11 | 18.05 | 0.741 |
| Canny+Hough [classical_hcmc] | custom | 187 | 0.0054 | 0.0385 | 0.0107 | 60.49 | 47.51 | 0.419 |
| AutoPath-mobilenet_v3_small [mbv3s_aug_hcmc] | custom | 187 | 0.0000 | 0.0000 | 0.0000 | 60.86 | 46.86 | 0.365 |
| AutoPath-resnet18 [res18_aug_hcmc] | custom | 187 | 0.0000 | 0.0000 | 0.0000 | 53.91 | 41.62 | 0.363 |
| AutoPath-resnet18 [res18_noaug_hcmc] | custom | 187 | 0.0000 | 0.0000 | 0.0000 | 51.46 | 40.39 | 0.341 |

| Phương pháp | Nhóm độ cong | n | mAP@50 | Sai số ngang TB (px) |
|---|---|---|---|---|

| Phương pháp | Điều kiện (thẻ) | Số khung | mAP@50 | Sai số ngang TB (px) | Tỉ lệ khung sai (d ≥ 0,05) |
|---|---|---|---|---|---|
| AutoSteer-2.0 [autosteer_hcmc] | intersection | 13 | 0.2559 | 29.29 | 0.692 |
| AutoSteer-2.0 [autosteer_hcmc] | night | 65 | 0.4762 | 49.14 | 0.477 |
| AutoSteer-2.0 [autosteer_hcmc] | no_marking | 49 | 0.3853 | 52.48 | 0.490 |
| AutoSteer-2.0 [autosteer_hcmc] | occlusion | 44 | 0.1736 | 32.24 | 0.727 |
| AutoSteer-2.0 [autosteer_hcmc] | rain | 54 | 0.4127 | 35.57 | 0.463 |
| Canny+Hough [classical_hcmc] | intersection | 13 | 0.0000 | 49.10 | 1.000 |
| Canny+Hough [classical_hcmc] | night | 65 | 0.0000 | 57.59 | 1.000 |
| Canny+Hough [classical_hcmc] | no_marking | 49 | 0.0000 | 68.32 | 1.000 |
| Canny+Hough [classical_hcmc] | occlusion | 44 | 0.0238 | 58.96 | 0.955 |
| Canny+Hough [classical_hcmc] | rain | 54 | 0.0192 | 71.25 | 0.963 |
| AutoPath-mobilenet_v3_small [mbv3s_aug_hcmc] | intersection | 13 | 0.0000 | 47.85 | 1.000 |
| AutoPath-mobilenet_v3_small [mbv3s_aug_hcmc] | night | 65 | 0.0000 | 61.94 | 1.000 |
| AutoPath-mobilenet_v3_small [mbv3s_aug_hcmc] | no_marking | 49 | 0.0000 | 55.30 | 1.000 |
| AutoPath-mobilenet_v3_small [mbv3s_aug_hcmc] | occlusion | 44 | 0.0000 | 76.02 | 1.000 |
| AutoPath-mobilenet_v3_small [mbv3s_aug_hcmc] | rain | 54 | 0.0000 | 79.90 | 1.000 |
| AutoPath-resnet18 [res18_aug_hcmc] | intersection | 13 | 0.0000 | 32.34 | 1.000 |
| AutoPath-resnet18 [res18_aug_hcmc] | night | 65 | 0.0000 | 48.48 | 1.000 |
| AutoPath-resnet18 [res18_aug_hcmc] | no_marking | 49 | 0.0000 | 55.59 | 1.000 |
| AutoPath-resnet18 [res18_aug_hcmc] | occlusion | 44 | 0.0000 | 67.00 | 1.000 |
| AutoPath-resnet18 [res18_aug_hcmc] | rain | 54 | 0.0000 | 78.25 | 1.000 |
| AutoPath-resnet18 [res18_noaug_hcmc] | intersection | 13 | 0.0000 | 34.97 | 1.000 |
| AutoPath-resnet18 [res18_noaug_hcmc] | night | 65 | 0.0000 | 46.80 | 1.000 |
| AutoPath-resnet18 [res18_noaug_hcmc] | no_marking | 49 | 0.0000 | 51.29 | 1.000 |
| AutoPath-resnet18 [res18_noaug_hcmc] | occlusion | 44 | 0.0000 | 55.92 | 1.000 |
| AutoPath-resnet18 [res18_noaug_hcmc] | rain | 54 | 0.0000 | 69.02 | 1.000 |
| AutoSteer-2.0 [autosteer_hcmc] (custom) | thẳng (<20px) | 175 | 0.4085 | 33.32 |
| AutoSteer-2.0 [autosteer_hcmc] (custom) | cong vừa (20-60px) | 11 | 0.5056 | 66.11 |
| AutoSteer-2.0 [autosteer_hcmc] (custom) | cong gắt (>60px) | 1 | 0.9950 | 8.41 |
| Canny+Hough [classical_hcmc] (custom) | thẳng (<20px) | 175 | 0.0058 | 60.66 |
| Canny+Hough [classical_hcmc] (custom) | cong vừa (20-60px) | 11 | 0.0000 | 57.04 |
| Canny+Hough [classical_hcmc] (custom) | cong gắt (>60px) | 1 | 0.0000 | nan |
| AutoPath-mobilenet_v3_small [mbv3s_aug_hcmc] (custom) | thẳng (<20px) | 175 | 0.0000 | 58.92 |
| AutoPath-mobilenet_v3_small [mbv3s_aug_hcmc] (custom) | cong vừa (20-60px) | 11 | 0.0000 | 78.42 |
| AutoPath-mobilenet_v3_small [mbv3s_aug_hcmc] (custom) | cong gắt (>60px) | 1 | 0.0000 | 208.20 |
| AutoPath-resnet18 [res18_aug_hcmc] (custom) | thẳng (<20px) | 175 | 0.0000 | 51.50 |
| AutoPath-resnet18 [res18_aug_hcmc] (custom) | cong vừa (20-60px) | 11 | 0.0000 | 80.88 |
| AutoPath-resnet18 [res18_aug_hcmc] (custom) | cong gắt (>60px) | 1 | 0.0000 | 178.93 |
| AutoPath-resnet18 [res18_noaug_hcmc] (custom) | thẳng (<20px) | 175 | 0.0000 | 49.90 |
| AutoPath-resnet18 [res18_noaug_hcmc] (custom) | cong vừa (20-60px) | 11 | 0.0000 | 61.64 |
| AutoPath-resnet18 [res18_noaug_hcmc] (custom) | cong gắt (>60px) | 1 | 0.0000 | 213.46 |

| alpha | Jitter (px) | Jitter thô (px) | Lag (px) |
|---|---|---|---|
| 0.1 | 0.975 | 6.599 | 8.711 |
| 0.2 | 1.671 | 6.599 | 6.633 |
| 0.35 | 2.577 | 6.599 | 4.749 |
| 0.5 | 3.427 | 6.599 | 3.400 |
| 0.7 | 4.582 | 6.599 | 1.948 |
| 1.0 | 6.599 | 6.599 | 0.000 |
