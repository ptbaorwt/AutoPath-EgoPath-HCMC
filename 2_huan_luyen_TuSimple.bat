@echo off
REM ==== SUA DUONG DAN NAY toi thu muc TuSimple da giai nen (chua label_data_0313.json va test_label.json) ====
set TUS=C:\Users\Thai Bao\Downloads\TUSimple
REM ===========================================================================================================
cd /d "%~dp0"
set PY=%USERPROFILE%\Anaconda3\envs\torch_env\python.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
echo [1/6] TN1: ResNet18 + augmentation
"%PY%" train.py --data-root "%TUS%" --backbone resnet18 --name res18_aug --workers 4 > logs\train_res18_aug.txt 2>&1
echo [2/6] TN1: MobileNetV3-Small + augmentation
"%PY%" train.py --data-root "%TUS%" --backbone mobilenet_v3_small --name mbv3s_aug --workers 4 > logs\train_mbv3s_aug.txt 2>&1
echo [3/6] TN2: ResNet18 khong augmentation
"%PY%" train.py --data-root "%TUS%" --backbone resnet18 --no-aug --name res18_noaug --workers 4 > logs\train_res18_noaug.txt 2>&1
echo [4/6] Danh gia tren test TuSimple
for %%R in (res18_aug mbv3s_aug res18_noaug) do "%PY%" evaluate.py --data-root "%TUS%" --method autopath --ckpt runs\%%R\best.pt --out results\%%R >> logs\evaluate.txt 2>&1
"%PY%" evaluate.py --data-root "%TUS%" --method classical --out results\classical >> logs\evaluate.txt 2>&1
echo [5/6] Do toc do tren RTX 3050
"%PY%" benchmark_fps.py --ckpts runs\res18_aug\best.pt runs\mbv3s_aug\best.pt --classical --out results\benchmark.json > logs\benchmark.txt 2>&1
echo [6/6] Ve hinh va bang
"%PY%" tools\make_figures.py --runs "runs/*" --results "results/*/results.json" --benchmark results\benchmark.json --out figures > logs\figures.txt 2>&1
echo XONG. Xem figures\tables.md
pause
