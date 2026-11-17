@echo off
cd /d "%~dp0"
set PY=%USERPROFILE%\Anaconda3\envs\torch_env\python.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set LOG=log_chay_thu.txt
echo === Chay thu tren du lieu gia (GPU) === > %LOG%
"%PY%" tools\make_fake_tusimple.py --out _test\fake --video >> %LOG% 2>&1
"%PY%" train.py --data-root _test\fake --cache-dir _test\cache --out-dir _test\runs --epochs 3 --batch 8 --workers 0 --no-pretrained --name smoke >> %LOG% 2>&1
"%PY%" evaluate.py --data-root _test\fake --cache-dir _test\cache --method autopath --ckpt _test\runs\smoke\best.pt --out _test\res >> %LOG% 2>&1
"%PY%" benchmark_fps.py --backbones resnet18 mobilenet_v3_small --classical --out _test\benchmark.json >> %LOG% 2>&1
"%PY%" demo_video.py --video _test\fake\fake_drive.mp4 --ckpt _test\runs\smoke\best.pt --alphas 0.1 0.35 1.0 --out _test\demo.mp4 >> %LOG% 2>&1
echo DONE >> %LOG%
