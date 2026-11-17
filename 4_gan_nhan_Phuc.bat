@echo off
REM Gan nhan phan cua: Ta Nguyen Phuc (danh sach khung: hcmc\phan_cong\phuc.txt)
REM Phim: click trai = them diem (tu gan ra xa) | n = doi vach TRAI/PHAI | u = xoa diem | 1-5 = the
REM       s/Enter = luu, sang khung tiep | k = bo qua khung | q = thoat (chay lai se lam tiep)
REM Quy tac: xem HUONG_DAN_GAN_NHAN.md
cd /d "%~dp0"
set PY=%USERPROFILE%\Anaconda3\envs\torch_env\python.exe
if not exist "%PY%" set PY=python
set PYTHONUTF8=1
echo Kiem tra OpenCV co giao dien...
"%PY%" -c "import cv2; cv2.namedWindow('t'); cv2.destroyAllWindows()" 2>nul || (
  echo Dang cai OpenCV ban co giao dien, doi 1-2 phut...
  "%PY%" -m pip uninstall -y -q opencv-python-headless opencv-contrib-python-headless opencv-python
  "%PY%" -m pip install -q opencv-python numpy
)
"%PY%" tools\label_egopath.py --frames hcmc\frames --list hcmc\phan_cong\phuc.txt --out hcmc\labels_phuc.json
pause
