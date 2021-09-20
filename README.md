#Hướng dẫn cách build `iMorph`
## Cài đặt môi trường python để build code 
####    `pip install -r requirements.txt`
## Tạo thư mục `build`
### Trong môi trường window
####    copy file `build_window.spec` vào thư mục `build` chạy lệnh `pyinstaller build_window.spec --onefile`
### Trong môi trường linux
####    copy file `build_linux.spec` vào thư mục `build` chạy lệnh `pyinstaller build_linux.spec --onefile`
### Sau khi build vào thư mục dist sẽ được file `iMorph.exe` trong win hoặc `iMorph` trong linux . Copy thư file conf.json vào OK