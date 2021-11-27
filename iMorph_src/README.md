# How to build `iMorph`
## Setup the python enviroment
####    `pip install -r requirements.txt`

## Setup qttools5-dev-tools
### Windows
pip install PyQt5
pip install PyQt5-tools
### Ubuntu (linux)
sudo apt-get install pyqt5-dev-tools
sudo apt-get install qttools5-dev-tools

## Build the executable file
### pyinstaller --onefile  --windowed iMorph.py
