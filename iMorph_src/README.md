## How to build iMorph
### Makesure python 3 is available the setup the python libraries with below command
`pip install -r requirements.txt`

### Setup qttools5-dev-tools
#### in Windows
`pip install -r requirements.txt`
`pip install PyQt5`
`pip install PyQt5-tools`
#### in Ubuntu (linux)
`sudo apt-get install pyqt5-dev-tools`
`sudo apt-get install qttools5-dev-tools`

### Build the executable file
`pyinstaller --onefile  --windowed iMorph.py`
