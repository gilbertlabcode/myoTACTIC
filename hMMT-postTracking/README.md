Python code for tracking myoTACTIC PDMS posts during hMMT contraction is found here. Code requirements: Python 3, opencv-contrib library, the matplotlib library.

**Python set-up instructions for Windows:**

1) Install a Python 3 distribution of your choice. I use WinPython (http://winpython.github.io/), as it's self-contained in the directory you
install it in. It can also be installed on a USB and run portably (i.e. only with the USB plugged in). 

2) Navigate to the location where WinPython was installed, and open the Command Prompt application or Windows Powershell application in 
the WinPython directory. Type `pip install matplotlib`, press Enter, and wait for the matplotlib library to finish installing.

3) Type `pip install opencv-contrib-python`, press Enter, and wait for the opencv library to finish installing.

**Running postTracking.py:**

*GUI method*

1) Download postTracking.py. 
2) In the WinPython directory, click File > Open..., then select postTracking.py and click Open.
3) In the toolbar click Run > Run, or press F5. The script should begin running.

*Command prompt method*

1) Download postTracking.py. 
2) Open the WinPython Command Prompt or Windows Powershell application, and navigate to the directory containing postTracking.py. 
3) Type `python postTracking.py` and press Enter. The script should begin running. 

**Python set-up instructions for macOS:**

1) Visit https://www.python.org/downloads/mac-osx/ to download and install the latest release of Python 3.
2) Open 'Terminal'. Type `pip install matplotlib`, press Enter, and wait for the matplotlib library to finish installing.
3) Type `pip install opencv-contrib-python`, press Enter, and wait for the opencv library to finish installing.
4) Type `which python` and press Enter to confirm that Python 3 is invoked by the 'python' keyword.

**Running postTracking.py:**

1) Download postTracking.py.
2) Using Terminal, navigate to the directory containing postTracking.py
3) Type `python postTracking.py` and press Enter. The script should begin running. If `which python` pointed to Python 2, type 
`python3 postTracking.py` and press Enter instead.

