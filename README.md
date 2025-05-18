# DeafCeption-Code
SVAN app software repository

*To be heard, To be expressed*

In Development

> ⚠️ This project is under a custom license.  
> It is **not open-source** for reuse. Forks are for personal viewing and learning only.  
> No redistribution or commercial use allowed without permission.

# DeafCeption Gesture X (DCGX) (Up To Date)
- Next step towards sign language interpretion
- Instead of relying on camera capture and a pure CV model, DCGX will interpret signs based on finger points.
- This removes the problems of lighting changes, background changes, different people, etc.
- Rapid Development Being Done on DCGX Model.
- Uploaded version supports **107 gestures.**
- Version 1.0.1 Released.
- Try out a micro-version [online](https://shubhayu-banerjee.github.io/DeafCeption-Demo/) right now! [Front End Git Repo](https://github.com/Shubhayu-Banerjee/DeafCeption-Demo)

# DeafCeption Fingerspell (DCF) (Phased Out)

Model detects ASL gestures for letters a,b,c and d (this is fingerspell! Individual letters)
- Model best performs with only hand in camera capture
- white/black (solid colour) uniform background preffered
- good lighting required
- speech output occurs every 20 cycles of the program
- press q to exit camera capture, then speak something into console clearly.
- speech converted to text

# Run DCF

- Download DCF Folder
- Simply install required libraries (if not already installed) and run

!! DCF Runner.py file does not work-standalone for some reason. Please open with an environment such as Pycharm/VSC !!

!! Ensure all files (DCF, labels, and keras model) are in the same folder !!

# DeafCeption Exec (DCE) (Phased Out)

- Version 1.0.1 Released
- Universal DeafCeption Model Executer.
- Has the capability to run all DeafCeption Gesture (DCG) Models.

# DeafCeption Gesture (DCG) (Phased Out)

- Version 0.4.1 Released
- Gestures are detected
- *Requires DCE to run*
- DCG 0.4.1 recognises ***thirteen gestures***.
