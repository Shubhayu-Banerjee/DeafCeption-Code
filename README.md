# DeafCeption-Code
SVAN app software repository

*To be heard, To be expressed*

In Development (Python)

# DeafCeption Fingerspell (DCF)

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

# DeafCeption Exec (DCE)

- Version 1.0.0 Released
- Universal DeafCeption Model Executer.
- Will have the capability to run all DeafCeption Models.

# DeafCeption Gesture (DCG)

- Version 0.4.1 Released
- Next step towards sign language interpretion
- Gestures are detected
- *Requires DCE to run (In Development)*
- Rapid Development Being Done on DeafCeption Gesture Model. DCG 0.4.1 recognises ***thirteen gestures***.

# DeafCeption Gesture X
- Instead of relying on camera capture and a pure CV model, DCGX will interpret signs based on finger points.
- This removes the problems of lighting changes, background changes, different people, etc.
- In very early development
