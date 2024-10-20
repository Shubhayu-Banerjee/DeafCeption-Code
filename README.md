# DeafCeption-Code
SVAN app software repository

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

# DeafCeption Gesture (DCG)

- Next step towards sign language interpretion
- Gestures to be detected rather than fingerspell
- *1st Version Release expected by 25th of October*

!!Rapid Development Being Done on DeafCeption Gesture Model. Currently recognises ***six gestures***!!

# DeafCeption Exec (DCE)

- in development
- universal DeafCeption model runner
- will make it such that upon loading DCE in an env., all that is required is specifying which model to run.
- *First Version Release expected by 1st of November.*
