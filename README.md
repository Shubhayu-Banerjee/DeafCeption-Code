# DeafCeption-Code
App software repository

*To be heard, To be expressed*

>In Development, Currently working with Indian Sign Language

> ⚠️ This project is under a custom license.  
> It is **not open-source** for reuse. Forks are for personal viewing and learning only.  
> No redistribution or commercial use allowed without permission.

# DeafCeption Gesture X (DCGX) (Up To Date)
- Next step towards live sign language interpretion
- Instead of relying on camera capture and a pure CV model, DCGX interprets signs based upon joint positions.
- This greatly reduces issues with of lighting changes, background changes, different people, etc.
- Rapid Development Being Done on DCGX Model.
- Uploaded version supports **109 gestures.**
- Version 2.2.1 Released.
- Try out a micro-version [online](https://shubhayu-banerjee.github.io/DeafCeption-Demo/) right now! [Front End Git Repo](https://github.com/Shubhayu-Banerjee/DeafCeption-Demo)
- Try out the latest demonstration app [(.exe)](https://drive.google.com/file/d/1_yepU6pERZnCmw1GLvZSQscsO7U6gYls/view?usp=sharing). It works with Indian Sign Language.
> ⚠️ Please keep in mind that single-hand gestures are still bound to the hand they were trained on. Most single hand signs are trained with the right hand, however signs like eat,cap,ball,egg,adult,energy etc. (in the main model (selected from dropdown) under sign->text) are trained with the left. For easy testing/proof of concept, select Numbers or Alphabet model from the drop down.

# DeafCeption Fingerspell (DCF) (Phased Out)

Model detects ASL gestures for letters a,b,c and d (this is fingerspell! Individual letters)
- Model best performs with only hand in camera capture
- White/black (solid colour) uniform background preffered
- Good lighting required
- Speech output occurs every 20 cycles of the program
- Press q to exit camera capture, then speak something into console clearly.
- Speech converted to text

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
