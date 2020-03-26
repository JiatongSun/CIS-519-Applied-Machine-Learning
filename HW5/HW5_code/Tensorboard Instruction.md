# Tensorboard Instruction

### Environment: Win10,  Anaconda,  Spyder 4.0,  Python 3.7

__Step:__
*If you haven't loaded tensorboard, please do the followings:*

1. Execute `%load_ext tensorboard` in Spyder Console
2. Create a new folder for save log, eg. `./logdir'`
3. Execute `%tensorboard --logdir ='./logdir' --host=127.0.0.1` (Be sure you're under the correct directory)
4. Open web browser and search <localhost:6006>
5. Execute the training process
6. View the updating data in the web browser

*If you have already loaded tensorboard, please do the followings:*

1. Open Win10 cmd (don't use powershell)
2. Execute `taskkill /IM "tensorboard.exe" /F` (If permission is needed, please run cmd as administrator)
3. Execute `del /q %TMP%\.tensorboard-info\*`
4. Close every software and restart computer
5. Delete your log directory and create a new empty one
6. Continue with the previous steps

*Be sure to delete every log before running a new one, or the graph may not be updated.*