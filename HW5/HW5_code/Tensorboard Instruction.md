# Tensorboard Instruction

### Environment: Win10,  Anaconda,  Spyder 4.0,  Python 3.7, tensorboard 2.1.0

*__Step:__*
*__If you haven't loaded tensorboard, please do the followings:__*

1. Change to working directory (Can simply run a `.py` file in Spyder and Spyder can automatically change folder )
2. Create a new folder for save log, eg. `./logdir'`
3. Execute `%load_ext tensorboard` in Spyder Console
4. Execute `%tensorboard --logdir="./logdir" --host=127.0.0.1`  (You should see `"Launching Tensorboard..."` and need to wait for a little bit)
5. Open web browser and search <localhost:6006> (An orange webpage saying "No dashboards are active")
6. Execute `%reload_ext tensorboard`
7. Execute the training process
8. View the updating data in the web browser

*__If you have already loaded tensorboard, please do the followings:__*

1. Restart computer 

   (The purpose of this step is to kill the tensorboard process. Another way is to execute `taskkill /IM "tensorboard.exe" /F` in win10 cmd window, but it doesn't work every time, so restarting is probably the easiest approach)

2. Delete your entire log directory

3. Continue with the basic steps in the previous section

*__Be sure to delete every log before running a new one, or the graph may not be updated.__*