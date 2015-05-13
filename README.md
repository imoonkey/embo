# embo
The files in **hack_copy/ ** should be placed in the root folder of spearmint. And instead of running "main.py", you should run "bo_w_extra_data.py".

Basically, every round, EM code writes the extra points into a json file named *extra_history.json* under the same folder of EM code, and BO reads it in and write it into database (mimicing the true point record), before doing the point selection.


