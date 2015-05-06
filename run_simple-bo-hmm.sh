#!/bin/bash
# mongo
# use spearmint
# db['simple-bo-hmm.jobs'].drop()
# db['simple-bo-hmm.hypers'].drop()
# exit

cd ~/Dropbox/Course/Optimization/Project/embo/
# mongo mongo_clean.js

rm bo_spearmint/output/*
rmdir bo_spearmint/output/
python ../Spearmint/spearmint/bo_w_extra_data.py bo_spearmint/