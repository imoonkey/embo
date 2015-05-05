#!/bin/bash
# mongo
# use spearmint
# db['simple-bo-hmm.jobs'].drop()
# db['simple-bo-hmm.hypers'].drop()
# exit

cd ~/Dropbox/Course/Optimization/Project/embo/
mongo mongo_clean.js

rm bo_em/output/*
rmdir bo_em/output/
python ../Spearmint/spearmint/bo_w_extra_data.py bo_em/