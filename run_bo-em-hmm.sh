#!/bin/bash

cd ~/Dropbox/Course/Optimization/Project/embo/
# mongo mongo_clean_bo_em.js

rm bo_em/output/*
rmdir bo_em/output/
rm bo_em/extra_history.json
python ../Spearmint/spearmint/bo_w_extra_data.py bo_em/