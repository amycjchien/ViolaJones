# ViolaJones

# Setup Environment (Python3)
python3 -m pip install opencv-python

python3 -m pip install matplot


# How to execute
## Training Usage:
### Baseline
python3 final_project.py \ 

-f ./dataset # path of image folder \

-t 5 # number of weak classifiers \ 

-e E # error type (E, FP, FN, E+FP, E+FN) \

### Cascading System
python3 final_project_cascade.py \ 

-f ./dataset # path of image folder \

## Visualizing Usage:
python3 final_project_visualization.py \ 

-f ./dataset # path of image folder \

-i 10 # need to have './classifier_' + str(args.iteration) + '.pkl' \
