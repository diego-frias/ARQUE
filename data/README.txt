==============================================================================
DATASET SETUP INSTRUCTIONS
==============================================================================

Due to copyright restrictions and file size limits, the image datasets (LIVE 
and CSIQ) are not included in this repository. You must download them from 
their official sources and organize them as described below.

------------------------------------------------------------------------------
1. LIVE Image Quality Assessment Database (Release 2)
------------------------------------------------------------------------------
Download: https://live.ece.utexas.edu/research/Quality/subjective.htm
(Look for "LIVE Image Quality Assessment Database Release 2")

Action:
1. Download the dataset.
2. Extract the contents into this 'data' folder.
3. Rename the root folder to 'LIVE' (if necessary).

Expected Directory Structure:
/data/
  └── LIVE/
       ├── dmos.mat          <-- Crucial file
       ├── gblur/
       │    └── img1.bmp ...
       ├── wn/
       ├── jpeg/
       ├── jp2k/
       └── fastfading/

------------------------------------------------------------------------------
2. CSIQ Image Quality Database
------------------------------------------------------------------------------
Download: http://vision.eng.shizuoka.ac.jp/mod/page/view.php?id=23
(Or search for "CSIQ Image Quality Database Larson")

Action:
1. Download the dataset.
2. Extract the contents into this 'data' folder.
3. Rename the root folder to 'CSIQ'.

Expected Directory Structure:
/data/
  └── CSIQ/
       ├── csiq_scores_by_image.csv   <-- Crucial file
       └── dst_imgs/
            ├── awgn/
            ├── blur/
            ├── jpeg/
            └── jpeg2000/

==============================================================================
TROUBLESHOOTING
==============================================================================
If you encounter "FileNotFoundError":
1. Ensure the folder names match exactly ("LIVE" and "CSIQ", case-sensitive).
2. Ensure you didn't create nested folders (e.g., data/LIVE/LIVE/gblur).
