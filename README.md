# Human boundary detection (classical OpenCV, no deep learning)

- `app.py`: finds the exact boundary of a person in a colour photo.
- `thermal.py`: finds the exact boundaries of people in a thermal image.
- `web.py`: web application running both scripts.

Each script has full instructions at the top of the file.

## How to run
    pip install -r requirements.txt
    python app.py myphoto.jpg
    python thermal.py thermal.jpg
    python -m streamlit run web.py      # web application

Add `--run-sam2` to either script to compare with SAM2
(needs: pip install git+https://github.com/facebookresearch/sam2.git).

Live web app: (paste link here)
