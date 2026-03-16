# Automotive Car Photo Editor (Flask)

Production-grade single-page car photo editor for dealerships using Flask + rembg (`u2net`) for server-side background removal.

## Requirements

- Python 3.10+ (or compatible Python 3)
- `pip`

## 1) Install dependencies

Run from the repository root:

`pip install -r requirements.txt`

If `pip` is mapped to Python 2 on your machine, use:

`pip3 install -r requirements.txt`

## 2) Start the app

Run:

`python app.py`

If `python` is not available, use:

`python3 app.py`

The app starts on:

`http://localhost:5000`

## 3) Use the app

1. Upload a car image via drag-and-drop or file picker.
2. Wait for AI background removal to complete.
3. Pick a style tile (or custom color).
4. Use **Show Original / Show Edited** to compare.
5. Click **Download PNG** to export `car-edited.png`.

## Troubleshooting

- **`python: command not found`**: use `python3`.
- **Background removal is slow on first run**: `u2net` model initialization can take extra time the first time `/remove-bg` is called.
- **Port already in use**: stop the process using port `5000` or run the app with a different port in `app.py`.
