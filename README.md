<!-- backend -->
uvicorn api:app --reload

<!-- frontend -->
python -m http.server 5500

<!-- link to app -->
http://localhost:5500/index.html
