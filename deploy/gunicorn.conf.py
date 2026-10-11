"""
Gunicorn settings (Linux). Chalane ke liye:
    gunicorn -c deploy/gunicorn.conf.py config.wsgi:application
"""
bind = "127.0.0.1:8000"          # sirf Nginx tak; bahar ki duniya Nginx (HTTPS) se aati hai

# Har worker apna AI model (PyTorch, ~500 MB RAM) load karta hai.
# 2 GB RAM server: 2 workers. 4 GB: 3. Zyada RAM ho tab hi barhayein.
workers = 2
threads = 2

# Ek inspection ke saare zones isi request mein AI se guzarte hain
# (CPU par ~3 sec/zone, 14 zones ~45 sec + pehli dafa model load). Is liye lamba timeout.
timeout = 180
graceful_timeout = 30

# Memory leak se bachne ke liye worker ko kuch requests ke baad naya kar do
max_requests = 200
max_requests_jitter = 20

accesslog = "-"
errorlog = "-"
