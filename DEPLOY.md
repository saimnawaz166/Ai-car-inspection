# Production deployment — PostgreSQL + Gunicorn/Waitress + HTTPS

Development (`python manage.py runserver`) waisa hi chalta hai. Production sirf
`.env` file se on hota hai — code badalne ki zarurat nahi.

| Cheez | Development | Production |
|---|---|---|
| Settings | defaults | `.env` (copy of `.env.example`) |
| Database | SQLite (`db.sqlite3`) | PostgreSQL (`POSTGRES_*`) |
| Web server | `runserver` | Linux: **Gunicorn** · Windows: **Waitress** |
| HTTPS | nahi | Nginx + Let's Encrypt (Linux) · IIS (Windows) |
| CSS/JS | Django | WhiteNoise (`collectstatic`, hashed file names) |
| Photos (`/media/`) | Django | Nginx (Linux) · Django with `DJANGO_SERVE_MEDIA=true` (Windows) |

**Server size:** kam az kam **2 vCPU, 4 GB RAM**, 20 GB disk. AI model (PyTorch) har
Gunicorn worker mein ~500 MB RAM leta hai. GPU ki zarurat nahi (CPU par ~0.4 sec/photo).

---

## Option A — Linux VPS (Ubuntu 22.04/24.04) · recommended

Domain ka A-record server ke IP par point hona chahiye (e.g. `inspect.example.com`).

### 1. Packages

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential libgl1 \
                    postgresql nginx certbot python3-certbot-nginx
```

### 2. PostgreSQL database

```bash
sudo -u postgres psql -c "CREATE USER autoinspect WITH PASSWORD 'STRONG-PASSWORD';"
sudo -u postgres psql -c "CREATE DATABASE autoinspect OWNER autoinspect;"
```

### 3. App code

```bash
sudo useradd --system --create-home --shell /bin/bash autoinspect
sudo mkdir -p /srv/autoinspect && sudo chown autoinspect:www-data /srv/autoinspect
# project files /srv/autoinspect mein copy karein (git clone / scp / zip)
# model file bhi: ml_models/car_damage_yolo11-seg.pt

sudo -u autoinspect -i
cd /srv/autoinspect
python3 -m venv venv
venv/bin/pip install --upgrade pip
# CPU-only PyTorch (chhota, GPU wala ~2 GB zyada hota hai)
venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
venv/bin/pip install -r requirements.txt
```

### 4. `.env`

```bash
cp .env.example .env
venv/bin/python -c "from django.core.management.utils import get_random_secret_key as k; print(k())"
nano .env     # DJANGO_SECRET_KEY, domain, POSTGRES_PASSWORD bharein
chmod 600 .env
```

### 5. Database tables, CSS, admin user

```bash
venv/bin/python manage.py migrate
venv/bin/python manage.py collectstatic --noinput
venv/bin/python manage.py createsuperuser
venv/bin/python manage.py check --deploy
mkdir -p media && exit
```

`check --deploy` mein `security.W005` / `W021` (HSTS subdomains/preload) aa sakte hain — yeh
domain owner ka faisla hai, zaroori nahi.

### 6. Gunicorn service

```bash
sudo cp /srv/autoinspect/deploy/autoinspect.service /etc/systemd/system/
sudo chown -R autoinspect:www-data /srv/autoinspect/media
sudo systemctl daemon-reload
sudo systemctl enable --now autoinspect
sudo systemctl status autoinspect        # "active (running)"
```

Workers/timeout: `deploy/gunicorn.conf.py` (default 2 workers, 180 sec timeout — 14 zones
ki inspection ~1 minute le sakti hai).

### 7. Nginx + HTTPS

```bash
sudo cp /srv/autoinspect/deploy/nginx-autoinspect.conf /etc/nginx/sites-available/autoinspect
sudo nano /etc/nginx/sites-available/autoinspect     # inspect.example.com -> apna domain
sudo ln -s /etc/nginx/sites-available/autoinspect /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d inspect.example.com           # free HTTPS certificate, auto-renew
```

Ab `https://inspect.example.com` khul jayega. HTTP khud HTTPS par redirect hota hai.
Sab theek chale to `.env` mein `DJANGO_HSTS_SECONDS=2592000` (30 din) kar dein aur
`sudo systemctl restart autoinspect`.

### Update (naya code)

```bash
cd /srv/autoinspect && sudo -u autoinspect git pull     # ya files copy
sudo -u autoinspect venv/bin/pip install -r requirements.txt
sudo -u autoinspect venv/bin/python manage.py migrate
sudo -u autoinspect venv/bin/python manage.py collectstatic --noinput
sudo systemctl restart autoinspect
```

### Backup (roz)

```bash
pg_dump -U autoinspect -h localhost autoinspect | gzip > /backups/db-$(date +%F).sql.gz
tar czf /backups/media-$(date +%F).tgz -C /srv/autoinspect media
```

---

## Option B — Windows Server / Windows PC

Gunicorn Windows par nahi chalta — uski jagah **Waitress** (`deploy/run_waitress.py`).

1. **PostgreSQL** Windows installer (postgresql.org) → pgAdmin se user `autoinspect` + database `autoinspect`.
2. Project folder mein:
   ```powershell
   python -m venv venv
   venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   venv\Scripts\pip install -r requirements.txt
   copy .env.example .env      # values bharein; DJANGO_SERVE_MEDIA=true
   venv\Scripts\python manage.py migrate
   venv\Scripts\python manage.py collectstatic --noinput
   venv\Scripts\python manage.py createsuperuser
   venv\Scripts\python deploy\run_waitress.py          # http://127.0.0.1:8000
   ```
3. **Hamesha chalta rahe:** Task Scheduler → "At startup" → `venv\Scripts\python.exe deploy\run_waitress.py`
   (Start in = project folder), ya NSSM se Windows service.
4. **HTTPS (IIS):** IIS + **URL Rewrite** + **Application Request Routing** install karein →
   site banayein, certificate lagayein (win-acme se free Let's Encrypt) →
   reverse-proxy rule `http://127.0.0.1:8000/{R:1}`, aur server variable
   `HTTP_X_FORWARDED_PROTO = https` set karein (Waitress isi header se HTTPS samajhta hai).
   IIS request limit: `maxAllowedContentLength` = 104857600 (100 MB), proxy timeout 180 sec.

> **Laptop warning:** is project ka laptop lambe CPU load par garam hokar band ho chuka hai.
> Production ke liye alag server ya VPS lein.

---

## Settings reference (`.env`)

| Variable | Default | Matlab |
|---|---|---|
| `DJANGO_DEBUG` | `true` | Production mein `false` |
| `DJANGO_SECRET_KEY` | dev key | Production mein zaroori, warna app start nahi hogi |
| `DJANGO_ALLOWED_HOSTS` | `*` (dev) | Comma se: `inspect.example.com,www.inspect.example.com` |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | — | `https://inspect.example.com` (forms ke liye zaroori) |
| `DJANGO_SECURE_SSL_REDIRECT` | `true` | HTTP → HTTPS redirect |
| `DJANGO_HSTS_SECONDS` | 2592000 | Browser ko HTTPS yaad rakhne ka waqt |
| `DJANGO_SERVE_MEDIA` | `false` | Photos Django se dein (Windows/Waitress par `true`) |
| `DJANGO_MEDIA_ROOT` | `./media` | Photos ka folder |
| `POSTGRES_DB` | — | Set ho to PostgreSQL, warna SQLite |
| `POSTGRES_USER` / `_PASSWORD` / `_HOST` / `_PORT` | autoinspect / — / localhost / 5432 | |
| `DAMAGE_MODEL_PATH` | `ml_models/car_damage_yolo11-seg.pt` | AI model file |

## Production se pehle zaroor

- **Login nahi hai:** abhi jo bhi URL khol sake woh inspections dekh aur delete kar sakta hai.
  Internet par lagane se pehle staff login add karein (ya kam az kam Nginx/IIS basic-auth / office VPN).
- **CarDD license:** maujooda model research/non-commercial data par train hua hai — business
  use se pehle apni photos par retrain karein.
- SQLite se data move karna ho: `python manage.py dumpdata --exclude contenttypes > data.json`
  (SQLite par) → `.env` mein Postgres → `migrate` → `python manage.py loaddata data.json` + `media/` folder copy.
