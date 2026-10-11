"""
Django settings for the AI Vehicle Damage Inspection project.

Saari production settings environment variables se aati hain (ya project ke
`.env` file se — dekhein `.env.example`). Kuch set na ho to development
defaults chalte hain: DEBUG on, SQLite, koi HTTPS nahi.

Production guide: DEPLOY.md
"""
import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path):
    """`.env` file ki KEY=value lines environment mein daalo (pehle se set wali nahi badalti)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_dotenv(BASE_DIR / ".env")


def env(key, default=None):
    return os.environ.get(key, default)


def env_bool(key, default=False):
    value = os.environ.get(key)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def env_list(key, default=""):
    return [item.strip() for item in os.environ.get(key, default).split(",") if item.strip()]


DEBUG = env_bool("DJANGO_DEBUG", True)

_DEV_KEY = "django-insecure-dev-only-key-do-not-use-in-production"
SECRET_KEY = env("DJANGO_SECRET_KEY", _DEV_KEY)
if not DEBUG and SECRET_KEY == _DEV_KEY:
    raise ImproperlyConfigured("Production mein DJANGO_SECRET_KEY set karna zaroori hai (dekhein DEPLOY.md).")

ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS", "*" if DEBUG else "")
# e.g. https://inspect.example.com — HTTPS par forms (CSRF) ke liye zaroori
CSRF_TRUSTED_ORIGINS = env_list("DJANGO_CSRF_TRUSTED_ORIGINS")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "inspection",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",  # CSS/JS production mein khud serve
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# ---- Database: POSTGRES_DB set ho to PostgreSQL, warna SQLite ----
if env("POSTGRES_DB"):
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": env("POSTGRES_DB"),
            "USER": env("POSTGRES_USER", "autoinspect"),
            "PASSWORD": env("POSTGRES_PASSWORD", ""),
            "HOST": env("POSTGRES_HOST", "localhost"),
            "PORT": env("POSTGRES_PORT", "5432"),
            "CONN_MAX_AGE": 60,
            "CONN_HEALTH_CHECKS": True,
            "OPTIONS": {"connect_timeout": 10},
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

AUTH_PASSWORD_VALIDATORS = [] if DEBUG else [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Karachi"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"  # `collectstatic` yahan jama karta hai

# Upload hone wali photos aur AI ki banayi hui annotated photos yahan save hongi
MEDIA_URL = "/media/"
MEDIA_ROOT = Path(env("DJANGO_MEDIA_ROOT", BASE_DIR / "media"))
# Nginx/IIS media serve na kare (e.g. Windows + Waitress) to Django khud serve kare
SERVE_MEDIA = env_bool("DJANGO_SERVE_MEDIA", DEBUG)

STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    # Production: file naam mein hash (cache-busting) + gzip/brotli
    "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage" if DEBUG
                    else "whitenoise.storage.CompressedManifestStaticFilesStorage"},
}

# Badi phone photos ke liye upload limit (20 MB)
DATA_UPLOAD_MAX_MEMORY_SIZE = 20 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 20 * 1024 * 1024

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---- HTTPS / security (sirf production) ----
if not DEBUG:
    # Nginx/IIS HTTPS khatam karke Django ko batata hai ke request https thi
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SECURE_SSL_REDIRECT = env_bool("DJANGO_SECURE_SSL_REDIRECT", True)
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_HSTS_SECONDS = int(env("DJANGO_HSTS_SECONDS", 60 * 60 * 24 * 30))  # 30 din
    SECURE_HSTS_INCLUDE_SUBDOMAINS = env_bool("DJANGO_HSTS_INCLUDE_SUBDOMAINS", False)
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_REFERRER_POLICY = "same-origin"
    X_FRAME_OPTIONS = "DENY"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": env("DJANGO_LOG_LEVEL", "INFO")},
}
