FROM python:3.9-slim

# Install redis-tools for redis-cli
RUN apt-get update && apt-get install -y redis-tools supervisor

# Créer un répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code de l'application dans le conteneur
COPY . .

# Environment variables for Flask and Celery
ENV FLASK_APP=app.py
ENV C_FORCE_ROOT=true
# Needed for Celery to run as root inside Docker


# Exposer le port 5000 pour Flask
EXPOSE 5000
EXPOSE 8501
EXPOSE 8502

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Commande pour exécuter à la fois Flask et les applications Streamlit
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]