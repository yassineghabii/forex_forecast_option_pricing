# Utiliser une image Java 17
FROM eclipse-temurin:17-jre

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier JAR généré dans l'image Docker
ARG JAR_FILE=target/back-0.0.1-SNAPSHOT.jar
COPY ${JAR_FILE} app.jar

# Exposer le port 8080
EXPOSE 8080

# Démarrer l'application Spring Boot
ENTRYPOINT ["java","-jar","/app/app.jar"]
