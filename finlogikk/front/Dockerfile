# Étape 1 : Construction de l'application Angular
FROM node:14-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build --prod

# Étape 2 : Serveur NGINX pour l'application Angular
FROM nginx:alpine
COPY --from=build /app/dist/finlogik /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
