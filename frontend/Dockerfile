# syntax=docker/dockerfile:1

# Stage 1: Build the React SPA
FROM node:20-alpine AS build

WORKDIR /app

# Install dependencies using package-lock.json for reproducible builds
COPY package.json package-lock.json ./
RUN npm ci

# Copy source code and build
COPY . .

# In production container, VITE_API_BASE_URL defaults to empty string
# so API calls use relative paths reverse-proxied by Nginx to the backend container
ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

RUN npm run build

# Stage 2: Serve production build with Nginx
FROM nginx:alpine

# Copy custom reverse proxy Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy compiled static assets from build stage
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

# Verified healthcheck for Nginx readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -q --spider http://127.0.0.1/nginx-health || exit 1

CMD ["nginx", "-g", "daemon off;"]
