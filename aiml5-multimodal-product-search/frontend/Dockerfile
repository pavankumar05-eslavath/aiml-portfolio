# syntax=docker/dockerfile:1.7
#
# Frontend image: Vite build served by nginx.
#
# nginx also reverse-proxies /api to the backend, so the browser talks to a single
# origin. That keeps CORS off the critical path in production and means the
# compiled JavaScript needs no build-time knowledge of the backend's address.

# ---------------------------------------------------------------- build stage
FROM node:22-alpine AS builder

WORKDIR /app

# Copy manifests first so `npm ci` is cached unless the lockfile changes.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Relative base URL: requests go to the same origin nginx serves, and nginx
# proxies them onward.
ENV VITE_API_BASE_URL=/api
RUN npm run build

# -------------------------------------------------------------- runtime stage
FROM nginx:1.27-alpine AS runtime

LABEL org.opencontainers.image.title="Multimodal Product Search UI" \
      org.opencontainers.image.licenses="MIT"

# Opt in to the base image's resolver detection. With this set, the entrypoint
# reads /etc/resolv.conf and exports NGINX_LOCAL_RESOLVERS, which the config
# template consumes. Without it the resolver address would have to be hardcoded,
# which breaks across container runtimes - see the note in the template.
ENV NGINX_ENTRYPOINT_LOCAL_RESOLVERS=1 \
    API_UPSTREAM=backend:8000

# Templates are rendered by the entrypoint's envsubst step at container start.
COPY nginx/default.conf.template /etc/nginx/templates/default.conf.template
# Included by every location block; kept out of conf.d/ because the base image
# includes conf.d/*.conf at http level, which would double-apply these headers.
COPY nginx/security-headers.conf /etc/nginx/snippets/security-headers.conf
COPY --from=builder /app/dist /usr/share/nginx/html

# nginx:alpine already runs its workers as the unprivileged `nginx` user; the
# master process needs root to bind port 80.
EXPOSE 80

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -q --spider http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
