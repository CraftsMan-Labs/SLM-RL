# Dev image for the Vue workshop UI (Vite HMR).
# Production static builds are out of scope — use `npm run build` locally if needed.
FROM node:22-bookworm-slim

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci \
 && cp package-lock.json node_modules/.package-lock.json

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

COPY . .

EXPOSE 5173

ENTRYPOINT ["docker-entrypoint.sh"]
# --host so HMR is reachable from the host browser via published port.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
