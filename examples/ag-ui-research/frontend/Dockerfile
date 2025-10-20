# Development Dockerfile for Next.js frontend
FROM node:22-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci
COPY . .
EXPOSE 3000

# Run in development mode with hot reload
CMD ["npm", "run", "dev"]
