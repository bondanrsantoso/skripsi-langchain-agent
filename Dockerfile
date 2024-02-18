# PRODUCTION DOCKERFILE
# -------------------------------------
# Dependency installation section
# only runs if dependency changed
FROM node:alpine as deps
RUN apk add libc6-compat g++ make py3-pip
WORKDIR /app
COPY package.json package-lock.json ./
COPY package-lock.json ./

# Run installation 
RUN npm ci 
# -------------------------------------
FROM node:alpine as runner
WORKDIR /app

COPY . .
COPY --from=deps /app/node_modules ./node_modules

EXPOSE 3000
CMD npm run start
