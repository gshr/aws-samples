set -eux

echo "Pulling and Running Redis Instance"
docker pull redis
docker run -d --name redis-container -p 6379:6379 redis