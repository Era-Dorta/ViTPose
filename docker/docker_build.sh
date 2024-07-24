CURRENT_FOLDER="$(dirname $0)"
cd $CURRENT_FOLDER

docker build -t vitpose/vitpose:latest --progress plain .