CURRENT_FOLDER="$(dirname $0)"
cd $CURRENT_FOLDER

docker build -t local/vitpose:mim --progress plain .