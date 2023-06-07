#!/bin/bash

fileid="1bCAJZ3POpsqDFBxFLRQ1DZTU7jVnl41k"
zipname="dataset.zip"
unzipdir=""

while [ $# -gt 0 ]; do
    case $1 in
        --unzipdir=*)
            unzipdir="${1#*=}"
            shift 1;;
        *)
            echo "Unknown parameter: $1"
            exit 1;;
    esac
done

if [ -z "$unzipdir" ]; then
    echo "Usage: ./download_dataset.sh --unzipdir=[unzipdir]"
    exit 1
fi

curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -L -b /tmp/cookies "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' /tmp/cookies`&id=${fileid}" -o ${zipname}

unzip ${zipname} -d ${unzipdir}