#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # no color
DIR="./data"

# Download and decompress datasets

Help()
{
   # Display Help
   echo "FedScale provides a large suite of FL datasets "
   echo "to evaluate today's FL performance"
   echo
   echo "Syntax: ./download.sh [--dataset_name]"
   echo "options:"
   echo "--help                         Print this Help."
   echo "--speech                       Download Speech Commands dataset (about 2.3GB)"
   echo "--open_images                  Download Open Images dataset (about 66GB)"
   echo "--amazon_review                Download Amazon Review dataset (about 11G)"
   echo "--charades                     Download Charades dataset (about 15G)"
   echo "--europarl                     Download Europarl dataset (about 458M)"
   echo "--go                           Download Go dataset (about 1.7G)"
   echo "--inaturalist                  Download Inaturalist 2019 dataset meta file (about 11M)"
   echo "--libriTTS                     Download LibriTTS dataset (about 78G)"
   echo "--open_images_detection        Download Open Images for detection (about 451M)"
   echo "--reddit                       Download Reddit dataset (about 25G)"
   echo "--taobao                       Download Taobao dataset (about 185M)"
   echo "--taxi                         Download Taxi Trajectory dataset (about 504M)"
   echo "--waymo                        Download Waymo Motion dataset meta file (about 74M)"
   echo "--femnist                      Download FEMNIST dataset (about 327M)"
   echo "--stackoverflow                Download StackOverflow dataset (about 13G)"
   echo "--blog                         Download Blog dataset (about 833M)"
}

speech()
{
    if [ ! -d "${DIR}/speech_commands/train/" ];
    then
        echo "Install dependencies"
        conda install -y numba=0.49.1
        conda install librosa=0.7.2

        echo "Downloading Speech Commands dataset(about 2.4GB)..."
        wget -O ${DIR}/speech_commands/google_speech.tar.gz https://fedscale.eecs.umich.edu/dataset/google_speech.tar.gz

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/speech_commands/google_speech.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/speech_commands/google_speech.tar.gz

        echo -e "${GREEN}Speech Commands dataset downloaded!${NC}"
    else
        echo -e "${RED}Speech Commands dataset already exists under ${DIR}/speech_commands/!"
fi
}

open_images()
{
    if [ ! -d "${DIR}/open_images/train/" ];
    then
        echo "Downloading Open Images dataset(about 66GB)..."
        wget -O ${DIR}/open_images.tar.gz https://fedscale.eecs.umich.edu/dataset/openImage.tar.gz

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/open_images.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/open_images.tar.gz

        echo -e "${GREEN}Open Images dataset downloaded!${NC}"
    else
        echo -e "${RED}Open Images dataset already exists under ${DIR}/open_images/!"
fi
}

amazon_review()
{
    if [ ! -d "${DIR}/amazon_review/train/" ];
    then
        echo "Downloading Amazon Review dataset(about 11GB)..."
        wget -O ${DIR}/amazon_review.tar.gz https://fedscale.eecs.umich.edu/dataset/amazon_review.tar.gz

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/amazon_review.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/amazon_review.tar.gz

        echo -e "${GREEN}Amazon Review dataset downloaded!${NC}"
    else
        echo -e "${RED}Amazon Review dataset already exists under ${DIR}/amazon_review/!"
fi
}

charades()
{
    if [ ! -d "${DIR}/charades/train/" ];
    then
        echo "Downloading Charades dataset(about 15GB)..."
        wget -O ${DIR}/charades.tar.gz https://fedscale.eecs.umich.edu/dataset/charades_v1.tar.gz

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/charades.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/charades.tar.gz

        echo -e "${GREEN}Charades dataset downloaded!${NC}"
    else
        echo -e "${RED}Charades dataset already exists under ${DIR}/charades/!"
fi
}

europarl()
{
    if [ ! -d "${DIR}/europarl/client_data_mapping/" ];
    then
        echo "Downloading europarl dataset(about 458M)..."
        wget -O ${DIR}/europarl.tar.gz https://fedscale.eecs.umich.edu/dataset/europarl.tar.gz

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/europarl.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/europarl.tar.gz

        echo -e "${GREEN}europarl dataset downloaded!${NC}"
    else
        echo -e "${RED}europarl dataset already exists under ${DIR}/europarl/!"
fi
}

go()
{
    if [ ! -d "${DIR}/go/train/" ];
    then
        echo "Downloading go dataset(about 1.7G)..."
        wget -O ${DIR}/go.tar.gz https://fedscale.eecs.umich.edu/dataset/go-dataset.tar.gz

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/go.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/go.tar.gz

        echo -e "${GREEN}go dataset downloaded!${NC}"
    else
        echo -e "${RED}go dataset already exists under ${DIR}/go/!"
fi
}

inaturalist()
{
    if [ ! -d "${DIR}/inaturalist/client_data_mapping/" ];
    then
        echo "Downloading inaturalist dataset ..."
        wget -O ${DIR}/inaturalist.tar.gz https://fedscale.eecs.umich.edu/dataset/inaturalist.tar.gz

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/inaturalist.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/inaturalist.tar.gz

        echo -e "${GREEN}inaturalist dataset downloaded!${NC}"
    else
        echo -e "${RED}inaturalist dataset already exists under ${DIR}/inaturalist/!"
fi
}


libriTTS()
{
    if [ ! -d "${DIR}/libriTTS/train/" ];
    then
        echo "Downloading libriTTS dataset(about 78G)..."
        wget -O ${DIR}/libriTTS.tar.gz https://fedscale.eecs.umich.edu/dataset/libriTTS.tar.gz

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/libriTTS.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/libriTTS.tar.gz

        echo -e "${GREEN}libriTTS dataset downloaded!${NC}"
    else
        echo -e "${RED}libriTTS dataset already exists under ${DIR}/libriTTS/!"
fi
}

open_images_detection()
{
    if [ ! -d "${DIR}/open_images_detection/client_data_mapping/" ];
    then

        echo "Install dependencies"
        conda install h5py

        echo "Downloading open_images_detection dataset(about 451M)..."
        wget -O ${DIR}/openimage_detection.tar.gz https://fedscale.eecs.umich.edu/dataset/openimage_detection.tar.gz

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/openimage_detection.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/openimage_detection.tar.gz

        echo -e "${GREEN}open_images_detection dataset downloaded!${NC}"
    else
        echo -e "${RED}open_images_detection dataset already exists under ${DIR}/open_images_detection/!"
fi
}

reddit()
{
    if [ ! -d "${DIR}/reddit/train/" ];
    then
        echo "Downloading reddit dataset(about 25G)..."
        wget -O ${DIR}/reddit.tar.gz https://fedscale.eecs.umich.edu/dataset/reddit.tar.gz

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/reddit.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/reddit.tar.gz

        echo -e "${GREEN}reddit dataset downloaded!${NC}"
    else
        echo -e "${RED}reddit dataset already exists under ${DIR}/reddit/!"
fi
}

blog()
{
    if [ ! -d "${DIR}/blog/train/" ];
    then
        echo "Downloading blog dataset(about 800M)..."
        wget -O ${DIR}/blog.tar.gz https://fedscale.eecs.umich.edu/dataset/blog.tar.gz

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/blog.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/blog.tar.gz

        echo -e "${GREEN}blog dataset downloaded!${NC}"
    else
        echo -e "${RED}blog dataset already exists under ${DIR}/blog/!"
fi
}

stackoverflow()
{
    if [ ! -d "${DIR}/stackoverflow/train/" ];
    then
        echo "Install dependencies"
        conda install h5py

        echo "Downloading stackoverflow dataset(about 800M)..."
        wget -O ${DIR}/stackoverflow.tar.gz https://fedscale.eecs.umich.edu/dataset/stackoverflow.tar.gz

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/stackoverflow.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/stackoverflow.tar.gz

        echo -e "${GREEN}stackoverflow dataset downloaded!${NC}"
    else
        echo -e "${RED}stackoverflow dataset already exists under ${DIR}/stackoverflow/!"
fi
}

taobao()
{
    if [ ! -d "${DIR}/taobao/client_data_mapping/" ];
    then
        echo "Downloading taobao dataset(about 185M)..."
        wget -O ${DIR}/taobao.tar.gz https://fedscale.eecs.umich.edu/dataset/taobao.tar.gz

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/taobao.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/taobao.tar.gz

        echo -e "${GREEN}taobao dataset downloaded!${NC}"
    else
        echo -e "${RED}taobao dataset already exists under ${DIR}/taobao/!"
fi
}

taxi()
{
    if [ ! -d "${DIR}/taxi_traj/client_data_mapping/" ];
    then
        echo "Downloading taxi prediction dataset(about 504M)..."
        wget -O ${DIR}/taxi_traj.tar.gz https://fedscale.eecs.umich.edu/dataset/taxi_traj.tar.gz

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/taxi_traj.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/taxi_traj.tar.gz

        echo -e "${GREEN}taxi_traj dataset downloaded!${NC}"
    else
        echo -e "${RED}taxi_traj dataset already exists under ${DIR}/taxi_traj/!"
fi
}

waymo()
{
    if [ ! -d "${DIR}/waymo/client_data_mapping/" ];
    then
        echo "Downloading waymo dataset(about 74M)..."
        wget -O ${DIR}/waymo.tar.gz https://fedscale.eecs.umich.edu/dataset/waymo.tar.gz

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/waymo.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/waymo.tar.gz

        echo -e "${GREEN}waymo dataset downloaded!${NC}"
    else
        echo -e "${RED}waymo dataset already exists under ${DIR}/waymo/!"
fi
}

femnist()
{
    if [ ! -d "${DIR}/femnist/client_data_mapping/" ];
    then
        echo "Downloading FEMNIST dataset(about 327M)..."
        wget -O ${DIR}/femnist.tar.gz https://fedscale.eecs.umich.edu/dataset/femnist.tar.gz

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/femnist.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/femnist.tar.gz

        echo -e "${GREEN}FEMNIST dataset downloaded!${NC}"
    else
        echo -e "${RED}FEMNIST dataset already exists under ${DIR}/femnist/! ${NC}"
fi
}

while true; do
   case "$1" in
      --help ) # display Help
         Help
         shift
         exit;;
      --open_images )
         open_images
         shift
         ;;
      --speech )
         speech
         shift
         ;;
      --amazon_review )
         amazon_review
         shift
         ;;
      --charades )
         charades
         shift
         ;;
      --europarl )
         europarl
         shift
         ;;
      --go )
         go
         shift
         ;;
      --inaturalist )
         inaturalist
         shift
         ;;
      --libriTTS )
         libriTTS
         shift
         ;;
      --open_images_detection )
         open_images_detection
         shift
         ;;
      --reddit )
         reddit
         shift
         ;;
      --taobao )
         taobao
         shift
         ;;
      --taxi )
         taxi
         shift
         ;;
      --waymo )
         waymo
         shift
         ;;
      --femnist )
         femnist
         shift
         ;;
      --blog )
         blog
         shift
         ;;
      --stackoverflow )
         stackoverflow
         shift
         ;;
      + )
         echo "${RED}Usage: check --help${NC}"
         break
         ;;
      * )
        exit
        ;;
   esac
done
