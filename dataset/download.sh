#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # no color
DIR="./data"

# Download and decompress datasets

Help()
{
   # Display Help
   echo "We provide four datasets (open_images, stackoverflow, and speech)"
   echo "to evalute the performance of Kuiper"
   echo 
   echo "Syntax: ./download.sh [-g|h|v|V]"
   echo "options:"
   echo "-h     Print this Help."
   echo "-A     Download all datasets (about 77GB)"
   echo "-t     Download Stack Overflow dataset (about 7.2GB)"
   echo "-p     Download Speech Commands dataset (about 2.3GB)"
   echo "-o     Download Open Images dataset (about 66GB)"
   echo
}

speech()
{
    if [ ! -d "${DIR}/speech_commands/train/" ]; 
    then
        echo "Downloading Speech Commands dataset(about 2.4GB)..."
        wget -O ${DIR}/speech_commands/google_speech.tar.gz https://www.dropbox.com/s/h9ss1ol10ikc6lx/google_speech.tar.gz?dl=0​​​​​​​

        echo "Dataset downloaded, now decompressing..." 
        tar -xf ${DIR}/speech_commands/google_speech.tar.gz

        echo "Removing compressed file..."
        rm -f ${DIR}/speech_commands/google_speech.tar.gz

        echo -e "${GREEN}Speech Commands dataset downloaded!${NC}"
    else
        echo -e "${RED}Speech Commands dataset already exists under ${DIR}/speech_commands/!"
fi
}

stackoverflow()
{
    if [ ! -d "${DIR}/stackoverflow/train/" ]; 
    then
        echo "Downloading Stack Overflow dataset(about 7.2GB)..."
        wget -O ${DIR}/stackoverflow/stackoverflow.tar.gz https://www.dropbox.com/s/7ct7u3yoy1n85uk/stackoverflow.tar.gz?dl=0

        echo "Dataset downloaded, now decompressing..." 
        tar -xf ${DIR}/stackoverflow/stackoverflow.tar.gz

        echo "Removing compressed file..."
        rm -f ${DIR}/stackoverflow/stackoverflow.tar.gz

        echo -e "${GREEN}Stack Overflow dataset downloaded!${NC}"
    else
        echo -e "${RED}Stack Overflow dataset already exists under ${DIR}/stackoverflow/!"
fi
}

open_images() 
{
    if [ ! -d "${DIR}/open_images/train/" ]; 
    then
        echo "Downloading Open Images dataset(about 66GB)..."   
        wget -O ${DIR}/open_images.tar.gz https://www.dropbox.com/s/635t55j74p3quhi/open_images.tar.gz?dl=0
        
        echo "Dataset downloaded, now decompressing..." 
        tar -xf ${DIR}/open_images.tar.gz

        echo "Removing compressed file..."
        rm -f ${DIR}/open_images.tar.gz

        echo -e "${GREEN}Open Images dataset downloaded!${NC}"
    else
        echo -e "${RED}Open Images dataset already exists under ${DIR}/open_images/!"
fi
}

while getopts ":hAotp" option; do
   case $option in
      h ) # display Help
         Help
         exit;;
      A )
         speech
         stackoverflow
         open_images
         exit;;
      o )
         open_images   
         ;;    
      t )
         stackoverflow   
         ;;  
      p )
         speech   
         ;;                    
      \? ) 
         echo -e "${RED}Usage: cmd [-h] [-A] [-o] [-t] [-p]${NC}"
         exit 1;;
   esac
done

if [ $OPTIND -eq 1 ]; then 
    echo -e "${RED}Usage: cmd [-h] [-A] [-o] [-t] [-p]${NC}"; 
fi
