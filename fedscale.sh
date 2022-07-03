#!/bin/bash
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # no color

DIR="$(dirname $0)"
ARGS=${@: 2};

Help()
{  
   echo
   echo -e "\033[1mFedScale is a real system to support FL evaluations and deployments.\033[0m"
   echo
   echo -e "${GREEN}Usage:${NC}"
   # Display Help
   echo
   echo "help                         Print this Help"
   echo "driver                       Manager jobs (refer to README)"
   echo "dataset                      Download FedScale datasets (refer to README)"
}

dataset()
{  
   bash $DIR/benchmark/dataset/download.sh $ARGS;
}

driver()
{  
   python $DIR/docker/driver.py $ARGS;
}

case "$1" in
   help ) # display Help
      Help
      ;;
   driver )
      driver
      ;;
   dataset )
      dataset
      ;;
   + )
      Help
      ;;
   * )
      Help
      ;;
esac
