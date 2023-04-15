# This is the build file for the docker. Note this should be run from the
# parent directory for the necessary files to be available

.PHONY: clean build run

DIR := ${CURDIR}

build:
	docker build -t peabody124/posepipe -f ./docker/Dockerfile .

run:
	docker run  -t --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=5 -v  /datajoint_external:/datajoint_external  peabody124/posepipe

#-v ${DIR}/data:/data -v ${DIR}/logs:/logs	
