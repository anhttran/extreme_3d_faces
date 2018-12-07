FROM mxnet/python:1.1.0_nccl
WORKDIR /app
ADD . /app

RUN apt-get update
RUN apt-get install -y libhdf5-serial-dev libboost-all-dev nano cmake libosmesa6-dev freeglut3-dev
#RUN apt install zip
RUN wget http://dlib.net/files/dlib-19.6.tar.bz2; \
	tar xvf dlib-19.6.tar.bz2; \
	cd dlib-19.6/; \
	mkdir build; \
	cd build; \
	cmake ..; \
	cmake --build . --config Release; \
	make install; \
	cd ..

RUN pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl
RUN pip install opencv-python torchvision scikit-image cvbase pandas mmdnn dlib

RUN mkdir build; \
	cd build; \
	cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=../demoCode ..; \
	make; \
	make install; \
	cd ..

WORKDIR /app/demoCode
EXPOSE 80

ENV NAME World

CMD ["python", "testBatchModel.py"]
