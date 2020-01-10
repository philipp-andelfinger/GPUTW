cp ./phold_ConsSync/SETTINGS.h ./

/opt/cuda/7.5/bin/nvcc -arch=sm_30 -rdc=true \
	./main.cu \
	./kernels.cu \
	./queues.cu \
	./random.cu \
	./nelder_mead_3d.cu \
\
	./phold_ConsSync/model.cu \
	./phold_ConsSync/Event.cu
