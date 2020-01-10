cp ./kademlia_RevsComp/SETTINGS.h ./

/opt/cuda/7.5/bin/nvcc -arch=sm_30 -rdc=true \
	./main.cu \
	./kernels.cu \
	./queues.cu \
	./random.cu \
	./nelder_mead_3d.cu \
\
	./kademlia_RevsComp/model.cu \
	./kademlia_RevsComp/Event.cu
