default: cudaMM
cudaMM: cudaMM.cu
	nvcc -O3 -Xcompiler "-Wall" -g -G cudaMM.cu -o cudaMM
clean:
	rm -f cudaMM