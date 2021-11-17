#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
int main(int argc, char** argv)
{

	fprintf(stdout, "CUDA Device Query\n");

	int deviceCount = 0;

	// Testa se existem dispositivos compatíveis com Cuda
	cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaGetDeviceCount retornou código: %d\n -> %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
		
	}

	// A função retorna 0 caso não exista hardware que suporte cuda.
	if (deviceCount == 0)
	{
		fprintf(stdout, "Não há dispositivo compatível com CUDA\n");
	}
	else
	{
		fprintf(stdout, "Detectado %d dispositivo(s) CUDA\n", deviceCount);
	}

	int dev, driverVersion = 0, runtimeVersion = 0;

	for (dev = 0; dev < deviceCount; ++dev)
	{
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		fprintf(stdout, "\nDevice %d: \"%s\"\n", dev, deviceProp.name);

		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		fprintf(stdout, "CUDA Driver Version / Runtime Version %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
		fprintf(stdout, "CUDA Capability Major/Minor version number: %d.%d\n", deviceProp.major, deviceProp.minor);
		fprintf(stdout, "QTD Multiprocessors: %d \n", deviceProp.multiProcessorCount);
		fprintf(stdout, "Total constant memory:%zu bytes\n", deviceProp.totalConstMem);
		fprintf(stdout, "Total shared memory per block:%zu bytes\n", deviceProp.sharedMemPerBlock);
		fprintf(stdout, "Shared memory per multiprocessor:%zu bytes\n", deviceProp.sharedMemPerMultiprocessor);
		fprintf(stdout, "Number of registers available per block:%d\n", deviceProp.regsPerBlock);
		fprintf(stdout, "maxThreadsPerMultiProcessor:%d\n", deviceProp.maxThreadsPerMultiProcessor);
		fprintf(stdout, "maxThreadsPerBlock:%d\n", deviceProp.maxThreadsPerBlock);
		fprintf(stdout, "Max thread dimensions : (% d, % d, % d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
		fprintf(stdout, "Max grid dimensions:  (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);

	
			
			
			
	}

	return 0;
}