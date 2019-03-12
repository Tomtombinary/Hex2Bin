
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>

#define TEXT_PROLOG_SIZE 8
#define TEXT_LINE_SIZE 58
#define BYTES_PER_LINE 16

#define NBLOCK 1024

/*
 * one block per text line
 * one thread per hexadecimal bytes in line
 */
__global__ void hexConvertKernel(unsigned char* result, char* input)
{
	unsigned char d0, d1;
	int lineId = blockIdx.x;
	int byteId = threadIdx.x;
	
	d0 = input[lineId * TEXT_LINE_SIZE + TEXT_PROLOG_SIZE + byteId * 3 + 1]; // * 3 because 1 escape + 2 hex digits
	d1 = input[lineId * TEXT_LINE_SIZE + TEXT_PROLOG_SIZE + byteId * 3 + 2];

	/*
	 * convert char digits to binary value in cuda core
	 */
	if (d0 == '?') d0 = 0;
	else if (d0 <= '9') d0 -= '0';
	else if (d0 <= 'F') d0 = d0 - 'A' + 10;
	
	if (d1 == '?') d1 = 0;
	else if (d1 <= '9') d1 -= '0';
	else if (d1 <= 'F') d1 = d1 - 'A' + 10;

	result[lineId * BYTES_PER_LINE + byteId] = d0 << 4 | d1;
}

/*
 * check if digit is a hexadecimal digit (lower char not allowed)
 */
bool isHexDigit(char digit)
{
	return ((digit >= '0' && digit <= '9') || (digit >= 'A' && digit <= 'F'));
}

/*
 * convert hexadecimal digit to 4 bits integer (lower char not allowed)
 */
unsigned char hexDigit2Byte(char digit)
{
	if (digit == '?') digit = 0;
	else if (digit <= '9') digit -= '0';
	else if (digit <= 'F') digit = digit - 'A' + 10;
	return digit;
}


long getFileSize(FILE* fd)
{
	long size;
	fseek(fd, 0, SEEK_END);
	size = ftell(fd);
	fseek(fd, 0, SEEK_SET);
	return size;
}

/**
 *  convert hexdecimal content (like above) with the CPU into binary format (by removing address and convert hexadecimals digits)
 *  00401000 56 8D 44 24 08 50 8B F1 E8 1C 1B 00 00 C7 06 08
 *  00401010 BB 42 00 8B C6 5E C2 04 00 CC CC CC CC CC CC CC
 *  00401020 C7 01 08 BB 42 00 E9 26 1C 00 00 CC CC CC CC CC 
 *  usr_in : user allocated buffer with hexadecimal content
 *  input_size : size of user allocated buffer
 *  usr_out: output allocated buffer to store binary content
 */
size_t CPU_Hex2Bin(char* usr_in,size_t input_size, unsigned char* usr_out)
{
	int nRound = input_size / TEXT_LINE_SIZE;
	int remainingBytes = input_size % TEXT_LINE_SIZE;
	int nHexBytes = (remainingBytes - TEXT_PROLOG_SIZE) / 3;

	/* convert remaining lines */
	for (int i = 0; i < nRound; i++)
	{
		for (int j = 0; j < BYTES_PER_LINE; j++)
		{
			unsigned char d0 = hexDigit2Byte(usr_in[i * TEXT_LINE_SIZE + TEXT_PROLOG_SIZE + j * 3 + 1]);
			unsigned char d1 = hexDigit2Byte(usr_in[i * TEXT_LINE_SIZE + TEXT_PROLOG_SIZE + j * 3 + 2]);

			usr_out[i * BYTES_PER_LINE + j] = d0 << 4 | d1;
		}
	}

	/* convert last partial line */
	for (int i = 0; i < nHexBytes; i++)
	{
		unsigned char d0 = hexDigit2Byte(usr_in[nRound * TEXT_LINE_SIZE + TEXT_PROLOG_SIZE + i * 3 + 1]);
		unsigned char d1 = hexDigit2Byte(usr_in[nRound * TEXT_LINE_SIZE + TEXT_PROLOG_SIZE + i * 3 + 2]);

		usr_out[nRound * BYTES_PER_LINE + i] = d0 << 4 | d1;
	}

	return nRound * BYTES_PER_LINE + nHexBytes;
}


/**
 *  convert hexdecimal content (like above) with the GPU into binary format (by removing address and convert hexadecimals digits)
 *  00401000 56 8D 44 24 08 50 8B F1 E8 1C 1B 00 00 C7 06 08
 *  00401010 BB 42 00 8B C6 5E C2 04 00 CC CC CC CC CC CC CC
 *  00401020 C7 01 08 BB 42 00 E9 26 1C 00 00 CC CC CC CC CC
 *  filename_in : path to hexadecimal file
 *  filename_out : path to output file
 *  
 *  file is divided in big chunk of size (NBLOCK * TEXT_LINE_SIZE)
 *  The GPU process the big chunk with NBLOCK blocks and BYTES_PER_LINE (16) threads per block.
 *  The remaining lines is processed with the CPU.
 */
cudaError_t CUDA_HexFile2Bin(char* filename_in,char* filename_out)
{
	// indicate error with GPU
	cudaError_t cudaStatus;
	// input and output file descriptor
	FILE *in = NULL, *out = NULL;
	// device input buffer and user input buffer
	char* dev_in = NULL, *usr_in = NULL;
	// device output buffer and user output buffer
	unsigned char* dev_out = NULL, *usr_out = NULL;

	// open input file in read mode
	in = fopen(filename_in, "rb");

	if (in == NULL)
	{
		perror("fopen input file");
		goto Error;
	}

	// open output file in write mode
	out = fopen(filename_out, "wb");

	if (out == NULL)
	{
		perror("fopen output file");
		goto Error;
	}


	long file_size = getFileSize(in);

	printf("[i] file size : %d bytes\n", file_size);
	printf("[i] block size: %d bytes\n", NBLOCK * TEXT_LINE_SIZE);

	// divide the file into big chunk (NBLOCK * TEXT_LINE_SIZE)
	unsigned int nRound = file_size / (NBLOCK * TEXT_LINE_SIZE);
	unsigned int remainingBytes = file_size % (NBLOCK * TEXT_LINE_SIZE);

	printf("[i] nRound : %d\n", nRound);
	printf("[i] remainingBytes : %d\n", remainingBytes);

	// allocate user buffer for one big chunk
	usr_in  = (char*)malloc(NBLOCK * TEXT_LINE_SIZE);
	// allocate output buffer for one big chunk
	usr_out = (unsigned char*)malloc(NBLOCK * BYTES_PER_LINE);

	// Allocate GPU buffers for two vectors (one input, one output).
	cudaStatus = cudaMalloc((void**)&dev_in, NBLOCK * TEXT_LINE_SIZE);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_out, NBLOCK * BYTES_PER_LINE);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	for (int i = 0; i < nRound; i++)
	{
		fread(usr_in, TEXT_LINE_SIZE, NBLOCK, in);

		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_in, usr_in, NBLOCK * TEXT_LINE_SIZE, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		// Launch a kernel on the GPU with one thread for each element.
		hexConvertKernel <<<NBLOCK, BYTES_PER_LINE >> > (dev_out, dev_in);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(usr_out, dev_out, NBLOCK * BYTES_PER_LINE, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		fwrite(usr_out, BYTES_PER_LINE, NBLOCK, out);
	}

	fread(usr_in,remainingBytes,1,in);
	size_t nbytes = CPU_Hex2Bin(usr_in, remainingBytes, usr_out);
	fwrite(usr_out, nbytes, 1, out);

Error:
	cudaFree(dev_in);
	cudaFree(dev_out);
	free(usr_out);
	free(usr_in);
	fclose(out);
	fclose(in);

	return cudaStatus;
}

int main(int argc,char** argv)
{
	cudaError_t cudaStatus;

	if (argc != 3)
	{
		printf("Usage: %s <hexadecimal input file> <output file>\n",argv[0]);
		exit(1);
	}

	cudaStatus = CUDA_HexFile2Bin(argv[1], argv[2]);
	// cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
