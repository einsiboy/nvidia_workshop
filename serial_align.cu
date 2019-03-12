//#include "cuda_runtime.h"
#include <cstdio>
#include <random>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <algorithm> 
#include <chrono>
#include <iostream>
#include <limits>
#include <cctype>
#include <algorithm>    // std::max



/* Cost Matrix - overkill for this exercise, so simpler -1, +1 being used
	  C  G  A  T
	  C 9  -5 -3 0
	  G -5 7  -1 -3
	  A -3 -1 10 -4
	  T 0  -3 -4 8
*/
//int cost_index[16] = { 9,-5,-3,0,-5,7,-1,-3,-3,-1,10,-4,0,-3,-4,8 };
// The cost of aligning each character to every other
int cost_index[16] = { 1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1 };
// The length of the reference genome which we are looking to align to
const int LENGTH_REFERENCE = 3000;
// The length of the snippet that we will try to align
const int LENGTH_SEQUENCE = 2000;
// The number of sequences that we will align
const int NUM_SEQUENCES = 5;

// Look-up table to convert ACGT to 0-3 number
int ascii_to_index[128];
// Look-up table to convert 0-3 index back to ACGT
char index_to_ascii[4];
// allocate memory for the reference sequence
char reference_string[LENGTH_REFERENCE + 1];
//allocate array for each sequence we are going to align
char* sequence_strings[NUM_SEQUENCES];

//--------------------------------------------
// Not need to change this part of the code except for any GPU memory allocation
__device__
int max(int a, int b, int c) {
	return a > b ? (a > c ? a : c) : (b > c ? b : c);
}


// returns the cost of aligning two characters
__device__
int getCost(char a, char b, int* cost_idx, int* ascii_to_idx) {
	return cost_idx[ascii_to_idx[a] + ascii_to_idx[b] * 4];
}

// Initialise the reference sequence with random characaters
void initReference()
{
    // Initialise the look-up tables
	ascii_to_index[(int)'C'] = 0;
	ascii_to_index[(int)'G'] = 1;
	ascii_to_index[(int)'A'] = 2;
	ascii_to_index[(int)'T'] = 3;
	index_to_ascii[0] = 'C';
	index_to_ascii[1] = 'G';
	index_to_ascii[2] = 'A';
	index_to_ascii[3] = 'T';

	for (int i = 0; i < LENGTH_REFERENCE; i++)
	{
		int ir = rand() % 4;
		reference_string[i] = index_to_ascii[ir];
	}
	reference_string[LENGTH_REFERENCE] = '\0';

}

void initSequences()
{
	int ioffset;
	int subs = 0;
	int ins = 0;
	int dels = 0;
	int max_ix = 0;
	
	for (int ix = 0; ix < NUM_SEQUENCES; ix++)
	{
		sequence_strings[ix] = (char*)malloc(sizeof(char) * LENGTH_SEQUENCE * 2 + 1);
		
		int ref_offset = rand() % (LENGTH_REFERENCE - LENGTH_SEQUENCE);
		ioffset = 0;

		for (int i = ref_offset; i < ref_offset+ LENGTH_SEQUENCE; i++)
		{
			int i_rand = rand() % 1000;

			if (i_rand < 22)
			{
				/* insertion of random length < 5 */
				int i_len = rand() % 4 + 1;
				for (int j = 0; j < i_len; j++)
				{
					sequence_strings[ix][ioffset] = index_to_ascii[rand() % 4];
					
					ioffset++;
					max_ix++;
					ins++;
				}
				sequence_strings[ix][ioffset] = reference_string[i];

				ioffset++;
				max_ix++;
			}
			else if (i_rand < 44)
			{
				/* substitution */
				int inew = rand() % 3;

				switch (reference_string[i])
				{
				case 'A':
					sequence_strings[ix][ioffset] = index_to_ascii[inew == 2 ? 3 : inew];
					break;
				case 'T':
					sequence_strings[ix][ioffset] = index_to_ascii[inew];
					break;
				case 'C':
					sequence_strings[ix][ioffset] = index_to_ascii[inew + 1];
					break;
				case 'G':
					sequence_strings[ix][ioffset] = index_to_ascii[inew == 1 ? 0 : inew];
					break;
				}


				ioffset++;
				subs++;
				max_ix++;
			}
			else if (i_rand < 66)
			{

				/* deletion */
				dels++;
				max_ix++;
			}
			else
			{
				sequence_strings[ix][ioffset] = reference_string[i];

				ioffset++;
				max_ix++;
			}

			
		}

		sequence_strings[ix][ioffset] = '\0';
		sequence_strings[ix] = (char*)realloc(sequence_strings[ix], sizeof(char) * (strlen(sequence_strings[ix]) + 1));

        std::cout << "Sequence " << ix + 1 << ": ";
		std::cout << subs << " subs, ";
		std::cout << dels << " dels, ";
		std::cout << ins << " ins" << "\n";
	}
}
// End of part that you should not optimise
//--------------------------------------------


__global__
void computeCost(char* seq, char* ref, int len_seq, int len_ref, int*
buf, int* cost, int *cost_idx, int* ascii_to_idx){

	const int cost_del = -1;

	int* row_1;
	int* row_2;
	int* row_tmp;

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

//	printf("idx %d\n", idx);

	for (int t = idx; t < len_ref - len_seq; t += stride){

		row_1 = &buf[2 * idx * (len_seq + 1)];
		row_2 = &row_1[len_seq + 1];

		for (int i = 0; i <= len_seq; ++i) row_1[i] = cost_del * i;

		for (int i = 1; i <= len_seq; ++i){

			row_2[0] = cost_del * i;

			for (int j = 1; j <= len_seq; ++j) row_2[j] =
													   max(row_2[j-1] + cost_del, row_1[j] + cost_del, row_1[j-1] +
																									   getCost(seq[i-1], ref[t+j-1], cost_idx, ascii_to_idx));

			row_tmp = row_1;
			row_1 = row_2;
			row_2 = row_tmp;
		}

		cost[t] = row_1[len_seq];

        // printf("%d %d\n" , idx, costs[idx]);
	}
}

int main()
{
    // Initilaise timer
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();


    // You may need to change the memory allocation in these functions
	initReference();
	initSequences();

    // allocate cuda
    int *d_ascii_to_index;
    int *d_cost_index;
    //char *d_index_to_ascii;
    char *d_reference_string;
    char *d_sequence_strings[NUM_SEQUENCES];

    cudaMalloc(&d_ascii_to_index, 128*sizeof(int));
    cudaMalloc(&d_cost_index, 16*sizeof(int));

    cudaMalloc(&d_reference_string, (LENGTH_REFERENCE + 1)*sizeof(char));
    for (int i = 0; i < NUM_SEQUENCES; i++){
		cudaMalloc(&d_sequence_strings[i], sizeof(char) * LENGTH_SEQUENCE * 2 + 1);
		cudaMemcpy(d_sequence_strings[i], sequence_strings[i], sizeof(char) * LENGTH_SEQUENCE * 2 + 1, cudaMemcpyHostToDevice);
	}

    //cudaMemcpy( dev_b, b, size*sizeof(float),
	//                              cudaMemcpyHostToDevice

    cudaMemcpy(d_ascii_to_index, ascii_to_index, 128*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cost_index, cost_index, 16*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_reference_string, reference_string, (LENGTH_REFERENCE + 1)*sizeof(char), cudaMemcpyHostToDevice);

	int deviceId;
	cudaDeviceProp d_prop;

	cudaGetDevice(&deviceId);
	cudaGetDeviceProperties(&d_prop, deviceId);

	int threads = 128;
	int block_size = ((LENGTH_REFERENCE - LENGTH_SEQUENCE + threads - 1) / threads) * threads;

    block_size = ((block_size + d_prop.multiProcessorCount - 1) / d_prop.multiProcessorCount);



    int *costs_seq[NUM_SEQUENCES];
    int *d_costs_seq[NUM_SEQUENCES];
    int *d_buf_seq[NUM_SEQUENCES];



    cudaStream_t streams[NUM_SEQUENCES];




    for (int i = 0; i < NUM_SEQUENCES; ++i){

        cudaMallocHost(&costs_seq[i], (LENGTH_REFERENCE - LENGTH_SEQUENCE) * sizeof(int));

        cudaMalloc(&d_costs_seq[i], (LENGTH_REFERENCE - LENGTH_SEQUENCE) * sizeof(int));
        cudaMalloc(&d_buf_seq[i],  2 * block_size * threads * (LENGTH_SEQUENCE+1) * sizeof(int));
        cudaStreamCreate(&streams[i]);

    }


	for (int i = 0; i < NUM_SEQUENCES; ++i)
	{
		std::cout << "Computing alignment score for sequence " << i + 1 << "\n";


        computeCost<<<block_size, threads, 0, streams[i]>>>(d_sequence_strings[i], d_reference_string,
				LENGTH_SEQUENCE, LENGTH_REFERENCE, d_buf_seq[i], d_costs_seq[i], d_cost_index, d_ascii_to_index);


		cudaMemcpyAsync(costs_seq[i], d_costs_seq[i], sizeof(int) * (LENGTH_REFERENCE - LENGTH_SEQUENCE), cudaMemcpyDeviceToHost, streams[i]);

    }

	cudaDeviceSynchronize();

	for (int i = 0; i < NUM_SEQUENCES; ++i){

        int max_cost = 0;
        int j_max = 0;

        for (int j = 0; j < (LENGTH_REFERENCE - LENGTH_SEQUENCE); ++j){
            if(costs_seq[i][j] > max_cost){
                max_cost = costs_seq[i][j];
                j_max = j;
            }
        }

        std::cout << "Optimal cost of " << max_cost << " found at offset " << j_max << "\n";



        cudaFree(d_buf_seq[i]);
        cudaFree(d_costs_seq[i]);
        cudaFree(costs_seq[i]);
        cudaStreamDestroy(streams[i]);
    }



	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> time_span = t2 - t1;

	std::cout << "It took " << time_span.count() << " milliseconds.\n";
	std::cout << "Press ENTER to continue... ";
	std::cin.ignore(std::numeric_limits <std::streamsize> ::max(), '\n');

	cudaFree(d_cost_index);
//	cudaFree(d_buf);
	cudaFree(d_ascii_to_index);
	cudaFree(d_reference_string);
//	cudaFree(d_costs);
	for (int i = 0; i < NUM_SEQUENCES; i++){
		cudaFree(d_sequence_strings[i]);
	}


	return 0;
}

