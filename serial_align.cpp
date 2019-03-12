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
inline int max(int a, int b, int c) {
	return a > b ? (a > c ? a : c) : (b > c ? b : c);
}

// Get the index of a specific character in the overall array of sequences
// which are concatenated 
inline int get1Dindex(int i, int j) {
	return j * (LENGTH_SEQUENCE +1) + i;
}

// returns the cost of aligning two characters
inline int getCost(char a, char b) {
	return cost_index[ascii_to_index[a] + ascii_to_index[b] * 4];
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


//  This is the part that you need to optimise
/* Compute the alignment cost using Needleman-Wunch */
int computeCost(char* a, char* b, int len_a, int len_b)
{
	int* matrix;
	const int cost_del = -1;
	// Allocate the cost matrix
	matrix = (int*)malloc(sizeof(int) * (len_a +1) * (len_b+1));
	
	// Initialise the values in the first row and column
	for (int i = 0; i <= len_a; i++)
		matrix[i] = cost_del * i;

	for (int j = 0; j <= len_b; j++)
		matrix[j * (len_a + 1) ] = cost_del  * j;

	for (int j = 1; j <= len_b; j++)
	{
		
		for (int i = 1; i <= len_a; i++)
		{
			int cost_m = getCost(a[i-1],b[j-1]) + matrix[get1Dindex(i-1,j-1)];
			int cost_i = matrix[get1Dindex(i - 1, j)] + cost_del;
			int cost_d = matrix[get1Dindex(i, j - 1)] + cost_del;

			matrix[get1Dindex(i, j)] = max(cost_m, cost_i, cost_d);
			/* uncomment to output the cost matrix values */
			//std::cout << max(cost_m, cost_i, cost_d)<< ',';

		}

	}
	int cost = matrix[(len_b + 1) * (len_a + 1)-1];

	free(matrix);
	
	return cost;
}

int main()
{
    // Initilaise timer
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	int max_cost = 0;
    int j_max = 0;

    // You may need to change the memory allocation in these functions
	initReference();
	initSequences();

	for (int i = 0; i < NUM_SEQUENCES; i++)
	{
        std::cout << "Computing alignment score for sequence " << i + 1 << "\n";
		max_cost = 0;
                j_max = 0;
		for (int j = 0; j < LENGTH_REFERENCE - LENGTH_SEQUENCE; j++)
		{
			
			int cost = computeCost(reference_string+j, sequence_strings[i], LENGTH_SEQUENCE, strlen(sequence_strings[i]));
			if (cost > max_cost) {
				max_cost = cost;
                j_max = j;
			}
		}
                std::cout << "Optimal cost of " << max_cost << " found at offset " << j_max << "\n";
		free(sequence_strings[i]);
	}

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> time_span = t2 - t1;

	std::cout << "It took " << time_span.count() << " milliseconds.\n";
	std::cout << "Press ENTER to continue... ";
	std::cin.ignore(std::numeric_limits <std::streamsize> ::max(), '\n');


	return 0;
}
