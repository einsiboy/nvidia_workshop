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

int cost_index[16] = { 1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1 };
#define LENGTH_REFERENCE 40000
#define LENGTH_SEQUENCE 500
#define NUM_SEQUENCES 2

int ascii_to_index[128];
char index_to_ascii[4];
char* reference_string;
char* sequence_strings;
int* offset_cost;
int* matrix;
int seq_length[NUM_SEQUENCES];

int max(int a, int b, int c) {
	return a > b ? (a > c ? a : c) : (b > c ? b : c);
}

int get1Dindex(int i, int j) {
	return j * LENGTH_SEQUENCE + i;
}

int get1Dindex(int seq_pos, int ref_pos, int len_seq) {
	return ref_pos * len_seq + seq_pos;
}

int getCost(char a, char b) {
	return cost_index[ascii_to_index[a] + ascii_to_index[b] * 4];
}

void initIndexes()
{
	ascii_to_index[(int)'C'] = 0;
	ascii_to_index[(int)'G'] = 1;
	ascii_to_index[(int)'A'] = 2;
	ascii_to_index[(int)'T'] = 3;
}

void initReference()
{
	reference_string = (char*)malloc((int)(LENGTH_REFERENCE + 1) * sizeof(char));

	index_to_ascii[0] = 'C';
	index_to_ascii[1] = 'G';
	index_to_ascii[2] = 'A';
	index_to_ascii[3] = 'T';

	/* Random string */
	for (int i = 0; i < LENGTH_REFERENCE; i++)
	{
		int ir = rand() % 4;
		reference_string[i] = index_to_ascii[ir];
	}
	reference_string[LENGTH_REFERENCE] = '\0';
}

void initSequences()
{
	long total_offset = 0;
	long total_matrix_size = 0;
	std::vector<char> ref;
	std::vector<char> seq;
	char sequences[LENGTH_SEQUENCE * NUM_SEQUENCES * 2];

	for (int ix = 0; ix < NUM_SEQUENCES; ix++)
	{
		int ref_offset = rand() % (LENGTH_REFERENCE - LENGTH_SEQUENCE);
		int subs = 0;
		int ins = 0;
		int dels = 0;
		int length = 0;

		std::cout << "Offset for sequence " << ix + 1 << " = " << ref_offset << "\n";

		for (int i = ref_offset; i < ref_offset + LENGTH_SEQUENCE; i++)
		{
			int i_rand = rand() % 1000;

			if (i_rand < 22)
			{
				/* insertion of random length < 5 */
				int i_len = rand() % 4 + 1;
				for (int j = 0; j < i_len; j++)
				{
					sequences[length + total_offset] = index_to_ascii[rand() % 4];
					ref.push_back('+');
					seq.push_back(sequences[length + total_offset]);
					length++;
					ins++;
				}
				sequences[length + total_offset] = reference_string[i];
				ref.push_back(reference_string[i]);
				seq.push_back(reference_string[i]);
				length++;
			}
			else if (i_rand < 44)
			{
				/* substitution */

				int inew = rand() % 3;

				/* Lower case denotes substitution */
				ref.push_back(std::tolower(reference_string[i]));

				switch (reference_string[i])
				{
				case 'A':
					sequences[length + total_offset] = index_to_ascii[inew == 2 ? 3 : inew];
					break;
				case 'T':
					sequences[length + total_offset] = index_to_ascii[inew];
					break;
				case 'C':
					sequences[length + total_offset] = index_to_ascii[inew + 1];
					break;
				case 'G':
					sequences[length + total_offset] = index_to_ascii[inew == 1 ? 0 : inew];
					break;
				}

				seq.push_back(std::tolower(sequences[length + total_offset]));

				length++;
				subs++;
			}
			else if (i_rand < 66)
			{
				/* deletion */
				ref.push_back(reference_string[i]);
				seq.push_back('_');
				dels++;
			}
			else
			{
				sequences[length + total_offset] = reference_string[i];
				ref.push_back(reference_string[i]);
				seq.push_back(reference_string[i]);
				length++;
			}


		}

		seq_length[ix] = length;
		total_offset += length;
		total_matrix_size += (length + 1) * (LENGTH_REFERENCE - length) * 2;

		ref.push_back('\0');
		seq.push_back('\0');


		std::cout << "Sequence " << ix + 1 << ": ";
		std::cout << subs << " subs, ";
		std::cout << dels << " dels, ";
		std::cout << ins << " ins" << "\n";

		std::cout << ref.data() << "\n";
		std::cout << seq.data() << "\n";
		ref.clear();
		seq.clear();
	}


	sequence_strings = (char*)malloc(sizeof(char) * total_offset);
	::memcpy(sequence_strings, sequences, total_offset);

}

/* Compute the alignment cost using Needleman-Wunch */
int computeCost(int* matrix, char* a, char* b, int len_a, int len_b) //a = ref, b = seq
{
	const int cost_del = -1;


	for (int i = 0; i <= len_a; i++)
		matrix[i] = cost_del * i;

	for (int j = 0; j <= len_b; j++)
		matrix[j * (len_a + 1)] = cost_del * j;

	for (int j = 1; j <= len_b; j++)
	{

		for (int i = 1; i <= len_a; i++)
		{
			int cost_m = getCost(a[i - 1], b[j - 1]) + matrix[get1Dindex(i - 1, j - 1)];
			int cost_i = matrix[get1Dindex(i - 1, j)] + cost_del;
			int cost_d = matrix[get1Dindex(i, j - 1)] + cost_del;

			matrix[get1Dindex(i, j)] = max(cost_m, cost_i, cost_d);

		}

	}
	int cost = matrix[(len_b + 1) * (len_a + 1) - 1];

	return cost;
}


void cudaAlign(char* seq, int len_seq, int* offset_cost, char* ref, int* matrix)
{
	int matrix_offset = 0;

	for (int i = 0; i < LENGTH_REFERENCE - len_seq; i++)
	{
		/* Offset is given by: global offset + size of current matrix x local offset */
		matrix_offset = (len_seq) * i;
		offset_cost[i] = computeCost(matrix + matrix_offset, ref + i, seq, LENGTH_SEQUENCE, len_seq);
	}
}

//innermost function - executed by one thread
void calculate_column(int* column_values, int col_height, char ref, char* seq, int col)
{
	int cost_del = -1;
	int tr = -col - 1;
	int tl = -col;

	for (int i = 0; i < col_height; i++)
	{
		char s = seq[i];
		int cost_m = getCost(s, ref) + tl;
		tl = column_values[i];
		column_values[i] = max(tr + cost_del, cost_m , column_values[i] + cost_del);
		tr = column_values[i];

	}
}

void calculate_offset_cost(int* column_values, int col_height, char* ref, char* seq, int ref_offset)
{
	for (int i = 0; i < col_height; i++)
	{
		calculate_column(column_values, col_height, ref[ref_offset + i], seq, i);
	}

}

//outermost function that computes the optimal alignment
int calculate_alignment(char* ref, char* seq, int len_seq)
{
	int matrix_size = (LENGTH_REFERENCE - len_seq) * (len_seq + 1);
	int* matrix = (int*)malloc(sizeof(int) * matrix_size);
	for (int ref_pos = 0; ref_pos < LENGTH_REFERENCE - len_seq; ref_pos++)
	{
		//initialise whole grid (in parallel)
		for (int seq_pos = 0; seq_pos < len_seq; seq_pos++)
		{
			// set each colum to 0, -1, -2, etc...
			int index = get1Dindex(seq_pos, ref_pos, len_seq);
			matrix[index] = -(seq_pos + 1);
		}
	}

	for (int ref_pos = 0; ref_pos < LENGTH_REFERENCE - len_seq; ref_pos++)
	{
		//for each possible alignment compute the cost for the current column
		int matrix_offset = get1Dindex(0, ref_pos, len_seq);
		calculate_offset_cost(matrix + matrix_offset, len_seq, ref, seq, ref_pos);

	}

	int max_cost = matrix[len_seq - 1];
	int offset = 0;

	for (int ref_pos = 1; ref_pos < LENGTH_REFERENCE - len_seq; ref_pos++)
	{
		const int index = get1Dindex(0, ref_pos, len_seq);
		const int cost = matrix[index + len_seq - 1];
		if (cost > max_cost) {
			max_cost = cost;
			offset = ref_pos;
		}
	}

	free(matrix);
	return offset;

}


int main()
{
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	long offset = 0;
	int num_blocks = 1;
	int num_threads = 1;
	int seq_offset = 0;

	initIndexes();
	initReference();
	initSequences();

	for (int i = 0; i < NUM_SEQUENCES; i++)
	{
		offset = calculate_alignment(reference_string, sequence_strings + seq_offset, seq_length[i]);
		seq_offset += seq_length[i];

		std::cout << "Optimal cost of " << 0 << " found at offset " << offset << "\n";
	}

	free(sequence_strings);
	free(reference_string);

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();


	std::chrono::duration<double, std::milli> time_span = t2 - t1;

	std::cout << "It took " << time_span.count() << " milliseconds.\n";
	//std::cout << "Press ENTER to continue... ";
	//std::cin.ignore(std::numeric_limits <std::streamsize> ::max(), '\n');


	return 0;
}