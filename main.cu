/*  This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#include "main.h"
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernels.h"
//#include "nelder_mead.h"
#include "nelder_mead_3d.h"
#include "statistics.cu"

/* Global variables*/
__device__ uint		g_n_nodes;
__device__ uint		g_n_lps;
__device__ uint		g_nodes_per_lp;

static	uint	nodes_per_lp;
static	uint	n_nodes;
static	uint	n_lps;

static	uint	events_per_node;
static	uint	states_per_node;
static	uint	antimsgs_per_node;

static	float	committed_events_threshold;
static	float	inactive_lps_percent;
static	int	window_size;

static	uint	threads_per_block;
static	uint	n_blocks;

/* Private functions */
char*	get_time();
size_t	get_free_memory();
void	print_size(size_t size);
uint	get_number_blocks(uint n_threads);
int	get_gvt(int *d_ts_temp);
char	change_nodes_per_lp(uint target_value_log, char *d_can_split);
void	merge_lps();
char	split_lps(char *d_can_split);

int main() {

// n -> number of nodes
for (int n = 1048576; n >= 16384; n /= 2) {

	float results_avg[50];
	float results_min[50];
	float results_max[50];
	int n_tests = 3;

// r -> number of runs
for (int r = 0; r < n_tests; r++) {

	float	initial_p1[] = { 0.6,    1,  0.6,  0.6};

/* For PHOLD model with lambda = 1 or 100 */
//	float	initial_p2[] = {1000, 1000, 5000, 1000};
/* For PHOLD model with lambda = 10000 */
	float	initial_p2[] = {1000, 1000, 50000, 1000};
/* For Kademlia models */	
//	float	initial_p2[] = { 100,  100, 1000,  100};

	float	initial_p3[] = { 0.5,  0.5,  0.5,  4.5};

	nodes_per_lp = pow(2, (int) initial_p3[0]);
	n_nodes = n;
	n_lps = n_nodes / nodes_per_lp;

/* For PHOLD models */
	events_per_node = 50;
	states_per_node = 30;
	antimsgs_per_node = 30;

/* For Kademlia models */
//	events_per_node = 30;
//	states_per_node = 15;
//	antimsgs_per_node = 30;

	committed_events_threshold = (float) 1000 * 1000000;
	inactive_lps_percent = initial_p1[0];
	window_size = initial_p2[0];

/* For PHOLD models
 * Parameters are: population, lookahead, mean
 */
	int model_params[] = {n, 1000, 10000};
	uint n_params = 3;

/* For Kademlia models */
//	int model_params[] = {600000};
//	uint n_params = 1;

#if (OPTM_SYNC == 0)
	states_per_node = 0;
	antimsgs_per_node = 0;
#endif

#if (ALLOW_ME == 0)
	inactive_lps_percent = 0;
#endif

	threads_per_block = 256;
	n_blocks = get_number_blocks(n_lps);

	int	*d_model_params;
	int	*d_lookahead;
	int	*d_ts_temp;
	uint	*d_n_events_cmt;
	uint	*d_inac_1, *d_inac_2, *d_inac_3;
	uint	*d_inac_4, *d_inac_5, *d_inac_6;
	char	*d_rollback_performed;
	char	*d_can_split;

	int	h_lookahead;
	uint	h_n_events_cmt;
	uint	h_inac_1, h_inac_2, h_inac_3;
	uint	h_inac_4, h_inac_5, h_inac_6;
	char 	h_rollback_performed;

	cudaDeviceReset();

	// cudaMalloc
	if (
	cudaMalloc(&d_model_params, sizeof(int) * n_params) != cudaSuccess ||
	cudaMalloc(&d_lookahead,	sizeof(int))	!= cudaSuccess ||
	cudaMalloc(&d_ts_temp,	sizeof(int) * n_blocks) != cudaSuccess ||
	cudaMalloc(&d_n_events_cmt,	sizeof(uint))	!= cudaSuccess ||
	cudaMalloc(&d_inac_1,		sizeof(uint))	!= cudaSuccess ||
	cudaMalloc(&d_inac_2,		sizeof(uint))	!= cudaSuccess ||
	cudaMalloc(&d_inac_3,		sizeof(uint))	!= cudaSuccess ||
	cudaMalloc(&d_inac_4,		sizeof(uint))	!= cudaSuccess ||
	cudaMalloc(&d_inac_5,		sizeof(uint))	!= cudaSuccess ||
	cudaMalloc(&d_inac_6,		sizeof(uint))	!= cudaSuccess ||
	cudaMalloc(&d_can_split,	sizeof(char))	!= cudaSuccess ||
	cudaMalloc(&d_rollback_performed, sizeof(char))	!= cudaSuccess) {
		return 0;
	}

	cudaMemcpy(d_model_params, model_params, sizeof(int) * n_params,
		cudaMemcpyHostToDevice);

	h_n_events_cmt = 0;
	cudaMemcpy(d_n_events_cmt, &h_n_events_cmt, sizeof(uint),
		cudaMemcpyHostToDevice);

	h_rollback_performed = 0;
	cudaMemcpy(d_rollback_performed, &h_rollback_performed, sizeof(char),
		cudaMemcpyHostToDevice);

	char res = 0;
	res += malloc_nodes(n_nodes);
	res += malloc_queues(n_nodes,
		events_per_node, states_per_node, antimsgs_per_node);

	if (res != 2) {
		free_nodes();
		free_queues();

		printf("ERROR: Memory not enough.\n");
		return 1;
	}

	// Initialization
	kernel_set_params<<<1,1>>>(
		n_nodes, n_lps, nodes_per_lp,
		events_per_node, states_per_node, antimsgs_per_node,
		d_model_params, n_params);

	kernel_get_lookahead<<<1,1>>>(d_lookahead);
	cudaMemcpy(&h_lookahead, d_lookahead, sizeof(int),
		cudaMemcpyDeviceToHost);

	kernel_init_queues<<<n_blocks, threads_per_block>>>();
	kernel_init_nodes<<<n_blocks, threads_per_block>>>();
	kernel_sort_event_queues<<<n_blocks, threads_per_block>>>();

	// Time measurement
	cudaEvent_t start, stop, start_1, stop_1, start_2, stop_2;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&start_1);
	cudaEventCreate(&stop_1);
	cudaEventCreate(&start_2);
	cudaEventCreate(&stop_2);
	float time;

	// Measurement
	char warm_up = 0;
	int setup = 0;
	char flag_continue = 0;	

	float total_events = 0;
	float n_events_since_change = 0;

	float max_rate = 0;
	float min_rate = 1000;

	// Simulation
	cudaDeviceSynchronize();

	cudaEventRecord(start);
	cudaEventRecord(start_2);

	while (1) {
		// Get minimal timestamp of all next events
		int gvt = get_gvt(d_ts_temp);

		// Delete past events
		h_n_events_cmt = 0;
		cudaMemcpy(d_n_events_cmt, &h_n_events_cmt, sizeof(uint),
			cudaMemcpyHostToDevice);

		kernel_clean_queues<<<n_blocks, threads_per_block>>>(
			gvt, d_n_events_cmt);

		cudaMemcpy(&h_n_events_cmt, d_n_events_cmt, sizeof(uint),
			cudaMemcpyDeviceToHost);
		total_events += h_n_events_cmt;
		n_events_since_change += h_n_events_cmt;

		if (total_events > committed_events_threshold) {
			break;
		}

		// Change parameters
		cudaEventRecord(stop_2);
		cudaEventSynchronize(stop_2);
		cudaEventElapsedTime(&time, start_2, stop_2);
		float current_rate = n_events_since_change / time / 1000;

		if (warm_up == 0 && time > 200) {
			warm_up = 1;

			n_events_since_change = 0;
			cudaEventRecord(start_2);

			flag_continue = 1;
		} else if (warm_up == 1 && time > 300){
			warm_up = 0;

//			printf("%4.2f %6d %2u %10.3f %10.0f %10d",
//				inactive_lps_percent, window_size,
//				nodes_per_lp, current_rate, total_events, gvt);

			if (current_rate < min_rate) {
				min_rate = current_rate;
//				printf(" --> MIN");
			} else if (current_rate > max_rate) {
				max_rate = current_rate;
//				printf(" --> MAX");
			}

//			printf("\n");

			if (setup < 4) {
				nm_start(setup,
					initial_p1[setup], initial_p2[setup],
					initial_p3[setup], current_rate);

				setup ++;
				if (setup < 4) {
					inactive_lps_percent =
						initial_p1[setup];
					window_size =
						initial_p2[setup];
					change_nodes_per_lp(
						initial_p3[setup],
						d_can_split);

					n_events_since_change = 0;
					cudaEventRecord(start_2);
					flag_continue = 1;
				}
			}

			if (flag_continue == 0) {
				float next_p1, next_p2, next_p3;

				char flag = 0;

				while (flag == 0) {

				nm_get_next_point(
					current_rate, &next_p1,
					&next_p2, &next_p3);

				flag = 1;

				if (next_p1 < 0 || next_p1 > 1 ||
				next_p2 < 0 || next_p3 < 0) {
					flag = 0;
					current_rate = 0;
					continue;
				}

				inactive_lps_percent = next_p1;
				window_size = next_p2;
				char res=change_nodes_per_lp(next_p3, d_can_split);
				if (res == 0) {
					flag = 0;
					current_rate = 0;
					continue;
				}

				}

				n_events_since_change = 0;
				cudaEventRecord(start_2);

				flag_continue = 1;
			}
		}

		if (flag_continue == 1) {
			flag_continue = 0;
			continue;
		}

		// Handle next event
		while (1) {
			h_inac_1 = h_inac_2 = h_inac_3 = 0;
			h_inac_4 = h_inac_5 = h_inac_6 = 0;
			cudaMemcpy(d_inac_1, &h_inac_1, sizeof(uint),
				cudaMemcpyHostToDevice);
			cudaMemcpy(d_inac_2, &h_inac_2, sizeof(uint),
				cudaMemcpyHostToDevice);
			cudaMemcpy(d_inac_3, &h_inac_3, sizeof(uint),
				cudaMemcpyHostToDevice);
			cudaMemcpy(d_inac_4, &h_inac_4, sizeof(uint),
				cudaMemcpyHostToDevice);
			cudaMemcpy(d_inac_5, &h_inac_5, sizeof(uint),
				cudaMemcpyHostToDevice);
			cudaMemcpy(d_inac_6, &h_inac_6, sizeof(uint),
				cudaMemcpyHostToDevice);

#if (OPTM_SYNC == 1)
			kernel_handle_next_event<<<
				n_blocks, threads_per_block>>>(
					gvt, window_size,
					d_inac_1, d_inac_2, d_inac_3,
					d_inac_4, d_inac_5, d_inac_6);
#else		
			kernel_handle_next_event<<<
				n_blocks, threads_per_block>>>(
					gvt, h_lookahead,
					d_inac_1, d_inac_2, d_inac_3,
					d_inac_4, d_inac_5, d_inac_6);
#endif

			cudaMemcpy(&h_inac_1, d_inac_1, sizeof(uint),
				cudaMemcpyDeviceToHost);
			cudaMemcpy(&h_inac_2, d_inac_2, sizeof(uint),
				cudaMemcpyDeviceToHost);
			cudaMemcpy(&h_inac_3, d_inac_3, sizeof(uint),
				cudaMemcpyDeviceToHost);
			cudaMemcpy(&h_inac_4, d_inac_4, sizeof(uint),
				cudaMemcpyDeviceToHost);
			cudaMemcpy(&h_inac_5, d_inac_5, sizeof(uint),
				cudaMemcpyDeviceToHost);
			cudaMemcpy(&h_inac_6, d_inac_6, sizeof(uint),
				cudaMemcpyDeviceToHost);

#if (ALLOW_ME == 1)
			uint inac =
				h_inac_1 + h_inac_2 + h_inac_3 +
				h_inac_4 + h_inac_5 + h_inac_6;
			if (inac >= n_lps * inactive_lps_percent) { break; }
#else
			break;
#endif
		}

#if (OPTM_SYNC == 1)
		// Roll back
		while (1) {
			h_rollback_performed = 0;
			cudaMemcpy(d_rollback_performed, &h_rollback_performed,
				sizeof(char), cudaMemcpyHostToDevice);

			kernel_roll_back<<<n_blocks, threads_per_block>>>(
				d_rollback_performed);

			cudaMemcpy(&h_rollback_performed, d_rollback_performed,
				sizeof(char), cudaMemcpyDeviceToHost);
			if (h_rollback_performed == 0) { break; }
		}
#endif

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("FATAL ERROR: %s\n", cudaGetErrorString(err));
			return 1;
		}

		// Sort
		kernel_sort_event_queues<<<n_blocks, threads_per_block>>>();
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	float avg_rate = total_events / time / 1000;
//	printf("%8d %6.2f %6.2f %6.2f\n",
//		n, min_rate, avg_rate, max_rate);

	free_queues();
	free_nodes();

	cudaFree(d_model_params);
	cudaFree(d_lookahead);
	cudaFree(d_ts_temp);
	cudaFree(d_n_events_cmt);
	cudaFree(d_inac_1); cudaFree(d_inac_2); cudaFree(d_inac_3);
	cudaFree(d_inac_4); cudaFree(d_inac_5); cudaFree(d_inac_6);
	cudaFree(d_rollback_performed);

	results_avg[r] = avg_rate;
	results_min[r] = min_rate;
	results_max[r] = max_rate;
} // r

	float mean_avg = get_mean(n_tests, results_avg);
	float mean_min = get_mean(n_tests, results_min);
	float mean_max = get_mean(n_tests, results_max);

	float i_max = n_tests == 1 ? 0 : get_interval_95(n_tests, results_max);
	float i_avg = n_tests == 1 ? 0 : get_interval_95(n_tests, results_avg);
	float i_min = n_tests == 1 ? 0 : get_interval_95(n_tests, results_min);

//	printf("%8d %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f\n",
//		n, results_min[0], results_avg[0], results_max[0],
//		mean_min, mean_avg, mean_max, i_min, i_avg, i_max);
	printf("%8d %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f\n",
		n, mean_min, mean_avg, mean_max, i_min, i_avg, i_max);

} // n

	return 0;
}

char* get_time() {
	time_t now = time(NULL);
	char *time_str = ctime(&now);
	time_str[strlen(time_str) - 1] = 0;
	return time_str;
}

size_t get_free_memory() {
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	return free;
}

void print_size(size_t size) {
	size_t numbers[3];
	numbers[0] = size / 1000000;
	numbers[1] = (size % 1000000) / 1000;
	numbers[2] = size % 1000;

	printf("%zu,%03zu,%03zu", numbers[0], numbers[1], numbers[2]);
}

uint get_number_blocks(uint n_threads) {
	return n_threads / threads_per_block +
		(n_threads % threads_per_block == 0 ? 0 : 1);
}

int get_gvt(int *d_ts_temp) {
	kernel_get_gvt_1<<<n_blocks, threads_per_block,
			threads_per_block * sizeof(int)>>>(d_ts_temp);

	uint next_n_blocks = get_number_blocks(n_blocks);
	uint n_left = n_blocks;
	uint distance = 1;

	while (n_left != 1) {
		kernel_get_gvt_2<<<next_n_blocks, threads_per_block,
			threads_per_block * sizeof(int)>>>(
				d_ts_temp, n_left, distance);
		n_left = next_n_blocks;
		next_n_blocks = get_number_blocks(next_n_blocks);
		distance *= threads_per_block;
	} 

	int gvt;
	cudaMemcpy(&gvt, d_ts_temp, sizeof(int),
		cudaMemcpyDeviceToHost);

	return gvt;
}

char change_nodes_per_lp(uint target_value_log, char *d_can_split) {
	uint target = pow(2, target_value_log);
	if (nodes_per_lp == target) { return 1; }

	uint nodes_per_lp_before = nodes_per_lp;

	if (nodes_per_lp > target) {
		while (nodes_per_lp != target) {
			if (split_lps(d_can_split) == 0) {
				while (nodes_per_lp != nodes_per_lp_before) {
					merge_lps();
				}

				return 0;
			}
		}
	} else {
		while (nodes_per_lp != target) {
			merge_lps();
		}
	}

	return 1;
}

void merge_lps() {
#if (OPTM_SYNC == 1)
	kernel_roll_back_all<<<n_blocks, threads_per_block>>>();
	kernel_sort_event_queues<<<n_blocks, threads_per_block>>>();
#endif

	kernel_rotate_queues<<<n_blocks, threads_per_block>>>();
	kernel_merge_queues<<<n_blocks, threads_per_block>>>();
	kernel_adjust_queues_after_merge<<<n_blocks, threads_per_block>>>();
	kernel_set_params_after_merge<<<1,1>>>();

	nodes_per_lp = nodes_per_lp * 2;
	n_lps = n_lps / 2 + (n_lps % 2 == 0 ? 0 : 1);
	n_blocks = get_number_blocks(n_lps);

	cudaDeviceSynchronize();
	kernel_sort_event_queues<<<n_blocks, threads_per_block>>>();
}

char split_lps(char *d_can_split) {
	char h_can_split = 1;
	cudaMemcpy(d_can_split, &h_can_split, sizeof(char),
		cudaMemcpyHostToDevice);
	kernel_check_queues_before_split<<<n_blocks, threads_per_block>>>(
		d_can_split);
	cudaMemcpy(&h_can_split, d_can_split, sizeof(char),
		cudaMemcpyDeviceToHost);

	if (h_can_split == 0) {
//		printf("IMPOSSBILE TO SPLIT.\n");
		return 0;
	}

#if (OPTM_SYNC == 1) 
	kernel_roll_back_all<<<n_blocks, threads_per_block>>>();
	kernel_sort_event_queues<<<n_blocks, threads_per_block>>>();
#endif

	kernel_rotate_queues<<<n_blocks, threads_per_block>>>();
	kernel_adjust_queues_before_split<<<n_blocks, threads_per_block>>>();
	kernel_split_queues<<<n_blocks, threads_per_block>>>();
	kernel_set_params_after_split<<<1,1>>>();

	nodes_per_lp =  nodes_per_lp / 2;
	n_lps = n_lps * 2;
	n_blocks = get_number_blocks(n_lps);

	cudaDeviceSynchronize();
	kernel_sort_event_queues<<<n_blocks, threads_per_block>>>();

	return 1;
}
