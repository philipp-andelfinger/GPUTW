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

#include "kernels.h"

__global__
void kernel_set_params(
uint n_nodes, uint n_lps, uint nodes_per_lp,
uint events_per_node, uint states_per_node, uint antimsgs_per_node,
int model_params[], uint n_params) {
	g_n_nodes = n_nodes;
	g_n_lps = n_lps; 
	g_nodes_per_lp = nodes_per_lp;

	set_queues_params(events_per_node, states_per_node, antimsgs_per_node);
	set_model_params(model_params, n_params);
}

__global__
void kernel_get_lookahead(int *lookahead) {
	*lookahead = get_lookahead();
}

__global__
void kernel_init_queues() {
	uint lpid = blockIdx.x * blockDim.x + threadIdx.x;
	if (lpid >= g_n_lps) { return; }

	init_queues(lpid);
}

__global__
void kernel_init_nodes() {
	uint lpid = blockIdx.x * blockDim.x + threadIdx.x;
	if (lpid >= g_n_lps) { return; }

	for (uint i = 0; i < g_nodes_per_lp; i++) {
		uint nid = lpid * g_nodes_per_lp + i;
		if (nid < g_n_nodes) { init_node(nid); }		
	}
}

__global__
void kernel_handle_next_event(int gvt, int window_size,
uint *n_inac_1, uint *n_inac_2, uint *n_inac_3,
uint *n_inac_4, uint *n_inac_5, uint *n_inac_6) {
	uint lpid = blockIdx.x * blockDim.x + threadIdx.x;
	if (lpid >= g_n_lps) { return; }

	if (has_next_event(lpid) == 0) {
		atomicAdd(n_inac_1, 1);
		return;
	}

	Event *next_event = get_next_event(lpid);

	if (window_size != -1 && next_event->timestamp >= gvt + window_size) {
		atomicAdd(n_inac_2, 1);
		return;
	}

#if (OPTM_SYNC == 1)
	int rollback_timestamp = get_rollback_timestamp(lpid);
	if (next_event->timestamp >= rollback_timestamp) {
		atomicAdd(n_inac_2, 1);
		return;
	}
#endif

	char res = handle_event(next_event);
	if (res == 0) {
		atomicAdd(n_inac_3, 1);
		return;
	}

	// Event cannot be handled because event queue is full.
	if (res == 11) {
		atomicAdd(n_inac_4, 1);
		return;
	}

	// Event cannot be handled because state queue is full.
	if (res == 12) {
		atomicAdd(n_inac_5, 1);
		return;
	}

	// Event cannot be handled because antimsg queue is full.
	if (res == 13) {
		atomicAdd(n_inac_6, 1);
		return;
	}

	mark_next_event_as_processed(lpid);

#if (OPTM_SYNC == 1)
	set_lpts(lpid, next_event->timestamp);
#endif
}

#if (OPTM_SYNC == 1)
__global__
void kernel_roll_back(char *rollback_performed) {
	uint lpid = blockIdx.x * blockDim.x + threadIdx.x;
	if (lpid >= g_n_lps) { return; }

	int rollback_timestamp = get_rollback_timestamp(lpid);	
	if (get_lpts(lpid) < rollback_timestamp) { return; }

	while (has_processed_event(lpid)) {
		Event *last_processed_event = get_last_processed_event(lpid);

		if (
		abs(last_processed_event->timestamp) < rollback_timestamp) {
			break;
		}

		mark_last_processed_event_as_unprocessed(lpid);
		roll_back_event(last_processed_event);
		*rollback_performed = 1;
	}

	set_lpts(lpid, rollback_timestamp - 1);
}

__global__
void kernel_roll_back_all() {
	uint lpid = blockIdx.x * blockDim.x + threadIdx.x;
	if (lpid >= g_n_lps) { return; }

	while (has_processed_event(lpid)) {
		Event *last_processed_event = get_last_processed_event(lpid);
		mark_last_processed_event_as_unprocessed(lpid);
		roll_back_event(last_processed_event);
		set_lpts(lpid, abs(last_processed_event->timestamp) - 1);
	}
}
#endif

__global__
void kernel_rotate_queues() {
	uint lpid = blockIdx.x * blockDim.x + threadIdx.x;
	if (lpid >= g_n_lps) { return; }

	rotate_queues(lpid);
}

__global__
void kernel_merge_queues() {
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= g_n_lps / 2) { return; }

	merge_queues(tid * 2);
}

__global__
void kernel_adjust_queues_after_merge() {
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid > g_n_lps / 4) { return; }

	uint next_lpid = (tid * 2 + 1) * 2;

	while (next_lpid < g_n_lps) {
		adjust_queues_after_merge(next_lpid);
		next_lpid *= 2;
	}
}

__global__
void kernel_check_queues_before_split(char *can_split) {
	uint lpid = blockIdx.x * blockDim.x + threadIdx.x;
	if (lpid >= g_n_lps) { return; }

	if (can_split_queues(lpid) == 0) {
		*can_split = 0;
	}
}

__global__
void kernel_adjust_queues_before_split() {
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= g_n_lps || tid < (g_n_lps - g_n_lps / 2)) { return; }

	uint next_lpid = tid;

	while (1) {
		adjust_queues_before_split(next_lpid);
	
		if (next_lpid % 2 != 0) { break; }
		next_lpid /= 2;
	}
}

__global__
void kernel_split_queues() {
	uint lpid = blockIdx.x * blockDim.x + threadIdx.x;
	if (lpid >= g_n_lps) { return; }

	split_queues(lpid);
} 

__global__
void kernel_set_params_after_merge() {
	g_nodes_per_lp *= 2;
	g_n_lps = g_n_lps / 2 + (g_n_lps % 2 == 0 ? 0 : 1);

	set_queues_params_after_merge();
}	

__global__
void kernel_set_params_after_split() {
	g_nodes_per_lp /= 2;
	g_n_lps *= 2;

	set_queues_params_after_split();
}

__global__
void kernel_sort_event_queues() {
	uint lpid = blockIdx.x * blockDim.x + threadIdx.x;
	if (lpid >= g_n_lps) { return; }

	sort_event_queue(lpid);
}

__global__
void kernel_clean_queues(uint gvt, uint *n_events_cleaned) {
	uint lpid = blockIdx.x * blockDim.x + threadIdx.x;
	if (lpid >= g_n_lps) { return; }

	uint n_events = 0;

#if (OPTM_SYNC == 1)
	uint n_states = 0;
	uint n_antimsgs = 0;

	while (has_processed_event(lpid)) {
		Event *first_processed_event = get_first_processed_event(lpid);

		if (first_processed_event->timestamp >= gvt) {
			break;
		}

		delete_first_processed_event(lpid);

		n_events ++;		
		n_states += get_number_states(first_processed_event);
		n_antimsgs += get_number_antimsgs(first_processed_event);
	}

	delete_first_n_states(lpid, n_states);
	delete_first_n_antimsgs(lpid, n_antimsgs);
#else
	while (has_processed_event(lpid)) {
		delete_first_processed_event(lpid);
		n_events ++;
	}
#endif

	atomicAdd(n_events_cleaned, n_events);
}

__global__
void kernel_collect_statistics() {
	uint lpid = blockIdx.x * blockDim.x + threadIdx.x;
	if (lpid >= g_n_lps) { return; }

	for (uint i = 0; i < g_nodes_per_lp; i++) {
		uint nid = lpid * g_nodes_per_lp + i;
		if (nid < g_n_nodes) { collect_statistics(nid); }
	}
}

__global__
void kernel_print_statistics() {
	print_statistics();
}

__device__ //private
void warp_reduce(volatile int *sdata, uint tid) {
	sdata[tid] = min(sdata[tid], sdata[tid + 32]);
	sdata[tid] = min(sdata[tid], sdata[tid + 16]);
	sdata[tid] = min(sdata[tid], sdata[tid +  8]);
	sdata[tid] = min(sdata[tid], sdata[tid +  4]);
	sdata[tid] = min(sdata[tid], sdata[tid +  2]);
	sdata[tid] = min(sdata[tid], sdata[tid +  1]);
}

__global__
void kernel_get_gvt_1(int *ts_temp) {
	extern __shared__ int sdata[];

	uint tid = threadIdx.x;
	uint gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid >= g_n_lps) {
		sdata[tid] = INT_MAX;
	} else if (has_next_event(gid) == 0) {
		sdata[tid] = INT_MAX;
	} else {
		sdata[tid] = get_next_event(gid)->timestamp;
	}

	__syncthreads();

	for (uint s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sdata[tid] = min(sdata[tid], sdata[tid + s]);
		}

		__syncthreads();
	}

	if (tid < 32) { warp_reduce(sdata, tid); }

	if (tid == 0) { ts_temp[blockIdx.x] = sdata[0]; }
}

__global__
void kernel_get_gvt_2(int *ts_temp, uint n, uint distance) {
	extern __shared__ int sdata[];

	uint tid = threadIdx.x;
	uint gid = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = gid < n ? ts_temp[gid * distance] : INT_MAX;
	__syncthreads();

	for (uint s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sdata[tid] = min(sdata[tid], sdata[tid + s]);
		}

		__syncthreads();
	}

	if (tid < 32) { warp_reduce(sdata, tid); }

	if (tid == 0) { ts_temp[gid * distance] = sdata[0]; }
}

__global__
void kernel_print_event_queue(uint lpid) {
	print_event_queue(lpid);
}
