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


#include "model.h"

__device__ static Nodes nodes;

__device__ static int max_lookup_delay;

/* Statistics */
__device__ static float n_lookups = 0;
__device__ static float n_requests_per_lookup_sum = 0;
__device__ static float lookup_duration_sum = 0;
__device__ static float min_distance_sum = 0;

/* Private functions */
__device__
uint get_distance(uint pid_1, uint pid_2);
__device__
uint get_pid(uint nid);
__device__
uint get_bucket_id(uint s_nid, uint t_pid);
__device__
uint* get_bucket(uint nid, uint bucket_id);
__device__
LookupResult* get_lookup_results(uint nid);
__device__
int get_latency(curandState_t *cr_state);
__device__
void copy_old_state(State *state, uint nid, uint event_type);
__device__
void recover_old_state(State *state, uint nid, uint event_type);

char malloc_nodes(uint n_nodes) {
	cudaError_t err;

	Nodes h_nodes;

	err = cudaMalloc(&(h_nodes.cr_state),
		sizeof(curandState_t) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(nodes, &h_nodes, sizeof(Nodes));

	err = cudaMalloc(&(h_nodes.pid),
		sizeof(uint) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(nodes, &h_nodes, sizeof(Nodes));

	err = cudaMalloc(&(h_nodes.buckets),
		sizeof(uint) * n_nodes * MAX_BUCKETS * BUCKET_SIZE);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(nodes, &h_nodes, sizeof(Nodes));

	err = cudaMalloc(&(h_nodes.last_bucket_id),
		sizeof(uint) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(nodes, &h_nodes, sizeof(Nodes));

	err = cudaMalloc(&(h_nodes.lookup_pid),
		sizeof(uint) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(nodes, &h_nodes, sizeof(Nodes));

	err = cudaMalloc(&(h_nodes.lookup_start_timestamp),
		sizeof(int) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(nodes, &h_nodes, sizeof(Nodes));

	err = cudaMalloc(&(h_nodes.n_pending_requests),
		sizeof(uint) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(nodes, &h_nodes, sizeof(Nodes));

	err = cudaMalloc(&(h_nodes.n_requests_per_lookup),
		sizeof(uint) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(nodes, &h_nodes, sizeof(Nodes));

	err = cudaMalloc(&(h_nodes.lookup_results),
		sizeof(LookupResult) * n_nodes * BUCKET_SIZE);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(nodes, &h_nodes, sizeof(Nodes));

	// Statistics
	err = cudaMalloc(&(h_nodes.n_lookups),
		sizeof(float) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(nodes, &h_nodes, sizeof(Nodes));

	err = cudaMalloc(&(h_nodes.n_requests_per_lookup_sum),
		sizeof(float) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(nodes, &h_nodes, sizeof(Nodes));

	err = cudaMalloc(&(h_nodes.lookup_duration_sum),
		sizeof(float) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(nodes, &h_nodes, sizeof(Nodes));

	err = cudaMalloc(&(h_nodes.min_distance_sum),
		sizeof(float) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(nodes, &h_nodes, sizeof(Nodes));

	return 1;
}

void free_nodes() {
	Nodes h_nodes;
	cudaMemcpyFromSymbol(&h_nodes, nodes, sizeof(Nodes));

	cudaFree(h_nodes.cr_state);

	cudaFree(h_nodes.pid);
	cudaFree(h_nodes.buckets);
	cudaFree(h_nodes.last_bucket_id);

	cudaFree(h_nodes.lookup_pid);
	cudaFree(h_nodes.lookup_start_timestamp);
	cudaFree(h_nodes.n_pending_requests);
	cudaFree(h_nodes.n_requests_per_lookup);
	cudaFree(h_nodes.lookup_results);

	cudaFree(h_nodes.n_lookups);
	cudaFree(h_nodes.n_requests_per_lookup_sum);
	cudaFree(h_nodes.lookup_duration_sum);
	cudaFree(h_nodes.min_distance_sum);
}

__device__
void set_model_params(int params[], uint n_params) {
	max_lookup_delay = params[0];
}

__device__
int get_lookahead() {
	return LOOKAHEAD;
}

__device__
void init_node(uint nid) {
	curand_init(nid, 0, 0, &(nodes.cr_state[nid]));

	nodes.pid[nid] = UINT_MAX;

	for (uint i = 0; i < MAX_BUCKETS; i++) {
		uint *bucket = get_bucket(nid, i);

		for (int j = 0; j < BUCKET_SIZE; j++) {
			bucket[j] = UINT_MAX;
		}
	}

	nodes.last_bucket_id[nid] = 0;

	nodes.n_lookups[nid] = 0;
	nodes.n_requests_per_lookup_sum[nid] = 0;
	nodes.lookup_duration_sum[nid] = 0;
	nodes.min_distance_sum[nid] = 0;

	Event event;
	event.type = 0;
	event.sender = nid;
	event.receiver = nid;
	event.timestamp = 0;

	append_event_to_queue(&event);
}

__device__ // private
char handle_event_type_0(Event *event) {
	uint nid = event->receiver;

	Event new_event;
	new_event.type = 1;
	new_event.sender = nid;
	new_event.receiver = nid;
	new_event.timestamp = event->timestamp + 1;

	char res = append_event_to_queue(&new_event);

	if (res == 0) {
		return 11;
	}

	curandState_t *cr_state = &(nodes.cr_state[nid]);
	nodes.pid[nid] = random(cr_state, INT_MAX);

	return 1;
}

__device__ // private
void insert_node_into_bucket(uint s_nid, uint t_nid) {
	if (s_nid == t_nid) { return; }

	uint t_pid = get_pid(t_nid);

	while (1) {
		uint bucket_id = get_bucket_id(s_nid, t_pid);
		uint *bucket = get_bucket(s_nid, bucket_id);

		int insert_index = -1;
		for (int i = BUCKET_SIZE - 1; i >= 0; i--) {
			if (bucket[i] == t_nid) { // Node already in the bucket
				return;
			} else if (bucket[i] == UINT_MAX) {
				insert_index = i;
			}
		}

		if (insert_index != -1) {
			bucket[insert_index] = t_nid;
			return;
		}

		// If the function still not returns,
		// then the last bucket should be splitted.
		uint last_bucket_id = nodes.last_bucket_id[s_nid];
		if (bucket_id != last_bucket_id
		|| last_bucket_id == MAX_BUCKETS - 1) {
			return;
		}

		nodes.last_bucket_id[s_nid] ++;
		last_bucket_id ++;

		uint *new_bucket = get_bucket(s_nid, last_bucket_id);

		int n_moves = 0;
		for (int i = 0; i < BUCKET_SIZE; i++) {
			uint nid = bucket[i];
			uint pid = get_pid(nid);
			uint new_bucket_id = get_bucket_id(s_nid, pid);

			if (new_bucket_id == last_bucket_id) {
				new_bucket[n_moves] = nid;
				bucket[i] = UINT_MAX;
				n_moves ++;
			}
		}
	}
}

__device__ // private
char handle_event_type_1(Event *event) {
	uint nid = event->receiver;

	curandState_t *cr_state = &(nodes.cr_state[nid]);
	int lookup_delay = random(cr_state, max_lookup_delay + 1) + 1;

	Event new_event;
	new_event.type = 2;
	new_event.sender = nid;
	new_event.receiver = nid;
	new_event.timestamp = event->timestamp + lookup_delay;

	char res = append_event_to_queue(&new_event);

	if (res == 0) {
		reverse_state(cr_state);
		return 11;
	}

	for (int i = 0; i < N_BOOTSTRAPS; i++) {
		uint known_nid = random(cr_state, g_n_nodes);
		insert_node_into_bucket(nid, known_nid);
	}

	return 1;
}

__device__ // private
char handle_event_type_2(Event *event) {
// Start lookup
	uint nid = event->receiver;
	uint lpid = nid / g_nodes_per_lp;

	if (state_queue_is_full(lpid)) { return 12; }

	State old_state;
	copy_old_state(&old_state, nid, 2);
	uint n_antimsgs = 0;

	curandState_t *cr_state = &(nodes.cr_state[nid]);
	
	// Generate new lookup pid
	uint lookup_pid = random(cr_state, INT_MAX);
	nodes.lookup_pid[nid] = lookup_pid;

	// Initialize
	LookupResult *lookup_results = get_lookup_results(nid);
	for (int i = 0; i < BUCKET_SIZE; i++) {
		lookup_results[i].status = 2;	// 2 = Invalid
	}

	// Send requests
	uint bucket_id = get_bucket_id(nid, lookup_pid);
	uint *bucket = get_bucket(nid, bucket_id);

	int n_requests = 0;
	uint undo_offsets[BUCKET_SIZE];

	for (int i = 0; i < BUCKET_SIZE; i++) {
		int receiver_nid = bucket[i];
		if (receiver_nid == UINT_MAX) { continue; }

		Event new_event;
		new_event.type = 4;
		new_event.sender = nid;
		new_event.receiver = receiver_nid;
		new_event.timestamp = event->timestamp + get_latency(cr_state);
		new_event.data[0] = lookup_pid;

		char res_event = append_event_to_queue(
			&new_event, &(undo_offsets[n_requests]));
		char res_antimsg = antimsg_queue_is_full(lpid);

		if (res_event == 0 || res_antimsg == 1) {
			for (int j = i - 1; j >= 0; j--) {
				if (bucket[j] == UINT_MAX) { continue; }

				delete_last_antimsg(lpid);

				n_requests --;
				undo_event(
					bucket[j] / g_nodes_per_lp,
					undo_offsets[n_requests]);
			}

			recover_old_state(&old_state, nid, 2);

			if (res_event == 0) {
				return 11;
			} else if (res_antimsg == 1) {
				return 13;
			}
		}

		append_antimsg_to_queue(&new_event);
		n_antimsgs ++;

		n_requests ++;
	}

	nodes.n_pending_requests[nid] = n_requests;
	nodes.n_requests_per_lookup[nid] = n_requests;

	nodes.lookup_start_timestamp[nid] = event->timestamp;

	append_state_to_queue(&old_state, lpid);
	event->n_antimsgs = n_antimsgs;

	return 1;
}

__device__ // private
void insert_node_into_lookup_results(uint s_nid, uint t_nid) {
	uint lookup_pid = nodes.lookup_pid[s_nid];
	uint t_pid = get_pid(t_nid);
	uint distance_new = get_distance(lookup_pid, t_pid);

	LookupResult *lookup_results= get_lookup_results(s_nid);

	int insert_index = -1;
	for (int i = 0; i < BUCKET_SIZE; i++) {
		if (lookup_results[i].status == 2) {
			insert_index = i;
			break;
		}

		uint nid = lookup_results[i].nid;
		uint pid = get_pid(nid);		
		if (pid == t_pid) { return; }

		uint distance_old = get_distance(lookup_pid, pid);
		if (distance_new < distance_old) {
			insert_index = i;
			break;
		}
	}

	if (insert_index != -1) {
		for (int i = BUCKET_SIZE - 2; i >= insert_index; i--) {
			lookup_results[i + 1] = lookup_results[i];
		}

		lookup_results[insert_index].nid = t_nid;
		lookup_results[insert_index].status = 0;
	}
}

__device__ // private
char handle_event_type_3(Event *event) {
// Receive response
	uint nid = event->receiver;
	uint lpid = nid / g_nodes_per_lp;

	if (state_queue_is_full(lpid)) { return 12; }

	State old_state;
	copy_old_state(&old_state, nid, 3);
	uint n_antimsgs = 0;

	nodes.n_pending_requests[nid] --;

	// Update lookup result
	for (int i = 0; i < BUCKET_SIZE; i++) {
		uint t_nid = event->data[i];
		if (t_nid == UINT_MAX) { continue; }

		insert_node_into_lookup_results(nid, t_nid);
	}

	// Send new requests
	LookupResult *lookup_results= get_lookup_results(nid);
	uint n_pending_requests = nodes.n_pending_requests[nid];
	curandState_t *cr_state = &(nodes.cr_state[nid]);
	
	uint n_requests = 0;
	uint undo_offsets[BUCKET_SIZE];

	for (int i = 0; i < BUCKET_SIZE; i++) {
		LookupResult *lookup_result = &(lookup_results[i]);

		// 3 = Newly.status, 2 = Invalid, 1 = Already.status
		if (lookup_result->status == 2) { break; }
		if (lookup_result->status == 3) { lookup_result->status = 1; }
		if (lookup_result->status == 1) { continue; }

		if (n_pending_requests + n_requests >= BUCKET_SIZE) {continue;}

		Event new_event;
		new_event.type = 4;
		new_event.sender = nid;
		new_event.receiver = lookup_result->nid;
		new_event.timestamp =
			event->timestamp + get_latency(cr_state);
		new_event.data[0] = nodes.lookup_pid[nid];

		char res_event = append_event_to_queue(
			&new_event, &(undo_offsets[n_requests]));
		char res_antimsg = antimsg_queue_is_full(lpid);

		if (res_event == 0 || res_antimsg == 1) {
			for (int j = i - 1; j >= 0; j--) {
				if (lookup_results[j].status != 3) {continue;}

				delete_last_antimsg(lpid);

				n_requests --;
				undo_event(
					lookup_results[j].nid / g_nodes_per_lp,
					undo_offsets[n_requests]);
			}

			recover_old_state(&old_state, nid, 3);

			if (res_event == 0) {
				return 11;
			} else if (res_antimsg == 1) {
				return 13;
			}
		}

		append_antimsg_to_queue(&new_event);
		n_antimsgs ++;

		lookup_result->status = 3;
		n_requests ++;
	}

	nodes.n_pending_requests[nid] += n_requests;
	nodes.n_requests_per_lookup[nid] += n_requests;

	// Issue new lookup
	if (nodes.n_pending_requests[nid] == 0) {
		int lookup_delay = random(cr_state, max_lookup_delay + 1) + 1;

		Event new_event;
		new_event.type = 2;
		new_event.sender = nid;
		new_event.receiver = nid;
		new_event.timestamp = event->timestamp + lookup_delay;

		char res_event = append_event_to_queue(&new_event);
		char res_antimsg = antimsg_queue_is_full(lpid);

		if (res_event == 0 || res_antimsg == 1) {
			recover_old_state(&old_state, nid, 3);

			if (res_event == 0) {
				return 11;
			} else if (res_antimsg == 1) {
				return 13;
			}
		}

		append_antimsg_to_queue(&new_event);
		n_antimsgs ++;

		// Statistics
		uint min_distance = 0;
		if (lookup_results[0].status == 1) {
			min_distance = get_distance(
				nodes.lookup_pid[nid],
				get_pid(lookup_results[0].nid));
		}

		nodes.n_lookups[nid] ++;
		nodes.n_requests_per_lookup_sum[nid] +=
			nodes.n_requests_per_lookup[nid];
		nodes.lookup_duration_sum[nid] +=
			event->timestamp - nodes.lookup_start_timestamp[nid];
		nodes.min_distance_sum[nid] += min_distance;
	}

	append_state_to_queue(&old_state, lpid);
	event->n_antimsgs = n_antimsgs;

	return 1;
}

__device__ // private
char handle_event_type_4(Event *event) {
// Receive request
	uint nid = event->receiver;
	uint lpid = nid / g_nodes_per_lp;

	if (state_queue_is_full(lpid)) { return 12; }
	if (antimsg_queue_is_full(lpid)) { return 13; }

	State old_state;
	copy_old_state(&old_state, nid, 4);

	curandState_t *cr_state = &(nodes.cr_state[nid]);

	Event new_event;
	new_event.type = 3;
	new_event.sender = nid;
	new_event.receiver = event->sender;
	new_event.timestamp = event->timestamp + get_latency(cr_state);
 
	uint lookup_pid = event->data[0];
	uint bucket_id = get_bucket_id(nid, lookup_pid);
	uint *bucket = get_bucket(nid, bucket_id);

	for (int i = 0; i < BUCKET_SIZE; i++) {
		new_event.data[i] = bucket[i];
	}

	char res = append_event_to_queue(&new_event);

	if (res == 0) {
		recover_old_state(&old_state, nid, 4);
		return 11;
	}

	append_antimsg_to_queue(&new_event);

	append_state_to_queue(&old_state, lpid);
	event->n_antimsgs = 1;

	return 1;
}	

__device__
char handle_event(Event *event) {
	uint type = event->type;

	if (type == 0) { 
		return handle_event_type_0(event);
	} else if (type == 1) {
		return handle_event_type_1(event);
	} else if (type == 2) {
		return handle_event_type_2(event);
	} else if (type == 3) {
		return handle_event_type_3(event);
	} else if (type == 4) {
		return handle_event_type_4(event);
	} else {
		return 0;
	}
}

__device__
void roll_back_event(Event *event) {
	uint type = event->type;

	if (type == 0 || type ==1) {
		printf("ERROR: roll_back_event()\n");
		asm("trap;");
	}

	int nid = event->receiver;
	int lpid = nid / g_nodes_per_lp;

	State *old_state = delete_last_state(lpid);
	recover_old_state(old_state, nid, type);

	uint n_antimsgs = event->n_antimsgs;
	for (uint i = 0; i < n_antimsgs; i++) {
		Event *antimsg = delete_last_antimsg(lpid);
		undo_event(antimsg);
	}
}

__device__
uint get_number_states(Event *event) {
	uint type = event->type;

	if (type == 2 || type == 3 || type == 4) {
		return 1;
	} else {
		return 0;
	}
}

__device__
uint get_number_antimsgs(Event *event) {
	uint type = event->type;

	if (type == 2 || type == 3 || type == 4) {
		return event->n_antimsgs;
	} else {
		return 0;
	}
}

__device__
void collect_statistics(uint nid) {
	atomicAdd(&n_lookups, nodes.n_lookups[nid]);
	atomicAdd(&n_requests_per_lookup_sum,
		nodes.n_requests_per_lookup_sum[nid]);
	atomicAdd(&lookup_duration_sum, nodes.lookup_duration_sum[nid]);
	atomicAdd(&min_distance_sum, nodes.min_distance_sum[nid]);
}

__device__
void print_statistics() {
	printf("TOTAL NUMBER OF LOOKUPS : %.1f\n", n_lookups);
	printf("AVG. REQESTS PER LOOKUP : %.1f\n",
		n_requests_per_lookup_sum / n_lookups);
	printf("AVG. DURATION PER LOOKUP: %.1f ms\n",
		lookup_duration_sum / n_lookups);
	printf("AVG. MINIMUM DISTANCE   : %.1f\n",
		min_distance_sum/ n_lookups);

	printf("SUM. REQESTS PER LOOKUP : %.1f\n",
		n_requests_per_lookup_sum);
	printf("SUM. DURATION PER LOOKUP: %.1f ms\n",
		lookup_duration_sum);
	printf("SUM. MINIMUM DISTANCE   : %.1f\n",
		min_distance_sum);
}

__device__ // private
uint get_distance(uint pid_1, uint pid_2) {
	return pid_1 ^ pid_2;
}

__device__ // private
uint get_pid(uint nid) {
	return nodes.pid[nid];
}

__device__ // private
uint get_bucket_id(uint s_nid, uint t_pid) {
	uint s_pid = get_pid(s_nid);
	uint distance = get_distance(s_pid, t_pid);
	uint n_same_bits = __clz(distance) - 1;
	uint last_bucket_id = nodes.last_bucket_id[s_nid];

	return min(n_same_bits, last_bucket_id);
}

__device__ // private
uint* get_bucket(uint nid, uint bucket_id) {
	return &(nodes.buckets[
		nid * MAX_BUCKETS * BUCKET_SIZE +
		bucket_id * BUCKET_SIZE]);
}

__device__ // private
LookupResult* get_lookup_results(uint nid){
	return &(nodes.lookup_results[nid * BUCKET_SIZE]);
}

__device__ // private
int get_latency(curandState_t *cr_state) {
	return LOOKAHEAD + random(cr_state, MAX_LATENCY + 1 - LOOKAHEAD);
}

__device__ // private
void copy_old_state(State *state, uint nid, uint event_type) {
	state->cr_state = nodes.cr_state[nid];

	if (event_type == 2) {
		state->lookup_pid = nodes.lookup_pid[nid];
		state->lookup_start_timestamp =
			nodes.lookup_start_timestamp[nid];
	}

	if (event_type == 3) {
		state->n_lookups = nodes.n_lookups[nid];
		state->n_requests_per_lookup_sum =
			nodes.n_requests_per_lookup_sum[nid];
		state->lookup_duration_sum = nodes.lookup_duration_sum[nid];
		state->min_distance_sum = nodes.min_distance_sum[nid];
	}

	if (event_type == 2 || event_type == 3) {
		state->n_pending_requests =
			nodes.n_pending_requests[nid];
		state->n_requests_per_lookup =
			nodes.n_requests_per_lookup[nid];

		LookupResult *lookup_results= get_lookup_results(nid);
		for (int i = 0; i < BUCKET_SIZE; i++) {
			state->lookup_results[i] = lookup_results[i];
		}
	}
}

__noinline__ __device__
void recover_old_state(State *state, uint nid, uint event_type) {
	nodes.cr_state[nid] = state->cr_state;

	if (event_type == 2) {
		nodes.lookup_pid[nid] = state->lookup_pid;
		nodes.lookup_start_timestamp[nid] =
			state->lookup_start_timestamp;

	}

	if (event_type == 3) {
		nodes.n_lookups[nid] = state->n_lookups;
		nodes.n_requests_per_lookup_sum[nid] =
			state->n_requests_per_lookup_sum;
		nodes.lookup_duration_sum[nid] = state->lookup_duration_sum;
		nodes.min_distance_sum[nid] = state->min_distance_sum;
	}

	if (event_type == 2 || event_type == 3) {
		nodes.n_pending_requests[nid] = state->n_pending_requests;
		nodes.n_requests_per_lookup[nid] =
			state->n_requests_per_lookup;

		LookupResult *lookup_results= get_lookup_results(nid);
		for (int i = 0; i < BUCKET_SIZE; i++) {
			lookup_results[i] = state->lookup_results[i];
		}
	}
}
