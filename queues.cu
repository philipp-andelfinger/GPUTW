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

#include "queues.h"

__device__ static EQs	eq;
__device__ static SQs	sq;
__device__ static AMQs	amq;

__device__ static uint	events_per_lp = 0;
__device__ static uint	states_per_lp = 0;
__device__ static uint	antimsgs_per_lp = 0;

__device__
Event* get_event(uint lpid, uint offset);
__device__
uint get_state_index(uint lpid, uint offset1, uint offset2);
__device__
uint get_antimsg_index(uint lpid, uint offset1, uint offset2);
__device__
char compare_events(Event *event_1, Event *event_2);

char malloc_queues(uint n_nodes,
uint events_per_node, uint states_per_node, uint antimsgs_per_node) {
	cudaError_t err;

	EQs h_eq;

	err = cudaMalloc(&(h_eq.bo),
		sizeof(uint) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(eq, &h_eq, sizeof(EQs));

	err = cudaMalloc(&(h_eq.so),
		sizeof(uint) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(eq, &h_eq, sizeof(EQs));

	err = cudaMalloc(&(h_eq.uo),
		sizeof(uint) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(eq, &h_eq, sizeof(EQs));

	err = cudaMalloc(&(h_eq.ql),
		sizeof(uint) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(eq, &h_eq, sizeof(EQs));

#if (OPTM_SYNC == 1)
	err = cudaMalloc(&(h_eq.famo),
		sizeof(uint) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(eq, &h_eq, sizeof(EQs));

	err = cudaMalloc(&(h_eq.rbts),
		sizeof(int) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(eq, &h_eq, sizeof(EQs));

	err = cudaMalloc(&(h_eq.lpts),
		sizeof(int) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(eq, &h_eq, sizeof(EQs));
#endif

	err = cudaMalloc(&(h_eq.events),
		sizeof(Event) * n_nodes * events_per_node);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(eq, &h_eq, sizeof(EQs));

	SQs h_sq;

	err = cudaMalloc(&(h_sq.bo),
		sizeof(uint) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(sq, &h_sq, sizeof(SQs));

	err = cudaMalloc(&(h_sq.ql),
		sizeof(uint) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(sq, &h_sq, sizeof(SQs));

	err = cudaMalloc(&(h_sq.states),
		sizeof(State) * n_nodes * states_per_node);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(sq, &h_sq, sizeof(SQs));

	AMQs h_amq;

	err = cudaMalloc(&(h_amq.bo),
		sizeof(uint) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(amq, &h_amq, sizeof(AMQs));

	err = cudaMalloc(&(h_amq.ql),
		sizeof(uint) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(amq, &h_amq, sizeof(AMQs));

	err = cudaMalloc(&(h_amq.antimsgs),
		sizeof(Event) * n_nodes * antimsgs_per_node);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(amq, &h_amq, sizeof(AMQs));

	return 1;
}

void free_queues() {
	EQs h_eq;
	cudaMemcpyFromSymbol(&h_eq, eq, sizeof(EQs));

	cudaFree(h_eq.bo);
	cudaFree(h_eq.so);
	cudaFree(h_eq.uo);
	cudaFree(h_eq.ql);

#if (OPTM_SYNC == 1)
	cudaFree(h_eq.famo);
	cudaFree(h_eq.rbts);
	cudaFree(h_eq.lpts);
#endif

	cudaFree(h_eq.events);

	SQs h_sq;
	cudaMemcpyFromSymbol(&h_sq, sq, sizeof(SQs));

	cudaFree(h_sq.bo);
	cudaFree(h_sq.ql);
	cudaFree(h_sq.states);

	AMQs h_amq;
	cudaMemcpyFromSymbol(&h_amq, amq, sizeof(AMQs));

	cudaFree(h_amq.bo);
	cudaFree(h_amq.ql);
	cudaFree(h_amq.antimsgs);
}

__device__
void set_queues_params(
uint events_per_node, uint states_per_node, uint antimsgs_per_node) {
	events_per_lp = events_per_node * g_nodes_per_lp;
	states_per_lp = states_per_node * g_nodes_per_lp;
	antimsgs_per_lp = antimsgs_per_node * g_nodes_per_lp;
}

__device__
void init_queues(uint lpid) {
	eq.bo[lpid] = 0;
	eq.so[lpid] = 0;
	eq.uo[lpid] = 0;
	eq.ql[lpid] = 0;

#if (OPTM_SYNC == 1)
	eq.famo[lpid] = UINT_MAX;
	eq.rbts[lpid] = INT_MAX;
	eq.lpts[lpid] = 0;
#endif

	sq.bo[lpid] = 0;
	sq.ql[lpid] = 0;

	amq.bo[lpid] = 0;
	amq.ql[lpid] = 0;
}

__device__ // private
void rotate_event_queue(uint lpid) {
	uint first = 0;
	uint middle = eq.bo[lpid];
	uint last = events_per_lp;

	eq.bo[lpid] = 0;

	uint next = middle;
	uint swap = 0;
	uint eq_ql = eq.ql[lpid];
	Event tmp;

	while (first != next) {
		Event *event_first = get_event(lpid, first);
		Event *event_next = get_event(lpid, next);

		tmp = *event_first;
		*event_first = *event_next;
		*event_next = tmp;

		first ++; next ++; swap ++;

		if (next == last) { next = middle; }
		else if (first == middle) { middle = next; }

		if (swap == eq_ql) { break; }
	}
}

__device__
void rotate_queues(uint lpid) {
	rotate_event_queue(lpid);
}

__device__
void merge_queues(uint lpid) {
	for (uint i = 0; i < eq.ql[lpid + 1]; i++) {
		Event *right_event = get_event(lpid + 1, i);

		uint insert_index = lpid * events_per_lp + eq.ql[lpid] + i;
		eq.events[insert_index] = *right_event;
	}

	eq.ql[lpid] += eq.ql[lpid + 1];

#if (OPTM_SYNC== 1)
	eq.lpts[lpid] = max(eq.lpts[lpid], eq.lpts[lpid + 1]);
#endif
}

__device__
void adjust_queues_after_merge(uint lpid) {
	eq.uo[lpid / 2] = eq.uo[lpid];
	eq.ql[lpid / 2] = eq.ql[lpid];

#if (OPTM_SYNC == 1)
	eq.lpts[lpid / 2] = eq.lpts[lpid];
#endif
}

__device__
char can_split_queues(uint lpid) {
	uint nodes_per_lp_new = g_nodes_per_lp / 2;
	uint events_per_lp_new = events_per_lp / 2;

	uint n_events_left = 0;
	uint n_events_right = 0;

	for (uint i = 0; i < eq.ql[lpid]; i++) {
		Event *event = get_event(lpid, i);
		uint new_lpid = event->receiver / nodes_per_lp_new;

		if (new_lpid % 2 == 0)	{
			n_events_left ++;
			if (n_events_left > events_per_lp_new) { return 0; }
		} else {
			n_events_right ++;
			if (n_events_right > events_per_lp_new) { return 0; }
		}
	}

	return 1;
}

__device__
void adjust_queues_before_split(uint lpid) {
	eq.bo[lpid * 2] = 0;
	eq.so[lpid * 2] = 0;
	eq.uo[lpid * 2] = eq.uo[lpid];
	eq.ql[lpid * 2] = eq.ql[lpid];

#if (OPTM_SYNC == 1)
	eq.famo[lpid * 2] = UINT_MAX;
	eq.lpts[lpid * 2] = eq.lpts[lpid];
#endif
}

__device__
void split_queues(uint lpid) {
	uint lpid_left = lpid * 2;
	uint lpid_right = lpid_left + 1;

	if (lpid_right < g_n_nodes) {
		eq.bo[lpid_right] = 0;
		eq.so[lpid_right] = 0;
		eq.uo[lpid_right] = 0;
		eq.ql[lpid_right] = 0;

#if (OPTM_SYNC == 1)
		eq.famo[lpid_right] = UINT_MAX;
		eq.lpts[lpid_right] = eq.lpts[lpid_left];
#endif
	}

	uint nodes_per_lp_new = g_nodes_per_lp / 2;
	uint events_per_lp_new = events_per_lp / 2;

	if (eq.ql[lpid_left] > events_per_lp_new) {
		eq.bo[lpid_right] = eq.ql[lpid_left] - events_per_lp_new;
	}

	for (uint i = 0; i < eq.ql[lpid_left]; i++) {
		uint index = lpid * events_per_lp + i;
		Event *event = &(eq.events[index]);

		if (event->receiver / nodes_per_lp_new == lpid_right) {
			Event event_temp = *event;

			for (uint j = i; j < eq.ql[lpid_left] - 1; j++) {
				eq.events[index] = eq.events[index + 1];
				index ++; 
			}

			eq.uo[lpid_left] --;
			eq.ql[lpid_left] --;
			i --;

			if (eq.bo[lpid_right] != 0) {
				for (uint j = 0; j < eq.ql[lpid_right]; j++) {
					eq.events[index] = eq.events[index +1];
					index ++;
				}

				eq.bo[lpid_right] --;
			} else {
				index = lpid_right * events_per_lp_new
					+ eq.ql[lpid_right];
			}

			eq.events[index] = event_temp;

			eq.uo[lpid_right] ++;
			eq.ql[lpid_right] ++;
		}
	}
}

__device__
void set_queues_params_after_merge() {
	events_per_lp *= 2;
	states_per_lp *= 2;
	antimsgs_per_lp *= 2;
}

__device__
void set_queues_params_after_split() {
	events_per_lp /= 2;
	states_per_lp /= 2;
	antimsgs_per_lp /= 2;
}

__device__
void print_event_queue(uint lpid) {
	printf("LP=%u BO=%u SO=%u UO=%u QL=%u\n",
		lpid, eq.bo[lpid], eq.so[lpid], eq.uo[lpid], eq.ql[lpid]);

	for (uint i = 0; i < eq.ql[lpid]; i++) {
		if (i % 3 == 0) { printf("\n"); }
		Event *event = get_event(lpid, i);
		printf("(S=%u R=%u T=%d) ",
			event->sender, event->receiver, event->timestamp);
	}

	printf("\n");
}

__device__ // private
uint get_insert_position(uint lpid, Event *event) {
	// Compare with the last sorted event before binary search
	Event *last_sorted_event = get_event(lpid, eq.uo[lpid] - 1);
	if (compare_events(last_sorted_event, event) != 1) {
		return eq.uo[lpid];
	}

	// Binary search
	uint left = eq.so[lpid];
	uint right = eq.uo[lpid] - 1;
	uint middle;

	while (left != right) {
		middle = (left + right) / 2;
		Event *middle_event = get_event(lpid, middle);

		if (compare_events(middle_event, event) != 1) {
			left = middle + 1;
		} else {
			right = middle;
		}
	}

	return left;
}

__device__
void sort_event_queue(uint lpid) {
	uint eq_uo_old = eq.uo[lpid];
	uint eq_ql_old = eq.ql[lpid];

#if (OPTM_SYNC == 1)
	// Remove antimsgs among sorted events
	uint n_antimsgs = 0;
	for (uint i = eq.famo[lpid]; i < eq_uo_old; i++) {
		Event *event = get_event(lpid, i);

		if (event->timestamp < 0) {
			n_antimsgs ++;
		} else if (n_antimsgs != 0) {
			Event *target_event = get_event(lpid, i - n_antimsgs);
			*target_event = *event;
		}
	}

	eq.uo[lpid] -= n_antimsgs;
	eq.ql[lpid] -= n_antimsgs;
#endif

	// Insert unsorted events into sorted events
	for (uint i = eq_uo_old; i < eq_ql_old; i++) {
		Event event = *(get_event(lpid, i));

		// Ignore unsorted events with negative timestamp
		if (event.timestamp < 0) {
			eq.ql[lpid] --;
			continue;
		}

		// For case there is no sorted event
		if (eq.uo[lpid] == eq.so[lpid]) {
			Event *target_event = get_event(lpid, eq.uo[lpid]);
			*target_event = event;
			eq.uo[lpid] ++;
			continue;
		}

		uint insert_offset =
			get_insert_position(lpid,  &event);

		// Move sorted events
		uint sorted_event_offset = eq.uo[lpid] - 1;
		Event *target_event = get_event(lpid, sorted_event_offset + 1);

		while (sorted_event_offset >= insert_offset) {
			Event *sorted_event =
				get_event(lpid, sorted_event_offset);

			*target_event = *sorted_event;
			target_event = sorted_event;

			if (sorted_event_offset == 0) { break; }
			else { sorted_event_offset --; }
		}

		// Insert the unsorted event
		*target_event = event;
		eq.uo[lpid] ++;

// Old method without using binary search.
/*
		// For case there are sorted events.
		for (uint j = eq.uo[lpid] - 1; j >= eq.so[lpid]; j--) {
			Event *sorted_event = get_event(lpid, j);

			if (compare_events(&event, sorted_event) == -1) {
				Event *target_event = get_event(lpid, j + 1);
				*target_event = *sorted_event;

				if (j == eq.so[lpid]) {
					Event *target_event = get_event(
						lpid, j);
					*target_event = event;
					eq.uo[lpid] ++;
					break;
				}
			} else {
				Event *target_event = get_event(lpid, j + 1);
				*target_event = event;
				eq.uo[lpid] ++;
				break;
			}
		}
*/
	}

#if (OPTM_SYNC == 1)
	eq.rbts[lpid] = INT_MAX;
	eq.famo[lpid] = UINT_MAX;
#endif
}

__device__
char has_next_event(uint lpid) {
	return eq.so[lpid] != eq.uo[lpid];
}

__device__
Event* get_next_event(uint lpid) {
	return get_event(lpid, eq.so[lpid]);
}

__device__
void mark_next_event_as_processed(uint lpid) {
	eq.so[lpid] ++;
}

#if (OPTM_SYNC == 1)
__device__
void set_lpts(uint lpid, int timestamp) {
	eq.lpts[lpid] = timestamp;
}

__device__
int get_lpts(uint lpid) {
	return eq.lpts[lpid];
}
#endif

__device__
char append_event_to_queue(Event *event) {
	uint lpid = event->receiver / g_nodes_per_lp;

	uint eq_ql_old = atomicAdd(&(eq.ql[lpid]), 1);
	if (eq_ql_old >= events_per_lp) {
		atomicSub(&(eq.ql[lpid]), 1);
		return 0;
	}

#if (OPTM_SYNC == 1)
	atomicMin(&(eq.rbts[lpid]), event->timestamp);
#endif

	Event *target_event = get_event(lpid, eq_ql_old);
	*target_event = *event;
	return 1;
}

__device__
char append_event_to_queue(Event *event, uint *undo_offset) {
	uint lpid = event->receiver / g_nodes_per_lp;

	uint eq_ql_old = atomicAdd(&(eq.ql[lpid]), 1);
	if (eq_ql_old >= events_per_lp) {
		atomicSub(&(eq.ql[lpid]), 1);
		return 0;
	}

#if (OPTM_SYNC == 1)
	atomicMin(&(eq.rbts[lpid]), event->timestamp);
#endif

	Event *target_event = get_event(lpid, eq_ql_old);
	*target_event = *event;
	
	*undo_offset = eq_ql_old;
	return 1;
}

#if (OPTM_SYNC == 1)
__device__
void undo_event(Event *event) {
	uint lpid = event->receiver / g_nodes_per_lp;

	for (uint i = 0; i < eq.ql[lpid]; i++) {
		uint offset = eq.ql[lpid] - i - 1;
		Event *e = get_event(lpid, offset);

		if (events_are_equal(e, event)) {
			e->timestamp = -(e->timestamp);

			atomicMin(&(eq.rbts[lpid]), event->timestamp);
			atomicMin(&(eq.famo[lpid]), offset);

			return;
		}
	}

	printf("ERROR: undo_event(Event %7d %7d)\n",
		event->sender, event->receiver);
	asm("trap;");
}
#endif

__device__
void undo_event(uint lpid, uint undo_offset) {
	Event *event = get_event(lpid, undo_offset);
	event->timestamp = -(event->timestamp);
}

#if (OPTM_SYNC == 1)
__device__
int get_rollback_timestamp(uint lpid) {
	return eq.rbts[lpid];
}
#endif

__device__
char has_processed_event(uint lpid) {
	return eq.so[lpid] != 0;
}

__device__
Event* get_first_processed_event(uint lpid) {
	return get_event(lpid, 0);
}

__device__
void delete_first_processed_event(uint lpid) {
	eq.bo[lpid] = (eq.bo[lpid] + 1) % events_per_lp;
	eq.so[lpid] --;
	eq.uo[lpid] --;
	eq.ql[lpid] --;
}

__device__
Event* get_last_processed_event(uint lpid) {
	return get_event(lpid, eq.so[lpid] - 1);
}

__device__
void mark_last_processed_event_as_unprocessed(uint lpid) {
	eq.so[lpid] --;
}

__device__
char state_queue_is_full(uint lpid) {
	return sq.ql[lpid] == states_per_lp;
}

__device__
void append_state_to_queue(State *state, uint lpid) {
	uint index = get_state_index(lpid, sq.bo[lpid], sq.ql[lpid]);
	sq.ql[lpid] ++;

	sq.states[index] = *state;
}

__device__
void delete_first_n_states(uint lpid, uint n) {
	if (n == 0) { return; }

	sq.bo[lpid] = (sq.bo[lpid] + n) % states_per_lp;
	sq.ql[lpid] -= n;
}

__device__
State* delete_last_state(uint lpid) {
	sq.ql[lpid] --;

	uint index = get_state_index(lpid, sq.bo[lpid], sq.ql[lpid]);
	return &(sq.states[index]);
}

__device__
char antimsg_queue_is_full(uint lpid) {
	return amq.ql[lpid] == antimsgs_per_lp;
}

__device__
void append_antimsg_to_queue(Event *antimsg) {
	uint lpid = antimsg->sender / g_nodes_per_lp;

	uint index = get_antimsg_index(lpid, amq.bo[lpid], amq.ql[lpid]);
	amq.ql[lpid] ++;

	amq.antimsgs[index] = *antimsg;
}

__device__
void delete_first_n_antimsgs(uint lpid, uint n) {
	if (n == 0) { return; }

	amq.bo[lpid] = (amq.bo[lpid] + n) % antimsgs_per_lp;
	amq.ql[lpid] -= n;
}

__device__
Event* delete_last_antimsg(uint lpid) {
	amq.ql[lpid] --;

	uint index = get_antimsg_index(lpid, amq.bo[lpid], amq.ql[lpid]);
	return &(amq.antimsgs[index]);
}

__device__ // private
uint get_event_index(uint lpid, uint offset1, uint offset2) {
	return lpid * events_per_lp + (offset1 + offset2) % events_per_lp;
}

__device__ // private
Event* get_event(uint lpid, uint offset) {
	uint index = get_event_index(lpid, eq.bo[lpid], offset);
	return &(eq.events[index]);
}

__device__ // private
uint get_state_index(uint lpid, uint offset1, uint offset2) {
	return lpid * states_per_lp + (offset1 + offset2) % states_per_lp;
}

__device__ // private
uint get_antimsg_index(uint lpid, uint offset1, uint offset2) {
	return lpid * antimsgs_per_lp +
		(offset1 + offset2) % antimsgs_per_lp;
}

/* Returns -1 if event 1 should be placed before event 2.
 * Returns  1 if event 2 should be placed before event 1.
 * Returns  0 if two events have equal order.
 */
__device__ // private
char compare_events(Event *event_1, Event *event_2) {
	if (event_1->timestamp > event_2->timestamp)	{ return  1; }
	if (event_1->timestamp < event_2->timestamp)	{ return -1; }

//	if (event_1->receiver > event_2->receiver)	{ return  1; }
//	if (event_1->receiver < event_2->receiver)	{ return -1; }

	if (event_1->sender > event_2->sender)		{ return  1; }
	if (event_1->sender < event_2->sender)		{ return -1; }

//	if (event_1->type > event_2->type)		{ return  1; }
//	if (event_1->type < event_2->type)		{ return -1; }

	return 0;
}
