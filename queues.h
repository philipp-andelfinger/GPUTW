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

#ifndef queues_h
#define queues_h

#include <stdio.h>
#include "main.h"
#include "SETTINGS.h"
#include EVENT_HEADER
#include STATE_HEADER

typedef struct {
	uint	*bo;	// Begin offset
	uint	*so;	// Sorted offset
	uint	*uo;	// Unsorted offset
	uint	*ql;	// Queue length

#if (OPTM_SYNC == 1)
	uint	*famo;	// First antimsg offset
	int	*rbts;	// Rollback timestamp
	int	*lpts;	// Last processed timestamp;
#endif

	Event	*events;
} EQs;

typedef struct {
	uint	*bo;	// Begin offset
	uint	*ql;	// Queue length
	State	*states;
} SQs;

typedef struct {
	uint	*bo;	// Begin offset
	uint	*ql;	// Queue length
	Event	*antimsgs;
} AMQs;

char malloc_queues(uint n_nodes,
uint events_per_node, uint states_per_node, uint antimsgs_per_node);

void free_queues();

__device__
void set_queues_params(
uint events_per_node, uint states_per_node, uint antimsgs_per_node);

__device__
void init_queues(uint lpid);

__device__
void rotate_queues(uint lpid);

__device__
void merge_queues(uint lpid);

__device__
void adjust_queues_after_merge(uint lpid);

__device__
char can_split_queues(uint lpid);

__device__
void adjust_queues_before_split(uint lpid);

__device__
void split_queues(uint lpid);

__device__
void set_queues_params_after_merge();

__device__
void set_queues_params_after_split();

__device__
void print_event_queue(uint lpid);

// ********************
/* Event Queue */

__device__
void sort_event_queue(uint lpid);

__device__
char has_next_event(uint lpid);

__device__ // Make sure there is next event.
Event* get_next_event(uint lpid);

__device__ // Make sure there is next event.
void mark_next_event_as_processed(uint lpid);

#if (OPTM_SYNC == 1)
__device__
void set_lpts(uint lpid, int timestamp);

__device__
int get_lpts(uint lpid);
#endif

__device__ // Returns 0 if event queue is full.
char append_event_to_queue(Event *event);

__device__ // Returns 0 if event queue is full.
char append_event_to_queue(Event *event, uint *undo_offset);

#if (OPTM_SYNC == 1)
__device__
void undo_event(Event *event);
#endif

__device__
void undo_event(uint nid, uint undo_offset);

#if (OPTM_SYNC == 1)
__device__
int get_rollback_timestamp(uint lpid);
#endif

__device__
char has_processed_event(uint lpid);

__device__
Event* get_first_processed_event(uint lpid);

__device__
void delete_first_processed_event(uint lpid);

__device__
Event* get_last_processed_event(uint lpid);

__device__
void mark_last_processed_event_as_unprocessed(uint lpid);

// ********************
/* State Queue */

__device__
char state_queue_is_full(uint lpid);

__device__
void append_state_to_queue(State *state, uint lpid);

__device__
void delete_first_n_states(uint lpid, uint n);

__device__
State* delete_last_state(uint lpid);

// ********************
/* Antimsg Queue */

__device__
char antimsg_queue_is_full(uint lpid);

__device__
void append_antimsg_to_queue(Event *antimsg);

__device__
void delete_first_n_antimsgs(uint lpid, uint n);

__device__
Event* delete_last_antimsg(uint lpid);

#endif
