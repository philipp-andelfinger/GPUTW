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


#ifndef model_h
#define model_h

#include <stdio.h>
#include "../main.h"
#include "../queues.h"
#include "../random.h"
#include "PARAMETERS.h"
#include "Event.h"
#include "State.h"

typedef struct {
	curandState_t	*cr_state;

	uint		*pid;
	uint		*buckets;
	uint		*last_bucket_id;

	uint		*lookup_pid;
	int		*lookup_start_timestamp;
	uint		*n_pending_requests;
	uint		*n_requests_per_lookup;
	LookupResult	*lookup_results;

	// Statistics
	float		*n_lookups;
	float		*n_requests_per_lookup_sum;
	float		*lookup_duration_sum;
	float		*min_distance_sum;
} Nodes;

char malloc_nodes(uint n_nodes);

void free_nodes();

__device__
void set_model_params(int params[], uint n_params);

__device__
int get_lookahead();

__device__
void init_node(uint nid);

__device__
char handle_event(Event *event);

__device__
void roll_back_event(Event *event);

__device__
uint get_number_states(Event *event);

__device__
uint get_number_antimsgs(Event *event);

__device__
void collect_statistics(uint nid);

__device__
void print_statistics();

#endif
