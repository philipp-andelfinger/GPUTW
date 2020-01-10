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


#ifndef State_h
#define State_h

#include <curand_kernel.h>
#include "PARAMETERS.h"
#include "LookupResult.h"

typedef struct {
	curandState_t	cr_state;

	uint		lookup_pid;
	int		lookup_start_timestamp;

	uint		n_pending_requests;
	uint		n_requests_per_lookup;
	LookupResult	lookup_results[BUCKET_SIZE];

//	float		n_lookups;
//	float		n_requests_per_lookup_sum;
//	float		lookup_duration_sum;
//	float		min_distance_sum;
} State;

#endif
