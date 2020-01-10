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

#ifndef kernels_h
#define kernels_h

#include <stdio.h>
#include "main.h"
#include "queues.h"
#include "SETTINGS.h"
#include MODEL_HEADER
#include EVENT_HEADER

__global__
void kernel_set_params(
uint n_nodes, uint n_lps, uint nodes_per_lp,
uint events_per_node, uint states_per_node, uint antimsgs_per_node,
int model_params[], uint n_params);

__global__
void kernel_get_lookahead(int *lookahead);

__global__
void kernel_init_queues();

__global__
void kernel_init_nodes();

__global__
void kernel_handle_next_event(int gvt, int window_size,
uint *n_inac_1, uint *n_inac_2, uint *n_inac_3,
uint *n_inac_4, uint *n_inac_5, uint *n_inac_6);

#if (OPTM_SYNC == 1)
__global__
void kernel_roll_back(char *rollback_performed);

__global__
void kernel_roll_back_all();
#endif

__global__
void kernel_rotate_queues();

__global__
void kernel_merge_queues();

__global__
void kernel_adjust_queues_after_merge();

__global__
void kernel_check_queues_before_split(char *can_split);

__global__
void kernel_adjust_queues_before_split();

__global__
void kernel_split_queues();

__global__
void kernel_set_params_after_merge();

__global__
void kernel_set_params_after_split();

__global__
void kernel_sort_event_queues();

__global__
void kernel_clean_queues(uint gvt, uint *n_events_cleaned);

__global__
void kernel_collect_statistics();

__global__
void kernel_print_statistics();

__global__
void kernel_get_gvt_1(int *ts_temp);

__global__
void kernel_get_gvt_2(int *ts_temp, uint n, uint distance);

__global__
void kernel_print_event_queue(uint lpid);

#endif
