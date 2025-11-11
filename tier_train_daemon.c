#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <sys/mman.h>
#include <numaif.h>
#include <errno.h>
#include <signal.h>
#include <stddef.h>

#define GET_CHUNK_POINTER(base_ptr, element_type, chunk_size, chunk_number) \
    ((element_type*)((char*)(base_ptr) + (chunk_size) * (chunk_number) * sizeof(element_type)))


#define MAX_OBJECT_NAME_LEN 1000
#define MAX_OBJECTS 1000
// Number of threads to migrate cache objects
#define NUM_THREADS 10
#define NUM_CHUNKS_PER_THREAD 200
/*
#define ST // STAY_TIME

 */


#define FAST_MEM_NODE 0
#define SLOW_MEM_NODE 3

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

int start_time = 0;
int current_epoch = 0;
int layers = 0;
int fwd_id = 0;
int bwd_id = 0;
int FWD = 0;
int BWD = 0;

struct ThreadArgs {
    double start_time;
    int pid;
    const char* out_file;
    const char* out_cache_file;
};

struct MonitorThreadArgs {
    const char* filename;
    int pid;
};

double bytesToGB(unsigned long long bytes) {
    return (double)bytes / (1024 * 1024 * 1024);
}


struct ThreadData {
    int pid;
    unsigned long count;	// chunk size
    void** pages;		// pointer to corresponding chunk of page pointers
    int* nodes; 		// pointer to corresponding chunk of node array
    int* status; 		// pointer to corresponding chunk of status array 
    int num_threads;
    int num_chunks;
};

int eff_bwd_id(int bwd_id, int num_layers) {
    // Epoch number is (((bwd_id - 1) / num_layers) + 1)
    return ((((bwd_id - 1) / num_layers) + 1) * num_layers) - ((bwd_id - 1) % num_layers);
}


double get_monotonic_time() {
    struct timespec current_time;

    if (clock_gettime(CLOCK_MONOTONIC, &current_time) != 0) {
        perror("clock_gettime");
        return -1.0; // Return -1.0 to indicate an error
    }

    return current_time.tv_sec + current_time.tv_nsec / 1e9;
}


void* move_pages_chunk(void* arg) {
    struct ThreadData* data = (struct ThreadData*)arg;

    int pid = data->pid;
    int count = data->count;
    void** pages = data->pages;
    int* nodes = data->nodes;
    int* status = data->status;
    int num_chunks = data->num_chunks;


    int chunk_size = count / num_chunks;


    int* result = (int*)malloc(sizeof(int));
    *result = 0;

    for (int i = 0; i < num_chunks; ++i) {
        void** pages_chunk = GET_CHUNK_POINTER(pages, void*, chunk_size, i);
        int* nodes_chunk;
        if (nodes != NULL) {
            //printf("Nodes chunk is not NULL\n");
            nodes_chunk = GET_CHUNK_POINTER(nodes, int, chunk_size, i);
        }
        else {
            //printf("Nodes chunk is NULL\n");
            nodes_chunk = NULL;
        }
        int* status_chunk = GET_CHUNK_POINTER(status, int, chunk_size, i);
        *result = *result | move_pages(pid, chunk_size, pages_chunk, nodes_chunk, status_chunk, MPOL_MF_MOVE);

        //*result = *result | 0;
    }


    return (void*)result;
}



int get_pages_nodes(unsigned long int virtual_address, size_t num_pages, double start_time, int pid, const char* out_file, char* name, int id, const char* out_cache_file, int dest, int* moved) {

    //pthread_mutex_lock(&lock);

    if (*moved == 0) {
        FILE* file = fopen(out_file, "a");
        FILE* file_cache = fopen(out_cache_file, "a");

        double start_t = get_monotonic_time();
        unsigned long count = num_pages;
        void** pages = (void**)malloc(count * sizeof(void*));
        int* status = (int*)malloc(count * sizeof(int));

        int* nodes = (int*)malloc(count * sizeof(int));
        int result = 0;


        int num_threads;
        if (dest == 0) {
            num_threads = 10;

        }
        else {
            num_threads = 10;
        }

        int chunk_size = num_pages / num_threads;


        if (!pages || !status) {
            perror("malloc");
            exit(EXIT_FAILURE);
        }

        // Populate the pages array with virtual addresses
        for (size_t i = 0; i < num_pages; ++i) {
            pages[i] = (void*)(virtual_address + i * getpagesize());
        }


         for (size_t i = 0; i < num_pages; ++i) {
            nodes[i] = dest;
        }

        if (dest == FAST_MEM_NODE) {
            fprintf(file_cache, "Epoch: %d, FWD: %d, BWD: %d, FWD ID: %d, BWD ID: %d\n", current_epoch, FWD, BWD, fwd_id, bwd_id);
            fprintf(file_cache, "Prefetch on %s\n", name);
        }
        else if (dest == SLOW_MEM_NODE) {
            fprintf(file_cache, "Epoch: %d, FWD: %d, BWD: %d, FWD ID: %d, BWD ID: %d\n", current_epoch, FWD, BWD, fwd_id, bwd_id);
            fprintf(file_cache, "Eviction on %s\n", name);
        }
        else {
            if (strstr(name, "Cache") != NULL) {
                fprintf(file_cache, "Epoch: %d, FWD: %d, BWD: %d, FWD ID: %d, BWD ID: %d\n", current_epoch, FWD, BWD, fwd_id, bwd_id);
                fprintf(file_cache, "Testing on %s\n", name);
            }
            else {
                fprintf(file, "Epoch: %d, FWD: %d, BWD: %d, FWD ID: %d, BWD ID: %d\n", current_epoch, FWD, BWD, fwd_id, bwd_id);
                fprintf(file, "Testing on %s\n", name);
            }
        }


        // pthread_t threads[num_threads];
        // struct ThreadData thread_data[num_threads];

        pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
        if (threads == NULL) {
            perror("Failed to allocate memory for threads");
            exit(1);
        }

        // Dynamic allocation of thread_data array
        struct ThreadData* thread_data = (struct ThreadData*)malloc(num_threads * sizeof(struct ThreadData));
        if (thread_data == NULL) {
            perror("Failed to allocate memory for thread data");
            free(threads); // Clean up previously allocated memory
            exit(1);
        }

        // Spawn threads for parallel move_pages
        for (int i = 0; i < num_threads; ++i) {
            void** pages_chunk = GET_CHUNK_POINTER(pages, void*, chunk_size, i);
            int* nodes_chunk;
            if (dest >= 0) {
                nodes_chunk = GET_CHUNK_POINTER(nodes, int, chunk_size, i);
            }
            else {
                nodes_chunk = NULL;
            }
            int* status_chunk = GET_CHUNK_POINTER(status, int, chunk_size, i);

            thread_data[i].pid = pid;
            thread_data[i].count = chunk_size;
            thread_data[i].pages = pages_chunk;
            thread_data[i].nodes = nodes_chunk;
            thread_data[i].status = status_chunk;

            if (dest == 0) {

                thread_data[i].num_chunks = 100;

            }
            else {

                thread_data[i].num_chunks = 150;
            }

            int thread_create_result = pthread_create(&threads[i], NULL, move_pages_chunk, (void*)&thread_data[i]);

            if (thread_create_result != 0) {
                fprintf(stderr, "Error creating thread %d\n", i);
                exit(EXIT_FAILURE);
            }
        }


        for (int i = 0; i < num_threads; ++i) {
            int* result_ptr;
            pthread_join(threads[i], (void**)&result_ptr);

            if (result_ptr != NULL) {
                result = result || *result_ptr;
                free(result_ptr);
            }
            else {
                result = 1;
                break;
            }
        }

        // Comment this later
        //result = 1;

        if (dest >= 0) {
            if (result == 0) {
                fprintf(file_cache, "Transfer is successful\n");
            }
            else {
                fprintf(file_cache, "Transfer is unsuccessful\n");
            }
        }
        else {
            if (result == 0) {
                fprintf(file, "Transfer is successful\n");
            }
            else {
                fprintf(file, "Transfer is unsuccessful\n");
            }
        }

        double end_t = get_monotonic_time();
        double elapsed_time = end_t - start_t;


        if (dest >= 0)
        {
            fprintf(file_cache, "Time taken: %.2f seconds\n", elapsed_time);
        }
        else {
            fprintf(file, "Time taken: %.2f seconds\n", elapsed_time);
        }


        int current_node = status[0];
        size_t start_page = 0;

        int count_dram = 0;
        int count_optane = 0;
        int count_fault = 0;
        int total_count = 0;
        int flag = 0;

        if (strstr(name, "Cache") != NULL) {
            flag = 1;
        }

        for (size_t i = 0; i < num_pages; ++i) {
            if (strstr(name, "Cache") != NULL && !(status[i] < 0) && (status[i] != current_node)) {
                // fprintf(file_cache, "%lf,%p - %p,%d\n", (double)(time(NULL) - start_time), pages[start_page], pages[i], status[i]);
                current_node = status[i];
                start_page = i;
            }

            if (status[i] == 0 || status[i] == 1) {
                count_dram += 1;
                total_count += 1;
            }
            else if (status[i] == 2 || status[i] == 3) {
                count_optane += 1;
                total_count += 1;
            }
            else {
                count_fault += 1;
                total_count += 1;
            }

        }

        if (start_page == 0 && strstr(name, "Cache") != NULL) {
            fprintf(file_cache, "%lf,%p - %p,%d\n", (double)(time(NULL) - start_time), pages[start_page], pages[num_pages - 1], current_node);
            fprintf(file_cache, "------------------------------------------------------------\n");
        }

        if (flag == 0 && total_count != 0) {
            fprintf(file, "Object: %s, ID: %d\n", name, id);
            fprintf(file, "DRAM percent = %.2f, Optane percent = %.2f\n", ((double)count_dram / total_count) * 100, ((double)count_optane / total_count) * 100);
            fprintf(file, "------------------------------------------------------------\n");
        }

        if (flag == 1 && total_count != 0) {
            fprintf(file_cache, "Object: %s, ID: %d\n", name, id);
            fprintf(file_cache, "DRAM percent = %.2f, Optane percent = %.2f\n", ((double)count_dram / total_count) * 100, ((double)count_optane / total_count) * 100);
            fprintf(file_cache, "------------------------------------------------------------\n");
        }




        fclose(file);
        fclose(file_cache);
        free(pages);
        free(status);
        free(nodes);

        //   pthread_mutex_unlock(&lock);

        return result;
    }

    //pthread_mutex_unlock(&lock);

    return 0;


}

void removeLastNCharacters(char* str, int n) {
    int len = strlen(str);

    if (len >= n) {
        str[len - n] = '\0';
    }
    else {
        printf("Error: String is too short to remove %d characters.\n", n);
    }
}

struct Object {
    char name[MAX_OBJECT_NAME_LEN];
    unsigned long int addr;
    int pages;
    int id;
    int evicted;
    int prefetched;
};

struct Object objects_dict[MAX_OBJECTS];
int objects_count = 0;

int get_layer(int obj_id) {
    return (obj_id % layers + (!(obj_id % layers)) * layers);
}

int get_epoch(int obj_id) {
    if (obj_id % layers != 0) {
        return (obj_id / layers + 1);
    }
    else {
        return (obj_id / layers);
    }
}

/* Function to evict data from DRAM to Optane */
void evict_data(void* arg) {
    int i;
    int current_layer;

    struct ThreadArgs* thread_args = (struct ThreadArgs*)arg;
    double start_time = thread_args->start_time;
    int pid = thread_args->pid;
    const char* out_file = thread_args->out_file;
    const char* out_cache_file = thread_args->out_cache_file;

    while (1) {
        if (kill(pid, 0) == -1 && current_epoch > 0) {
            perror("TierTrain: Training workload has ended\n");
            exit(EXIT_FAILURE);
        }

        if (current_epoch == 0 || layers == 0) {
            // You might want to sleep here to avoid busy waiting
            sleep(1);
            continue;
        } else {
            FILE* file_cache = fopen(out_cache_file, "a");
            if (file_cache == NULL) {
                perror("Failed to open cache file");
                continue;
            }

            for (i = 0; i < objects_count; ++i) {
                struct Object obj = objects_dict[i];

                if (get_layer(obj.id) == layers || obj.evicted) {
                    continue;
                }

                current_layer = FWD ? get_layer(fwd_id) : get_layer(bwd_id);

                if (get_epoch(obj.id) == current_epoch) {
                    if (FWD) {
                        if (strstr(obj.name, "Cache") != NULL) {
                            if (get_layer(obj.id) == current_layer - 1 && obj.evicted != 1) {

                                int result = get_pages_nodes(obj.addr, obj.pages, start_time, pid, 
                                                            out_file, obj.name, obj.id, out_cache_file, SLOW_MEM_NODE, &obj.evicted);

                                objects_dict[i].evicted = !result;

                                if (result != 0) {
                                    fprintf(file_cache, "Either error if -ve or %d pages have not been moved\n\n", result);
                                }
                            }
                        }
                    }
                }
            }
            fclose(file_cache);
        }
        // uncomment this sleep to avoid busy spinning
        // sleep(2);
    }
}


/* Function to prefetch data from Optane to DRAM */
void prefetch_data(void* arg) {
    int i;
    int current_layer;

    struct ThreadArgs* thread_args = (struct ThreadArgs*)arg;
    double start_time = thread_args->start_time;
    int pid = thread_args->pid;
    const char* out_file = thread_args->out_file;
    const char* out_cache_file = thread_args->out_cache_file;


    while (1) {
        if (kill(pid, 0) == -1 && current_epoch > 0) {
            perror("TierTrain: Training workload has ended\n");
            exit(EXIT_FAILURE);
        }

        if (current_epoch == 0 || layers == 0) {
        }
        else if (BWD) {
            FILE* file_cache = fopen(out_cache_file, "a");
            for (i = 0; i < objects_count; ++i) {
                struct Object obj = objects_dict[i];

                if (obj.prefetched == 1) {
                    continue;
                }

                current_layer = get_layer(bwd_id);

                if (current_layer == 1) {
                    continue;
                }

                if (get_epoch(obj.id) == current_epoch) {
                    if ((get_layer(obj.id) == current_layer - 1) && strstr(obj.name, "Cache") != NULL) {

                        int result = get_pages_nodes(obj.addr, obj.pages, start_time, pid, out_file, obj.name, obj.id, out_cache_file, FAST_MEM_NODE, &obj.prefetched);
                        objects_dict[i].prefetched = !result;

                        if (result != 0) {
                            fprintf(file_cache, "Either error if -ve or %d pages have not been moved\n\n", result);
                        }

                    }
                }
            }
            fclose(file_cache);
        }
        else
        {
            // do nothing
        }

        // uncomment later
// sleep(2);
    }
}


/*
    Function to get node location of saved tensors at any instant
*/
void process_dict(void* arg) {
    int i;
    int current_layer;

    struct ThreadArgs* thread_args = (struct ThreadArgs*)arg;
    double start_time = thread_args->start_time;
    int pid = thread_args->pid;
    const char* out_file = thread_args->out_file;
    const char* out_cache_file = thread_args->out_cache_file;



    while (1) {
        if (current_epoch == 0 || layers == 0) {
        }
        else {
            FILE* file = fopen(out_file, "a");
            FILE* file_cache = fopen(out_cache_file, "a");
            for (i = 0; i < objects_count; ++i) {
                struct Object obj = objects_dict[i];
                current_layer = FWD ? get_layer(fwd_id) : get_layer(bwd_id);

                // if (get_epoch(obj.id) == current_epoch) {
                    //if (FWD) {
                        // if (get_layer(obj.id) == get_layer(fwd_id) || strstr(obj.name, "Cache") != NULL) {

                            //if(strstr(obj.name, "Cache") == NULL){
                                    // fprintf(file_cache, "Currently executing Epoch: %d Layer : %d FWD\n", current_epoch, current_layer);
                                    // fprintf(file_cache, "T = %lf : Testing on %s (%lu) Epoch = %d Layer = %d Phase = %c%c%c\n\n",
                                    // (double)(time(NULL) - start_time), obj.name, obj.addr,  get_epoch(obj.id), get_layer(obj.id), obj.name[0], obj.name[1], obj.name[2]);
                int moved = 0;
                get_pages_nodes(obj.addr, obj.pages, start_time, pid, out_file, obj.name, obj.id, out_cache_file, -1, &moved);
                //  }
              // else {
              //     fprintf(file, "Currently executing Epoch: %d Layer : %d FWD\n", current_epoch, current_layer);
              //     fprintf(file, "T = %lf : move_pages on %s (%lu) Epoch = %d Layer = %d Phase = %c%c%c\n\n",
              //         (double)(time(NULL) - start_time), obj.name, obj.addr, get_epoch(obj.id), get_layer(obj.id), obj.name[0], obj.name[1], obj.name[2]);
              //     get_pages_nodes(obj.addr, obj.pages, start_time, pid, out_file, obj.name, obj.id, out_cache_file, -1, NULL);
              // }
          //}
  //     } 
  //}

            }
            fclose(file);
            fclose(file_cache);
        }
        sleep(3);
    }
}



/*  Function to monitor the input tensor and layer execution data.
    Stores the tensors in a queue that can be accessed by the eviction and the prefetching threads
*/
void monitor_file(void* arg) {
    struct MonitorThreadArgs* args = (struct MonitorThreadArgs*)arg;
    const char* filename = args->filename;
    int pid = args->pid;

    FILE* file;
    char line[256];
    char* token;
    char* phase;
    FWD = 0;
    BWD = 0;
    int file_position = 0;
    int id;

    while (1) {
        if (kill(pid, 0) == -1 && current_epoch > 0) {
            perror("TierTrain: Training workload has ended\n");
            exit(EXIT_FAILURE);
        }

        file = fopen(filename, "r");

        if (fseek(file, file_position, SEEK_SET) != 0) {
            printf("Error seeking in file");
            fclose(file);
        }

        while (fgets(line, sizeof(line), file) != NULL) {
            if (strstr(line, "Layers")) {
                sscanf(line, "Layers : %d", &layers);
            }
            else if (strstr(line, "Epoch")) {
                sscanf(line, "Epoch : %d", &current_epoch);
                /*
                    if current_epoch == 2
                        compute layer wise execution time from collected timestamps in epoch 1
                        fill it in the quite time array (qt_l)
                */
            }
            else if (strstr(line, "FWD ID")) {
                /*
                    Get timestamp here for FWD Layer - FWD ID
                */
                sscanf(line, "FWD ID :  %d", &fwd_id);
                FWD = 1;
                BWD = 0;
            }
            else if (strstr(line, "BWD ID")) {
                sscanf(line, "BWD ID: %d", &bwd_id);
                //bwd_id = bwd_id + (current_epoch - 1) * layers;
                /*
                    Get timestamp here for BWD Layer - BWD ID
                */
                bwd_id = eff_bwd_id(bwd_id, layers);
                FWD = 0;
                BWD = 1;
            }
            else if (strstr(line, "Object")) {
                struct Object obj;
                char num_str[10];
                char object_key[100];
                obj.name[0] = '\0';

                if (FWD) {
                    phase = "FWD";
                    id = fwd_id;
                }
                else {
                    phase = "BWD";
                    id = bwd_id;
                }

                obj.id = id;
                obj.evicted = 0;
                obj.prefetched = 0;
                sprintf(num_str, "%d", id);
                sscanf(line, "Object: %s %ld, %d\n", obj.name, &obj.addr, &obj.pages);
                removeLastNCharacters(obj.name, 1);
                strcpy(object_key, phase);
                strcat(object_key, num_str);
                strcat(object_key, obj.name);
                strcpy(obj.name, object_key);
                objects_dict[objects_count++] = obj;
            }
        }

        file_position = ftell(file);
        sleep(1);
        fclose(file);
    }
}

/*
Input file: Contains the layer execution information and tensor data
Output file: NOT USED
Output file cache: Status of eviction adn prefetching
 */
int main(int argc, char* argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <input filename> <output filename> <output_filename_cache> pid\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const char* filename = argv[1];
    const char* out_file = argv[2];
    const char* out_cache_file = argv[3];
    int pid = atoi(argv[4]);

    sleep(5);


    FILE* start_time_file = fopen("log_files/utc_start_time.txt", "r");
    fscanf(start_time_file, "%d", &start_time);
    fclose(start_time_file);

    FILE* file = fopen(out_file, "w");
    fclose(file);

    struct ThreadArgs thread_args;
    thread_args.start_time = start_time;
    thread_args.pid = pid;
    thread_args.out_file = out_file;
    thread_args.out_cache_file = out_cache_file;

    struct MonitorThreadArgs monitor_args;
    monitor_args.filename = filename;
    monitor_args.pid = pid;

    pthread_t monitor_thread, process_dict_thread, evict_data_thread, prefetch_data_thread;

    pthread_create(&monitor_thread, NULL, (void* (*)(void*))monitor_file, (void*)&monitor_args);
    //pthread_create(&process_dict_thread, NULL, (void *(*)(void *))process_dict, (void *)&thread_args);
    pthread_create(&evict_data_thread, NULL, (void* (*)(void*))evict_data, (void*)&thread_args);
    pthread_create(&prefetch_data_thread, NULL, (void* (*)(void*))prefetch_data, (void*)&thread_args);

    pthread_join(monitor_thread, NULL);
    //pthread_join(process_dict_thread, NULL);
    pthread_join(evict_data_thread, NULL);
    pthread_join(prefetch_data_thread, NULL);

    /*
    Create another thread that reads the realtime available DRAM (BW_D) and PMM (BW_PMM) bandwidth?
    Compute IBW = ((1/BW_D) + (1/BW_O))
    */

    return 0;
}
