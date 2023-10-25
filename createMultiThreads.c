#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>
#include <pthread.h> 


struct ThreadInfo {
    float* a;
    float* b;
    float* r;
    int num;
    int thread_id;
    int num_threads; // Number of threads
};



void mult_std(float* a, float* b, float* r, int num) {
    for (int i = 0; i < num; i++) {
        r[i] = a[i] * b[i];
    }
}


void* worker_thread(void* arg) {
    struct ThreadInfo* thread_info = (struct ThreadInfo*)arg;

    // Calculate the range of array elements for this thread
    int start = thread_info->thread_id * (thread_info->num / thread_info->num_threads);
    int end = (thread_info->thread_id + 1) * (thread_info->num / thread_info->num_threads);

    // Call either mult_std or mult_vect here based on your choice
    mult_std(thread_info->a + start, thread_info->b + start, thread_info->r + start, end - start);

    // In the mult_vect case, remember to adjust for vectorization step size (4 elements per iteration)

    return NULL;
}


void mult_vect(float* a, float* b, float* r, int num) {
    float32x4_t va, vb, vr;
    for (int i = 0; i < num; i += 4) {
        va = vld1q_f32(&a[i]);
        vb = vld1q_f32(&b[i]);
        vr = vmulq_f32(va, vb);
        vst1q_f32(&r[i], vr);
    }
}

int main(int argc, char *argv[]) {
    int num = 100000000;
    int num_threads = 4; 
    float *a = (float*)aligned_alloc(16, num * sizeof(float));
    float *b = (float*)aligned_alloc(16, num * sizeof(float));
    float *r = (float*)aligned_alloc(16, num * sizeof(float));

    for (int i = 0; i < num; i++) {
        a[i] = (i % 127) * 0.1457f;
        b[i] = (i % 331) * 0.1231f;
    }

    pthread_t threads[num_threads];
    struct ThreadInfo thread_info[num_threads];


    struct timespec ts_start;
    struct timespec ts_end_1;
    struct timespec ts_end_2;

    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    for (int i = 0; i < num_threads; i++) {
        // Initialize the ThreadInfo struct for each thread
        thread_info[i].a = a;
        thread_info[i].b = b;
        thread_info[i].r = r;
        thread_info[i].num = num;
        thread_info[i].thread_id = i;
        thread_info[i].num_threads = num_threads;

        // Create the threads, and pass the ThreadInfo struct to the worker function
        pthread_create(&threads[i], NULL, worker_thread, &thread_info[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_end_1);

    mult_vect(a, b, r, num);

    clock_gettime(CLOCK_MONOTONIC, &ts_end_2);

    double duration_std = (ts_end_1.tv_sec - ts_start.tv_sec) +
                         (ts_end_1.tv_nsec - ts_start.tv_nsec) * 1e-9;
    double duration_vec = (ts_end_2.tv_sec - ts_end_1.tv_sec) +
                         (ts_end_2.tv_nsec - ts_end_1.tv_nsec) * 1e-9;

    printf("Elapsed time std: %f\n", duration_std);
    printf("Elapsed time vec: %f\n", duration_vec);

    free(a);
    free(b);
    free(r);

    return 0;
}
