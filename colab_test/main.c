/*
 * Copyright (c) 2018-2020, Jianjia Ma
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2019-03-29     Jianjia Ma   first implementation
 *
 * Notes:
 * This is a keyword spotting example using NNoM
 *
 */

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "nnom.h"
#include "kws_weights.h"

#include "mfcc.h"
#include "math.h"

#ifndef NNOM_USING_EFABLESS_ACCELERATOR
#define     NNOM_USING_EFABLESS_ACCELERATOR
#endif

#ifdef NNOM_USING_STATIC_MEMORY
    uint8_t static_buf[200500];
#endif


// NNoM model
nnom_model_t *model;

// 10 labels-1
const char label_name[][30] =  {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown"};
#define NUM_CLASSES 11  // Adjust this based on your label count
mfcc_t mfcc;
char ground_truth[12000][10];
int dma_audio_buffer[AUDIO_FRAME_LEN]; //512
int16_t audio_buffer_16bit[(int)(AUDIO_FRAME_LEN)]; // an easy method for 50% overlapping

float mfcc_features_f[MFCC_COEFFS];             // output of mfcc
int8_t mfcc_features[MFCC_LEN][MFCC_COEFFS];     // ring buffer
uint32_t mfcc_feat_index = 0;

void thread_kws_serv(int *frame);

int check_label_in_array(const char *label) {
    const char label_name[][10] =  {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown"};
    int num_labels = sizeof(label_name) / sizeof(label_name[0]);

    for (int i = 0; i < num_labels; i++) {
        if (strcmp(label, label_name[i]) == 0) {
            return 1;  // Found a match
        }
    }

    return 0;  // No match found
}

int confusion_matrix[NUM_CLASSES][NUM_CLASSES] = {0};
// 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight','five', 'follow', 'forward',
                    //  'four','go','happy','house','learn','left','marvin','nine','no','off','on','one','right',
                    //  'seven','sheila','six','stop','three','tree','two','up','visual','yes','zero'
char Unknown_labels[][30]= {"backward", "bed", "bird", "cat", "dog", "eight","five", "follow", "forward",
                    "four","happy","house","learn","marvin","wow","nine","one","seven","sheila","six","three","tree","two","visual","zero"};
// Helper function to get the index of a label
int get_label_index(const char *label, char label_name[][30], int num_labels) {
    char temp_label[30];
    for (int i = 0; i < 25; i++) {
        if (strcmp(label, Unknown_labels[i]) == 0) {
            return 10;
        }
    }
    for (int i = 0; i < num_labels; i++) {
        if (strcmp(label, label_name[i]) == 0) {
            return i;
        }
    }
    return -1;  // Label not found
}

void print_confusion_matrix(int matrix[NUM_CLASSES][NUM_CLASSES], char label_name[][30], int num_labels) {
    printf("\nConfusion Matrix:\n");
    printf("      ");
    for (int i = 0; i < num_labels; i++) {
        printf("%10s", label_name[i]);
    }
    printf("\n");

    for (int i = 0; i < num_labels; i++) {
        printf("%10s", label_name[i]);
        for (int j = 0; j < num_labels; j++) {
            printf("%10d", matrix[i][j]);
        }
        printf("\n");
    }
}

void calculate_metrics(int matrix[NUM_CLASSES][NUM_CLASSES], char label_name[][30], int num_labels) {
    float total_correct = 0, total_samples = 0;
    printf("\nMetrics:\n");
    for (int i = 0; i < num_labels; i++) {
        int tp = matrix[i][i];
        int fn = 0, fp = 0, tn = 0;

        for (int j = 0; j < num_labels; j++) {
            if (j != i) {
                fn += matrix[i][j];
                fp += matrix[j][i];
            }
            for (int k = 0; k < num_labels; k++) {
                if (j != i && k != i) {
                    tn += matrix[j][k];
                }
            }
        }

        float precision = (tp + fp > 0) ? (float)tp / (tp + fp) : 0;
        float recall = (tp + fn > 0) ? (float)tp / (tp + fn) : 0;
        float accuracy = (tp + tn > 0) ? (float)(tp + tn) / (tp + tn + fp + fn) : 0;

        printf("Label: %s\n", label_name[i]);
        printf("  Precision: %.2f\n", precision * 100);
        printf("  Recall: %.2f\n", recall * 100);
        printf("  Accuracy: %.2f\n\n", accuracy * 100);

        total_correct += tp;
        total_samples += tp + fn;
    }

    printf("Overall Accuracy: %.2f%%\n", (total_correct / total_samples) * 100);
}

int main(void)
{

    
#ifdef NNOM_USING_STATIC_MEMORY
    // when use static memory buffer, we need to set it before create
    nnom_set_static_buf(static_buf, sizeof(static_buf));
#endif

    

    // create and compile the model
    model = nnom_model_create();
    mfcc_create(&mfcc, MFCC_COEFFS_LEN, MFCC_COEFFS_FIRST, MFCC_TOTAL_NUM_BANK, AUDIO_FRAME_LEN, 0.97f, true);


    uint32_t label;
    float prob;
    int audios_processed = 0;
    int correct_audios = 0;
    FILE * file;
    FILE * ground_truth_f;
    
    // Open files for reading audio data and ground truth labels
    file = fopen ("Colab_Test_Audio/test_x.bin","r"); //the audio data stored in a textfile
    ground_truth_f = fopen ("Colab_Test_Audio/test_y.bin","r"); //the ground truth textfile
    // Read ground truth labels from file into an array
    for (int j = 0; fscanf(ground_truth_f, "%s", ground_truth[j++]) != EOF;) ;
    fclose (ground_truth_f);
    int one_audio_data[16000];



    // Main loop for processing audio samples and performing inference
    while (1)
    {
        // Read audio samples from file into an array
        for (int p = 0; p < 16000 ; fscanf(file, "%d", &one_audio_data[p++]));

        for (mfcc_feat_index = 0; mfcc_feat_index<49; mfcc_feat_index++)
            thread_kws_serv(&one_audio_data[mfcc_feat_index* AUDIO_SLIDE_LEN]);

        // Copy MFCC features to input data buffer for the neural network
        memcpy(nnom_input_data, mfcc_features, MFCC_FEAT_SIZE);
        
        // Make a prediction using the neural network
        nnom_predict(model, &label, &prob);
        printf("%d %s : %d%% - Ground Truth is: %s\n", audios_processed, (char*)&label_name[label], (int)(prob * 100),ground_truth[audios_processed]);
        
        int y_true_index = get_label_index(ground_truth[audios_processed], label_name, NUM_CLASSES);
        int y_pred_index = label;
        if (y_true_index != -1 && y_pred_index != -1) {
            confusion_matrix[y_true_index][y_pred_index]++;
            if (y_true_index == y_pred_index) {
                correct_audios++;
                printf("\t Correct \n");
            } else {
                printf("\t Wrong \n");
            }
        } else {
            printf("\t Error in label matching \n");
            printf("\t y_true_index: %d, y_pred_index: %d\n", y_true_index, y_pred_index);
        }
        
        // Reset audio sample index and frame size
        audios_processed++;// Increment sample counter
        // break;
        // Exit the loop when a sufficient number of samples are processed
        if(audios_processed>=11000) break;
    }
    
    printf("Accuracy: %.6f%%\n", ((float)correct_audios / audios_processed) * 100);

    print_confusion_matrix(confusion_matrix, label_name, NUM_CLASSES);
    calculate_metrics(confusion_matrix, label_name, NUM_CLASSES);
    
    // Close the file containing audio samples
    fclose(file);
}
void thread_kws_serv(int *frame)
{
    for(int i = 0; i < AUDIO_FRAME_LEN; i++)
        audio_buffer_16bit[i] = frame[i];

    for(int i = AUDIO_FRAME_LEN; i < 512; i++)
        audio_buffer_16bit[i] = 0;

    mfcc_compute(&mfcc, audio_buffer_16bit, mfcc_features_f);
    quantize_data(mfcc_features_f, mfcc_features[mfcc_feat_index], MFCC_COEFFS, 0);
}