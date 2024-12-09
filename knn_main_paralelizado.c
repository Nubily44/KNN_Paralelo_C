// Bibliotecas:
#include <math.h> // Matemática
#include <stdio.h> // Padrão
#include <stdlib.h> // Padrão 
#include <strings.h> // Strings
#include <limits.h> // Limites
#include <omp.h>    // OpenMP

// Parametros
#define K 3
#define W 4
#define H 1

#define NUM_THREADS 12
#define TRAIN_COUNT 500

double *lerArquivo(char *nomeArquivo, int num_elementos){
     FILE *arquivo;
     arquivo = fopen(nomeArquivo, "r");
     if (arquivo == NULL) {
          printf("Erro ao abrir o arquivo.\n");
          return NULL;
     }

     double *valores = (double *) malloc(num_elementos * sizeof(double));

     for (int i = 0; i < num_elementos; i++) {
          fscanf(arquivo, "%lf", &valores[i]);
     }
     fclose(arquivo);
     return valores;
}

void criaMatrizes(double **xtrain, double *ytrain, double **xtest, double *bufferTrain, double *bufferTest, int testCount){
     
     int base = 0;
     // linha, coluna

     for (int i=0;i<TRAIN_COUNT - W - H + 1;i++){
          for (int j=0;j<W;j++){
               xtrain[i][j] = bufferTrain[base+j];
               
          }
          ytrain[i] = bufferTrain[base+W];
          base++;
     }
     base = 0;
     for (int i=0;i<testCount - W - H + 1;i++){
          for (int j=0;j<W;j++){
               xtest[i][j] = bufferTest[base+j];
          }
          base++;
     }
}

double * calculaDistancias(double** xTrain, double* linha_Xtest, int train_numrows){ 
     double *distancias = (double *)malloc(train_numrows * sizeof(double)); 
     double temp;
     #pragma omp parallel for private(temp)
     for(int i = 0; i < train_numrows; i++){
          temp = 0;
          for (int j = 0; j < W; j++){
               temp += pow(xTrain[i][j] - linha_Xtest[j], 2);
          }
     distancias[i] = sqrt(temp);
     }
     return distancias;
}

double * KNN (double** xTrain, double** xTest, double* yTrain, int train_numrows, int test_numrows){

     double *y_test = (double *)malloc(train_numrows * sizeof(double));
     double * vetor_distancias = (double *)malloc(train_numrows * sizeof(double)); 

     #pragma omp parallel for private(vetor_distancias, soma, menor, indicemenor, l)
     for(int i = 0; i < test_numrows; i++){

          vetor_distancias = calculaDistancias(xTrain, xTest[i], train_numrows);  
          printf("AAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n");
          double soma = 0;
          for(int j = 0; j<K; j++){
               double menor = INT_MAX;
               int indicemenor = -1;


               for (int l = 0; l<train_numrows;l++){
                    if (vetor_distancias[l]<menor){
                    indicemenor = l;
                    menor = vetor_distancias[l];                   
                    }
               }

               vetor_distancias[indicemenor] = INT_MAX;
               soma += yTrain[indicemenor];             
          }
          y_test[i] = soma/K;
          
          free(vetor_distancias);
     }

     return y_test;
}

 


int main(int argc,char *argv[]){

     omp_set_num_threads(NUM_THREADS);

     FILE *filetempo;
     filetempo = fopen("tempos.txt", "w");

     if (argc != 3) {
          printf("Uso correto: %s <xtrain_datapath> <xtest_datapath> (que varia para cada teste)\n", argv[0]);
          return 1;
     }
     
     int test_count;
     printf("Quantos testes serão realizados?\n");
     scanf("%d", &test_count);
     int train_nrows = TRAIN_COUNT - W - H + 1;
     int test_nrows = test_count - W - H + 1;


     double **xTrain = (double **) malloc(train_nrows * sizeof(double *)); 
     for (int i = 0; i < train_nrows; i++) {
          xTrain[i] = (double *) malloc(W * sizeof(double));
     }

     double *yTrain = (double *) malloc(TRAIN_COUNT * sizeof(double));

     double **xTest = (double **) malloc(test_nrows * sizeof(double *));
     for (int i = 0; i < test_nrows; i++) {
          xTest[i] = (double *) malloc(W * sizeof(double));
     }

     double *yTest = (double *) malloc(TRAIN_COUNT * sizeof(double));

     double *bufferTrain = (double *) malloc(TRAIN_COUNT * sizeof(double));
     double *bufferTest = (double *) malloc(test_count * sizeof(double));

     bufferTrain = lerArquivo(argv[2], TRAIN_COUNT);
     bufferTest = lerArquivo(argv[3], test_count);

     criaMatrizes(xTrain, yTrain, xTest, bufferTrain, bufferTest, test_count);

     double start_time, end_time;

     start_time = omp_get_wtime();

     yTest = KNN(xTrain, xTest, yTrain, train_nrows, test_nrows);

     end_time = omp_get_wtime();

     printf("KNN computation time = %f seconds\n", end_time - start_time);

     return 0;
}