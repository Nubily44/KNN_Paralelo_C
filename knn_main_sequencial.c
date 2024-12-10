// Bibliotecas:
#include <math.h> // Matemática
#include <stdio.h> // Padrão
#include <stdlib.h> // Padrão 
#include <strings.h> // Strings
#include <limits.h> // Limites
#include <float.h> // Limites
#include <time.h> // Tempo

// Parametros
#define K 3
#define W 4
#define H 1

#define TRAIN_COUNT 500

double *lerArquivo(char *nomeArquivo, int num_elementos){
     FILE *arquivo;
     arquivo = fopen(nomeArquivo, "r");
     if (arquivo == NULL) {
          printf("Erro ao abrir o arquivo: %s\n", nomeArquivo);
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

     double *y_test = (double *)malloc(test_numrows * sizeof(double));
     

     for(int i = 0; i < test_numrows; i++){

          double * vetor_distancias = (double *)malloc(train_numrows * sizeof(double));
          vetor_distancias = calculaDistancias(xTrain, xTest[i], train_numrows);
          double soma = 0;
          for(int j = 0; j<K; j++){
               double menor = DBL_MAX;
               int indicemenor = -1;


               for (int l = 0; l<train_numrows;l++){
                    if (vetor_distancias[l]<menor){
                    indicemenor = l;
                    menor = vetor_distancias[l];                   
                    }
               }

               vetor_distancias[indicemenor] = DBL_MAX;
               soma += yTrain[indicemenor];             
          }
          y_test[i] = soma/K;
          free(vetor_distancias);
     }
     return y_test;
}

int main(int argc,char *argv[]){

     FILE *filetempo;
     filetempo = fopen("tempos.txt", "a");

     FILE *fileytest;
     fileytest = fopen("ytest.txt", "w");

     if (argc != 3) {
          printf("Uso correto: %s <xtrain_datapath> <xtest_datapath> (que varia para cada teste)\n", argv[0]);
          return 1;
     }

     int test_count;
     printf("Quantos testes serao realizados?\n");
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

     double *bufferTrain = lerArquivo(argv[1], TRAIN_COUNT);
     double *bufferTest = lerArquivo(argv[2], test_count);

     criaMatrizes(xTrain, yTrain, xTest, bufferTrain, bufferTest, test_count);

     free(bufferTrain);
     free(bufferTest);

     clock_t start_time, end_time;
     double tempo_usado;
     start_time = clock();
     for(int i=0; i<5; i++){
          yTest = KNN(xTrain, xTest, yTrain, train_nrows, test_nrows);
     } 
     end_time = clock();

     tempo_usado = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
     tempo_usado /= 5;

     printf("Tempo de execucao: %lf\n", tempo_usado);

     fprintf(filetempo, "%lf\n", tempo_usado);

     for (int i = 0; i < test_nrows; i++) {
          fprintf(fileytest, "%lf\n", yTest[i]);
     }


     // I WANT TO BREEAAAAAK FREEEEEEEE
     for (int i = 0; i < train_nrows; i++) {
          free(xTrain[i]);
     }
     free(xTrain);

     for (int i = 0; i < test_nrows; i++) {
          free(xTest[i]);
     }
     free(xTest);

     free(yTrain);
     free(yTest);

     fclose(filetempo);
     fclose(fileytest);
     return 0;
}