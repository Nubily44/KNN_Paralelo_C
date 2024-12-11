// Bibliotecas:
#include <math.h> // Matemática
#include <stdio.h> // Padrão
#include <stdlib.h> // Padrão 
#include <strings.h> // Strings
#include <limits.h> // Limites
#include <float.h> // Limites
#include <omp.h>    // OpenMP

// Parametros
int K;
int W;
int H;

#define NUM_THREADS 12 // AJUSTAR DE ACORDO COM O NUMERO DE THREADS DISPONIVEIS
#define TRAIN_COUNT 500


// Lê os arquivos xtrain e todos os xtest
double *lerArquivo(char *nomeArquivo, int num_elementos){
     FILE *arquivo;
     arquivo = fopen(nomeArquivo, "r");
     if (arquivo == NULL) {
          printf("Erro ao abrir o arquivo %s\n", nomeArquivo);
          return NULL;
     }

     double *valores = (double *) malloc(num_elementos * sizeof(double)); //Buffers para os valores

     for (int i = 0; i < num_elementos; i++) {
          fscanf(arquivo, "%lf", &valores[i]);
     }
     fclose(arquivo);
     return valores; //Retorna o buffer
}

// Cria as matrizes xtrain, xtest, e o vetor ytrain baseado em K, W e H
void criaMatrizes(double **xtrain, double *ytrain, double **xtest, double *bufferTrain, double *bufferTest, int testCount){
     
     int base = 0; // Var auxiliar para preencher as matrizes (representa basicamente a base pra onde os valores vão ser preenchidos)
     // Exemplo: para preencher a linha [1 2 3] no xtrain e [4] no ytrain, base = 1

     for (int i=0;i<TRAIN_COUNT - W - H + 1;i++){
          for (int j=0;j<W;j++){
               xtrain[i][j] = bufferTrain[base+j]; // xtrain
          }
          ytrain[i] = bufferTrain[base+W];   // ytrain
          base++;
     }
     base = 0;
     for (int i=0;i<testCount - W - H + 1;i++){
          for (int j=0;j<W;j++){
               xtest[i][j] = bufferTest[base+j];  // xtest
          }
          base++;
     }
}

// Distancias euclidianas, poderia ser otimizado com o método de newton e/ou pré calculando as distancias
double * calculaDistancias(double** xTrain, double* linha_Xtest, int train_numrows){ 
     double *distancias = (double *)malloc(train_numrows * sizeof(double)); 

     #pragma omp parallel for // paralelizado, todas as variaveis declaradas internamente são privadas de cada thread
     for(int i = 0; i < train_numrows; i++){
          double temp = 0;
          for (int j = 0; j < W; j++){
               temp += pow(xTrain[i][j] - linha_Xtest[j], 2);
          }
     distancias[i] = sqrt(temp);
     }
     return distancias;
}

// KNN com as distancias euclidianas, O(n²) no pior caso. Poderia ser otimizado com uma fila de prioridade com heapsort para achar os melhores valores
double * KNN (double** xTrain, double** xTest, double* yTrain, int train_numrows, int test_numrows){

     double *y_test = (double *)malloc(test_numrows * sizeof(double));
     double *vetor_distancias; // declarado antes pois o vetor de distancias precisa ser private na paralelização
     #pragma omp parallel for private(vetor_distancias) // paralelizado, todas as variaveis declaradas internamente são privadas de cada thread (além do vetor_distancias)

     for(int i = 0; i < test_numrows; i++){
          double * vetor_distancias = (double *)malloc(train_numrows * sizeof(double)); // alocado paralelamente
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

     omp_set_num_threads(NUM_THREADS);

     FILE *filetempo;
     filetempo = fopen("tempos.txt", "a"); // Abre o arquivo tempos.txt para escrita sem sobrescrever os tempos
                                           // Basicamente o log de tempos de exec
     FILE *fileytest;
     fileytest = fopen("ytest.txt", "w"); // Escrita do ytest

     if (argc != 7) {
          printf("Uso correto: %s <xtrain_datapath> <xtest_datapath> (que varia para cada teste) <numero de entradas do xtest> <k> <w> <h> \n", argv[0]);
          return 1;
     }
     
     int test_count = atoi(argv[3]);

     K = atoi(argv[4]);
     W = atoi(argv[5]);
     H = atoi(argv[6]);


     int train_nrows = TRAIN_COUNT - W - H + 1; // Numero de linhas do xtrain
     int test_nrows = test_count - W - H + 1;   // Numero de linhas do xtest


     //Alocações de memória
     double **xTrain = (double **) malloc(train_nrows * sizeof(double *)); 
     for (int i = 0; i < train_nrows; i++) {
          xTrain[i] = (double *) malloc(W * sizeof(double));
     }

     double *yTrain = (double *) malloc(train_nrows * sizeof(double));

     double **xTest = (double **) malloc(test_nrows * sizeof(double *));
     for (int i = 0; i < test_nrows; i++) {
          xTest[i] = (double *) malloc(W * sizeof(double));
     }

     double *yTest = (double *) malloc(test_nrows * sizeof(double));

     double *bufferTrain = lerArquivo(argv[1], TRAIN_COUNT);
     double *bufferTest = lerArquivo(argv[2], test_count);




     criaMatrizes(xTrain, yTrain, xTest, bufferTrain, bufferTest, test_count);

     free(bufferTrain);
     free(bufferTest);

     double start_time, end_time; // tempo


     // KNN PRINCIPAL
     start_time = omp_get_wtime();
     for(int i = 0; i < 5 ; i++){
          yTest = KNN(xTrain, xTest, yTrain, train_nrows, test_nrows); // Roda 5 KNNs, tira a média dos tempos
     }
     end_time = omp_get_wtime();
     // KNN PRINCIPAL



     double tempo_usado = (end_time - start_time)/5;

     printf("Tempo de execucao: %lf segundos\n", tempo_usado);


     // Escrita dos resultados
     fprintf(filetempo, "%lf\n", tempo_usado);

     for (int i = 0; i < test_nrows; i++) {
          fprintf(fileytest, "%lf\n", yTest[i]);
     }



     // I WANT TO BREEAAAAAK FREEEEEEEE (Todos os frees necessários)
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