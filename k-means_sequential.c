#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#define MAX_ITER 100
#define THRESHOLD 1e-6
#define min(a, b) \
    ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

int number_of_points_global;
int number_of_iterations_global;
double delta_global = THRESHOLD + 1;
int K_global;
int *data_points_global;
float *iter_centroids_global;
int *data_point_cluster_global;

void kmeans_sequential_execution()
{
    printf("Sequential k-means start\n");

    int i = 0, j = 0;
    double min_dist, current_dist;

	//ID de cluster associé à chaque point
    int *point_to_cluster_id = (int *)malloc(number_of_points_global * sizeof(int));

	// Cluster location or centroid (x,y,z) coordinates for K clusters in a iteration
    float *cluster_points_sum = (float *)malloc(K_global * 3 * sizeof(float));

	// Nombre de points dans un cluster pour une itération
    int *points_inside_cluster_count = (int *)malloc(K_global * sizeof(int));

	// Début de boucle
    int iter_counter = 0;
    double temp_delta = 0.0;
    while ((delta_global > THRESHOLD) && (iter_counter < MAX_ITER)) //+1 is for the last assignment to cluster centroids (from previous iter)
    {
		// Initialiser cluster_points_sum ou centroïde à 0,0
        for (i = 0; i < K_global * 3; i++)
            cluster_points_sum[i] = 0.0;

		// Initialiser le nombre de points pour chaque cluster à 0
        for (i = 0; i < K_global; i++)
            points_inside_cluster_count[i] = 0;

        for (i = 0; i < number_of_points_global; i++)
        {
            //Affecter le reste des points à leur cluster le plus proche
            min_dist = DBL_MAX;
            for (j = 0; j < K_global; j++)
            {
                current_dist = pow((double)(iter_centroids_global[(iter_counter * K_global + j) * 3] - (float)data_points_global[i * 3]), 2.0) +
                               pow((double)(iter_centroids_global[(iter_counter * K_global + j) * 3 + 1] - (float)data_points_global[i * 3 + 1]), 2.0) +
                               pow((double)(iter_centroids_global[(iter_counter * K_global + j) * 3 + 2] - (float)data_points_global[i * 3 + 2]), 2.0);
                if (current_dist < min_dist)
                {
                    min_dist = current_dist;
                    point_to_cluster_id[i] = j;
                }
            }

             //Mettre à jour le nombre local de points à l’intérieur du cluster
            points_inside_cluster_count[point_to_cluster_id[i]] += 1;

			// Mettre à jour la somme locale des points de données de cluster
            cluster_points_sum[point_to_cluster_id[i] * 3] += (float)data_points_global[i * 3];
            cluster_points_sum[point_to_cluster_id[i] * 3 + 1] += (float)data_points_global[i * 3 + 1];
            cluster_points_sum[point_to_cluster_id[i] * 3 + 2] += (float)data_points_global[i * 3 + 2];
        }

        //Calculer le centroïde à partir de cluster_points_sum et stocker à l’intérieur iter_centroids_global dans une itération
        for (i = 0; i < K_global; i++)
        {
            assert(points_inside_cluster_count[i] != 0);
            iter_centroids_global[((iter_counter + 1) * K_global + i) * 3] = cluster_points_sum[i * 3] / points_inside_cluster_count[i];
            iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 1] = cluster_points_sum[i * 3 + 1] / points_inside_cluster_count[i];
            iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 2] = cluster_points_sum[i * 3 + 2] / points_inside_cluster_count[i];
        }

	/*
    	Delta est la somme de la distance au carré entre le centroïde de l’itération précédente et actuelle.
    	La formule de soutien est la suivante :
        	delta = (iter1_centroid1_x - iter2_centroid1_x)^2 + (iter1_centroid1_y - iter2_centroid1_y)^2 + (iter1_centroid1_z - iter2_centroid1_z)^2 + (iter1_centroid2_x - iter2_centroid2_x)^2 + (iter1_centroid2_y - iter2_centroid2_y)^2 + (iter1_centroid2_z - iter2_centroid2_z)^2
    	Mettre à jour delta_global avec le nouveau delta
	*/
        temp_delta = 0.0;
        for (i = 0; i < K_global; i++)
        {
            temp_delta += (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3] - iter_centroids_global[((iter_counter)*K_global + i) * 3]) * (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3] - iter_centroids_global[((iter_counter)*K_global + i) * 3]) + (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 1] - iter_centroids_global[((iter_counter)*K_global + i) * 3 + 1]) * (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 1] - iter_centroids_global[((iter_counter)*K_global + i) * 3 + 1]) + (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 2] - iter_centroids_global[((iter_counter)*K_global + i) * 3 + 2]) * (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 2] - iter_centroids_global[((iter_counter)*K_global + i) * 3 + 2]);
        }
        delta_global = temp_delta;

        iter_counter++;
    }

	// Stocker le nombre d’itérations effectuées dans la variable globale
    number_of_iterations_global = iter_counter;

    // Affecter des points au choix final pour les centroïdes de cluster
    for (i = 0; i < number_of_points_global; i++)
    {
        // Affecter des points à des clusters
        data_point_cluster_global[i * 4] = data_points_global[i * 3];
        data_point_cluster_global[i * 4 + 1] = data_points_global[i * 3 + 1];
        data_point_cluster_global[i * 4 + 2] = data_points_global[i * 3 + 2];
        data_point_cluster_global[i * 4 + 3] = point_to_cluster_id[i];
        assert(point_to_cluster_id[i] >= 0 && point_to_cluster_id[i] < K_global);
    }
}

void kmeans_sequential(int N,
					int K,
					int* data_points,
					int** data_point_cluster_id,
					float** iter_centroids,
					int* num_iterations
					)
{

    // Initialiser les variables globales
    number_of_points_global = N;
    number_of_iterations_global = *num_iterations;
    K_global = K;
    data_points_global = data_points;

	//Allocation d’espace de 4 unités chacune pour N points de données
    *data_point_cluster_id = (int *)malloc(N * 4 * sizeof(int));
    data_point_cluster_global = *data_point_cluster_id;

    /*
        Allocation d’espace d’unités 3K pour chaque itération
        Puisque point de données tridimensionnel et K nombre de clusters 
    */
    iter_centroids_global = (float *)calloc((MAX_ITER + 1) * K * 3, sizeof(float));

    // Assigner les premiers points K aux centroïdes initiaux
    int i = 0;
    for (i = 0; i < K; i++)
    {
        iter_centroids_global[i * 3] = data_points[i * 3];
        iter_centroids_global[i * 3 + 1] = data_points[i * 3 + 1];
        iter_centroids_global[i * 3 + 2] = data_points[i * 3 + 2];
    }

    // afficher les centroïdes initiaux
    for (i = 0; i < K; i++)
    {
        printf("initial centroid #%d: %f,%f,%f\n", i + 1, iter_centroids_global[i * 3], iter_centroids_global[i * 3 + 1], iter_centroids_global[i * 3 + 2]);
    }

    // Exécuter la fonction séquentielle k-means
    kmeans_sequential_execution();

    // Enregistrer le nombre d’itérations et stocker les données iter_centroids_global dans iter_centroids
    *num_iterations = number_of_iterations_global;
    int centroids_size = (*num_iterations + 1) * K * 3;
    printf("number of iterations:%d\n", number_of_iterations_global);
    *iter_centroids = (float *)calloc(centroids_size, sizeof(float));
    for (i = 0; i < centroids_size; i++)
    {
        (*iter_centroids)[i] = iter_centroids_global[i];
    }

    // afficher les centroïdes finaux
    for (i = 0; i < K; i++)
    {
        printf("centroid #%d: %f,%f,%f\n", i + 1, (*iter_centroids)[((*num_iterations) * K + i) * 3], (*iter_centroids)[((*num_iterations) * K + i) * 3 + 1], (*iter_centroids)[((*num_iterations) * K + i) * 3 + 2]);
    }
}

void dataset_in(const char *dataset_filename, int *N, int **data_points)
{
	FILE *fin = fopen(dataset_filename, "r");
	fscanf(fin, "%d", N);
	*data_points = (int *)malloc(sizeof(int) * ((*N) * 3));
    int i = 0;
	for (i = 0; i < (*N) * 3; i++)
	{
		fscanf(fin, "%d", (*data_points + i));
	}
	fclose(fin);
}

void clusters_out(const char *cluster_filename, int N, int *cluster_points)
{
	FILE *fout = fopen(cluster_filename, "w");
    int i = 0;
	for (i = 0; i < N; i++)
	{
		fprintf(fout, "%d %d %d %d\n",
				*(cluster_points + (i * 4)), *(cluster_points + (i * 4) + 1),
				*(cluster_points + (i * 4) + 2), *(cluster_points + (i * 4) + 3));
	}
	fclose(fout);
}

void centroids_out(const char *centroid_filename, int K, int number_of_iterations, float *iter_centroids)
{
	FILE *fout = fopen(centroid_filename, "w");
    int i = 0;
	for (i = 0; i < number_of_iterations + 1; i++)
	{
        int j = 0;
		for (j = 0; j < K; j++)
		{
			fprintf(fout, "%f %f %f, ",
					*(iter_centroids + (i * K + j) * 3),		 //x coordinate
					*(iter_centroids + (i * K + j) * 3 + 1),  //y coordinate
					*(iter_centroids + (i * K + j) * 3 + 2)); //z coordinate
		}
		fprintf(fout, "\n");
	}
	fclose(fout);
}

int main()
{

	//---------------------------------------------------------------------
	int N;					// Nombre de points de données (input)
	int K;					// Nombre de clusters à former (input)
	int* data_points;		//Points de données (input)
	int* cluster_points;	//points de données en cluster
	float* iter_centroids;		//centroïdes de chaque itération 
	int number_of_iterations;     //nombre des itérations effectuées par algo 
	//---------------------------------------------------------------------

	double start_time, end_time;
	double computation_time;

	printf("Enter No. of Clusters: ");
    scanf("%d", &K);

	printf("\nFollowing files should be in the same directory as of program\n");
    printf("1 for 10000 datapoints\n");
    printf("2 for 50000 datapoints\n");
    printf("3 for 100000 datapoints\n");
    printf("4 for 200000 datapoints\n");
    printf("5 for 400000 datapoints\n");
    printf("6 for 500000 datapoints\n");
    printf("7 for 600000 datapoints\n");
    printf("8 for 800000 datapoints\n");
    printf("9 for 1000000 datapoints\n");
    printf("\nEnter the number of dataset file to input: ");

    int x;
    scanf("%d",&x);

	char *dataset_filename = "dataset-10000.txt";

    switch (x)
    {
    case 1:
        dataset_filename = "dataset-10000.txt";
        break;
    case 2:
        dataset_filename = "dataset-50000.txt";
        break;
    case 3:
        dataset_filename = "dataset-100000.txt";
        break;
    case 4:
        dataset_filename = "dataset-200000.txt";
        break;
    case 5:
        dataset_filename = "dataset-400000.txt";
        break;
    case 6:
        dataset_filename = "dataset-500000.txt";
        break;
    case 7:
        dataset_filename = "dataset-600000.txt";
        break;
    case 8:
        dataset_filename = "dataset-800000.txt";
        break;
    case 9:
        dataset_filename = "dataset-1000000.txt";
        break;
    default:
        dataset_filename = "dataset-10000.txt";
        break;
    }


	/*
        La fonction lit dataset_file et stocke les données dans data_points tableau. Chaque point a trois indices consécutifs associés dans le tableau.
        data_points tableau ressemble à : [pt_1_x, pt_1_y, pt_1_z, pt_2_x, pt_2_y, pt_2_z]
	*/
	dataset_in (dataset_filename, &N, &data_points);

	start_time = omp_get_wtime();
	kmeans_sequential(N, K, data_points, &cluster_points, &iter_centroids, &number_of_iterations);
	end_time = omp_get_wtime();	

	// Création de noms de fichiers pour différents jeux de données

    char file_index_char[2];
    snprintf(file_index_char,10,"%d", x);

    char cluster_filename[105] = "cluster_output_dataset";
    strcat(cluster_filename,file_index_char);
    strcat(cluster_filename,".txt");

    char centroid_filename[105] = "centroid_output_dataset";
    strcat(centroid_filename,file_index_char);
    strcat(centroid_filename,".txt");

	/*
        Les points groupés sont enregistrés dans cluster_filename.
        Chaque point est associé à l’index de cluster auquel il appartient.
        cluster_points tableau ressemble à : [pt_1_x, pt_1_y, pt_1_z, pt_1_cluster_index, pt_2_x, pt_2_y, pt_2_z, pt_2_cluster_index]
        Format de fichier de sortie:
            pt_1_x pt_1_y, pt_1_z, pt_1_cluster_index
	*/
	clusters_out (cluster_filename, N, cluster_points);

	/*
        Les points centroïdes sont stockés dans centroid_filename.
        Chaque ligne du fichier représente les coordonnées centroïdes des clusters après chaque itération.
        Format de fichier de sortie:
            centroid_1_x, centroid_1_y, centroid_1_z, centroid_2_x, centroid_2_y, centroid_2_z
    */
	centroids_out (centroid_filename, K, number_of_iterations, iter_centroids);

	/*
        Le temps de calcul est stocké dans 'compute_time_openmp.txt'.
    */
   	computation_time = end_time - start_time;
	printf("Time Taken: %lf \n", computation_time);
    
	char time_file_omp[100] = "compute_time_openmp_dataset";
    strcat(time_file_omp,file_index_char);
    strcat(time_file_omp,".txt");

	FILE *fout = fopen(time_file_omp, "a");
	fprintf(fout, "%f\n", computation_time);
	fclose(fout);
    
	printf("Cluster Centroid point output file '%s' saved\n", centroid_filename);
    printf("Clustered points output file '%s' saved\n", cluster_filename);
    printf("Computation time output file '%s' saved\n", time_file_omp);
	
	return 0;
}
