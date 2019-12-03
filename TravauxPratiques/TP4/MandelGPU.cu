# include <cuda.h>
# include <cuda_runtime.h>
# include <vector_types.h>
# include <device_launch_parameters.h>
# include <cstdio>
# ifdef WIN32
# include <time.h>
# else
# include <sys/time.h>
# endif


__global__ void iterMandel(int niterMax, int n, int* mandel )
{
  int   t_x, t_y, iter;
  float step, zr;
  float2 z0,c0;
  iter = 0;
  t_x = threadIdx.x+blockIdx.x*blockDim.x;
  t_y = threadIdx.y+blockIdx.y*blockDim.y;
  if ((t_x<n) && (t_y<n)) {
    step = 2.5f/(n-1.f);
    c0.x = -2.00 + step*t_x;
    c0.y = -1.25 + step*t_y;
    z0 = c0;
    while ((z0.x*z0.x+z0.y*z0.y < 4.f)&&(iter<niterMax)) {
      iter += 1;
      zr = z0.x*z0.x-z0.y*z0.y+c0.x;
      z0.y = 2.f*z0.x*z0.y+c0.y;
      z0.x = zr;
    }
    mandel[t_x+t_y*n] = iter;
  }
}

/* Petite routine renvoyant un instant
   en seconde. En faisant la différence
   entre deux instants, on peut mesure le temps
   pris par une routine.
   La précision est de l'ordre de la micro-seconde.
*/
float topChrono()
{
# ifdef WIN32
  clock_t chrono;
  chrono = clock();
  return ((float)chrono)/CLOCKS_PER_SEC;
# else
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return tv.tv_sec+1.E-6*tv.tv_usec;
# endif
}

/* Calcul les coordonnees des sommets de la grille */
void compGrid(int n, float*& xh, float*& yh)
{
  xh = new float[n];
  yh = new float[n];
  float step = 2.5/(n-1.);
  for (int i = 0; i < n; i++) {
    xh[i] = -2.+i*step;
    yh[i] = -1.25+i*step;
  }
}

/* Sauve l'ensemble de mandelbrot au format Gnuplot
 */
void saveMandel(int n, const float* xh, const float* yh,
		const int* mandel)
{
  FILE* fich = fopen("Mandel.dat", "w");
  for (int j = 0; j < n; j++) {
    for (int i  = 0; i < n; i++) {
      fprintf(fich,"%f\t%f\t%d\n",yh[i], xh[j], mandel[i*n+j]);
    }
    fprintf(fich, "\n");
  }
  fclose(fich);
}

int main(int nargc, char* argv[])
{
  float t1,t2;
  float *xh, *yh;
  int n = 1280;
  int maxIter = 1000;
  int* gmandel, *cmandel;
  cudaEvent_t start, stop; 
  float time;
  if (nargc > 1) n = atoi(argv[1]);
  if (nargc > 2) maxIter = atoi(argv[2]);

  compGrid(n,xh,yh);

  cudaMalloc((void**)&gmandel, n*n*sizeof(int));
  cmandel = new int[n*n];
  
  cudaEventCreate(&start); cudaEventCreate(&stop);
  dim3 blockDim(16,16);
  dim3 gridDim((n+15)/16, (n+15)/16);
  cudaEventRecord( start, 0 );
  iterMandel<<<gridDim,blockDim>>>(maxIter, n, gmandel);
  cudaMemcpy(cmandel, gmandel, n*n*sizeof(int), cudaMemcpyDeviceToHost);
  cudaEventRecord( stop, 0 ); cudaEventSynchronize( stop );
  cudaFree(gmandel);
  cudaEventElapsedTime( &time, start, stop );
  cudaEventDestroy( start ); cudaEventDestroy( stop );
  printf("Temps de calcul du Bouddha : %f secondes\n",time/1000.);
  saveMandel(n,xh,yh,cmandel);
  delete [] cmandel;
  delete [] xh;
  delete [] yh;
  return EXIT_SUCCESS;
};
