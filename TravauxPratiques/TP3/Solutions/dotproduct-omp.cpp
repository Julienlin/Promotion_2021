# include <vector>
# include <cassert>
# include <string>
# include <iostream>
# include <chrono>

#include <omp.h>

double dot( std::vector<double>& u, std::vector<double>& v )
{
  assert(u.size() == v.size());
  double scal = 0.;
  for ( size_t i = 0; i < u.size(); ++i ) {
    scal += u[i]*v[i];
  }
  return scal;
}

int main( int nargs, char* vargs[])
{
  std::chrono::time_point<std::chrono::system_clock> start, end;
  int N = 1023;
  int nbSamples = 1024;
  if (nargs > 1) {
    nbSamples = std::stoi(vargs[1]);
  }
  
  start = std::chrono::system_clock::now();
  std::vector<std::vector<double>> U(nbSamples), V(nbSamples);
  for ( int iSample = 0; iSample < nbSamples; ++iSample ) {
    U[iSample] = std::vector<double>(N);
    V[iSample] = std::vector<double>(N);

    // seq: 1.62982 ; static: 2.54972 ; dynamic: 12.1787
    // moralite: ne pas tout threader !
    //  #pragma omp parallel for schedule(static)
    //  #pragma omp parallel for schedule(dynamic)
    for ( int i = 0; i < N; ++i ) {
      U[iSample][i] = (iSample + 1 + i)%N;
      V[iSample][i] = (nbSamples - iSample + i)%N;
    }
  }
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "Temps assemblage vecteurs : " << elapsed_seconds.count() 
              << std::endl;

  start = std::chrono::system_clock::now();
  std::vector<double> result(nbSamples);

  int nbTaches = 1; 

  #pragma omp parallel
  nbTaches = omp_get_num_threads();

  std::cout << "nb threads = " << nbTaches << std::endl;

  #pragma omp parallel for schedule(dynamic)
  for ( int iSample = 0; iSample < nbSamples; ++iSample )
    result[iSample] = dot(U[iSample],V[iSample]);

  end = std::chrono::system_clock::now();
  elapsed_seconds = end-start;
  std::cout << "Temps produits scalaires : " << elapsed_seconds.count() 
              << std::endl;

  start = std::chrono::system_clock::now();
  double ref = result[0];
  double sum = 0;;
  for ( const auto& val : result )
    sum += val;
  sum /= ref;
  end = std::chrono::system_clock::now();
  elapsed_seconds = end-start;
  std::cout << "Temps sommation : " << elapsed_seconds.count() 
              << std::endl;
  std::cout << "sum : " << sum << std::endl;
  return 0;
}
