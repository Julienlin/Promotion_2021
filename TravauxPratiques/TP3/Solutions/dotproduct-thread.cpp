# include <vector>
# include <cassert>
# include <string>
# include <iostream>
# include <chrono>
#include <thread>

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
  int nbThreads = 1 ; 

  if (nargs > 1) {
    nbSamples = std::stoi(vargs[1]);
  }

  if (nargs > 2) {
    nbThreads = std::stoi(vargs[2]);
  }
  
  start = std::chrono::system_clock::now();
  std::vector<std::vector<double>> U(nbSamples), V(nbSamples);
  for ( int iSample = 0; iSample < nbSamples; ++iSample ) {
    U[iSample] = std::vector<double>(N);
    V[iSample] = std::vector<double>(N);
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


  if (nbThreads > 1) {
    std::vector<std::thread> threads;
    for (int iThread=0; iThread<nbThreads ; iThread++) {
      threads.push_back( std::thread ([&U,&V,&result,iThread,nbThreads,nbSamples]() {
	    int sub = nbSamples / nbThreads+1;
	    int beg = sub * iThread;
	    int end = (iThread+1) * sub ; 
	    if (end > nbSamples) end=nbSamples;
	    std::cout << "thread "<< iThread << " beg="<<beg << " end="<<end << std::endl; 
	    for ( int iSample = beg; iSample < end; ++iSample ){
	      result[iSample] = dot(U[iSample], V[iSample]);
	      //	      std::cout << iSample << " " << result[iSample] << std::endl; 
	    }
	  } ) );
    }
    for (auto& t: threads){ t.join(); }
  }
  else {
    for ( int iSample = 0; iSample < nbSamples; ++iSample )
      result[iSample] = dot(U[iSample],V[iSample]);
  }
  //  std::cout << "======================" << std::endl;     
  //  for (auto&x: result) std::cout << x << " "  ;
  //  std::cout << std::endl; 



  end = std::chrono::system_clock::now();
  elapsed_seconds = end-start;
  std::cout << "Temps produits scalaires : " << elapsed_seconds.count() 
	    << std::endl;


  start = std::chrono::system_clock::now();
  double ref = result[0];
  double sum = 0.;
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
