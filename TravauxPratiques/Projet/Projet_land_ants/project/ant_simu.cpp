#include "ant.hpp"
#include "display.hpp"
#include "fractal_land.hpp"
#include "gui/colors.hpp"
#include "gui/context.hpp"
#include "gui/event_manager.hpp"
#include "gui/point.hpp"
#include "gui/quad.hpp"
#include "gui/segment.hpp"
#include "gui/triangle.hpp"
#include "pheromone.hpp"
#include "utils.hpp"
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <random>
#include <vector>

using ratio = std::ratio<1, 1000>;

void advance_time(
    const fractal_land &land, pheromone &phen, const position_t &pos_nest,
    const position_t &pos_food, std::vector<ant> &ants, std::size_t &cpteur,
    std::chrono::time_point<std::chrono::high_resolution_clock> &timer,
    bool &is_first_time) {
  for (size_t i = 0; i < ants.size(); ++i) {

    ants[i].advance(phen, land, pos_food, pos_nest, cpteur);
    if (cpteur == 1 && is_first_time) {
      timer = std::chrono::high_resolution_clock::now();
      is_first_time = false;
    }
  }
  phen.do_evaporation();
  phen.update();
}

int main(int nargs, char *argv[]) {
  MPI_Init(&nargs, &argv);
  int rank, nbp;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nbp);

  // Creating communicator for slaves
  MPI_Group world_group;
  MPI_Comm_group(comm, &world_group);

  std::vector<int> slaves_ranks(nbp - 1);
  for (int i = 0; i < nbp - 1; i++) {
    slaves_ranks[i] = i + 1;
  }

  MPI_Group slaves_group;
  MPI_Group_incl(world_group, nbp - 1, slaves_ranks.data(), &slaves_group);

  MPI_Comm slaves_comm;
  MPI_Comm_create_group(comm, slaves_group, 0, &slaves_comm);

  std::chrono::duration<double, ratio> init_fractal_time;
  std::chrono::duration<double, ratio> init_ant_time;
  std::chrono::time_point<std::chrono::high_resolution_clock> first_food_time;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_finding;
  bool is_first_time = true;
  const int nb_ants = 5000; // Nombre de fourmis
  const double eps = 0.8;   // Coefficient d'exploration
  const double alpha = 0.7; // Coefficient de chaos
  // const double beta=0.9999; // Coefficient d'évaporation
  const double beta = 0.999; // Coefficient d'évaporation
  // Location du nid
  position_t pos_nest{256, 256};
  // const int i_nest = 256, j_nest = 256;
  // Location de la nourriture
  position_t pos_food{500, 500};

  int nb_ants_loc = nb_ants / (nbp - 1);
  int size_sub_matrix;
  int fract_dim;

  if (rank == 0) {
    auto start = std::chrono::high_resolution_clock::now(); // Compute init
                                                            // fractal land time
    // const int i_food = 500, j_food = 500;
    // Génération du territoire 512 x 512 ( 2*(2^8) par direction )
    fractal_land land(8, 2, 1., 1024);
    double max_val = 0.0;
    double min_val = 0.0;
    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i)
      for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j) {
        max_val = std::max(max_val, land(i, j));
        min_val = std::min(min_val, land(i, j));
      }
    double delta = max_val - min_val;
    /* On redimensionne les valeurs de fractal_land de sorte que les valeurs
    soient comprises entre zéro et un */
    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i)
      for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j) {
        land(i, j) = (land(i, j) - min_val) / delta;
      }

    auto end = std::chrono::high_resolution_clock::now(); // Compute init
                                                          // fractal land time
    init_fractal_time = end - start;

    // put display part here

    fract_dim = land.dimensions();
    // Broadcasting fractal dimension
    MPI_Bcast(&fract_dim, 1, MPI_UNSIGNED_LONG, 0, comm);
    // Broadcasting fractal data
    MPI_Bcast(land.data(), fract_dim * fract_dim, MPI_DOUBLE, 0, comm);

  } else {
    MPI_Bcast(&fract_dim, 1, MPI_UNSIGNED_LONG, 0, comm);
    std::vector<double> buffer_mpi(fract_dim * fract_dim);
    MPI_Bcast(buffer_mpi.data(), fract_dim * fract_dim, MPI_DOUBLE, 0, comm);
    fractal_land land(fract_dim, buffer_mpi);
  }

  // // Définition du coefficient d'exploration de toutes les fourmis.
  // ant::set_exploration_coef(eps);
  // auto start = std::chrono::high_resolution_clock::now(); // Compute init ant
  // time

  // // On va créer des fourmis un peu partout sur la carte :
  // std::vector<ant> ants;
  // ants.reserve(nb_ants);
  // std::random_device
  //     rd; // Will be used to obtain a seed for the random number engine
  // std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with
  // rd() std::uniform_int_distribution<size_t> ant_pos(0, land.dimensions() -
  // 1); for (size_t i = 0; i < nb_ants; ++i)
  //   ants.push_back({{ant_pos(gen), ant_pos(gen)}});
  // auto end = std::chrono::high_resolution_clock::now();
  // init_ant_time = end - start;

  // // On crée toutes les fourmis dans la fourmilière.
  // pheromone phen(land.dimensions(), pos_food, pos_nest, alpha, beta);
  MPI_Finalize();
  return 0;
}
// gui::context graphic_context(nargs, argv);
// gui::window &win = graphic_context.new_window(2 * land.dimensions() + 10,
//                                               land.dimensions() + 266);
// display_t displayer(land, phen, pos_nest, pos_food, ants, win);
// // Compteur de la quantité de nourriture apportée au nid par les fourmis
// size_t food_quantity = 0;
// int ind = 0;
// int SIZE_VEC = 7000;
// std::vector<std::chrono::duration<double, ratio>> duration_advance(SIZE_VEC);
// std::vector<std::chrono::duration<double, ratio>> duration_display(SIZE_VEC);
// std::chrono::duration<double, ratio> tot;
// gui::event_manager manager;
// manager.on_key_event(int('q'), [](int code) { exit(0); });
// manager.on_key_event(int('t'), [&](int code) {
//   std::cout << "Iteration : " << ind << std::endl
//             << "Advance time : " << duration_advance[ind - 1].count() << "
//             ms"
//             << ", average : "
//             << utils::avg<std::chrono::duration<double, ratio>>(
//                    duration_advance, ind)
//                    .count()
//             << " ms" << std::endl
//             << "Display time : " << duration_display[ind - 1].count() << "
//             ms"
//             << std::endl
//             << "Total time : " << tot.count() / 1000 << " s" << std::endl;
//   if (!is_first_time) {
//     std::chrono::duration<double, ratio> res =
//         first_food_time - start_finding;
//     std::cout << "time to find food : " << res.count() << " ms" << std::endl;
//   }
// });
// manager.on_display([&] {
//   displayer.display(food_quantity);
//   win.blit();
// });
// manager.on_idle([&]() {
//   auto start = std::chrono::high_resolution_clock::now();
//   advance_time(land, phen, pos_nest, pos_food, ants, food_quantity,
//                first_food_time, is_first_time);
//   auto end = std::chrono::high_resolution_clock::now();
//   if (ind > SIZE_VEC) {
//     duration_advance.push_back(end - start);
//   } else {
//     duration_advance[ind] = end - start;
//   }
//   start = std::chrono::high_resolution_clock::now();
//   displayer.display(food_quantity);
//   end = std::chrono::high_resolution_clock::now();
//   if (ind > SIZE_VEC) {
//     duration_display.push_back(end - start);
//   } else {
//     duration_display[ind] = end - start;
//   }
//   tot += duration_advance[ind] + duration_display[ind];
//   ind++;
//   win.blit();
// });
// start_finding = std::chrono::high_resolution_clock::now();
// manager.loop();
