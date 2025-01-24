#include <iostream>
#include <neso_particles.hpp>
#include <mpi.h>

namespace neso = NESO::Particles;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    //scope bo trzeba zwolnic pamiec przed mpi finalize a te obiekty zwalniaja pamiec w destruktorach
    {
        //siatka, dla tej kartezjanzskiej podajemy globalne wymiary on to sobie rozklada
        std::vector<int> cells_course = {20, 10};
        std::shared_ptr<neso::CartesianHMesh> mesh_ptr = std::make_shared<neso::CartesianHMesh>(MPI_COMM_WORLD, 2, cells_course);

        //domena (wsm to nic nie robi ale jest potrzebne bo dalej bedzie domena podawana jako argument a nie siatka, ale to wsm
        //ma tyle samo informacji co siatka)
        std::shared_ptr<neso::Domain> domain_ptr = std::make_shared<neso::Domain>(mesh_ptr);

        //target, "1" bo kazalem mu wybrac gpu
        //std::shared_ptr<neso::SYCLTarget> target_ptr = std::make_shared<neso::SYCLTarget>(1, MPI_COMM_WORLD);
        std::shared_ptr<neso::SYCLTarget> target_ptr = std::make_shared<neso::SYCLTarget>(-1, MPI_COMM_WORLD);

        //wlasciwosci czastek
        neso::Sym<neso::REAL> position("position");
        neso::Sym<neso::INT> id("id");
        neso::Sym<neso::REAL> z_dim("z");

        neso::ParticleProp particle_position(position, 2, true);
        neso::ParticleProp particle_cell_id(id, 1, true);
        neso::ParticleProp particle_z_dim(z_dim, 1, false);

        std::vector<neso::ParticleProp<neso::INT>> int_props = {particle_cell_id};
        std::vector<neso::ParticleProp<neso::REAL>> real_props = {particle_position, particle_z_dim};

        neso::ParticleSpec particle_spec(real_props, int_props);

        //particle group
        std::shared_ptr<neso::ParticleGroup> particle_group_ptr = std::make_shared<neso::ParticleGroup>(domain_ptr, particle_spec, target_ptr);

        //to co powzyej moze zostac ewentualnie ilosc celli zmienic

        //napisac gettera cart_commu z siatki, to mi ulatwi inicjalizacje czastek na kazdym corze (moze inicliazowac je kawalek od sciany)
        //inicjalizowac czastki rownoodlegle od siebie, moze byc odrazu w cellach albo wszytskie wladowac do zerowego i potem zmapkowac
        //narazie sprobowac rk4, czastki bez bezwladosci, pole predkosci to co mi przeslal double gyre
        //termin pracy: polowa stycznia!!!!!
        //jakby wyszlo z tego cos ladnego to zapisac wyniki pod prace inzynierska (np dane do pokazania jak sie czastki przemieszczaly)
        //wszytskie wzory sa na stronie co mi wyslal, sprobowac to dzisaj zrobic!!!!


        MPI_Comm cart_comm = mesh_ptr->get_comm();
        int rank_cords[2];
        int rank_dims[2];
        int periods[2];

        MPI_Cart_get(cart_comm, 2, rank_dims, periods, rank_cords);
        std::cout<<rank_cords[0]<<", "<<rank_cords[1]<<":,  ("<<mesh_ptr->cell_starts[0]<<", "<<mesh_ptr->cell_starts[1]<<")"<<std::endl;

        int n_particles = 10000;
        int comm_size = rank_dims[0] * rank_dims[1];
        int n_particles_local = n_particles/comm_size;

        neso::ParticleSet particles_to_add(n_particles, particle_spec);

        double cell_size = mesh_ptr->get_cell_width_fine();
        int rank_x_cells = 40/rank_dims[0];
        int rank_y_cells = 20/rank_dims[1];
        double rank_x_size = rank_x_cells * cell_size;
        double rank_y_size = rank_y_cells * cell_size;

        double x = rank_x_size * rank_cords[0] + cell_size;
        double y = rank_y_size * rank_cords[1] + cell_size;
        double wall_sep = 1.5 * cell_size;
        int x_particles = 50;
        int y_particles = 50;
        double particle_sep_x = (rank_x_size - cell_size)/x_particles;
        double particle_sep_y = (rank_y_size - cell_size)/y_particles;
        int local_cells_x = mesh_ptr->cell_ends[0] - mesh_ptr->cell_starts[0];
        int local_cells_y = mesh_ptr->cell_ends[1] - mesh_ptr->cell_starts[1];

        int i = 0;
        int x_counter = 0;
        int cell_id_x;
        int cell_id_y;
        int linear_local_cell_id;
        for(i=0;i<n_particles_local;i++)
        {
            particles_to_add.at(particle_position.sym, i, 0) = x;
            particles_to_add.at(particle_position.sym, i, 1) = y;
            
            cell_id_x = (x/cell_size) - mesh_ptr->cell_starts[0];
            cell_id_y = (y/cell_size) - mesh_ptr->cell_starts[1];
            linear_local_cell_id = cell_id_x + cell_id_y * local_cells_x;

            particles_to_add.at(particle_cell_id.sym, i, 0) = linear_local_cell_id;

            x = x + particle_sep_x;
            x_counter++;
            if(x_counter == x_particles)
            {
                x_counter = 0;
                x = rank_x_size * rank_cords[0] + cell_size;

                y = y + particle_sep_y;
            }
        }
        particle_group_ptr->add_particles_local(particles_to_add);

        // neso::H5Part save_position_initial("out0.h5part", particle_group_ptr, position, id, velocity, z_dim);
        // save_position_initial.write();
        // save_position_initial.close();

        float* time = (float*)sycl::malloc_device(1 * sizeof(float), target_ptr->queue);
        float dt = 0.001;

        sycl::event set_initial_time = target_ptr->queue.submit([&](sycl::handler &cgh) {
            cgh.single_task<>([=]() { time[0] = 0; });
        });
        set_initial_time.wait();

        auto kernel = [=] (auto position){
            float pi = 3.1415;

            float x = position[0]/10;
            float y = position[1]/10;

            float u = -pi * 1 * sycl::sin(pi * (x * x * sycl::sin(time[0])
            + x - 2 * x * sycl::sin(time[0]))) * sycl::cos(pi * y);
            float v = pi * 1 * sycl::cos(pi * (x * x * sycl::sin(time[0])
            + x - 2 * x * sycl::sin(time[0]))) * sycl::sin(pi * y)
            * (2 * x * sycl::sin(time[0]) + 1 - 2 * sycl::sin(time[0]));

            position[0] = position[0] + u * dt;
            position[1] = position[1] + v * dt;
        };

        neso::ParticleLoop loop("advection", particle_group_ptr, kernel, neso::Access::write(neso::Sym<neso::REAL>("position")));
        loop.execute();
        neso::H5Part save_position("out.h5part", particle_group_ptr, position, z_dim);

        int step = 0;
        int save_interval = 0;
        int save_count = 0;
        for(step=0;step<100;step++)
        {
            loop.execute();
            if(save_interval == 20)
            {
                save_position.write(save_count);
                save_count++;
                save_interval = 0;
            }
            
            sycl::event set_time = target_ptr->queue.submit([&](sycl::handler &cgh) {
                cgh.single_task<>([=]() { time[0] = time[0] + dt; });
            });

            set_time.wait();
            save_interval++;
        }
        save_position.close();
    }
    MPI_Finalize();
}
