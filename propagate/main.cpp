#include <iostream>
#include <neso_particles.hpp>
#include <mpi.h>
#include <fftw3.h>

namespace neso = NESO::Particles;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //scope bo trzeba zwolnic pamiec przed mpi finalize a te obiekty zwalniaja pamiec w destruktorach
    
    {
        //siatka, dla tej kartezjanzskiej podajemy globalne wymiary on to sobie rozklada
        std::vector<int> cells_course = {3, 1, 5};
        std::shared_ptr<neso::CartesianHMesh> mesh_ptr = std::make_shared<neso::CartesianHMesh>(MPI_COMM_WORLD, 3, cells_course, 2.0, 1);

        //domena (wsm to nic nie robi ale jest potrzebne bo dalej bedzie domena podawana jako argument a nie siatka, ale to wsm
        //ma tyle samo informacji co siatka)
        std::shared_ptr<neso::Domain> domain_ptr = std::make_shared<neso::Domain>(mesh_ptr);

        std::shared_ptr<neso::SYCLTarget> target_ptr = std::make_shared<neso::SYCLTarget>(1, MPI_COMM_WORLD);

        enum moje::timeInterpolationMethod interpolation = moje::fourier;
        moje::ChflowReader reader("ChflowCoeffs", target_ptr, 31, 0.5, interpolation);

        neso::Sym<neso::REAL> position("position");
        neso::Sym<neso::REAL> position_old("position_old");
        neso::Sym<neso::REAL> k("k");
        neso::Sym<neso::INT> id("id");
        neso::Sym<neso::INT> color_rank("color");
        neso::Sym<neso::REAL> velocity("velocity");

        neso::ParticleProp particle_position(position, 3, true);
        neso::ParticleProp particle_position_old(position_old, 3, false);
        neso::ParticleProp particle_k(k, 3, false);
        neso::ParticleProp particle_cell_id(id, 1, true);
        neso::ParticleProp particle_color(color_rank, 1, false);
        neso::ParticleProp particle_velocity(velocity, 3, false);


        std::vector<neso::ParticleProp<neso::INT>> int_props = {particle_cell_id, particle_color};
        std::vector<neso::ParticleProp<neso::REAL>> real_props = {particle_position, particle_k, particle_position_old, particle_velocity};

        neso::ParticleSpec particle_spec(real_props, int_props);

        std::shared_ptr<neso::ParticleGroup> particle_group_ptr = std::make_shared<neso::ParticleGroup>(domain_ptr, particle_spec, target_ptr);

        MPI_Comm cart_comm = mesh_ptr->get_comm();
        int rank_cords[3];
        int rank_dims[3];
        int periods[3];
        MPI_Cart_get(cart_comm, 3, rank_dims, periods, rank_cords);

        int n_particles = 27436;
        int comm_size = rank_dims[0] * rank_dims[1] * rank_dims[2];
        int n_particles_local = n_particles/comm_size;

        neso::ParticleSet particles_to_add(n_particles_local, particle_spec);

        int n_particles_per_dim = std::pow(n_particles_local, (double)1/3);
        if(std::pow(n_particles_per_dim, 3) != n_particles_local)
        {
            n_particles_per_dim++;
        }

        double cell_size = mesh_ptr->get_cell_width_fine();
        double rank_lenght[3];
        int i = 0;
        for(i=0;i<3;i++)
        {
            rank_lenght[i] = (mesh_ptr->cell_ends[i] - mesh_ptr->cell_starts[i]) * cell_size;
        }
        
        double x = rank_lenght[0] * rank_cords[0];
        double y = rank_lenght[1] * rank_cords[1];
        double z = rank_lenght[2] * rank_cords[2];

        int x_counter = 0;
        int y_counter = 0;
        int linear_local_cell_id;
        int cell_id_x;
        int cell_id_y;
        int cell_id_z;
        double dx = rank_lenght[0]/(n_particles_per_dim - 1);
        double dy = rank_lenght[1]/(n_particles_per_dim - 1);
        double dz = rank_lenght[2]/(n_particles_per_dim - 1);
        for(i=0;i<n_particles_local;i++)
        {
            particles_to_add.at(particle_position.sym, i, 0) = x;
            particles_to_add.at(particle_position.sym, i, 1) = y - 1.0;
            particles_to_add.at(particle_position_old.sym, i, 0) = x;
            particles_to_add.at(particle_position_old.sym, i, 1) = y - 1.0;
            particles_to_add.at(particle_position.sym, i, 2) = z;
            particles_to_add.at(particle_position_old.sym, i, 2) = z;
            particles_to_add.at(particle_color.sym, i, 0) = rank_cords[0];

            cell_id_x = (x/cell_size) - mesh_ptr->cell_starts[0];
            cell_id_y = (y/cell_size) - mesh_ptr->cell_starts[1];
            cell_id_z = (z/cell_size) - mesh_ptr->cell_starts[2];

            if(cell_id_x == mesh_ptr->cell_counts_local[0])
            {
                cell_id_x--;
            }
            if(cell_id_y == mesh_ptr->cell_counts_local[1])
            {
                cell_id_y--;
            }
            if(cell_id_z == mesh_ptr->cell_counts_local[2])
            {
                cell_id_z--;
            }

            linear_local_cell_id = cell_id_x + cell_id_y * mesh_ptr->cell_counts_local[0] + 
                                   cell_id_z * mesh_ptr->cell_counts_local[0] * mesh_ptr->cell_counts_local[1];

            particles_to_add.at(particle_cell_id.sym, i, 0) = linear_local_cell_id;

            x = x + dx;
            x_counter++;
            if(x_counter == n_particles_per_dim)
            {
                x_counter = 0;
                y_counter++;
                x = rank_lenght[0] * rank_cords[0];

                y = y + dy;
                
                if(y_counter == n_particles_per_dim)
                {
                    y = rank_lenght[1] * rank_cords[1];
                    y_counter = 0;
                    z = z + dz;
                }
            }
        }

        particle_group_ptr->add_particles_local(particles_to_add);

        moje::ChflowDeviceAccesor accesor;
        moje::AccesorInit(accesor, reader);

        double dt = 0.05;
        
        auto calc_k = [=] (auto position_old, auto k) {
            double x = position_old[0] + k[0];
            double y = position_old[1] + k[1];
            double z = position_old[2] + k[2];

            int mx;
            int my;
            int mz;

            double T0 = 1;
            double T1 = y;
            double pi = 3.141592;
            
            double* t0 = &T0;
            double* t1 = &T1;
            double* swap;
            
            double sin_val;
            double cos_val;
            double cos_val2;

            //for(my=1;my<accesor.n_mody;my++)
            my = 0;
            k[0] = k[0] + t0[0] * accesor.at(accesor.n_modx/2 - 1, my, 0, 0, 0);
            k[1] = k[1] + t0[0] * accesor.at(accesor.n_modx/2 - 1, my, 0, 1, 0);
            k[2] = k[2] + t0[0] * accesor.at(accesor.n_modx/2 - 1, my, 0, 2, 0);
            for(my=1;my<accesor.n_mody;my++)
            {
                k[0] = k[0] + t1[0] * accesor.at(accesor.n_modx/2 - 1, my, 0, 0, 0);
                k[1] = k[1] + t1[0] * accesor.at(accesor.n_modx/2 - 1, my, 0, 1, 0);
                k[2] = k[2] + t1[0] * accesor.at(accesor.n_modx/2 - 1, my, 0, 2, 0);

                t0[0] = 2 * y * t1[0] - t0[0];
                swap = t0;
                t0 = t1;
                t1 = swap;
            }
            my = 0;
            t0[0] = 1;
            t1[0] = y;


            k[0] = k[0] - 2 * t0[0] * accesor.at(accesor.n_modx/2 - 1, my, 0, 0, 0);
            k[1] = k[1] - 2 * t0[0] * accesor.at(accesor.n_modx/2 - 1, my, 0, 1, 0);
            k[2] = k[2] - 2 * t0[0] * accesor.at(accesor.n_modx/2 - 1, my, 0, 2, 0);
            for(my=1;my<accesor.n_mody;my++)
            {
                k[0] = k[0] - 2 * t1[0] * accesor.at(accesor.n_modx/2 - 1, my, 0, 0, 0);
                k[1] = k[1] - 2 * t1[0] * accesor.at(accesor.n_modx/2 - 1, my, 0, 1, 0);
                k[2] = k[2] - 2 * t1[0] * accesor.at(accesor.n_modx/2 - 1, my, 0, 2, 0);

                t0[0] = 2 * y * t1[0] - t0[0];
                swap = t0;
                t0 = t1;
                t1 = swap;
            }
            my = 0;
            t0[0] = 1;
            t1[0] = y;


            cos_val = sycl::cos(2 * pi * (accesor.n_modz-1) * z/accesor.Lz);
            cos_val2 = sycl::cos(pi * accesor.n_modx * x/accesor.Lx);
            k[0] = k[0] + t0[0] * cos_val * (accesor.at(accesor.n_modx/2 - 1, my, accesor.n_modz-1, 0, 0));
            k[0] = k[0] + t0[0] * cos_val2 * (accesor.at(accesor.n_modx-1, my, 0, 0, 0));
            k[1] = k[1] + t0[0] * cos_val * (accesor.at(accesor.n_modx/2 - 1, my, accesor.n_modz-1, 1, 0));
            k[1] = k[1] + t0[0] * cos_val2 * (accesor.at(accesor.n_modx-1, my, 0, 1, 0));
            k[2] = k[2] + t0[0] * cos_val * (accesor.at(accesor.n_modx/2 - 1, my, accesor.n_modz-1, 2, 0));
            k[2] = k[2] + t0[0] * cos_val2 * (accesor.at(accesor.n_modx-1, my, 0, 2, 0));
            for(my=1;my<accesor.n_mody;my++)
            {
                k[0] = k[0] + t1[0] * cos_val * (accesor.at(accesor.n_modx/2 - 1, my, accesor.n_modz-1, 0, 0));
                k[0] = k[0] + t1[0] * cos_val2 * (accesor.at(accesor.n_modx-1, my, 0, 0, 0));
                k[1] = k[1] + t1[0] * cos_val * (accesor.at(accesor.n_modx/2 - 1, my, accesor.n_modz-1, 1, 0));
                k[1] = k[1] + t1[0] * cos_val2 * (accesor.at(accesor.n_modx-1, my, 0, 1, 0));
                k[2] = k[2] + t1[0] * cos_val * (accesor.at(accesor.n_modx/2 - 1, my, accesor.n_modz-1, 2, 0));
                k[2] = k[2] + t1[0] * cos_val2 * (accesor.at(accesor.n_modx-1, my, 0, 2, 0));

                t0[0] = 2 * y * t1[0] - t0[0];
                swap = t0;
                t0 = t1;
                t1 = swap;
            }
            my = 0;
            t0[0] = 1;
            t1[0] = y;


            k[0] = k[0] - 2 * t0[0] * cos_val * (accesor.at(accesor.n_modx/2 - 1, my, accesor.n_modz-1, 0, 0));
            k[0] = k[0] - 2 * t0[0] * cos_val2 * (accesor.at(accesor.n_modx-1, my, 0, 0, 0));
            k[1] = k[1] - 2 * t0[0] * cos_val * (accesor.at(accesor.n_modx/2 - 1, my, accesor.n_modz-1, 1, 0));
            k[1] = k[1] - 2 * t0[0] * cos_val2 * (accesor.at(accesor.n_modx-1, my, 0, 1, 0));
            k[2] = k[2] - 2 * t0[0] * cos_val * (accesor.at(accesor.n_modx/2 - 1, my, accesor.n_modz-1, 2, 0));
            k[2] = k[2] - 2 * t0[0] * cos_val2 * (accesor.at(accesor.n_modx-1, my, 0, 2, 0));
            for(my=1;my<accesor.n_mody;my++)
            {
                k[0] = k[0] - 2 * t1[0] * cos_val * (accesor.at(accesor.n_modx/2 - 1, my, accesor.n_modz-1, 0, 0));
                k[0] = k[0] - 2 * t1[0] * cos_val2 * (accesor.at(accesor.n_modx-1, my, 0, 0, 0));
                k[1] = k[1] - 2 * t1[0] * cos_val * (accesor.at(accesor.n_modx/2 - 1, my, accesor.n_modz-1, 1, 0));
                k[1] = k[1] - 2 * t1[0] * cos_val2 * (accesor.at(accesor.n_modx-1, my, 0, 1, 0));
                k[2] = k[2] - 2 * t1[0] * cos_val * (accesor.at(accesor.n_modx/2 - 1, my, accesor.n_modz-1, 2, 0));
                k[2] = k[2] - 2 * t1[0] * cos_val2 * (accesor.at(accesor.n_modx-1, my, 0, 2, 0));

                t0[0] = 2 * y * t1[0] - t0[0];
                swap = t0;
                t0 = t1;
                t1 = swap;
            }
            my = 0;
            t0[0] = 1;
            t1[0] = y;

            
            for(mx=0;mx<accesor.n_modx/2 - 1;mx++)
            {
                for(mz=1;mz<accesor.n_modz-1;mz++)
                {
                    sin_val = sycl::sin(2 * pi *(x * (mx - accesor.n_modx/2 + 1)/accesor.Lx + mz * z/accesor.Lz));
                    cos_val = sycl::cos(2 * pi *(x * (mx - accesor.n_modx/2 + 1)/accesor.Lx + z * mz/accesor.Lz));
                    k[0] = k[0] + 2 * t0[0] * (accesor.at(mx, my, mz, 0, 0) * cos_val - accesor.at(mx, my, mz, 0, 1) * sin_val);
                    k[1] = k[1] + 2 * t0[0] * (accesor.at(mx, my, mz, 1, 0) * cos_val - accesor.at(mx, my, mz, 1, 1) * sin_val);
                    k[2] = k[2] + 2 * t0[0] * (accesor.at(mx, my, mz, 2, 0) * cos_val - accesor.at(mx, my, mz, 2, 1) * sin_val);
                    for(my=1;my<accesor.n_mody;my++)
                    {
                        k[0] = k[0] + 2 * t1[0] * (accesor.at(mx, my, mz, 0, 0) * cos_val - accesor.at(mx, my, mz, 0, 1) * sin_val);
                        k[1] = k[1] + 2 * t1[0] * (accesor.at(mx, my, mz, 1, 0) * cos_val - accesor.at(mx, my, mz, 1, 1) * sin_val);
                        k[2] = k[2] + 2 * t1[0] * (accesor.at(mx, my, mz, 2, 0) * cos_val - accesor.at(mx, my, mz, 2, 1) * sin_val);

                        t0[0] = 2 * y * t1[0] - t0[0];
                        swap = t0;
                        t0 = t1;
                        t1 = swap;
                    }
                    my = 0;
                    t0[0] = 1;
                    t1[0] = y;
                }
            }
            for(mx=accesor.n_modx/2 - 1;mx<accesor.n_modx - 1;mx++)
            {
                mz = 0;
                sin_val = sycl::sin(2 * pi *(x * (mx - accesor.n_modx/2 + 1)/accesor.Lx));
                cos_val = sycl::cos(2 * pi *(x * (mx - accesor.n_modx/2 + 1)/accesor.Lx));
                k[1] = k[1] + 2 * t0[0] * (accesor.at(mx, my, mz, 1, 0) * cos_val - accesor.at(mx, my, mz, 1, 1) * sin_val);
                k[2] = k[2] + 2 * t0[0] * (accesor.at(mx, my, mz, 2, 0) * cos_val - accesor.at(mx, my, mz, 2, 1) * sin_val);
                for(my=1;my<accesor.n_mody;my++)
                {
                    k[0] = k[0] + 2 * t1[0] * (accesor.at(mx, my, mz, 0, 0) * cos_val - accesor.at(mx, my, mz, 0, 1) * sin_val);
                    k[1] = k[1] + 2 * t1[0] * (accesor.at(mx, my, mz, 1, 0) * cos_val - accesor.at(mx, my, mz, 1, 1) * sin_val);
                    k[2] = k[2] + 2 * t1[0] * (accesor.at(mx, my, mz, 2, 0) * cos_val - accesor.at(mx, my, mz, 2, 1) * sin_val);

                    t0[0] = 2 * y * t1[0] - t0[0];
                    swap = t0;
                    t0 = t1;
                    t1 = swap;
                }
                my = 0;
                t0[0] = 1;
                t1[0] = y;

                for(mz=1;mz<accesor.n_modz-1;mz++)
                {
                    sin_val = sycl::sin(2 * pi *(x * (mx - accesor.n_modx/2 + 1)/accesor.Lx + mz * z/accesor.Lz));
                    cos_val = sycl::cos(2 * pi *(x * (mx - accesor.n_modx/2 + 1)/accesor.Lx + z * mz/accesor.Lz));

                    k[0] = k[0] + 2 * t0[0] * (accesor.at(mx, my, mz, 0, 0) * cos_val - accesor.at(mx, my, mz, 0, 1) * sin_val);
                    k[1] = k[1] + 2 * t0[0] * (accesor.at(mx, my, mz, 1, 0) * cos_val - accesor.at(mx, my, mz, 1, 1) * sin_val);
                    k[2] = k[2] + 2 * t0[0] * (accesor.at(mx, my, mz, 2, 0) * cos_val - accesor.at(mx, my, mz, 2, 1) * sin_val);
                    for(my=1;my<accesor.n_mody;my++)
                    {
                        k[0] = k[0] + 2 * t1[0] * (accesor.at(mx, my, mz, 0, 0) * cos_val - accesor.at(mx, my, mz, 0, 1) * sin_val);
                        k[1] = k[1] + 2 * t1[0] * (accesor.at(mx, my, mz, 1, 0) * cos_val - accesor.at(mx, my, mz, 1, 1) * sin_val);
                        k[2] = k[2] + 2 * t1[0] * (accesor.at(mx, my, mz, 2, 0) * cos_val - accesor.at(mx, my, mz, 2, 1) * sin_val);

                        t0[0] = 2 * y * t1[0] - t0[0];
                        swap = t0;
                        t0 = t1;
                        t1 = swap;
                    }
                    my = 0;
                    t0[0] = 1;
                    t1[0] = y;
                }

                cos_val = sycl::cos(2 * pi * (accesor.n_modz-1) * z/accesor.Lz);
                sin_val = sycl::sin(2 * pi * x * (mx - accesor.n_modx/2 + 1)/accesor.Lx);
                cos_val2 = sycl::cos(2 * pi * x * (mx - accesor.n_modx/2 + 1)/accesor.Lx);
                k[0] = k[0] + 2 * t0[0] * cos_val * (accesor.at(mx, my, mz, 0, 0) * cos_val2 - accesor.at(mx, my, mz, 0, 1) * sin_val);
                k[1] = k[1] + 2 * t0[0] * cos_val * (accesor.at(mx, my, mz, 1, 0) * cos_val2 - accesor.at(mx, my, mz, 1, 1) * sin_val);
                k[2] = k[2] + 2 * t0[0] * cos_val * (accesor.at(mx, my, mz, 2, 0) * cos_val2 - accesor.at(mx, my, mz, 2, 1) * sin_val);
                for(my=1;my<accesor.n_mody;my++)
                {
                    k[0] = k[0] + 2 * t1[0] * cos_val * (accesor.at(mx, my, mz, 0, 0) * cos_val2 - accesor.at(mx, my, mz, 0, 1) * sin_val);
                    k[1] = k[1] + 2 * t1[0] * cos_val * (accesor.at(mx, my, mz, 1, 0) * cos_val2 - accesor.at(mx, my, mz, 1, 1) * sin_val);
                    k[2] = k[2] + 2 * t1[0] * cos_val * (accesor.at(mx, my, mz, 2, 0) * cos_val2 - accesor.at(mx, my, mz, 2, 1) * sin_val);

                    t0[0] = 2 * y * t1[0] - t0[0];
                    swap = t0;
                    t0 = t1;
                    t1 = swap;
                }
                my = 0;
                t0[0] = 1;
                t1[0] = y;
            }
            for(mz=0;mz<accesor.n_modz-1;mz++)
            {
                cos_val = sycl::cos(pi * accesor.n_modx * x/accesor.Lx);
                sin_val = sycl::sin(2 * pi * z * mz/accesor.Lz);
                cos_val2 = sycl::cos(2 * pi * z * mz/accesor.Lz);
                k[0] = k[0] + 2 * t0[0] * cos_val * (accesor.at(mx, my, mz, 0, 0) * cos_val2 - accesor.at(mx, my, mz, 0, 1) * sin_val);
                k[1] = k[1] + 2 * t0[0] * cos_val * (accesor.at(mx, my, mz, 1, 0) * cos_val2 - accesor.at(mx, my, mz, 1, 1) * sin_val);
                k[2] = k[2] + 2 * t0[0] * cos_val * (accesor.at(mx, my, mz, 2, 0) * cos_val2 - accesor.at(mx, my, mz, 2, 1) * sin_val);
                for(my=1;my<accesor.n_mody;my++)
                {
                    k[0] = k[0] + 2 * t1[0] * cos_val * (accesor.at(mx, my, mz, 0, 0) * cos_val2 - accesor.at(mx, my, mz, 0, 1) * sin_val);
                    k[1] = k[1] + 2 * t1[0] * cos_val * (accesor.at(mx, my, mz, 1, 0) * cos_val2 - accesor.at(mx, my, mz, 1, 1) * sin_val);
                    k[2] = k[2] + 2 * t1[0] * cos_val * (accesor.at(mx, my, mz, 2, 0) * cos_val2 - accesor.at(mx, my, mz, 2, 1) * sin_val);

                    t0[0] = 2 * y * t1[0] - t0[0];
                    swap = t0;
                    t0 = t1;
                    t1 = swap;
                }
                my = 0;
                t0[0] = 1;
                t1[0] = y;
            }

            cos_val = sycl::cos(2 * pi * (accesor.n_modz-1) * z/accesor.Lz);
            cos_val2 = sycl::cos(pi * accesor.n_modx * x/accesor.Lx);
            k[0] = k[0] + t0[0] * (accesor.at(mx, my, mz, 0, 0) * cos_val * cos_val2);
            k[1] = k[1] + t0[0] * (accesor.at(mx, my, mz, 1, 0) * cos_val * cos_val2);
            k[2] = k[2] + t0[0] * (accesor.at(mx, my, mz, 2, 0) * cos_val * cos_val2);
            for(my=1;my<accesor.n_mody;my++)
            {
                k[0] = k[0] + t1[0] * (accesor.at(mx, my, mz, 0, 0) * cos_val * cos_val2);
                k[1] = k[1] + t1[0] * (accesor.at(mx, my, mz, 1, 0) * cos_val * cos_val2);
                k[2] = k[2] + t1[0] * (accesor.at(mx, my, mz, 2, 0) * cos_val * cos_val2);

                t0[0] = 2 * y * t1[0] - t0[0];
                swap = t0;
                t0 = t1;
                t1 = swap;
            }
            my = 0;
            t0[0] = 1;
            t1[0] = y;
        };

        auto euler = [=] (auto position, auto k, auto position_old)
        {
            position[0] = position[0] + dt * k[0];
            position[1] = position[1] + dt * k[1];

            k[0] = 0;
            k[1] = 0;

            position_old[0] = position[0];
            position_old[1] = position[1];
            position_old[2] = position[2];

            if(position[0] > accesor.Lx)
            {
                position[0] = 0;
            }
            if(position[0] < 0)
            {
                position[0] = accesor.Lx;
            }
            if(position[2] > accesor.Lz)
            {
                position[2] = 0;
            }
            if(position[2] < 0)
            {
                position[2] = accesor.Lz;
            }
        };

        auto add_k1 = [=] (auto position, auto k) {
            k[0] = dt * k[0]/2;
            position[0] = position[0] + k[0]/3.0;

            k[1] = dt * k[1]/2;
            position[1] = position[1] + k[1]/3.0;

            k[2] = dt * k[2]/2;
            position[2] = position[2] + k[2]/3.0;
        };

        auto add_k2 = [=] (auto position, auto k) {
            k[0] = dt * k[0]/2;
            position[0] = position[0] + 2.0 * k[0]/3.0;

            k[1] = dt * k[1]/2;
            position[1] = position[1] + 2.0 * k[1]/3.0;

            k[2] = dt * k[2]/2;
            position[2] = position[2] + 2.0 * k[2]/3.0;
        };

        auto add_k3 = [=] (auto position, auto k) {
            k[0] = dt * k[0];
            position[0] = position[0] + k[0]/3.0;

            k[1] = dt * k[1];
            position[1] = position[1] + k[1]/3.0;

            k[2] = dt * k[2];
            position[2] = position[2] + k[2]/3.0;
        };

        auto add_k4 = [=] (auto position, auto k, auto position_old) {
            position[0] = position[0] + dt * k[0]/6.0;

            position[1] = position[1] + dt * k[1]/6.0;

            position[2] = position[2] + dt * k[2]/6.0;

            position_old[0] = position[0];
            position_old[1] = position[1];
            position_old[2] = position[2];

            k[0] = 0;
            k[1] = 0;
            k[2] = 0;

            if(position[0] > accesor.Lx)
            {
                position[0] = 0;
            }
            if(position[0] < 0)
            {
                position[0] = accesor.Lx;
            }
            if(position[2] > accesor.Lz)
            {
                position[2] = 0;
            }
            if(position[2] < 0)
            {
                position[2] = accesor.Lz;
            }
        };

        auto vel = [=] (auto k1, auto veloci)
        {
            veloci[0] = k1[0];
            veloci[1] = k1[1];
            veloci[2] = k1[2];
        };

        neso::ParticleLoop calculate_f("advection", particle_group_ptr, calc_k, neso::Access::write(neso::Sym<neso::REAL>("position_old")), 
                                       neso::Access::write(neso::Sym<neso::REAL>("k")));
        neso::ParticleLoop k1("advection", particle_group_ptr, add_k1, neso::Access::write(neso::Sym<neso::REAL>("position")),
                              neso::Access::write(neso::Sym<neso::REAL>("k")));
        neso::ParticleLoop k2("advection", particle_group_ptr, add_k2, neso::Access::write(neso::Sym<neso::REAL>("position")),
                              neso::Access::write(neso::Sym<neso::REAL>("k")));
        neso::ParticleLoop k3("advection", particle_group_ptr, add_k3, neso::Access::write(neso::Sym<neso::REAL>("position")),
                              neso::Access::write(neso::Sym<neso::REAL>("k")));
        neso::ParticleLoop k4_finalize("advection", particle_group_ptr, add_k4, neso::Access::write(neso::Sym<neso::REAL>("position")),
                              neso::Access::write(neso::Sym<neso::REAL>("k")), neso::Access::write(neso::Sym<neso::REAL>("position_old")));
        neso::ParticleLoop euler_step("advection", particle_group_ptr, euler, neso::Access::write(neso::Sym<neso::REAL>("position")),
                              neso::Access::write(neso::Sym<neso::REAL>("k")), neso::Access::write(neso::Sym<neso::REAL>("position_old")));

        neso::ParticleLoop get_velocity("advection", particle_group_ptr, vel, neso::Access::write(neso::Sym<neso::REAL>("k")), 
                                       neso::Access::write(neso::Sym<neso::REAL>("velocity")));

        neso::H5Part save_position("testy_zmiana_petli.h5part", particle_group_ptr, position, velocity, color_rank);
        //save_position.write(0);
        
        int step = 0;
        int save_interval = 0;
        int save_count = 0;
        double time = 0;
        
        reader.set_time(0);
        reader.host_to_device();
        
        for(step=0;step<80;step++)
        {
            calculate_f.submit();
            reader.set_time(time + dt/2.0);
            calculate_f.wait();
            get_velocity.execute();
            k1.execute();
            
            reader.host_to_device();
            calculate_f.execute();
            k2.execute();

            calculate_f.submit();
            reader.set_time(time + dt);
            calculate_f.wait();
            k3.execute();
            
            reader.host_to_device();
            calculate_f.execute();

            k4_finalize.execute();

            time = time + dt;
            save_interval++;
            if(save_interval == 2)
            {
                save_position.write(save_count);
                save_count++;
                save_interval = 0;
            }
            //save_interval++;
            if(rank == 0)
            {
                std::cout<<step<<std::endl;
            }
            
        }
        
        save_position.close();
    }
    
    MPI_Finalize();
}
