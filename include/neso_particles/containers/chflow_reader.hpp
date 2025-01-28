#include <iostream>
#include <fstream>
#include <mpi.h>
#include <memory>
#include <fftw3.h>
#include "../compute_target.hpp"

namespace neso = NESO::Particles;

namespace moje {
    class ChflowReader;

    enum timeInterpolationMethod {
        fourier,
        linear
    };

    class NodeFFTdata
    {
        friend class ChflowReader;
        public:
        NodeFFTdata()
        {

        }

        void set(int start, int blocks, int positive_block_size, int negative_block_size, int time_steps, double step_len,
                 MPI_Comm intra, std::string generic_name, enum timeInterpolationMethod method)
        {
            this->start_bytes = (positive_block_size + negative_block_size) * start;
            this->positive_block_size_bytes = positive_block_size;
            this->negative_block_size_bytes = negative_block_size;
            this->n_blocks = blocks;
            this->n_time_steps = time_steps;
            this->intra_comm = intra;
            this->file_head_bytes = 2 * sizeof(double) + 4 * sizeof(int);
            this->Tk = time_steps * step_len;
            this->step_len = step_len;
            this->method = method;

            int size;
            int rank_intra;
            MPI_Comm_rank(intra_comm, &rank_intra);
            if(rank_intra == 0)
            {
                size = (this->positive_block_size_bytes + this->negative_block_size_bytes) * n_blocks * n_time_steps;
            }
            else
            {
                size = 0;
            }

            MPI_Win_allocate_shared(size, sizeof(fftw_complex), MPI_INFO_NULL,
                                    intra_comm, &this->values, &this->loaded);

            MPI_Aint tmp_s;
            int tmp_d;
            MPI_Win_shared_query(loaded, 0, &tmp_s, &tmp_d, &this->values);

            int i = 0;
            int j = 0;
            for(i=0;i<n_time_steps;i++)
            {
                MPI_File file;
                std::string file_name = generic_name + std::to_string(i) + ".ff";

                MPI_File_open(intra_comm, file_name.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
                MPI_File_seek(file, start_bytes + file_head_bytes, MPI_SEEK_SET);

                //loaduje tylko proces zerowy
                int full_block_size = this->positive_block_size_bytes + this->negative_block_size_bytes;
                if(rank_intra == 0)
                {
                    char* buffer = (char*)this->values;
                    for(j=0;j<n_blocks;j++)
                    {
                        //zamieniam kolejnoscia zeby byly od ujemnych do dodatnich
                        buffer = buffer + this->negative_block_size_bytes;
                        MPI_File_read(file, buffer + i * n_blocks * full_block_size + j * full_block_size,
                                      this->positive_block_size_bytes/sizeof(double), MPI_DOUBLE, MPI_STATUS_IGNORE);
                        buffer = buffer - this->negative_block_size_bytes;
                        MPI_File_read(file, buffer + i * n_blocks * full_block_size + j * full_block_size,
                                      this->negative_block_size_bytes/sizeof(double), MPI_DOUBLE, MPI_STATUS_IGNORE);
                    }
                }

                MPI_Barrier(intra_comm);
                MPI_File_close(&file);
            }
            //na tym etapie mamy zaladowane wspolczynniki ze wszytskich krokow czasowych
            this->values_to_coeffs();
        }

        ~NodeFFTdata()
        {
            MPI_Win_free(&loaded);
            MPI_Win_free(&interpolated);

            fftw_free(coeffs);            
        }

        private:
        MPI_Comm intra_comm;

        MPI_Win loaded;
        MPI_Win interpolated;

        double* values; //pobrana z pliku duza tablica, shared memory
        fftw_complex* coeffs; //kazdy ma swoj kawalek po zrobieniu swoich dft

        fftw_complex* interpolated_values;

        int start_bytes;
        int positive_block_size_bytes;
        int negative_block_size_bytes;
        int n_blocks;
        int n_time_steps;
        int file_head_bytes;
        int file_size_bytes;

        int my_start;
        int my_end;

        double Tk;
        double step_len;

        enum timeInterpolationMethod method;

        void values_to_coeffs()
        {
            int intra_size;
            int intra_rank;
            MPI_Comm_size(intra_comm, &intra_size);
            MPI_Comm_rank(intra_comm, &intra_rank);
            
            switch(this->method)
            {
                case fourier:
                {
                    neso::get_decomp_1d(intra_size, (int)(n_blocks * (this->positive_block_size_bytes + this->negative_block_size_bytes)/sizeof(double)),
                                intra_rank, &my_start, &my_end);

                    int n_dfts = my_end - my_start;
                    int n_non_negative_mods = n_time_steps/2 + 1;
                    this->coeffs = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n_dfts * n_non_negative_mods);

                    int n[] = {n_time_steps};
                    int stride = n_blocks * (this->positive_block_size_bytes + this->negative_block_size_bytes)/sizeof(double);

                    fftw_plan plan = fftw_plan_many_dft_r2c(1, n, n_dfts, this->values + my_start, NULL, stride,
                                                            1, this->coeffs, NULL, 1, n_non_negative_mods, FFTW_ESTIMATE);

                    fftw_execute(plan);

                    fftw_destroy_plan(plan);
                }break;

                case linear:
                {
                    neso::get_decomp_1d(intra_size, (int)(n_blocks * (this->positive_block_size_bytes + this->negative_block_size_bytes)/sizeof(fftw_complex)),
                                intra_rank, &my_start, &my_end);
                    this->coeffs = (fftw_complex*)fftw_malloc(0);
                }break;
            }

            int interp_size;
            if(intra_rank == 0)
            {
                interp_size = n_blocks * (this->positive_block_size_bytes + this->negative_block_size_bytes);
            }
            else
            {
                interp_size = 0;
            }
            MPI_Win_allocate_shared(interp_size, sizeof(fftw_complex), MPI_INFO_NULL, intra_comm, &this->interpolated_values,
                                    &this->interpolated);
            MPI_Aint tmp_s;
            int tmp_d;
            MPI_Win_shared_query(this->interpolated, 0, &tmp_s, &tmp_d, &this->interpolated_values);
            //teraz kazdy ma juz wspolczynniki i wie gdzie pisac (my_start i my end) gdy bedzie transformate robil
        }

        void interpolate_my(double t)
        {
            switch(this->method)
            {
                case fourier:
                {
                    int k = 0;
                    int dft_id = 0;
                    double* val = (double*)this->interpolated_values + my_start;

                    double real;
                    double imag;
                    double y;
                    
                    int n_non_negative = n_time_steps/2 + 1;
                    for(dft_id=0;dft_id<my_end - my_start;dft_id++)
                    {
                        real = this->coeffs[dft_id * n_non_negative][0];
                        val[dft_id] = real/n_time_steps;
                        for(k=1;k<n_non_negative;k++)
                        {
                            real = this->coeffs[dft_id * n_non_negative + k][0];
                            imag = this->coeffs[dft_id * n_non_negative + k][1];
                            y = 2 * 3.141592 * k * t/this->Tk;

                            val[dft_id] = val[dft_id] + 2 * (real * std::cos(y) - imag * std::sin(y))/n_time_steps;
                        }
                    }
                    MPI_Barrier(this->intra_comm);
                }break;

                case linear:
                {
                    fftw_complex* inter_val = this->interpolated_values + my_start;
                    fftw_complex* val = (fftw_complex*)this->values + my_start;
                    int stride = n_blocks * (positive_block_size_bytes + negative_block_size_bytes)/sizeof(fftw_complex);

                    int n = t/this->step_len;
                    int dft_id = 0;
                    for(dft_id=0;dft_id<my_end - my_start;dft_id++)
                    {
                        inter_val[dft_id][0] = val[dft_id + n * stride][0] + (val[dft_id + (n + 1) * stride][0] - val[dft_id + n * stride][0]) * (t - n * this->step_len)/this->step_len;
                        inter_val[dft_id][1] = val[dft_id + n * stride][1] + (val[dft_id + (n + 1) * stride][1] - val[dft_id + n * stride][1]) * (t - n * this->step_len)/this->step_len;
                    }
                    MPI_Barrier(this->intra_comm);
                }break;
            }
        }
    };


    //tu nie ma wszytskiego bo to wlasciwei tylko po to zeby sprawdzac czy jest ok wczytywanie
    struct ChflowHostAccesor
    {
        int n_modx;
        int n_mody;
        int n_modz;
        int n_dims;

        double* coeffs;

        double& at(int mx, int my, int mz, int comp, int part) const
        {
            return coeffs[2 * (comp * n_mody * n_modx * n_modz +
                          my * n_modx * n_modz + mx * n_modz + mz) + part];
        }
    };

    struct ChflowDeviceAccesor
    {
        int n_modx;
        int n_mody;
        int n_modz;
        int n_dims;

        double Lx;
        double Lz;

        double* coeffs;

        //part - rzeczywsita czy urojona
        // 0 rzeczywista, 1 urojona
        double at(int mx, int my, int mz, int comp, int part) const
        {
            return coeffs[2 * (comp * n_mody * n_modx * n_modz +
                          my * n_modx * n_modz + mx * n_modz + mz) + part];
        }
    };

    class ChflowReader
    {
        friend void AccesorInit(ChflowHostAccesor& acc, ChflowReader& read);
        friend inline void AccesorInit(ChflowDeviceAccesor& acc, ChflowReader& read);

        private:
        NodeFFTdata fft_data;

        neso::CommPair fft_comm;

        MPI_Win shared;

        std::shared_ptr<neso::SYCLTarget> target;        

        double* h_coeffs; //to bedzie wczytane z pliku do shared memory dla kazdego intra commu
        double* d_coeffs; //to bedzie musial miec kazdy proces w swoim kawalku pamieci na device (kopia h_coeffs)

        //przy modx jest rozroznienie bo te ujemne tez sa zapisane w pliku a te z z juz nie
        int n_modx;
        int n_modx_non_negative;
        int n_mody;
        int n_modz;
        int n_dims;
        
        double Lx;
        double Lz;
        //na ten moment zaloze ze po y jest zawsze -1 do 1

        public:
        ChflowReader(std::string generic_filename, std::shared_ptr<neso::SYCLTarget> target_ptr, int n_time_steps, double step_len,
                     enum timeInterpolationMethod method) : fft_comm(MPI_COMM_WORLD)
        {
            std::string file = generic_filename + std::to_string(0) + ".ff";
            MPI_File case_data;
            MPI_File_open(fft_comm.comm_intra, file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &case_data);

            int* buffer_int = new int[4];
            double* buffer_double = new double[2];
            MPI_File_read(case_data, buffer_int, 4, MPI_INT, MPI_STATUS_IGNORE);
            MPI_File_read(case_data, buffer_double, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);

            this->n_modx = buffer_int[0];
            this->n_mody = buffer_int[1];
            this->n_modz = buffer_int[2]/2 + 1;
            this->n_dims = buffer_int[3];

            this->n_modx_non_negative = this->n_modx/2 + 1;

            this->Lx = buffer_double[0];
            this->Lz = buffer_double[1];

            delete[] buffer_double;
            delete[] buffer_int;

            if(fft_comm.rank_intra == 0)
            {
                std::cout<<"chflow info:"<<std::endl;
                std::cout<<"field dims = "<<this->n_dims<<std::endl;
                std::cout<<"number of mods = ("<<this->n_modx<<","<<this->n_mody<<","<<this->n_modz<<")"<<std::endl;
                std::cout<<"domain dims = ("<<this->Lx<<","<<2<<","<<this->Lz<<")"<<std::endl;
            }
            MPI_File_close(&case_data);

            int data_blocks_per_file = this->n_dims * this->n_mody; //zakladam narazie ze to bedzie zawsze wiecej niz liczba nodow na ktorym to sie odpala
            int positive_data_block_size_bytes = 2 * sizeof(double) * this->n_modz * this->n_modx_non_negative;
            int negative_data_block_size_bytes = 2 * sizeof(double) * this->n_modz * (this->n_modx - this->n_modx_non_negative);
            
            int n_nodes;
            if(fft_comm.is_intra())
            {
                MPI_Comm_size(fft_comm.comm_inter, &n_nodes);
            }

            MPI_Win start_end;
            int start_end_size;
            if(fft_comm.is_intra())
            {
                start_end_size = 2 * sizeof(int);
            }
            else
            {
                start_end_size = 0;
            }
            int* s_e;
            MPI_Win_allocate_shared(start_end_size, sizeof(int), MPI_INFO_NULL, fft_comm.comm_intra, &s_e, &start_end);
            MPI_Aint tmp_s;
            int tmp_d;
            MPI_Win_shared_query(start_end, 0, &tmp_s, &tmp_d, &s_e);
            if(fft_comm.is_intra())
            {
                neso::get_decomp_1d(n_nodes, data_blocks_per_file, fft_comm.rank_inter, &s_e[0], &s_e[1]);
            }
            MPI_Barrier(fft_comm.comm_intra);

            int our_start = s_e[0];
            int our_end = s_e[1];
            MPI_Win_free(&start_end);
            
            this->fft_data.set(our_start, our_end - our_start, positive_data_block_size_bytes, negative_data_block_size_bytes,
                               n_time_steps, step_len, fft_comm.comm_intra, generic_filename, method);

            MPI_Barrier(fft_comm.comm_intra);

            int size = 2 * this->n_dims * this->n_modx * this->n_mody * this->n_modz;
            int size_bytes;
            if(fft_comm.rank_intra == 0)
            {
                size_bytes = size * sizeof(double);
            }
            else
            {
                size_bytes = 0;
            }

            MPI_Win_allocate_shared(size_bytes, sizeof(double), MPI_INFO_NULL, this->fft_comm.comm_intra, &this->h_coeffs, &this->shared);

            MPI_Aint size_tmp;
            int disp_tmp;
            MPI_Win_shared_query(this->shared, 0, &size_tmp, &disp_tmp, &this->h_coeffs);

            int size_device = size * sizeof(double);
            this->d_coeffs = (double*)sycl::malloc_device(size_device, target_ptr->queue);

            this->target = target_ptr;
        }

        void set_time(double time)
        {
            this->fft_data.interpolate_my(time);
            MPI_Barrier(MPI_COMM_WORLD);

            if(this->fft_comm.is_intra())
            {
                int size = fft_comm.size_inter;

                int i = 0;
                int send_size_bytes = this->fft_data.n_blocks * (this->fft_data.positive_block_size_bytes + this->fft_data.negative_block_size_bytes);
                int* sizes = new int[size];
                MPI_Allgather(&send_size_bytes, 1, MPI_DOUBLE, sizes, 1, MPI_DOUBLE, this->fft_comm.comm_inter);

                int* displacements = new int[size];
                displacements[0] = 0;
                for(i=1;i<size;i++)
                {
                    displacements[i] = displacements[i-1] + sizes[i-1]/sizeof(double);
                }

                MPI_Allgatherv(this->fft_data.interpolated_values, send_size_bytes/sizeof(double), MPI_DOUBLE,
                               this->h_coeffs, sizes, displacements, MPI_DOUBLE, this->fft_comm.comm_inter);

                delete[] sizes;
                delete[] displacements;
            }
        }

        void host_to_device()
        {
            int size = 2 * this->n_dims * this->n_modx * this->n_mody * this->n_modz * sizeof(double);
            this->target->queue.memcpy(this->d_coeffs, this->h_coeffs, size).wait();
        }
        
        ~ChflowReader()
        {
            MPI_Win_free(&this->shared);
            
            sycl::free(this->d_coeffs, this->target->queue);
        }
        
    };

    void AccesorInit(ChflowHostAccesor& acc, ChflowReader& read)
    {
        acc.coeffs = read.h_coeffs;

        acc.n_dims = read.n_dims;
        acc.n_modx = read.n_modx;
        acc.n_mody = read.n_mody;
        acc.n_modz = read.n_modz;
    }

    inline void AccesorInit(ChflowDeviceAccesor& acc, ChflowReader& read)
    {
        acc.coeffs = read.d_coeffs;

        acc.n_dims = read.n_dims;
        acc.n_modx = read.n_modx;
        acc.n_mody = read.n_mody;
        acc.n_modz = read.n_modz;

        acc.Lx = read.Lx;
        acc.Lz = read.Lz;
    }
}