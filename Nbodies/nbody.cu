/*
   Running without arguments is equivalent to 1000 iterations with the
   5 celestial objects declared in the golden_bodies array.

   $ nbody.exe 1000 5

   The output of this shows the energy before and after the simulation,
   and should be:

   -0.169075164
   -0.169087605
*/

#include "Timer.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>

#define GPU 0
#define DP 1

#if DP
using type = double;
#else
using type = float;
#endif

const type pi{3.141592653589793};
const type solar_mass{4 * pi * pi};
const type days_per_year{365.24};

int threadsPerWarp = 0;
int maxThreadsPerBlock = 0;
int maxSharedMemPerBlock = 0;

template <typename T>
struct planet {
  T x, y, z;
  T vx, vy, vz;
  T mass;
};

template <typename T>
__global__ void advanceVelocities(int nbodies, planet<T> *bodies, int offset)
{
  int j = threadIdx.x + offset;
  int i = blockIdx.x;
  //T valid = fmax(0.0, fmin(1.0, (T)(j - i))); //Returns either 1 or 0. 1 if j is valid, 0 if it is not. 

  if (i < nbodies && j < nbodies && j>i)
  {
    planet<T> &b = bodies[i];
    planet<T> &b2 = bodies[j];
    T dx = b.x - b2.x;
    T dy = b.y - b2.y;
    T dz = b.z - b2.z;
    T inv_distance = 1.0 / sqrt(dx * dx + dy * dy + dz * dz);
    T mag = inv_distance * inv_distance * inv_distance;
    b.vx -= dx * b2.mass * mag;
    b.vy -= dy * b2.mass * mag;
    b.vz -= dz * b2.mass * mag;
    b2.vx += dx * b.mass  * mag;
    b2.vy += dy * b.mass  * mag;
    b2.vz += dz * b.mass  * mag;
  }
}

template <typename T>
__global__ void advancePositions(int nbodies, planet<T> *bodies, int offset)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x + offset;

  if (i < nbodies)
  {
    planet<T> &b = bodies[i];
    b.x += b.vx;
    b.y += b.vy;
    b.z += b.vz;
  }
}

template <typename T>
void advance_gpued(int nbodies, planet<T> *bodies)
{
  Timer timer;
  timer.start("advance_gpued");
  //Advance velocities
  int numIterations = ceil(nbodies / maxThreadsPerBlock);
  for (int i = 0; i < numIterations; ++i)
  {
    advanceVelocities << <nbodies, min(nbodies, maxThreadsPerBlock) >> >(nbodies, bodies, i*maxThreadsPerBlock);
  }
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    std::cout << "Error in velocity kernal: " << cudaGetErrorString(error) << std::endl;
  }
  //Advance positions
  for (int i = 0; i < numIterations; ++i)
  {
    advancePositions << <1, min(nbodies, maxThreadsPerBlock) >> >(nbodies, bodies, i*maxThreadsPerBlock);
  }
  cudaDeviceSynchronize();
  error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    std::cout << "Error in position kernal: " << cudaGetErrorString(error) << std::endl;
  }

  timer.end();
}

template <typename T>
void advance(int nbodies, planet<T> *bodies)
{
  Timer timer; timer.start("advance");
  int i, j;

  for (i = 0; i < nbodies; ++i) {
    planet<T> &b = bodies[i];
    for (j = i + 1; j < nbodies; j++) {
      planet<T> &b2 = bodies[j];
      T dx = b.x - b2.x;
      T dy = b.y - b2.y;
      T dz = b.z - b2.z;
      T inv_distance = 1.0/sqrt(dx * dx + dy * dy + dz * dz);
      T mag = inv_distance * inv_distance * inv_distance;
      b.vx  -= dx * b2.mass * mag;
      b.vy  -= dy * b2.mass * mag;
      b.vz  -= dz * b2.mass * mag;
      b2.vx += dx * b.mass  * mag;
      b2.vy += dy * b.mass  * mag;
      b2.vz += dz * b.mass  * mag;
    }
  }
  for (i = 0; i < nbodies; ++i) {
    planet<T> &b = bodies[i];
    b.x += b.vx;
    b.y += b.vy;
    b.z += b.vz;
  }
  timer.end();
}

template <typename T>
T energy(int nbodies, planet<T> *bodies)
{
  Timer timer; timer.start("energy");
  T e = 0.0;
  for (int i = 0; i < nbodies; ++i) {
    planet<T> &b = bodies[i];
    e += 0.5 * b.mass * (b.vx * b.vx + b.vy * b.vy + b.vz * b.vz);
    for (int j = i + 1; j < nbodies; j++) {
      planet<T> &b2 = bodies[j];
      T dx = b.x - b2.x;
      T dy = b.y - b2.y;
      T dz = b.z - b2.z;
      T distance = sqrt(dx * dx + dy * dy + dz * dz);
      e -= (b.mass * b2.mass) / distance;
    }
  }
  timer.end();
  return e;
}
template <typename T>
void offset_momentum(int nbodies, planet<T> *bodies)
{
  Timer timer; timer.start("offset_momentum");
  T px = 0.0, py = 0.0, pz = 0.0;
  for (int i = 0; i < nbodies; ++i) {
    px += bodies[i].vx * bodies[i].mass;
    py += bodies[i].vy * bodies[i].mass;
    pz += bodies[i].vz * bodies[i].mass;
  }
  bodies[0].vx = - px / solar_mass;
  bodies[0].vy = - py / solar_mass;
  bodies[0].vz = - pz / solar_mass;
  timer.end();
}

struct planet<type> golden_bodies[5] = {
  {                               /* sun */
    0, 0, 0, 0, 0, 0, solar_mass
  },
  {                               /* jupiter */
    4.84143144246472090e+00,
    -1.16032004402742839e+00,
    -1.03622044471123109e-01,
    1.66007664274403694e-03 * days_per_year,
    7.69901118419740425e-03 * days_per_year,
    -6.90460016972063023e-05 * days_per_year,
    9.54791938424326609e-04 * solar_mass
  },
  {                               /* saturn */
    8.34336671824457987e+00,
    4.12479856412430479e+00,
    -4.03523417114321381e-01,
    -2.76742510726862411e-03 * days_per_year,
    4.99852801234917238e-03 * days_per_year,
    2.30417297573763929e-05 * days_per_year,
    2.85885980666130812e-04 * solar_mass
  },
  {                               /* uranus */
    1.28943695621391310e+01,
    -1.51111514016986312e+01,
    -2.23307578892655734e-01,
    2.96460137564761618e-03 * days_per_year,
    2.37847173959480950e-03 * days_per_year,
    -2.96589568540237556e-05 * days_per_year,
    4.36624404335156298e-05 * solar_mass
  },
  {                               /* neptune */
    1.53796971148509165e+01,
    -2.59193146099879641e+01,
    1.79258772950371181e-01,
    2.68067772490389322e-03 * days_per_year,
    1.62824170038242295e-03 * days_per_year,
    -9.51592254519715870e-05 * days_per_year,
    5.15138902046611451e-05 * solar_mass
  }
};

const type DT{1e-2};
const type RECIP_DT{(type)1.0/DT};

/*
 * Rescale certain properties of bodies. That allows doing
 * consequential advance()'s as if dt were equal to 1.0.
 *
 * When all advances done, rescale bodies back to obtain correct energy.
 */
template <typename T>
void scale_bodies(int nbodies, planet<T> *bodies, T scale)
{
  Timer timer; timer.start("scale_bodies");
  for (int i = 0; i < nbodies; ++i) {
    bodies[i].mass *= scale*scale;
    bodies[i].vx   *= scale;
    bodies[i].vy   *= scale;
    bodies[i].vz   *= scale;
  }
  timer.end();
}

template <typename T>
void init_random_bodies(int nbodies, planet<T> *bodies)
{
  Timer timer; timer.start("init_random_bodies");
  for (int i = 0; i < nbodies; ++i) {
    bodies[i].x    =  (T)rand()/RAND_MAX;
    bodies[i].y    =  (T)rand()/RAND_MAX;
    bodies[i].z    =  (T)rand()/RAND_MAX;
    bodies[i].vx   =  (T)rand()/RAND_MAX;
    bodies[i].vy   =  (T)rand()/RAND_MAX;
    bodies[i].vz   =  (T)rand()/RAND_MAX;
    bodies[i].mass =  (T)rand()/RAND_MAX;
  }
  timer.end();
}

int main(int argc, char ** argv)
{
	cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
	cudaDeviceGetAttribute(&maxSharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);
	cudaDeviceGetAttribute(&threadsPerWarp, cudaDevAttrWarpSize, 0);
#if GPU
	std::cout << "GPU\n";
	std::cout << "Max Threads/Block: " << maxThreadsPerBlock << std::endl;
  std::cout << "Max Shared Mem/Block: " << maxSharedMemPerBlock << std::endl;
  std::cout << "Warp Size: " << threadsPerWarp << std::endl;
#else
	std::cout << "CPU\n";
#endif

  int niters = 1000, nbodies = 5;
  if (argc > 1) { niters  = atoi(argv[1]); }
  if (argc > 2) { nbodies = atoi(argv[2]); }

  std::cout << "niters=" << niters << " nbodies=" << nbodies << '\n';

  std::string outputName = "outputTimes";
#if GPU
  outputName += "_GPU";
#else
  outputName += "_CPU";
#endif
#if DP
  outputName += "_double";
#else
  outputName += "_float";
#endif
  outputName += ".csv";

  std::ofstream outStream(outputName);
  Timer::setFileStream(&outStream);

  planet<type> *bodies;
  if (argc == 1) { 
    bodies = golden_bodies; // Check accuracy with 1000 solar system iterations
  } else {
    bodies = new planet<type>[nbodies];
    init_random_bodies(nbodies, bodies);
  }

  Timer timerMain; timerMain.start("Main");
  offset_momentum(nbodies, bodies);
  type e1 = energy(nbodies, bodies);
  scale_bodies(nbodies, bodies, DT);
  Timer timerAdvance; timerAdvance.start("arch_advance");

#if GPU
  //Allocate memory on the GPU for the list of bodies. 
  planet<type> *devBodies = nullptr;
  cudaError_t error = cudaMalloc(&devBodies, nbodies*sizeof(planet<type>));
  if (error != cudaSuccess)
  {
    std::cout << "Failed to allocate global memory for CUDA. \n";
    return -1;
  }
  //Copy the list of bodies to the GPU memory
  error = cudaMemcpy(devBodies, bodies, nbodies*sizeof(planet<type>), cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
  {
    std::cout << "Failed to copy from host to device memory: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }
#endif

  for (int i = 1; i <= niters; ++i)  {

#if GPU
    advance_gpued(nbodies, devBodies);
#else
    advance(nbodies, bodies);
#endif

  }

#if GPU
  //Set all memory for the list of bodies on the CPU to zero to ensure there are no random errors. 
  memset(bodies, 0, nbodies*sizeof(planet<type>));
  //Copy data from GPU memory to CPU memory
  error = cudaMemcpy(bodies, devBodies, nbodies*sizeof(planet<type>), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess)
  {
    std::cout << "Failed to copy from device to host memory: " << cudaGetErrorString(error) << std::endl;
  }
  //Free memory on the GPU
  error = cudaFree(devBodies);
  if (error != cudaSuccess)
  {
    std::cout << "Failed to free device memory. \n";
  }
#endif

  timerAdvance.end();
  scale_bodies(nbodies, bodies, RECIP_DT);

  type e2 = energy(nbodies, bodies);

  std::cout << std::setprecision(9);
  std::cout << e1 << '\n' << e2 << '\n';
  timerMain.end();

  if (argc != 1) { delete [] bodies; }
  outStream.close();
  return 0;
}
