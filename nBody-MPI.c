#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define BODIES 5000
#define TIMESTEPS 100

#define GRAVCONST 0.0000001

// global vars
float mass[BODIES];					//mass of bodies
float vx[BODIES], vy[BODIES];		//velocity in x-axis position, y-axis position of bodies
float x[BODIES], y[BODIES];			//x-axis position, y-axis position of body
float dx, dy, d, F, ax, ay;			//distance in x-axis position, y-axis position, d distance, Force, accelleration in x-axis, velocity in y-axis of a body

// vars for MPI
int numOfRanks, rankID; 	//number of ranks (processes), and rank ID
double t1,t2,t3,t4,t5,t6;			//for timing		
int *numOfBODIESperRank, *offsets;	//array of size of BODIES, and for index/displacements

float *scatteredMass;
float *scatteredVx;
float *scatteredVy;
float *scatteredX;
float *scatteredY;


void randomInit();			//initialisation of mass, x-axis position, y-axis position, velocity in x-axis, velocity in y-axis
void outputBody(int);		//output the result of the latest x-axis position, y-axis position, velocity in x-axis, velocity in y-axis of a body 
int main(void) {
  int time, i, j;									//timestep, i-body, j-body

  MPI_Init(NULL, NULL);								//initialise MPI parallel environment
  MPI_Comm_size(MPI_COMM_WORLD, &numOfRanks);		//set the number of processes in the communicator
  MPI_Comm_rank(MPI_COMM_WORLD, &rankID);			//sets the rankID of the process in the communicator MPI_COMM_WORLD
  
   t1=MPI_Wtime(); 									//start timing of the WHOLE CODE


offsets=(int *) malloc(sizeof(int)*numOfRanks);
numOfBODIESperRank=(int *) malloc(sizeof(int)*numOfRanks);

 if(rankID==0) {

		int remainder = BODIES % numOfRanks;					//calculate the remainder of indivisible amount of the total number of BODIES divided by number of ranks
			int sum = 0;											//accumulation of each of number of BODIES per rank 
			for(i = 0; i < numOfRanks; i++) {						
				numOfBODIESperRank[i] = BODIES / numOfRanks;		//split the amount of BODIES per rank
				if (remainder > 0) {								//if not divisible by number of rank, the remainder will distributed into ranks, start from rankID=0 (root), the next remainder will be given to next rank, etc
					numOfBODIESperRank[i] += 1;
					remainder--;
				}
				offsets[i] = sum;									//displacement/offset/ start of the index to be distributed 
				sum += numOfBODIESperRank[i];
			}
}

MPI_Bcast(numOfBODIESperRank,numOfRanks,MPI_FLOAT,0,MPI_COMM_WORLD); //broadcast all length and index of the portions of the bodies for each processes
MPI_Bcast(offsets,numOfRanks,MPI_FLOAT,0,MPI_COMM_WORLD);  				//broadcast all length and index of the portions of the bodies for each processes


scatteredMass=(float *) malloc(sizeof(float)*numOfBODIESperRank[rankID]); 
 scatteredVx=(float *) malloc(sizeof(float)*numOfBODIESperRank[rankID]); 
 scatteredVy=(float *) malloc(sizeof(float)*numOfBODIESperRank[rankID]); 
 scatteredX=(float *) malloc(sizeof(float)*numOfBODIESperRank[rankID]); 
 scatteredY=(float *) malloc(sizeof(float)*numOfBODIESperRank[rankID]);

MPI_Scatterv(mass,numOfBODIESperRank,offsets, MPI_FLOAT,scatteredMass,numOfBODIESperRank[rankID],MPI_FLOAT,0,MPI_COMM_WORLD);
MPI_Scatterv(vx,numOfBODIESperRank,offsets, MPI_FLOAT,scatteredVx,numOfBODIESperRank[rankID],MPI_FLOAT,0,MPI_COMM_WORLD);
MPI_Scatterv(vy,numOfBODIESperRank,offsets, MPI_FLOAT,scatteredVy,numOfBODIESperRank[rankID],MPI_FLOAT,0,MPI_COMM_WORLD);
MPI_Scatterv(x,numOfBODIESperRank,offsets, MPI_FLOAT,scatteredX,numOfBODIESperRank[rankID],MPI_FLOAT,0,MPI_COMM_WORLD);
MPI_Scatterv(y,numOfBODIESperRank,offsets, MPI_FLOAT,scatteredY,numOfBODIESperRank[rankID],MPI_FLOAT,0,MPI_COMM_WORLD);

t5=MPI_Wtime(); //start of the timing of PARALLEL INITIALISATION
randomInit();
t6=MPI_Wtime(); //end of the timing of PARALLEL INITIALISATION

MPI_Allgatherv(scatteredMass, numOfBODIESperRank[rankID], MPI_FLOAT, &mass, numOfBODIESperRank, offsets, MPI_FLOAT, MPI_COMM_WORLD);
MPI_Allgatherv(scatteredX, numOfBODIESperRank[rankID], MPI_FLOAT, &x, numOfBODIESperRank, offsets, MPI_FLOAT, MPI_COMM_WORLD);
MPI_Allgatherv(scatteredY, numOfBODIESperRank[rankID], MPI_FLOAT, &y, numOfBODIESperRank, offsets, MPI_FLOAT, MPI_COMM_WORLD);
MPI_Allgatherv(scatteredVx, numOfBODIESperRank[rankID], MPI_FLOAT, &vx, numOfBODIESperRank, offsets, MPI_FLOAT, MPI_COMM_WORLD);
MPI_Allgatherv(scatteredVy, numOfBODIESperRank[rankID], MPI_FLOAT, &vy, numOfBODIESperRank, offsets, MPI_FLOAT, MPI_COMM_WORLD);

t2=MPI_Wtime(); //start of the timing of LOOP ONLY
  for (time=0; time<TIMESTEPS; time++) {
    printf("Timestep %d\n",time);
    for (i=offsets[rankID];i<offsets[rankID]+numOfBODIESperRank[rankID];i++) { //each process calculate its own portion
		//printf("rank=%d offset=%d num=%d i=%d\n",rankID,offsets[rankID],numOfBODIESperRank[rankID],i);
      // calc forces on body i due to bodies (j != i)
	      for (j=0; j<BODIES; j++) {
		if (j != i) {
			//calculate the x-axis & y-axis distances (r) body i due to bodies (j != i)
		  dx = x[j] - x[i];
		  dy = y[j] - y[i];
		  //r = dx^2 + dy^2
		  float temp=dx*dx + dy*dy;	
		  //catch if the distance is too narrow/small		
		  d = temp>0.01 ? temp : 0.01; 
		   //calculate the Force by applying Newton's Law
		  F = GRAVCONST * mass[i] * mass[j] / (d*d);
		  //calculate the acceleration of body i in x-axis and y-axis
		  ax = (F/mass[i]) * dx/d;
		  ay = (F/mass[i]) * dy/d;
		  //the integral of acceleration (over time) of i body, become its velocity. We calculate velocity in x-axis and y-axis.
		  vx[i] += ax;
		  vy[i] += ay;
		}
	      } // body j
    } // body i

//scatter the position and velocity to be calculated locall
MPI_Scatterv(vx,numOfBODIESperRank,offsets, MPI_FLOAT,scatteredVx,numOfBODIESperRank[rankID],MPI_FLOAT,0,MPI_COMM_WORLD);
MPI_Scatterv(vy,numOfBODIESperRank,offsets, MPI_FLOAT,scatteredVy,numOfBODIESperRank[rankID],MPI_FLOAT,0,MPI_COMM_WORLD);
MPI_Scatterv(x,numOfBODIESperRank,offsets, MPI_FLOAT,scatteredX,numOfBODIESperRank[rankID],MPI_FLOAT,0,MPI_COMM_WORLD);
MPI_Scatterv(y,numOfBODIESperRank,offsets, MPI_FLOAT,scatteredY,numOfBODIESperRank[rankID],MPI_FLOAT,0,MPI_COMM_WORLD);

    // having worked out all velocities we now apply and determine new position
    for (i=0;i<numOfBODIESperRank[rankID];i++) {
      scatteredX[i] += scatteredVx[i];
      scatteredY[i] += scatteredVy[i];
      //DEBUG ONLY: outputBody(i);
    }
    
    //gather and send the new position of each body to all other ranks
    MPI_Allgatherv(scatteredX, numOfBODIESperRank[rankID], MPI_FLOAT, &x, numOfBODIESperRank, offsets, MPI_FLOAT, MPI_COMM_WORLD);
	MPI_Allgatherv(scatteredY, numOfBODIESperRank[rankID], MPI_FLOAT, &y, numOfBODIESperRank, offsets, MPI_FLOAT, MPI_COMM_WORLD);

    printf("---\n");
  } // time
t3=MPI_Wtime(); //end of the timing of LOOP ONLY

//rankID==0 output the results and timings
if (rankID==0){
  printf("Final data\n");
  for (i=0; i<BODIES; i++) {
    outputBody(i);
  }
t4=MPI_Wtime(); 
printf("time elapsed (WHOLE CODE):%f seconds\n",t4-t1); //t WHOLE CODE
printf("time elapsed (LOOP ONLY):%f seconds\n",t3-t2); //t (LOOP ONLY) PARALLEL
printf("time elapsed (PARALLEL INIT):%f seconds\n",t6-t5); //t PARALLEL INIT
}

//free all of the dynamic arrays
free(numOfBODIESperRank);
free(offsets);
free(scatteredMass);
free(scatteredX);
free(scatteredY);
free(scatteredVx);
free(scatteredVy);

MPI_Finalize();		//end of the MPI
}


void randomInit() {
  int i;
  srand(rankID+1);
   for (i=0; i<numOfBODIESperRank[rankID]; i++) {
    scatteredMass[i] = 0.001 + (float)rand()/(float)RAND_MAX;            // 0.001 to 1.001

    scatteredX[i] = -250.0 + 500.0*(float)rand()/(float)RAND_MAX;   //  -10 to +10 per axis
    scatteredY[i] = -250.0 + 500.0*(float)rand()/(float)RAND_MAX;   //

    scatteredVx[i] = -0.2 + 0.4*(float)rand()/(float)RAND_MAX;   // -0.25 to +0.25 per axis
    scatteredVy[i] = -0.2 + 0.4*(float)rand()/(float)RAND_MAX;  
    
  }
  printf("Randomly initialised\n");
  return;
}

void outputBody(int i) {
printf("Body %d: Position=(%f,%f) Velocity=(%f,%f)\n", i, x[i],y[i], vx[i],vy[i]);
  return;
}
