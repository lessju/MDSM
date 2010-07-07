#ifndef DEDISPERSE_KERNEL
#define DEDISPERSE_KERNEL

// The dedispersion loop
__global__ dedisperse_tree(float** ddata, float** output, int nsamp, int nchans, int shift)
{
   int nstages, ns, nn, mm, j, i;
   float jsum[50];

   // Calculate number of stages
   nstages = (int) (log((float) maxlag) / log(2.0) + 0.5);

/* GO THROUGH LOOP nstage TIMES, ALTERNATING ddata AND output AS
   INPUT AND OUTPUT TO ROUTINE (TO SAVE SPACE)                 */

   for (ns = 1; ns < nstages + 1; ns++) {
      if (ns % 2 == 1)
         tree_stage(ddata, nsamp, nchans, ns, output, shift);
      else
         tree_stage(output, nsamp, nchans, ns, ddata, shift);
   }


/* SINCE ddata SHOULD CONTAIN DEDISPERSED DATA,
   WE NEED TO WRITE output INTO ddata IF nstages IS ODD */

   if (nstages % 2 == 1) {
      for (nn = 0; nn < n; nn++) {
         for (mm = 0; mm < m; mm++)
            ddata[nn][mm] = output[nn][mm];
      }
   }

}

void tree_stage(float **ddata, float **output, int nsamp, int nchans, int ns, int shift)
{
   int k, j, l, lj1, ns;
   int j_index1, j_index2;
   int k_index1, k_index2;
   int ngroupsize, ngroups, ndiff;
   float x;

/* FREQUENCY CHANNELS ARE PROCESSED IN GROUPS WHOSE SIZE 
   IS DETERMINED BY WHICH STAGE IS BEING PROCESSED    */

   ngroupsize = (int) pow(2.0, (float) ns);
   ngroups = nchans / ngroupsize;
   ndiff = ngroupsize / 2;


/* LOOP OVER TIME SAMPLES */

   for (k = 0; k < nsamp; k++) {

      /* LOOP OVER GROUP */

      for (l = 1; l < ngroupsize + 1; l++) {

         k_index1 = k;
         k_index2 = k - l * shift / 2;
         lj1 = l - 1 - (l / 2);

         /* LOOP OVER NUMBER OF GROUPS */

         for (j = 0; j < nchans; j = j + ngroupsize) {

            j_index1 = j + lj1;
            j_index2 = j_index1 + ndiff;

            /* NEED TO GUARD AGAINST GOING OUT OF ARRAY BOUNDARIES
               ON THE TIME SAMPLES (FREQUENCY CHANNELS SHOULD BE OK) */

            x = ddata[k_index1][j_index1];

            if (k_index2 >= 0 && k_index2 < nsamp)
               x = x + ddata[k_index2][j_index2];

            output[k][j + l - 1] = x;

         }
      }
   }
}

#endif
