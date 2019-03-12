# OxfordNanWorkshop

Code to form the basis for a hands-on CUDA acceleration session.

The exercise is built around a fictional alignment task in which a 2k sequence needs to be aligned with a longer reference.
The sequence is aligned by progressively moving it along the reference and, at each position, calculating a match score using Needleman-Wunsch.

This naive aligner is initally presented as a sequential code and the purpose of the exercise is to analyse the serial performance and identify the hotspots t
hat are most appropriate for parallelisation.

cuda_inline_nw.cu uses a fairly simple Unified Memory implementation to gain a 90x speed up over the serial version.

Further tunings for the exercise (with time) will
 involve using shared memory for each offset alignment.
  It's unlikely that folks will have time, but they could
  also experiment with thread-spawning for as opposed to the inline matrix calculation.
