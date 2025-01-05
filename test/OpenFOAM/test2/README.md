# Test 2 - openFOAM implementation
  1. Load the openFOAM environment
  2. `blockMesh`
  3. use the icoFoam solver (on my machine works `$FOAM_APPBIN/icoFoam`)
  4. examine the results with `paraFoam -builtin`

The domain has been set with h=1, the number of points for each dimension can be edited at 
  ./system/blockMeshDict

The solver parameters can be edited at
  ./system/fvSolution

The number of step and step size can be edited at
  ./system/controlDict


## ParaFoam
In paraFoam open the whole folder (`paraFoam -builtin` in this directory), apply the filter and select the U.
I suggest to watch a slice and not the wall, with Z normal, to have similar results of `race_rules.pdf`.
