# MetaMat3D

## OOP structure 

- Metamaterial:
	
	Contains all necessary parameters to determine material properties in accordance with continuous medium theory and also methods for its calculation. As main properties were chosen:

	- **Magnetic permeability** verified only for quasi-static mode
	- **Magnetic polarizability** : applicable only for quasi-static mode
	- **Scattering cross-section** : applicable only with taking into account retardation.

	These parameters were calculated for continuous medium structures with the same properties as efficiently calculated for metamaterial of split-ring resonators. The chosen shape is:

	- Sphere and Cube (3D Structures)
	- Disk and Square (2D Structures)
	- Needle (1D structure)

	Also containing functions for calculating mutual inductance between resonators in two cases - with or not retardation effects.

- Metastructure 

	Discrete finite model of studying metastructures. As long as unit cell size is fixed in metamaterials, contains number of metacell along each axes as for rectangular area. Then it is possible to change shape:

	- Cubic (rectangular) : all area filled with split-rings
	- Spherical (or Elliptic) : create a sphere (or ellipse) inscribed into rectangular area.
	- Cylindrical (x, y, z oriented) : crates a cylinder along chosen axes inscribed into rectangular area. 
	- Custom shape (coming soon) - an a 3D mask describes which ring is should be or not if metastructure (if 1 ring stay same in accordance to rectangular grid, if 0 ring is removed)

	The second property describes unit-cell structure and border condition:

	- Smooth shifted structure - centers of SSR from same unit cell is not at the same place, also center of the unit cell is not a center for any of rings. Structure covered with smooth border of resonators, so only 1 type of rings is on border.
	- Ragged shifted structure - unit cell is same as in smooth, but border condition is different. Last layers for each face is removed, so at the border is two different oriented rings.
	- Centred structure - unit cell constructed with rings at the same centers, but different normal vectors. Also the center of unit cell is same as for each ring, so because of symmetry between each orientation there are no possibility to change border conditions. 
	*Note -rings in same unit cells are not coupled to each other*  

	There also protected attribute `MutualInductance` contained all independent inductances for each orientation, that are necessary to fast-fill matrix for straight solution and to apply operator for iterative solver.

So finally structures looks like this:

- Ring
	- Attributes: 
	 
		- Radius 
		- Width 
		- Self-capacitance (C)
		- Self-inductance (L)
		
	- Methods: 
		- Mutual Inductance function 
		- Self-Frequency
	

- Metamaterial
	
	- Attributes:
		
		- Ring 
		- Mesh step
		- Mesh shifts 
	
	- Methods
		
		- Magnetic Permeability
		- Magnetic Permeability resounance frequency
		- Sphere Magnetic Polarizability 
		- Sphere Magnetic Polarizability resounance frequency
		- Disk Magnetic Polarizability
		- Disk Magnetic Polarizability resounance frequency
		- Needle Magnetic Polarizability
		- Needle Magnetic Polarizability resounance frequency 
		- Sphere Cross section parameters
		- Sphere internal fields 
	
- Metastructure
	- Attributes
		
		- Metamatertials 
		- Size (x, y, z)
		- Anizotropic/izotropic
		- Shape 
		- Rings Centers Array
		- Precalculated MutualInductance Array
	- Methods 
		
		- Calculate Mutual Inductance
		- Calculate EigVals of Extended circulant 

- DirectSolver
	- Attributes
		- Metastructure
		- Matrix
		- Frequency 
		- Currents 
		- Voltages
		- Sources
	- Methods 
		- FillMatrix
		- Solve

- IterativeSolver
	- Attributes 
		- SolverType (lgmres/gmres/bicgstab)
		- Frequencies 
		- PreconditionerType (diagonal/circulant/NearestInteraction)
		- Frequencies
	- Methods
		- Fill Circulant Vectors and eigvalues
		- ApplyOperator
		- Solve(Number of frequencies)

- MetaSolution 
	- Attributes 
		- Frequencies
		- Memory Restriction
		- Currents to save 
		- Other effective params to calculate and save 

	- Methods 
		- CalcPolarizability 
		- CalcCrossSection 
		- CalcField 
		- Save (Npz and scv)
		- Open (Npz and scv)
	

