import numpy as np
import colormaps as cmaps
from plotly import graph_objects as go


# Object class for rings
class Ring:
	def __init__(self, x:float, y:float, z:float, pos:str, radius:float, width:float, R:float, L:float, C:float) -> None:
		"""Constructor for the Ring class

		Parameters
		----------
		x : float
			x position of the ring
		y : float
			y position of the ring
		z : float
			z position of the ring
		pos : str
			direction of the normal vector for the ring - "x" or "y" or "z"
		radius : float
			Radius of the ring
		width : float
			Width of the strip
		L : float
			Self-inductance
		C : float
			Capacitance
		R : float
			Resistance
		"""        
		self.x = x      # x position of the ring
		self.y = y      # y position of the ring
		self.z = z      # z position of the ring

		self.pos = pos  # orientation of the ring - "x" or "y" or "z"

		self.radius = radius # Radius of the ring
		self.width = width  # Width of the strip

		self.L = L      # Self-inductance
		self.C = C      # Capacitance
		self.R = R      # Resistance

	def __repr__(self) -> str:
		"""Representation of the object

		Returns
		-------
		str
			String representation of the object
		"""
		return f"x = {self.x} y = {self.y} z = {self.z} orientation: {self.pos} Radius: {self.radius}"
	
	def __str__(self) -> str:
		"""String representation of the object

		Returns
		-------
		str
			String representation of the object
		"""        
		return f"x = {self.x} y = {self.y} z = {self.z} orientation: {self.pos} Radius: {self.r}"

def visualize_rings(Rings: list,
				  AirRingColor: tuple[float, float, float] = (0.9, 0.9, 0.9),
				  ZRingColor: tuple[float, float, float] = cmaps.voltage(0.1),
				  YRingColor: tuple[float, float, float] = cmaps.voltage(0.7),
				  XRingColor: tuple[float, float, float] = cmaps.voltage(0.4),
				  ) -> go.Figure:	
	"""Function to visualize list of rings in 3D space
	Parameters
	----------
	Rings : list
		list of ring objects to visualize
	AirRingColor : tuple
		color for air rings
	ZRingColor : tuple
		color for Z-oriented rings
	YRingColor : tuple
		color for Y-oriented rings
	XRingColor : tuple
		color for X-oriented rings

	Returns
	-------
	go.Figure
		Plotly figure object containing the 3D visualization
	"""
	fig = go.Figure()
	added_orientations = {'x': False, 'y': False, 'z': False}

	# Generate rings based on metastructure geometry and type
	if len(Rings) == 0:
		raise ValueError("Before visualizing us fill_grid method to obtain rings array")

	for ring in Rings:
		x, y, z = [], [], []
		theta = np.linspace(0, 2*np.pi, 100)
		alpha = 1

		if ring.pos == "x":
			x.extend(np.full((100,), ring.x))
			y.extend(ring.y + ring.radius * np.cos(theta))
			z.extend(ring.z + ring.radius * np.sin(theta))
			color = f'rgb({int(XRingColor[0]*255)}, {int(XRingColor[1]*255)}, {int(XRingColor[2]*255)})'
			name = 'X orientation' if not added_orientations['x'] else ''
			added_orientations['x'] = True

		elif ring.pos == "y":
			x.extend(ring.x + ring.radius * np.cos(theta))
			y.extend(np.full((100,), ring.y))
			z.extend(ring.z + ring.radius * np.sin(theta))
			color = f'rgb({int(YRingColor[0]*255)}, {int(YRingColor[1]*255)}, {int(YRingColor[2]*255)})'
			name = 'Y orientation' if not added_orientations['y'] else ''
			added_orientations['y'] = True

		elif ring.pos == "z":
			x.extend(ring.x + ring.radius * np.cos(theta))
			y.extend(ring.y + ring.radius * np.sin(theta))
			z.extend(np.full((100,), ring.z))
			color = f'rgb({int(ZRingColor[0]*255)}, {int(ZRingColor[1]*255)}, {int(ZRingColor[2]*255)})'
			name = 'Z orientation' if not added_orientations['z'] else ''
			added_orientations['z'] = True

		# Check if ring has high resistance (air ring)
		if ring.R > 1e3:
			color = f'rgb({int(AirRingColor[0]*255)}, {int(AirRingColor[1]*255)}, {int(AirRingColor[2]*255)})'
			alpha = 1.0

		fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines',
									   line=dict(color=color, width=4), name=name,
									   showlegend=bool(name), opacity=alpha))
	return fig

class SRR_metamaterial:
	'''Class to define SRR-based metamaterial parameters
	
	Parameters
	----------
	Resistance : float
		Resistance of single SRR
	SelfInductance : float
		Self-inductance of single SRR
	Capacitance : float
		Capacitance of single SRR
	Radius : float
		Radius of single SRR
	Width : float
		Width of single SRR
	period : tuple
		Period of the metamaterial (dz, dy, dx)
	lattice_sum : float
		Lattice sum for given grid, dependent on grid geomerty (hexagonal, rectangular, etc.)
	alpha_per_cell : float
		magnetic moment per unit cell
	omega_0 : float
		Resonance frequency of single SRR
	'''
	def __init__(self,
			  	 Resistance:float,
			  	 SelfInductance:float,
				 Capacitance:float,
				 Radius:float,
				 period:tuple,
				 width:float = 0
				 ):
		"""Class to define SRR-based metamaterial parameters
	
		Parameters
		----------
		Resistance : float
			Resistance of single SRR
		SelfInductance : float
			Self-inductance of single SRR
		Capacitance : float
			Capacitance of single SRR
		Radius : float
			Radius of single SRR
		"""    
		# Set parameters of the ring
		self.R, self.L, self.C = Resistance, SelfInductance, Capacitance
		self.radius, self.width = Radius, width

		# Initialize grid parameters
		self.dz, self.dy, self.dx = period
		# TODO: add calculation of lattice sum. This one is for rectangular grid
		self.lattice_sum = -0.06
		self.alpha_per_cell = np.pi * self.radius ** 2 /(self.dx * self.dy * self.dz)
		self.omega_0 = 1 / np.sqrt(self.L * self.C)

	def set_period(self, period:tuple) -> None:
		"""Function to set grid parameters for metamaterial
	
		Parameters
		----------
		period : tuple
			Period of the metamaterial (dz, dy, dx)
		"""
 
		# Check if grid parameters are valid
		if not all(isinstance(i, (int, float)) and i > 0 for i in period):
			raise ValueError("Grid parameters must be positive numbers.")
		self.dz, self.dy, self.dx = period
		# TODO: add calculation of lattice sum
		self.lattice_sum = -0.06

	def get_params(self) -> dict:
		"""Function to get metamaterial parameters

		Returns
		-------
		dict
			dictionary with metamaterial parameters
		"""    
		return {
			'R': self.R,
			'L': self.L,
			'C': self.C,
			'radius': self.radius,
			'width': self.width,
			'period': (self.dz, self.dy, self.dx)
		}

	def effective_mu(self, Omega: np.ndarray) -> np.ndarray:
		"""Function to calculate effective magnetic permeability for anisotropic system
		Parameters
		----------
		Omega : np.ndarray
			frequency array
		Returns
		-------
		np.ndarray
			effective mu array
		"""    
		mu_0 = 4 * np.pi * 10 ** -7
		Z = self.R - 1j * Omega * self.L + 1j/(Omega * self.C) - 1j * Omega * mu_0 * self.radius * self.lattice_sum
		Const =  1j * Omega * mu_0 * np.pi ** 2 * self.radius ** 4 /(self.dx * self.dy * self.dz)

		return (Z + 2/3 * Const)/(Z - 1/3 * Const)

	def spherical_polarizability(self, omega: np.ndarray) -> np.ndarray:
		"""Function to calculate polarizability for spherical structures

		Parameters
		----------
		omega : np.ndarray
			frequency array
		Returns
		-------
		np.ndarray
			polarizability for spherical structures
		"""    
		mu = self.effective_mu(omega)
		return 3 * (mu - 1) / (mu + 2)

	def disk_polarizability(self, omega: np.ndarray) -> np.ndarray:
		"""Function to calculate polarizability for disk structures

		Parameters
		----------
		omega : np.ndarray
			frequency array
		Returns
		-------
		np.ndarray
			polarizability for disk structures
		"""    
		mu = self.effective_mu(omega)
		return 1 - 1 / mu

	def needle_polarizability(self, omega: np.ndarray) -> np.ndarray:
		"""Function to calculate polarizability for needle structures

		Parameters
		----------
		omega : np.ndarray
			frequency array
		Returns
		-------
		np.ndarray
			polarizability for needle structures
		"""    
		mu = self.effective_mu(omega)
		return mu - 1