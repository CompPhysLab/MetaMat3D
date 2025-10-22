from Metamaterial import Ring, SRR_metamaterial, visualize_rings
import matplotlib.animation as anime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import colormaps as cmaps 
import numpy as np
from tqdm import tqdm
import os

class Metastructure:
	"""Class to define finite metastructure parameters
	
	Class Attributes
	----------------
	metamaterial : Metamaterial
		Metamaterial object defining the material properties
	shape : tuple[int, int, int]
		Shape of the metastructure (z, y, x)
	geometry : str
		Geometry of the metastructure (supported: 'rectangle', 'ellipse', 'cylinder-z', 'cylinder-y', 'cylinder-x')
	type : str
		Type of the metastructure (supported: 'smooth', 'ragged', 'centred', 'anizotropic')
	orientations : list[str]
		List of orientations for the rings in the metastructure (supported: 'z', 'y', 'x')
	rings : list[Ring]
		List of Ring objects in the metastructure (optional, only when metastructure is not provided)
	numbers : dict[str, int]
		Dictionary with number of rings for each orientation
		"""

	def __init__(self, metamaterial: SRR_metamaterial,
				 shape: tuple[int, int, int] = (2, 2, 2),
				 geometry: str = 'rectangle',
				 Type: str = 'smooth',
				 orientations = ['z', 'y', 'x'],
				 full_grid = False
				 ):
		"""Constructor to define finite metastructure parameters
		Parameters
		----------
		metamaterial :
			Metamaterial object defining the material properties
		shape : tuple[int, int, int]
			Shape of the metastructure (z, y, x)
		geometry : str
			Geometry of the metastructure (supported: 'rectangle', 'ellipse', 'cylinder-z', 'cylinder-y', 'cylinder-x')
		type : str
			Type of the metastructure (supported: 'smooth', 'ragged', 'centred', 'anizotropic'),
		orientations : list
			List of orientations for the rings in the metastructure (supported: 'z', 'y', 'x')
		"""  
		self.metamaterial = metamaterial
		self.orientations = orientations

		if geometry not in ['rectangle', 'ellipse', 'cylinder-z', 'cylinder-y', 'cylinder-x']:
			raise ValueError("Unsupported geometry type. Supported geometries are: 'rectangle', 'ellipse', 'cylinder-z', 'cylinder-y', 'cylinder-x'.")
		self.geometry = geometry
		self.rings = []
		self.numbers = {}

		if Type not in ['smooth', 'ragged', 'centred', 'anizotropic']:
			raise ValueError("Unsupported metastructure type. Supported types are: 'smooth', 'ragged', 'centred', 'anizotropic'")
		self.type = Type
		self.set_shape(shape)
		self.full_grid = full_grid

	@property
	def name(self):
		"""Function to get unique name based on metastructure parameters"""
		return f'{self.type}_{self.geometry}_{self.shape[0]}x{self.shape[1]}x{self.shape[2]}'

	def set_type(self, Type:str): 
		"""Function to set the type of the metastructure
		Parameters
		----------
		Type : str
			Type of the metastructure (supported: 'smooth', 'ragged', 'centred', 'anizotropic')
		"""
		if Type not in ['smooth', 'ragged', 'centred', 'anizotropic']:
			raise ValueError("Unsupported metastructure type. Supported types are: 'smooth', 'ragged', 'centred', 'anizotropic'")
		self.type = Type
		self.set_shape(self.shape)
		
	def set_shape(self, shape: tuple[int, int, int]) -> None:
		"""Function to set the shape of the metastructure
		Parameters
		----------
		shape : tuple[int, int, int]
			Shape of the metastructure (z, y, x)
		"""
		if not (isinstance(shape, tuple) and len(shape) == 3 and all(isinstance(i, int) and i > 0 for i in shape)):
			raise ValueError("Shape must be a tuple of three positive integers.")
		self.shape = shape
		self.Nz, self.Ny, self.Nx = shape
		self.dz, self.dy, self.dx = self.metamaterial.dz, self.metamaterial.dy, self.metamaterial.dx
		self.R_z, self.R_y, self.R_x = self.Nz * self.dz/2, self.Ny * self.dy/2, self.Nx * self.dx/2
		self.N, self.r0 = {}, {}
		if not all(orientation in ['z', 'y', 'x'] for orientation in self.orientations):
			raise ValueError("orientation must be a symbol 'z', 'y' or 'x'")
		if len(set(self.orientations)) < len(self.orientations):
			raise ValueError("Some orientation repeated many times")
		if self.type == 'anizotropic' and len(self.orientations) > 1:
			raise ValueError("anizotropic case suppors only one orientation")
		# Setting number of rings for each orientation along each axis
		if self.type == 'smooth':
			for orientation in self.orientations:
				self.N[orientation] = {
					'nz': self.Nz + (orientation == 'z'),
					'ny': self.Ny + (orientation == 'y'),
					'nx': self.Nx + (orientation == 'x')
				}
				self.r0[orientation] = {
					'nz': self.metamaterial.dz/2 * (1 - (orientation == 'z')),
					'ny': self.metamaterial.dy/2 * (1 - (orientation == 'y')),
					'nx': self.metamaterial.dx/2 * (1 - (orientation == 'x'))
				}
		elif self.type == 'ragged':
			for orientation in self.orientations:
				self.N[orientation] = {
					'nz': self.Nz - (orientation == 'z'),
					'ny': self.Ny - (orientation == 'y'),
					'nx': self.Nx - (orientation == 'x')
				}
				self.r0[orientation] = {
					'nz': self.metamaterial.dz/2 * (1 + (orientation == 'z')),
					'ny': self.metamaterial.dy/2 * (1 + (orientation == 'y')),
					'nx': self.metamaterial.dx/2 * (1 + (orientation == 'x'))
				}
		elif self.type == 'centred':
			for orientation in self.orientations:
				self.N[orientation] = {
					'nz': self.Nz, 
					'ny': self.Ny,
					'nx': self.Nx
				}
				self.r0[orientation] = {
					'nz': self.metamaterial.dz/2,
					'ny': self.metamaterial.dy/2,
					'nx': self.metamaterial.dx/2
				}
		elif self.type == 'anizotropic':
			for orientation in self.orientations:
				self.N[orientation] = {
					'nz': self.Nz, 
					'ny': self.Ny,
					'nx': self.Nx
				}
				self.r0[orientation] = {
					'nz': self.metamaterial.dz/2 * (1 - (orientation == 'z')),
					'ny': self.metamaterial.dy/2 * (1 - (orientation == 'y')),
					'nx': self.metamaterial.dx/2 * (1 - (orientation == 'x'))
				}
				if orientation == 'z': self.R_z -= self.dz/2
				if orientation == 'y': self.R_y -= self.dy/2
				if orientation == 'x': self.R_x -= self.dx/2

		self.set_geometry(self.geometry)

	def set_geometry(self, geometry):
		"""Function to set the geometry of the metastructure
		Parameters
		----------
		geometry : str
			Geometry type to set ('rectangle', 'ellipse', 'cylinder-z', 'cylinder-y', 'cylinder-x')
		"""
		if geometry not in ['rectangle', 'ellipse', 'cylinder-z', 'cylinder-y', 'cylinder-x']:
			raise ValueError("Unsupported geometry type. Supported geometries are: 'rectangle', 'ellipse', 'cylinder-z', 'cylinder-y', 'cylinder-x'.")
		self.geometry = geometry
		self.Rz, self.Ry, self.Rx = self.R_z, self.R_y, self.R_x
		if self.geometry == 'rectangle':
			self.Rz, self.Ry, self.Rx = np.inf, np.inf, np.inf
		elif self.geometry in ['cylinder-z', 'cylinder-y', 'cylinder-x']:
			if self.geometry[-1] == 'z': self.Rz = np.inf
			if self.geometry[-1] == 'y': self.Ry = np.inf
			if self.geometry[-1] == 'x': self.Rx = np.inf
		if self.type == 'anizotropic':
			if self.orientations[0] == 'z': self.Rz += self.dz/2
			if self.orientations[0] == 'y': self.Ry += self.dy/2
			if self.orientations[0] == 'x': self.Rx += self.dx/2
		self.rings = []
		self.numbers = {}

	def fill_grid(self) -> None:
		"""
		Function to fill the metastructure grid based on geometry and type
		"""
		dz, dy, dx = self.dz, self.dy, self.dx
		
		for orientation in self.orientations:
			self.numbers[orientation] = 0
			for k in range(self.N[orientation]['nz']):
				for j in range(self.N[orientation]['ny']): 
					for i in range(self.N[orientation]['nx']):
						self.rings.append(
							Ring(
								 i * dx  + self.r0[orientation]['nx'],
								 j * dy  + self.r0[orientation]['ny'],
								 k * dz  + self.r0[orientation]['nz'],
								orientation,
								self.metamaterial.radius, self.metamaterial.width,
								self.metamaterial.R, self.metamaterial.L,
								self.metamaterial.C)
						)
						self.numbers[orientation] += 1
		if self.geometry != 'rectangle':
			for orientation in self.orientations:
				for ring in self.rings[:]:
					# rings positions 
					r_x, r_y, r_z = ring.x, ring.y, ring.z
					distance = (r_x - self.R_x) ** 2/self.Rx**2 + \
							   (r_y - self.R_y) ** 2/self.Ry**2 + \
							   (r_z - self.R_z) ** 2/self.Rz**2
					if distance > 1.00:
						if self.full_grid:
							ring.R = 1e200
							ring.C = 1e200
							ring.L = 1e200
						else:
							self.rings.remove(ring)
						self.numbers[ring.pos] -= 1
		
	def visualize(self, 
				  AirRingColor: tuple[float, float, float] = (0.9, 0.9, 0.9),
				  ZRingColor: tuple[float, float, float] = cmaps.voltage(0.1),
				  YRingColor: tuple[float, float, float] = cmaps.voltage(0.7),
				  XRingColor: tuple[float, float, float] = cmaps.voltage(0.4),
				  ) -> go.Figure:	
		"""Function to visualize the metastructure with specified ring colors
		Parameters
		----------
		AirRingColor : tuple[float, float, float]
			Color of the air ring (R, G, B) - values between 0 and 1
		ZRingColor : tuple[float, float, float]
			Color of the Z ring (R, G, B) - values between 0 and 1
		YRingColor : tuple[float, float, float]
			Color of the Y ring (R, G, B) - values between 0 and 1
		XRingColor : tuple[float, float, float]
			Color of the X ring (R, G, B) - values between 0 and 1

		Returns
		-------
		go.Figure
			Plotly figure with 3D visualization of the metastructure
		"""
		fig = visualize_rings(self.rings,
									AirRingColor,
									ZRingColor,
									YRingColor,
									XRingColor)
		fig.update_layout(
			title=f'{self.geometry} {self.type} metastructure {self.shape}',
			scene=dict(
				xaxis_title='X',
				yaxis_title='Y',
				zaxis_title='Z',
				aspectmode='cube'
			)
		)

		return fig