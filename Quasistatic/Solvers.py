# Calculating currents in each ring for anizotropic system on straight way
import os 
import json
import numpy as np
from scipy.fft import fftn, ifftn
from scipy.linalg import solve
from RingsInductance import Mnm, MutualInductance
from scipy.sparse.linalg import lgmres, gmres, LinearOperator
from tqdm import tqdm

class MetaSolution:
	def __init__(self, Omega = np.array([]), params = None):
		self.currents = []
		self.polarizability = {}
		self.Omega = Omega
		if params is None:
			self.name = ''
		else: 
			self.params = params
			self.name = params['metastructure'] if 'metastructure' in params else 'OnlyRingsSolution'


	def save_sol(self, folder):
		# print(f'Saving data to {filename}/{name}')
		# If folder does not exist, create it
		os.makedirs(f'./{folder}/{self.name}', exist_ok=True)
		with open(f'./{folder}/{self.name}/Params.json', 'w') as f:
			json.dump(self.params, f, indent=4)
		np.savez_compressed(f'./{folder}/{self.name}/currents.npz', currents = self.currents)
		np.savez_compressed(f'./{folder}/{self.name}/polarizability.npz', polarizability = self.polarizability)

	def open_sol(self, folder, struct, load_currents = False):
		# print(f'Opening data from {filename}/{name}')
		with open(f'{folder}/{struct.name}/Params.json', 'r') as f:
			self.params = json.load(f)
			self.Omega = np.array(self.params['Omega'])
	
		if load_currents == True:
			with np.load(f'{folder}/{struct.name}/currents.npz', allow_pickle=True) as f:
				self.currents = f['currents']
			with np.load(f'{folder}/{struct.name}/polarizability.npz', allow_pickle=True) as f:
				self.polarizability = f['polarizability'].item()
		else:
			with np.load(f'{folder}/{struct.name}/polarizability.npz', allow_pickle=True) as f:
				self.polarizability = f['polarizability'].item()
		struct.currents = self.currents
		struct.polarizability = self.polarizability
		struct.omega = self.Omega
		

	def getSlice(self):
		pass

class MetaSolver:
	def __init__(self, 
			  	 Rings: list = None,
			  	 metastructure = None,
			  	 Omega:np.ndarray = None,
				 solve_for:str = 'currents',
				 print_logs:bool = True):
		self.Omega = Omega
		self.solver_for = solve_for
		if Rings is None and metastructure is None:
			raise ValueError("Either Rings array or metastructure must be provided.")
		if Rings is not None:
			self.params = {'Omega': Omega.tolist(),
						   'number_of_rings': len(Rings),
						   'solve_for': solve_for, 
						   'only_rings': True}
			self.rings = Rings
			self.only_rings = True
			self.InductanceData = {}
			self.R = np.array([ring.R for ring in Rings])
			self.L = np.array([ring.L for ring in Rings])
			self.C = np.array([ring.C for ring in Rings])
			self.solve_for = 'currents'
			self.print_logs = print_logs
			self.number = len(Rings)
			self.number_z = sum([1 for ring in Rings if ring.pos == 'z'])
			self.number_y = sum([1 for ring in Rings if ring.pos == 'y'])
			self.number_x = sum([1 for ring in Rings if ring.pos == 'x'])
			return 
		elif metastructure is not None:
			self.params = {'Omega': Omega.tolist(),
						   'metastructure': metastructure.name, 
						   'solve_for': solve_for,
						   'only_rings': False, 
						   'metamaterial': metastructure.metamaterial.get_params(),
						   'numbers': metastructure.numbers}
			self.struct = metastructure
			if len(self.struct.rings) == 0:
				self.struct.fill_grid()
			self.rings = self.struct.rings
			self.only_rings = False
			self.InductanceData = {}
			self.R = np.array([ring.R for ring in self.rings])
			self.L = np.array([ring.L for ring in self.rings])
			self.C = np.array([ring.C for ring in self.rings])
			self.solve_for = 'currents'
			self.print_logs = print_logs
			self.number = len(self.rings)
			self.number_z = self.struct.numbers['z']
			self.number_y = self.struct.numbers['y']
			self.number_x = self.struct.numbers['x']
			return
		else:
			raise NotImplementedError("Iterative solver for structures without created rings array are not implemented yet.")

	def direct_solve(self, Voltages:np.ndarray) -> MetaSolution:
		CURRENTS, Data = [], MetaSolution(self.Omega, self.params)
		self.calcRingsInductance()
		# Calculating flux for uniform field distribution 
		if self.solve_for == 'currents':
			if self.print_logs: print('Direct solving (Currents)')
			for omega in tqdm(self.Omega):
				# Solve equation (diag(Z_0)/jw + M)I = Phi_0z
				I = solve(np.diag(self.M_diag(omega)) + self.InductanceMatrix,
			  			  Voltages/(1j*omega))
				CURRENTS.append(I)
			if self.print_logs: print('Direct solving (Currents): Done')
		elif self.solve_for == 'voltage':
			if self.print_logs: print('Direct solving (Voltage)')
			# Solve equation 
			for omega in tqdm(self.Omega):
				M_diag = self.M_diag(omega)
				# Solve equation (1 + M/M_diag)I = Phi_0z/M_diag
				I = solve(np.eye(self.Number) + np.diag(1/M_diag)@self.InductanceMatrix, Voltages/(1j*omega*M_diag))
				CURRENTS.append(I)
			if self.print_logs: print('Direct solving (Voltage): Done')

		Data.currents = np.array(CURRENTS)
		return Data

	def direct_polarizability(self, axes:str, save_currents:bool = False) -> MetaSolution:
		'''
		Function to calculate direct polarizability for given axes
		Parameters
		----------
		axes : str
			axes of excitation ('x', 'y', 'z'). Voltage would be calculated as flux through rings for 1 A/m uniform field
		save_currents : bool, optional
			Flag to save currents distribution in each ring, by default False
		Returns
		-------
		MetaSolution
			Solution object with polarizability and currents (if save_currents=True)
		'''
		Alpha, CURRENTS = {pos: [] for pos in self.struct.orientations}, []
		Data = MetaSolution(self.Omega, self.params)
		self.calcRingsInductance()
		# Calculating flux for uniform field distribution 
		phi = np.zeros(self.number)
		for i, ring in enumerate(self.rings):
			phi[i] = np.pi * ring.radius ** 2 * 4 * np.pi * 1e-7 * (ring.pos in axes)
		if self.solve_for == 'currents':
			if self.print_logs: print('Direct solving (Currents)')
			for omega in tqdm(self.Omega):
				# Solve equation (diag(Z_0)/jw + M)I = Phi_0z
				I = solve(np.diag(self.M_diag(omega)) + self.InductanceMatrix, phi)
				if save_currents: CURRENTS.append(I)
				start = 0
				for pos in self.struct.orientations:
					end = start + self.struct.numbers[pos]
					Alpha[pos].append(np.sum(I[start:end])/self.struct.numbers[pos])
					start = end

			for pos in self.struct.orientations: 
				Alpha[pos] = np.array(Alpha[pos]) * self.struct.metamaterial.alpha_per_cell  
			if self.print_logs: print('Direct solving (Currents): Done')
		elif self.solve_for == 'voltage':
			if self.print_logs: print('Direct solving (Voltage)')
			# Solve equation 
			for omega in tqdm(self.Omega):
				M_diag = self.M_diag(omega)
				# Solve equation (1/jw + M/M_diag)I = Phi_0z/M_diag
				I = solve(np.eye(self.Number) + np.diag(1/M_diag)@self.InductanceMatrix, phi/M_diag)
				if save_currents: CURRENTS.append(I)
				start = 0
				for pos in self.struct.orientations:
					end = start + self.struct.numbers[pos]
					Alpha[pos].append(np.sum(I[start:end])/self.struct.numbers[pos])
					start = end
			for pos in self.struct.orientations: 
				Alpha[pos] = np.array(Alpha[pos]) * self.struct.metamaterial.alpha_per_cell     
			print('Direct solving (Voltage): Done')

		Data.currents = CURRENTS
		Data.polarizability = Alpha
		return Data
	
	def Z_diag(self, omega):
		''' 
		Function to calculate diagonal matrix elements for Metastructure
		'''
		return self.R + 1j*omega*self.L + 1/(1j*omega*self.C)

	def M_diag(self, omega):
		''' 
		Function to calculate diagonal matrix elements for Metastructure divided by j*omega
		'''
		return self.R/(1j*omega) + self.L - 1/(omega**2*self.C)

	def calcRingsInductance(self):
		'''
		Function to calculate Inductance Matrix for all rings in the structure
		'''
		# TODO: Save Inductance data to file and load if exists
		self.InductanceMatrix = np.zeros((self.number, self.number), dtype = complex)
		if self.print_logs : print('Calculating Inductance Matrix:')
		for n in tqdm(range(len(self.rings))):
			for m in range(n, len(self.rings)):
				self.InductanceMatrix[n][m] = Mnm(self.rings[n], self.rings[m],
									  self.InductanceData)
				self.InductanceMatrix[m][n] = self.InductanceMatrix[n][m]