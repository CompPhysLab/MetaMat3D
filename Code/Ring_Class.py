# Object class for rings
class Ring():
    def __init__(self, x:float, y:float, z:float, pos:str, radius:float, width:float, L:float, C:float, R:float) -> None:
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

        self.r = radius # Radius of the ring
        self.w = width  # Width of the strip

        self.L = L      # Self-inductance
        self.C = C      # Capacitance
        self.R = R      # Resistance

    def M(self, w:float) -> complex:
        """Function to calculate effective self-inductance

        Parameters
        ----------
        w : float
            Frequency of the signal

        Returns
        -------
        complex
            Effective self-inductance
        """         
        return self.R/1j/w - self.L + 1/(w ** 2 * self.C)

    def Z(self, w:float) -> complex:
        """Function to calculate self-impedance

        Parameters
        ----------
        w : float
            Frequency of the signal

        Returns
        -------
        complex
            Self-impedance
        """        
        return self.R - 1j * w * self.L + 1j/(w * self.C)

    def sigma(self, w:float) -> complex:
        """Function to calculate conductivity

        Parameters
        ----------
        w : float
            Frequency of the signal

        Returns
        -------
        complex
            Conductivity
        """        
        return 1/(self.R - 1j * w * self.L + 1j/(w * self.C))

#   Important to make parameters visible in console

    def __repr__(self) -> str:
        """Representation of the object

        Returns
        -------
        str
            String representation of the object
        """
        return f"x = {self.x} y = {self.y} z = {self.z} orientation: {self.pos} Radius: {self.r}"

    def __str__(self) -> str:
        """String representation of the object

        Returns
        -------
        str
            String representation of the object
        """        
        return f"x = {self.x} y = {self.y} z = {self.z} orientation: {self.pos} Radius: {self.r}"
