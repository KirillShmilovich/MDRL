from simtk import unit as u
import simtk.openmm as mm
from openmmtools.testsystems import LennardJonesFluid
import numpy as np
from sys import stdout

from featurizers import graphlet_featurizer
from utils import get_n_clusters


class LJSimualtion:
    def __init__(
            self,
            n_particles=100,
            pressure=80 * u.atmosphere,
            temperature=120 * u.kelvin,
            timestep=2.5 * u.femtoseconds,
            collision_rate=5.0 / u.picoseconds,
            sigma=3.4 * u.angstrom,
            epsilon=0.238 * u.kilocalories_per_mole,
            report=False,
    ):
        self.n_particles = n_particles
        self.pressure = pressure
        self.temperature = temperature
        self.collision_rate = collision_rate
        self.timestep = timestep
        self.sigma = sigma
        self.epsilon = epsilon

        self._setup(report)

    def sim(self, n_steps):
        self.simulation.step(n_steps)

    def changeTemp(self, temp):
        self.temperature = temp
        self.simulation.integrator.setTemperature(self.temperature)

    @property
    def xyz(self):
        return (self.simulation.context.getState(
            getPositions=True,
            enforcePeriodicBox=True).getPositions(asNumpy=True)._value)  # nm

    @property
    def dims(self):
        return np.diag(
            self.simulation.context.getState(
                getPositions=True,
                enforcePeriodicBox=True).getPeriodicBoxVectors(
                    asNumpy=True)._value)  # nm

    def graphlet_features(self, r_cut=None):
        if r_cut is None:
            r_cut = 1.15 * self.sigma.in_units_of(u.nanometers)._value
        return graphlet_featurizer(self.xyz, self.dims, r_cut=r_cut)

    def n_clusters(self, r_cut=None):
        if r_cut is None:
            r_cut = 1.15 * self.sigma.in_units_of(u.nanometers)._value
        return get_n_clusters(self.xyz, self.dims, r_cut=r_cut)

    def _get_platform(self):
        platform = mm.Platform.getPlatformByName("OpenCL")
        properties = {"DeviceIndex": "0", "OpenCLPrecision": "mixed"}
        return platform, properties

    def _setup(self, report=False):
        self.fluid = LennardJonesFluid(nparticles=self.n_particles,
                                       sigma=self.sigma,
                                       epsilon=self.epsilon)
        self.system = self.fluid.system
        self.integrator = mm.LangevinIntegrator(self.temperature,
                                                self.collision_rate,
                                                self.timestep)
        platform, properties = self._get_platform()
        self.simulation = mm.app.Simulation(self.fluid.topology, self.system,
                                            self.integrator, platform,
                                            properties)
        self.simulation.context.setPositions(self.fluid.positions)

        self.simulation.minimizeEnergy()
        self.simulation.context.setVelocitiesToTemperature(self.temperature)
        if report:
            self.simulation.reporters.append(
                mm.app.StateDataReporter(stdout,
                                         100,
                                         step=True,
                                         temperature=True,
                                         separator=" | "))
