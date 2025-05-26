# quantum_ml/models.py
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit.circuit.library import PauliFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from sklearn.svm import SVC

def make_vqc(reps: int, paulis: list, feature_dim: int = 2, seed=42, maxiter=200):
    algorithm_globals.random_seed = seed
    feature_map = PauliFeatureMap(feature_dimension=feature_dim, reps=reps, paulis=paulis)
    ansatz = RealAmplitudes(num_qubits=feature_dim, reps=reps)
    backend = AerSimulator()
    sampler = AerSampler()
    optimizer = COBYLA(maxiter=maxiter)
    return VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler
    )

def make_svm(seed=42):
    """Return a fresh classical RBF‐SVM."""
    return SVC(kernel='rbf', probability=True, random_state=seed)
