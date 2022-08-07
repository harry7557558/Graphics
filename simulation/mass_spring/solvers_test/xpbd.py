from integrators import *
import random


class XPBD(Base):
    """
    XPBD: Position-Based Simulation of Compliant Constrained Dynamics
    https://matthias-research.github.io/pages/publications/XPBD.pdf
    """

    def __init__(self, state: State):
        self.N_ITERS = 1  # requires more for stiffer springs
        # super().__init__(state, 1, 0)
        super().__init__(state, min(self.N_ITERS, 4), 0)

    def update(self, dt: float):
        state = self.current_state
        masses, springs = state.masses, state.springs
        x0 = [deepcopy(mass.x) for mass in masses]

        # viscous drag (implicit)
        state.calc_acceleration(ext_only=True)
        for mass in masses:
            k = mass.drag
            mass.a -= k * mass.v / (1.0 + k*dt)

        # get initial state
        x = self.state_to_vector_x(state)
        v = self.state_to_vector_v(state)
        a = self.state_to_vector_a(state)

        # predicted positions and Lagrange multipliers
        x_pred = x + v * dt + a * dt**2
        self.vector_to_state_x(x_pred, state)
        l0 = np.zeros(len(springs))

        # iteratively solve for spring constraints (Gauss-Seidel)
        for i in range(self.N_ITERS):
            indices = list(range(len(springs)))
            random.shuffle(indices)
            for j in indices:
                spring = springs[j]
                mi1, mi2 = spring.masses
                m1, m2 = masses[mi1], masses[mi2]

                # calculate constraint
                ks, kd, c, cx1, cx2 = state.get_spring_constraint(spring)

                # calculate dλ
                alpha = 1.0/(dt*dt)
                k = kd/dt + ks
                mc2 = m1.inv_m * cx1.length_squared() + \
                    m2.inv_m * cx2.length_squared()
                dcv = (cx1.dot(m1.x - x0[mi1]) +
                       cx2.dot(m2.x - x0[mi2])) / dt
                dl = (-ks*c - kd*dcv - alpha*l0[j]) / (k*mc2 + alpha)
                l0[j] += dl

                # update position
                m1.x += m1.inv_m * cx1 * dl
                m2.x += m2.inv_m * cx2 * dl

        # update velocity and time
        x1 = self.state_to_vector_x(state)
        v1 = (x1-x) / dt
        self.vector_to_state_v(v1, state)
        self.current_state.time += dt
