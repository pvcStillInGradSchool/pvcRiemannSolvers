"""Concrete implementations of temporal schemes.
"""
from concept import TemporalScheme


class ExplicitEuler(TemporalScheme):
    """The explicit Euler method.
    """

    def __init__(self):
        pass

    def update(self, ode_system, delta_t):
        u_curr = ode_system.get_solution_column()
        residual = ode_system.get_residual_column()
        u_next = u_curr
        u_next += residual * delta_t
        ode_system.set_solution_column(u_next)


class SspRungeKutta(TemporalScheme):
    """The explicit Runge--Kutta methods.
    """

    def __init__(self, order: int):
        assert 1 <= order <= 3
        self._order = order
        self._euler = ExplicitEuler()


    def update(self, ode_system, delta_t):
        if self._order == 1:
            self._rk1_update(ode_system, delta_t)
        elif self._order == 2:
            self._rk2_update(ode_system, delta_t)
        elif self._order == 3:
            self._rk3_update(ode_system, delta_t)
        else:
            raise NotImplementedError("Only 1st-, 2nd- and 3rd-order are implemented.")


    def _rk1_update(self, ode_system, delta_t):
        self._euler.update(ode_system, delta_t)


    def _rk2_update(self, ode_system, delta_t):
        u_curr = ode_system.get_solution_column()  # u_curr == U_{n}
        self._euler.update(ode_system, delta_t)
        # Now, ode_system holds U_{n + 1/2}
        self._euler.update(ode_system, delta_t)
        u_next = ode_system.get_solution_column()
        # Now, u_next == U_{n + 1/2} + R_{n + 1/2} * delta_t
        u_next += u_curr
        u_next /= 2
        # Now, u_next == U_{n + 1}
        ode_system.set_solution_column(u_next)


    def _rk3_update(self, ode_system, delta_t):
        u_curr = ode_system.get_solution_column()  # u_curr == U_{n}
        self._euler.update(ode_system, delta_t)
        # Now, ode_system holds U_{n + 1/3}
        self._euler.update(ode_system, delta_t)
        u_next = ode_system.get_solution_column()
        # Now, u_next == U_{n + 1/3} + R_{n + 1/3} * dt
        u_next += u_curr * 3
        u_next /= 4
        # Now, u_next == U_{n + 2/3}.
        ode_system.set_solution_column(u_next)
        self._euler.update(ode_system, delta_t)
        u_next = ode_system.get_solution_column()
        # Now, u_next == U_{n + 2/3} + R_{n + 2/3} * dt
        u_next *= 2
        u_next += u_curr
        u_next /= 3
        # Now, u_next == U_{n + 1}.
        ode_system.set_solution_column(u_next)


if __name__ == '__main__':
    pass