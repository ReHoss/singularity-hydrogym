import firedrake as fd


def compute_vorticity(flow) -> tuple:
    u, p = flow.u, flow.p
    return u, p, flow.vorticity()


# noinspection PyUnresolvedReferences
def postprocess_cavity(flow) -> list:
    dt = flow.dt
    ke = 0.5 * fd.assemble(fd.inner(flow.u, flow.u) * fd.dx)
    tke = flow.evaluate_objective()
    cfl = flow.max_cfl(dt)
    return [cfl, ke, tke]
