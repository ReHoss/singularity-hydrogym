import firedrake as fd  # type: ignore
import psutil  # pyright: ignore [reportMissingModuleSource]  # Provided by firedrake


def compute_vorticity_paraview(flow) -> tuple:
    u, p = flow.u, flow.p
    return u, p, flow.vorticity()


# noinspection PyUnresolvedReferences
def postprocess_cavity(flow) -> list:
    dt = flow.dt
    ke = 0.5 * fd.assemble(fd.inner(flow.u, flow.u) * fd.dx)
    tke = flow.evaluate_objective()
    cfl = flow.max_cfl(dt)
    return [cfl, ke, tke]


def postprocess_pinball(flow) -> tuple[float, float, float, float]:
    mem_usage = psutil.virtual_memory().percent
    obs = flow.get_observations()
    cl = [float(cl_i) for cl_i in obs[:3]]
    return cl[0], cl[1], cl[2], mem_usage
