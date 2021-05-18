import multiprocessing as mp
import os
import warnings
from collections import defaultdict
from multiprocessing.pool import ThreadPool

import _diffcp
import cvxpy.settings as cp_settings
import cvxpy.settings as s
import ecos
import mosek
import numpy as np
import ray
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import scs
import tqdm
from cvxpy.reductions.cone2cone import affine2direct as a2d
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.conic_solvers.mosek_conif import MOSEK
from threadpoolctl import threadpool_limits

import diffcp.cones as cone_lib


def recover_primal_variables(task, sol, K_dir):
    # This function applies both when slacks are introduced, and
    # when the problem is dualized.
    prim_vars = dict()
    idx = 0
    m_free = K_dir[a2d.FREE]
    if m_free > 0:
        temp = [0.0] * m_free
        task.getxxslice(sol, idx, len(temp), temp)
        prim_vars[a2d.FREE] = np.array(temp)
        idx += m_free
    if task.getnumintvar() > 0:
        return prim_vars  # Skip the slack variables.
    m_pos = K_dir[a2d.NONNEG]
    if m_pos > 0:
        temp = [0.0] * m_pos
        task.getxxslice(sol, idx, idx + m_pos, temp)
        prim_vars[a2d.NONNEG] = np.array(temp)
        idx += m_pos
    num_soc = len(K_dir[a2d.SOC])
    if num_soc > 0:
        soc_vars = []
        for dim in K_dir[a2d.SOC]:
            temp = [0.0] * dim
            task.getxxslice(sol, idx, idx + dim, temp)
            soc_vars.append(np.array(temp))
            idx += dim
        prim_vars[a2d.SOC] = soc_vars
    num_dexp = K_dir[a2d.DUAL_EXP]
    if num_dexp > 0:
        temp = [0.0] * (3 * num_dexp)
        task.getxxslice(sol, idx, idx + len(temp), temp)
        temp = np.array(temp)
        perm = expcone_permutor(num_dexp, MOSEK.EXP_CONE_ORDER)
        prim_vars[a2d.DUAL_EXP] = temp[perm]
        idx += 3 * num_dexp
    num_dpow = len(K_dir[a2d.DUAL_POW3D])
    if num_dpow > 0:
        temp = [0.0] * (3 * num_dpow)
        task.getxxslice(sol, idx, idx + len(temp), temp)
        temp = np.array(temp)
        prim_vars[a2d.DUAL_POW3D] = temp
        idx += 3 * num_dpow
    num_psd = len(K_dir[a2d.PSD])
    if num_psd > 0:
        psd_vars = []
        for j, dim in enumerate(K_dir[a2d.PSD]):
            xj = [0.0] * (dim * (dim + 1) // 2)
            task.getbarxj(sol, j, xj)
            psd_vars.append(np.array(xj))
        prim_vars[a2d.PSD] = psd_vars
    return prim_vars


def solve_via_data(data, warm_start, verbose, solver_opts, env, solver_cache=None):
    import mosek

    if "dualized" in data:
        if len(data[s.C]) == 0 and len(data["c_bar_data"]) == 0:
            # primal problem was unconstrained minimization of a linear function.
            if np.linalg.norm(data[s.B]) > 0:
                sol = Solution(s.INFEASIBLE, -np.inf, None, None, dict())
                return {"sol": sol}
            else:
                sol = Solution(s.OPTIMAL, 0.0, dict(), {s.EQ_DUAL: data[s.B]}, dict())
                return {"sol": sol}
        else:
            # env = mosek.Env()
            task = env.Task(0, 0)
            task = MOSEK._build_dualized_task(task, data)
    else:
        if len(data[s.C]) == 0:
            sol = Solution(s.OPTIMAL, 0.0, dict(), dict(), dict())
            return {"sol": sol}
        else:
            env = mosek.Env()
            task = env.Task(0, 0)
            task = MOSEK._build_slack_task(task, data)

    # Set parameters, optimize the Mosek Task, and return the result.
    save_file = MOSEK.handle_options(env, task, verbose, solver_opts)
    if save_file:
        task.writedata(save_file)
    task.optimize()

    if verbose:
        task.solutionsummary(mosek.streamtype.msg)

    return {"env": env, "task": task, "solver_options": solver_opts}


@ray.remote
def solve_wrapper(A, b, c, cone_dict, warm_start, dualized_data, mode, env, kwargs):
    """A wrapper around solve_and_derivative for the batch function."""
    return solve(
        A,
        b,
        c,
        cone_dict,
        warm_start=warm_start,
        env=env,
        dualized_data=dualized_data,
        mode=mode,
        **kwargs,
    )


def solve_batch(
    As,
    bs,
    cs,
    cone_dicts,
    dualized_data,
    n_jobs_forward=-1,
    n_jobs_backward=-1,
    mode="lsqr",
    warm_starts=None,
    **kwargs,
):
    """
    Solves a batch of cone programs and returns a function that
    performs a batch of derivatives. Uses a ThreadPool to perform
    operations across the batch in parallel.

    For more information on the arguments and return values,
    see the docstring for `solve_and_derivative` function.

    Args:
        As - A list of A matrices.
        bs - A list of b arrays.
        cs - A list of c arrays.
        cone_dicts - A list of dictionaries describing the cone.
        n_jobs_forward - Number of jobs to use in the forward pass. n_jobs_forward = 1
            means serial and n_jobs_forward = -1 defaults to the number of CPUs (default=-1).
        n_jobs_backward - Number of jobs to use in the backward pass. n_jobs_backward = 1
            means serial and n_jobs_backward = -1 defaults to the number of CPUs (default=-1).
        mode - Differentiation mode in ["lsqr", "dense"].
        warm_starts - A list of warm starts.
        kwargs - kwargs sent to scs.

    Returns:
        xs: A list of x arrays.
        ys: A list of y arrays.
        ss: A list of s arrays.
        D_batch: A callable with signature
                D_batch(dAs, dbs, dcs) -> dxs, dys, dss
            This callable maps lists of problem data derivatives to lists of solution derivatives.
        DT_batch: A callable with signature
                DT_batch(dxs, dys, dss) -> dAs, dbs, dcs
            This callable maps lists of solution derivatives to lists of problem data derivatives.
    """
    batch_size = len(As)
    if warm_starts is None:
        warm_starts = [None] * batch_size
    if n_jobs_forward == -1:
        n_jobs_forward = mp.cpu_count()
    if n_jobs_backward == -1:
        n_jobs_backward = mp.cpu_count()
    n_jobs_forward = min(batch_size, n_jobs_forward)
    n_jobs_backward = min(batch_size, n_jobs_backward)
    env = mosek.Env()

    if n_jobs_forward == 1:
        # serial
        xs, ys, ss, Ds, DTs = [], [], [], [], []
        for i in range(batch_size):
            x, y, s = solve(
                As[i],
                bs[i],
                cs[i],
                cone_dicts[i],
                dualized_data[i],
                warm_starts[i],
                env=env,
                mode=mode,
                **kwargs,
            )
            xs += [x]
            ys += [y]
            ss += [s]
    else:
        # thread pool
        # os.environ["OMP_NUM_THREADS"] = f"4"

        args = [
            (A, b, c, cone_dict, warm_start, d_data, mode, env, kwargs)
            for A, b, c, cone_dict, warm_start, d_data in zip(
                As, bs, cs, cone_dicts, warm_starts, dualized_data
            )
        ]

        results = ray.get([solve_wrapper.remote(*arg) for arg in args])
        xs = [r[0] for r in results]
        ys = [r[1] for r in results]
        ss = [r[2] for r in results]

    return xs, ys, ss


class SolverError(Exception):
    pass


def solve(
    A,
    b,
    c,
    cone_dict,
    dualized_data,
    warm_start=None,
    env=None,
    mode="lsqr",
    solve_method="SCS",
    **kwargs,
):
    """Solves a cone program, returns its derivative as an abstract linear map.

    This function solves a convex cone program, with primal-dual problems
        min.        c^T x                  min.        b^Ty
        subject to  Ax + s = b             subject to  A^Ty + c = 0
                    s \in K                            y \in K^*

    The problem data A, b, and c correspond to the arguments `A`, `b`, and `c`,
    and the convex cone `K` corresponds to `cone_dict`; x and s are the primal
    variables, and y is the dual variable.

    This function returns a solution (x, y, s) to the program. It also returns
    two functions that respectively represent application of the derivative
    (at A, b, and c) and its adjoint.

    The problem data must be formatted according to the SCS convention, see
    https://github.com/cvxgrp/scs.

    For background on derivatives of cone programs, see
    http://web.stanford.edu/~boyd/papers/diff_cone_prog.html.

    Args:
      A: A sparse SciPy matrix in CSC format; the first block of rows must
        correspondond to the zero cone, the next block to the positive orthant,
        then the second-order cone, the PSD cone, the exponential cone, and
        finally the exponential dual cone. PSD matrix variables must be
        vectorized by scaling the off-diagonal entries by sqrt(2) and stacking
        the lower triangular part in column-major order. WARNING: This
        function eliminates zero entries in A.
      b: A NumPy array representing the offset.
      c: A NumPy array representing the objective function.
      cone_dict: A dictionary with keys corresponding to cones, values
          corresponding to their dimensions. The keys must be a subset of
          diffcp.ZERO, diffcp.POS, diffcp.SOC, diffcp.PSD, diffcp.EXP;
          the values of diffcp.SOC, diffcp.PSD, and diffcp.EXP
          should be lists. A k-dimensional PSD cone corresponds to a k-by-k
          matrix variable; a value of k for diffcp.EXP corresponds to k / 3
          exponential cones. See SCS documentation for more details.
      warm_start: (optional) A tuple (x, y, s) at which to warm-start SCS.
      mode: (optional) Which mode to compute derivative with, options are
          ["dense", "lsqr"].
      solve_method: (optional) Name of solver to use; SCS or ECOS.
      kwargs: (optional) Keyword arguments to send to the solver.

    Returns:
        x: Optimal value of the primal variable x.
        y: Optimal value of the dual variable y.
        s: Optimal value of the slack variable s.
        derivative: A callable with signature
                derivative(dA, db, dc) -> dx, dy, ds
            that applies the derivative of the cone program at (A, b, and c)
            to the perturbations `dA`, `db`, `dc`. `dA` must be a SciPy sparse
            matrix in CSC format with the same sparsity pattern as `A`;
            `db` and `dc` are NumPy arrays.
        adjoint_derivative: A callable with signature
                adjoint_derivative(dx, dy, ds) -> dA, db, dc
            that applies the adjoint of the derivative of the cone program at
            (A, b, and c) to the perturbations `dx`, `dy`, `ds`, which must be
            NumPy arrays. The output `dA` matches the sparsity pattern of `A`.
    Raises:
        SolverError: if the cone program is infeasible or unbounded.
    """
    result = solve_internal(
        A,
        b,
        c,
        cone_dict,
        dualized_data,
        env=env,
        warm_start=warm_start,
        mode=mode,
        solve_method=solve_method,
        **kwargs,
    )
    x = result["x"]
    y = result["y"]
    s = result["s"]
    return x, y, s


def solve_internal(
    A,
    b,
    c,
    cone_dict,
    dualized_data,
    env=None,
    solve_method=None,
    warm_start=None,
    mode="lsqr",
    raise_on_error=True,
    **kwargs,
):
    if mode not in ["dense", "lsqr"]:
        raise ValueError(
            "Unsupported mode {}; the supported modes are "
            "'dense' and 'lsqr'".format(mode)
        )

    if np.isnan(A.data).any():
        raise RuntimeError("Found a NaN in A.")

    # set explicit 0s in A to np.nan
    A = sp.sparse.csc_matrix(A, copy=True)
    A.data[A.data == 0] = np.nan

    # compute rows and cols of nonzeros in A
    rows, cols = A.nonzero()

    # reset np.nan entries in A to 0.0
    A.data[np.isnan(A.data)] = 0.0

    # eliminate explicit zeros in A, we no longer need them
    A.eliminate_zeros()

    solve_method = "MOSEK"

    if solve_method is None:
        psd_cone = ("s" in cone_dict) and (cone_dict["s"] != [])
        ep_cone = ("ep" in cone_dict) and (cone_dict["ep"] != 0)
        ed_cone = ("ed" in cone_dict) and (cone_dict["ed"] != 0)
        if psd_cone or ep_cone or ed_cone:
            solve_method = "SCS"
        else:
            solve_method = "ECOS"
    if solve_method == "MOSEK":

        STATUS_MAP = {
            mosek.solsta.optimal: cp_settings.OPTIMAL,
            mosek.solsta.integer_optimal: cp_settings.OPTIMAL,
            mosek.solsta.prim_feas: cp_settings.OPTIMAL_INACCURATE,  # for integer problems
            mosek.solsta.prim_infeas_cer: cp_settings.INFEASIBLE,
            mosek.solsta.dual_infeas_cer: cp_settings.UNBOUNDED,
        }
        # "Near" statuses only up to Mosek 8.1
        if hasattr(mosek.solsta, "near_optimal"):
            STATUS_MAP[mosek.solsta.near_optimal] = cp_settings.OPTIMAL_INACCURATE
            STATUS_MAP[
                mosek.solsta.near_integer_optimal
            ] = cp_settings.OPTIMAL_INACCURATE
            STATUS_MAP[
                mosek.solsta.near_prim_infeas_cer
            ] = cp_settings.INFEASIBLE_INACCURATE
            STATUS_MAP[
                mosek.solsta.near_dual_infeas_cer
            ] = cp_settings.UNBOUNDED_INACCURATE
        STATUS_MAP = defaultdict(lambda: s.SOLVER_ERROR, STATUS_MAP)

        solver_output = solve_via_data(
            dualized_data,
            warm_start=None,
            verbose=False,
            env=env,
            solver_opts={
                "mosek_params": {
                    # "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1.0e-12,
                    # "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1.0e-12,
                    # "MSK_DPAR_SEMIDEFINITE_TOL_APPROX": 1.0e-12,
                    "MSK_IPAR_NUM_THREADS": 4,
                }
            },
        )
        env = solver_output["env"]
        task = solver_output["task"]
        solver_opts = solver_output["solver_options"]

        if task.getnumintvar() > 0:
            sol_type = mosek.soltype.itg
        elif "bfs" in solver_opts and solver_opts["bfs"] and task.getnumcone() == 0:
            sol_type = mosek.soltype.bas  # the basic feasible solution
        else:
            sol_type = mosek.soltype.itr  # the solution found via interior point method

        prim_vars = None
        dual_vars = None
        problem_status = task.getprosta(sol_type)
        if sol_type == mosek.soltype.itg and problem_status == mosek.prosta.prim_infeas:
            status = s.INFEASIBLE
            prob_val = np.inf
            raise SolverError("Solver scs returned status infeasible")
        else:
            solsta = task.getsolsta(sol_type)
            status = STATUS_MAP[solsta]
            prob_val = np.NaN
            if status in cp_settings.SOLUTION_PRESENT:
                prob_val = task.getprimalobj(sol_type)
                prim_vars = recover_primal_variables(
                    task, sol_type, dualized_data["K_dir"]
                )
                dual_vars = MOSEK.recover_dual_variables(task, sol_type)
            # Delete the mosek Task and Environment
            task.__exit__(None, None, None)
            env.__exit__(None, None, None)

            result = {}
            result["x"] = dual_vars["eq_dual"]
            result["y"] = np.array([])
            # print(prim_vars.keys())
            if "+" in prim_vars:
                result["y"] = prim_vars["+"]
            if "fr" in prim_vars:
                result["y"] = np.append(prim_vars["fr"], result["y"])
            if "q" in prim_vars:
                for d in prim_vars["q"]:
                    result["y"] = np.append(result["y"], d)
            if "s" in prim_vars:
                for d in prim_vars["s"]:
                    result["y"] = np.append(result["y"], d)

            # # print(prim_vars)
            s = b - A @ result["x"]
            result["s"] = s

            # print(jere)

    elif solve_method == "SCS":
        data = {"A": A, "b": b, "c": c}

        if warm_start is not None:
            data["x"] = warm_start[0]
            data["y"] = warm_start[1]
            data["s"] = warm_start[2]

        kwargs.setdefault("verbose", False)
        result = scs.solve(data, cone_dict, **kwargs)

        status = result["info"]["status"]
        if status == "Solved/Inaccurate" and "acceleration_lookback" not in kwargs:
            # anderson acceleration is sometimes unstable
            warnings.warn("Solved/Inaccurate 2.")
            result = scs.solve(data, cone_dict, acceleration_lookback=0, **kwargs)
            status = result["info"]["status"]

        if status == "Solved/Inaccurate":
            warnings.warn("Solved/Inaccurate.")
        elif status != "Solved":
            if raise_on_error:
                raise SolverError("Solver scs returned status %s" % status)
            else:
                result["D"] = None
                result["DT"] = None
                return result

        x = result["x"]
        y = result["y"]
        s = result["s"]
        # print(result["y"].shape)
        # print(result["y"])
    elif solve_method == "ECOS":
        if warm_start is not None:
            raise ValueError("ECOS does not support warmstart.")
        if ("s" in cone_dict) and (cone_dict["s"] != []):
            raise ValueError("PSD cone not supported by ECOS.")
        if ("ep" in cone_dict) and (cone_dict["ep"] != 0):
            raise NotImplementedError("Exponential cones not supported yet.")
        if ("ed" in cone_dict) and (cone_dict["ed"] != 0):
            raise NotImplementedError("Exponential cones not supported yet.")
        if warm_start is not None:
            raise ValueError("ECOS does not support warm starting.")
        len_eq = cone_dict["f"]
        C_ecos = c
        G_ecos = A[len_eq:]
        if 0 in G_ecos.shape:
            G_ecos = None
        H_ecos = b[len_eq:].flatten()
        if 0 in H_ecos.shape:
            H_ecos = None
        A_ecos = A[:len_eq]
        if 0 in A_ecos.shape:
            A_ecos = None
        B_ecos = b[:len_eq].flatten()
        if 0 in B_ecos.shape:
            B_ecos = None

        cone_dict_ecos = {}
        if "l" in cone_dict:
            cone_dict_ecos["l"] = cone_dict["l"]
        if "q" in cone_dict:
            cone_dict_ecos["q"] = cone_dict["q"]
        if A_ecos is not None and A_ecos.nnz == 0 and np.prod(A_ecos.shape) > 0:
            raise ValueError("ECOS cannot handle sparse data with nnz == 0.")

        kwargs.setdefault("verbose", False)
        solution = ecos.solve(
            C_ecos, G_ecos, H_ecos, cone_dict_ecos, A_ecos, B_ecos, **kwargs
        )
        x = solution["x"]
        y = np.append(solution["y"], solution["z"])
        s = b - A @ x

        result = {"x": x, "y": y, "s": s}
        status = solution["info"]["exitFlag"]
        STATUS_LOOKUP = {
            0: "Optimal",
            1: "Infeasible",
            2: "Unbounded",
            10: "Optimal Inaccurate",
            11: "Infeasible Inaccurate",
            12: "Unbounded Inaccurate",
        }

        if status == 10:
            warnings.warn("Solved/Inaccurate.")
        elif status < 0:
            raise SolverError("Solver ecos errored.")
        if status not in [0, 10]:
            raise SolverError("Solver ecos returned status %s" % STATUS_LOOKUP[status])
    else:
        raise ValueError("Solver %s not supported." % solve_method)

    return result
