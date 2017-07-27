import logging

import numpy as np
from scipy.misc import factorial, pade

from nengo import Node, Probe, Connection, Simulator
from nengo.networks import EnsembleArray
from nengo.neurons import Direct
from nengo.processes import WhiteSignal, PresentInput
from nengo.solvers import LstsqL2
from nengo.utils.numpy import rmse, rms


from nengolib import Network
from nengolib.neurons import PerfectLIF
from nengolib.signal import (Balanced, LinearSystem, cont2discrete,
                             canonical, nrmse)
from nengolib.networks import LinearNetwork
from nengolib.synapses import Alpha, Lowpass, DoubleExp, DiscreteDelay


def lambert_delay(delay, sub_delay, tau, p, q):
    """Returns F = p/q s.t. F((tau*s+1)/e^(-sb)) = e^(-sa)."""
    a, b = delay, sub_delay
    r = a / b
    c = np.exp(a / tau)
    d = (b / tau) * np.exp(b / tau)
    i = np.arange(1, p + q + 1)
    taylor = np.append([1./r], (i+r)**(i-1) / factorial(i))
    tf = pade(taylor, q)
    nds = np.poly1d([-d, 0])  # -ds
    return LinearSystem((c*r*tf[0](nds), tf[1](nds)), analog=True)


def delayed_synapse():
    a = 0.1  # desired delay
    b = 0.01  # synapse delay
    tau = 0.01  # recurrent tau
    hz = 15  # input frequency
    t = 1.0  # simulation time
    dt = 0.00001  # simulation timestep
    order = 6  # order of pade approximation
    tau_probe = 0.02

    dexp_synapse = DoubleExp(tau, tau / 5)

    sys_lambert = lambert_delay(a, b, tau, order - 1, order)
    synapse = (cont2discrete(Lowpass(tau), dt=dt) *
               DiscreteDelay(int(b / dt)))

    n_neurons = 2000
    neuron_type = PerfectLIF()

    A, B, C, D = sys_lambert.observable.transform(5*np.eye(order)).ss

    sys_normal = PadeDelay(a, order)
    assert len(sys_normal) == order

    with Network(seed=0) as model:
        stim = Node(output=WhiteSignal(t, high=hz, y0=0))

        x = EnsembleArray(n_neurons / order, len(A), neuron_type=neuron_type)
        output = Node(size_in=1)

        Connection(x.output, x.input, transform=A, synapse=synapse)
        Connection(stim, x.input, transform=B, synapse=synapse)
        Connection(x.output, output, transform=C, synapse=None)
        Connection(stim, output, transform=D, synapse=None)

        lowpass_delay = LinearNetwork(
            sys_normal, n_neurons_per_ensemble=n_neurons / order,
            synapse=tau, input_synapse=tau,
            dt=None, neuron_type=neuron_type, radii=1.0)
        Connection(stim, lowpass_delay.input, synapse=None)

        dexp_delay = LinearNetwork(
            sys_normal, n_neurons_per_ensemble=n_neurons / order,
            synapse=dexp_synapse, input_synapse=dexp_synapse,
            dt=None, neuron_type=neuron_type, radii=1.0)
        Connection(stim, dexp_delay.input, synapse=None)

        p_stim = Probe(stim, synapse=tau_probe)
        p_output_delayed = Probe(output, synapse=tau_probe)
        p_output_lowpass = Probe(lowpass_delay.output, synapse=tau_probe)
        p_output_dexp = Probe(dexp_delay.output, synapse=tau_probe)

    with Simulator(model, dt=dt, seed=0) as sim:
        sim.run(t)

    return (a, dt, sim.trange(), sim.data[p_stim],
            sim.data[p_output_delayed], sim.data[p_output_lowpass],
            sim.data[p_output_dexp])


def delay_example():
    seed = 2

    n_neurons = 1000
    theta = 1.0
    sys = PadeDelay(theta, 6)

    T = 20.0
    dt = 0.001
    freq = 1
    rms = 0.4

    tau = 0.1
    tau_probe = 0.02

    radii = np.ones(len(sys))  # initial guess
    desired_radius = 0.8  # aiming to get this as largest x
    num_iter = 5  # number of times to simulate and retry new radius

    # could also do this simply by the direct method in discrete_example
    # but this is just to demonstrate that you can do something iterative
    # within the same network
    for _ in range(num_iter):
        with Network(seed=seed) as model:
            signal = WhiteSignal(T, high=freq, rms=rms, y0=0)
            u = Node(output=signal)

            delay = LinearNetwork(
                sys, n_neurons_per_ensemble=n_neurons / len(sys), synapse=tau,
                input_synapse=tau, radii=radii, realizer=Balanced(), dt=None)
            Connection(u, delay.input, synapse=None)

            p_u = Probe(u, synapse=tau_probe)
            p_x = Probe(delay.state.input, synapse=None)
            p_a = Probe(delay.state.add_neuron_output(), synapse=None)
            p_y = Probe(delay.output, synapse=tau_probe)

        with Simulator(model, dt=dt, seed=seed) as sim:
            sim.run(T)

        # place the worst case at x=desired_radius and re-run
        worst_x = np.max(np.abs(sim.data[p_x]), axis=0)
        radii *= (worst_x / desired_radius)
        logging.info("Radii: %s\nWorst x: %s", radii, worst_x)

    return (theta, dt, sim.trange(), sim.data[p_u], sim.data[p_x],
            sim.data[p_a], sim.data[p_y])


def discrete_example(seed, dt):
    n_neurons = 1000
    theta = 0.1
    freq = 50
    q = 27
    radii = 1.0
    sys = PadeDelay(theta, q)

    T = 5000*(dt+0.001)
    rms = 1.0
    signal = WhiteSignal(T, high=freq, rms=rms, y0=0)

    tau = 0.1
    tau_probe = 0.02
    reg = 0.1

    # Determine radii using direct mode
    with LinearNetwork(
            sys, n_neurons_per_ensemble=1, input_synapse=tau, synapse=tau,
            dt=dt, neuron_type=Direct(),
            realizer=Balanced()) as model:
        Connection(Node(output=signal), model.input, synapse=None)
        p_x = Probe(model.state.input, synapse=None)

    with Simulator(model, dt=dt, seed=seed+1) as sim:
        sim.run(T)

    radii *= np.max(abs(sim.data[p_x]), axis=0)
    logging.info("Radii: %s", radii)

    with Network(seed=seed) as model:
        u = Node(output=signal)

        kwargs = dict(
            n_neurons_per_ensemble=n_neurons / len(sys),
            input_synapse=tau, synapse=tau, radii=radii,
            solver=LstsqL2(reg=reg), realizer=Balanced())
        delay_disc = LinearNetwork(sys, dt=dt, **kwargs)
        delay_cont = LinearNetwork(sys, dt=None, **kwargs)
        Connection(u, delay_disc.input, synapse=None)
        Connection(u, delay_cont.input, synapse=None)

        p_u = Probe(u, synapse=tau_probe)
        p_y_disc = Probe(delay_disc.output, synapse=tau_probe)
        p_y_cont = Probe(delay_cont.output, synapse=tau_probe)

    with Simulator(model, dt=dt, seed=seed) as sim:
        sim.run(T)

    return (theta, dt, sim.trange(), sim.data[p_u],
            sim.data[p_y_disc], sim.data[p_y_cont])


def time_cells(order):
    seed = 0
    n_neurons = 300
    theta = 4.784
    tau = 0.1
    radius = 0.3
    realizer = Balanced


    # The following was patched from nengolib commit
    # 7e204e0c305e34a4f63d0a6fbba7197862bbcf22, prior to
    # aee92b8fc45749f07f663fe696745cf0a33bfa17, so that
    # the generated PDF is consistent with the version that the
    # overlay was added to.
    def PadeDelay(c, q):
        j = np.arange(1, q+1, dtype=np.float64)
        u = (q + j - 1) * (q - j + 1) / (c * j)

        A = np.zeros((q, q))
        B = np.zeros((q, 1))
        C = np.zeros((1, q))
        D = np.zeros((1,))

        A[0, :] = B[0, 0] = -u[0]
        A[1:, :-1][np.diag_indices(q-1)] = u[1:]
        C[0, :] = - j / float(q) * (-1) ** (q - j)
        return LinearSystem((A, B, C, D), analog=True)

    F = PadeDelay(theta, order)
    synapse = Alpha(tau)

    pulse_s = 0
    pulse_w = 1.0
    pulse_h = 1.5

    T = 6.0
    dt = 0.001
    pulse = np.zeros(int(T/dt))
    pulse[int(pulse_s/dt):int((pulse_s + pulse_w)/dt)] = pulse_h

    with Network(seed=seed) as model:
        u = Node(output=PresentInput(pulse, dt))

        delay = LinearNetwork(
            F, n_neurons_per_ensemble=n_neurons / len(F), synapse=synapse,
            input_synapse=None, radii=radius, dt=dt, realizer=realizer())
        Connection(u, delay.input, synapse=None)

        p_x = Probe(delay.state.input, synapse=None)
        p_a = Probe(delay.state.add_neuron_output(), synapse=None)

    with Simulator(model, dt=dt) as sim:
        sim.run(T)

    return sim.trange(), sim.data[p_x], sim.data[p_a]
