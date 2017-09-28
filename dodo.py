import logging
import warnings

import numpy as np
from scipy.special import lambertw, binom
from scipy.stats import pearsonr

import statsmodels.api as sm

import pylab
import seaborn as sns
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.lines import Line2D
from matplotlib.mlab import PCA
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import nengo
from nengo.processes import WhiteSignal

import nengolib
from nengolib.signal import cont2discrete, LinearSystem, nrmse
from nengolib.stats import sphere
from nengolib.synapses import (DiscreteDelay, PadeDelay, Lowpass, DoubleExp,
                               ss2sim)
from main import (delayed_synapse, delay_example, discrete_example,
                  time_cells, lambert_delay)

###############################################################################

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.serif'] = 'cm'

logging.basicConfig(level=logging.INFO)

###############################################################################

if nengo.version.version_info != (2, 4, 0):
    warnings.warn("Expecting nengo version 2.4.0, saw %s" %
                  nengo.__version__)

if nengolib.version.version_info != (0, 4, 1):
    warnings.warn("Expecting nengolib version 0.4.1, saw %s" %
                  nengolib.__version__)

###############################################################################

LAMBERT_SIM = 'data/lambert.npz'

DISCRETE_SEEDS = 25
DISCRETE_DTS = np.linspace(0.0001, 0.002, 20)
DISCRETE_SIM = 'data/discrete_%d_%d.npz'

DELAY_EXAMPLE_SIM = 'data/delay_example.npz'

TIME_CELLS_ORDERS = range(3, 11)
TIME_CELLS_SIM = 'data/time_cells_%s.npz'

###############################################################################


class HandlerDashedLines(HandlerLineCollection):
    """Adapted from http://matplotlib.org/examples/pylab_examples/legend_demo5.html"""  # noqa: E501

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # figure out how many lines there are
        numlines = len(orig_handle.get_segments())
        xdata, xdata_marker = self.get_xdata(
            legend, xdescent, ydescent, width / numlines, height, fontsize)
        leglines = []
        for i in range(numlines):
            legline = Line2D(
                xdata + i * width / numlines,
                np.zeros_like(xdata.shape) - ydescent + height / 2)
            self.update_prop(legline, orig_handle, legend)
            # set color, dash pattern, and linewidth to that
            # of the lines in linecollection
            try:
                color = orig_handle.get_colors()[i]
            except IndexError:
                color = orig_handle.get_colors()[0]
            try:
                dashes = orig_handle.get_dashes()[i]
            except IndexError:
                dashes = orig_handle.get_dashes()[0]
            try:
                lw = orig_handle.get_linewidths()[i]
            except IndexError:
                lw = orig_handle.get_linewidths()[0]
            if dashes[0] is not None:
                legline.set_dashes(dashes[1])
            legline.set_color(color)
            legline.set_transform(trans)
            legline.set_linewidth(lw)
            leglines.append(legline)
        return leglines


def simulate_lambert(targets):
    delay, dt, t, stim, delayed, lowpass, dexp = delayed_synapse()
    np.savez(targets[0], delay=delay, dt=dt, t=t, stim=stim,
             delayed=delayed, lowpass=lowpass, dexp=dexp)


def simulate_delay_example(targets):
    theta, dt, t, u, x, a, y = delay_example()
    np.savez(targets[0], theta=theta, dt=dt, t=t, u=u, x=x, a=a, y=y)


def simulate_time_cells(targets, order):
    t, x, a = time_cells(order)
    np.savez(targets[0], t=t, x=x, a=a, order=order)


def savefig(name):
    pylab.savefig(name, dpi=600, transparent=True, bbox_inches='tight')


def ideal_delay(x, delay, dt):
    return DiscreteDelay(int(delay/dt)).filt(x, dt=dt)


def figure_lambert(targets):
    npfile = np.load(LAMBERT_SIM)
    delay = npfile['delay']
    dt = npfile['dt']
    t = npfile['t']
    stim = npfile['stim']
    delayed = npfile['delayed']
    lowpass = npfile['lowpass']
    dexp = npfile['dexp']

    target = ideal_delay(stim, delay, dt)

    e_delayed = nrmse(delayed, target=target)
    e_lowpass = nrmse(lowpass, target=target)
    e_dexp = nrmse(dexp, target=target)

    improvement = (e_lowpass - e_delayed) / e_lowpass * 100
    logging.info("Paper constant: Lambert improvement: %s", improvement)
    logging.info("Paper constant: Delayed NRMSE: %s", e_delayed)
    logging.info("Paper constant: Lowpass NRMSE: %s", e_lowpass)
    logging.info("Paper constant: Double Exp NRMSE: %s", e_dexp)

    sample_rate = 100
    t = t[::sample_rate]
    stim = stim[::sample_rate]
    delayed = delayed[::sample_rate]
    lowpass = lowpass[::sample_rate]
    dexp = dexp[::sample_rate]
    target = target[::sample_rate]

    tau_over_theta = 0.1
    lambda_over_tau = 1.0
    max_freq_times_theta = 4.0
    theta = 0.1  # <-- the graph is still the same, no matter theta!
    tau = tau_over_theta * theta
    lmbda = lambda_over_tau * tau
    max_freq = max_freq_times_theta / theta
    tau2 = tau / 5  # TODO: parameters copied from delayed_synapse()
    q = 6
    assert np.allclose(tau, 0.01)
    assert np.allclose(lmbda, 0.01)

    freqs = np.linspace(0, max_freq, 200)
    s = 2.j*np.pi*freqs
    axis = freqs * theta  # scale-invariant axis

    lw = 3
    alpha = 0.8
    cmap = sns.color_palette(None, 4)

    F_lamb = lambert_delay(theta, lmbda, tau, q-1, q)
    F_low = ss2sim(PadeDelay(theta, order=q), Lowpass(tau), dt=None)
    F_alpha = ss2sim(PadeDelay(theta, order=q), DoubleExp(tau, tau2), dt=None)

    y_low = F_low(tau*s + 1)
    y_lamb = F_lamb((tau*s + 1)*np.exp(lmbda*s))
    y_alpha = F_alpha((tau*s + 1)*(tau2*s + 1))
    y = np.exp(-theta*s)

    # Compute where lmbda*s + lmbda/tau is within the principal branch
    tx = lmbda*2*np.pi*freqs
    st = (lmbda/tau > -(tx/np.tan(tx))) & (tx < np.pi) | (tx == 0)
    p = lmbda*s[st] + lmbda/tau
    assert np.allclose(lambertw(p*np.exp(p)), p)

    with sns.axes_style('ticks'):
        with sns.plotting_context('paper', font_scale=2.8):
            pylab.figure(figsize=(18, 5))
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.618])
            gs.update(wspace=0.3)
            ax1 = plt.subplot(gs[1])
            ax2 = plt.subplot(gs[0])

            ax1.set_title(
                r"$0.1\,$s Delay of $15\,$Hz White Noise").set_y(1.05)
            ax1.plot(t, target, lw=4, c=cmap[0], zorder=4, linestyle='--')
            ax1.plot(t, lowpass, alpha=alpha, c=cmap[1], zorder=2)
            ax1.plot(t, dexp, alpha=alpha, c=cmap[3], zorder=2)
            ax1.plot(t, delayed, alpha=alpha, c=cmap[2], zorder=3)
            ax1.set_ylim(-0.5, 0.5)
            ax1.set_xlabel("Time [s]", labelpad=20)
            ax1.set_ylabel("Output")

            ax2.set_title("Delay Accuracy").set_y(1.05)
            ax2.plot(axis, np.zeros_like(freqs), lw=lw, c=cmap[0],
                     zorder=4, linestyle='--', label=r"Ideal")
            ax2.plot(axis, abs(y - y_low), lw=lw, alpha=alpha, c=cmap[1],
                     zorder=3, label=r"Lowpass")
            ax2.plot(axis, abs(y - y_alpha), lw=lw, alpha=alpha, c=cmap[3],
                     zorder=3, label=r"Double-exponential")
            ax2.plot(axis, abs(y - y_lamb), lw=lw, alpha=alpha, c=cmap[2],
                     zorder=3, label=r"Delayed Lowpass")

            s = 0.8
            pts = np.asarray([[1.5, 0],
                              [1.5-s, -s],
                              [1.5+s, -s]])
            ax2.add_patch(Polygon(pts, closed=True, color='black'))

            ax2.set_xlabel(r"Frequency $\times \, \theta$ [Hz $\times$ s]",
                           labelpad=20)
            ax2.set_ylabel(r"Absolute Error", labelpad=20)

            ax2.legend(
                loc='upper left', frameon=True).get_frame().set_alpha(0.8)

            sns.despine(offset=15)

            savefig(targets[0])


def figure_principle3(targets):
    theta = 0.1
    tau = 0.1 * theta
    lmbda = tau
    orders = range(6, 28)

    freqs = np.linspace(0.1 / theta, 16 / theta, 1000)
    s = 2.j * np.pi * freqs

    y = np.exp(-theta*s)
    Hinvs = (tau*s + 1)*np.exp(lmbda*s)

    cmap_lamb = sns.color_palette("GnBu_d", len(orders))[::-1]
    cmap_ignore = sns.color_palette("OrRd_d", len(orders))[::-1]

    data = np.empty((2, len(DISCRETE_DTS), DISCRETE_SEEDS))

    for seed in range(DISCRETE_SEEDS):
        for i, dt in enumerate(DISCRETE_DTS):
            npfile = np.load(DISCRETE_SIM % (seed, i))
            assert np.allclose(npfile['dt'], dt)
            delay = npfile['delay']
            # t = npfile['t']
            stim = npfile['stim']
            disc = npfile['disc']
            cont = npfile['cont']

            target = ideal_delay(stim, delay, dt)
            e_disc = nrmse(disc, target=target)
            e_cont = nrmse(cont, target=target)

            data[0, i, seed] = e_disc
            data[1, i, seed] = e_cont

    i = np.where(DISCRETE_DTS == 0.001)[0][0]
    assert np.allclose(DISCRETE_DTS[i], 0.001)
    e_disc = np.mean(data, axis=2)[0, i]
    e_cont = np.mean(data, axis=2)[1, i]
    improvement = (e_cont - e_disc) / e_cont * 100
    logging.info("Paper constant: Improvement at 1 ms: %s (%s -> %s)",
                 improvement, e_cont, e_disc)

    with sns.axes_style('ticks'):
        with sns.plotting_context('paper', font_scale=2.8):
            f, (ax1, ax2) = pylab.subplots(1, 2, figsize=(18, 5))

            ax1.set_title("Discrete Lowpass Improvement").set_y(1.05)

            for i, condition, cpal, marker in (
                    (1, 'Principle 3', sns.color_palette("OrRd_d"), 'X'),
                    (0, 'Extension', sns.color_palette("GnBu_d"), 'o')):
                sns.tsplot(data[i].T, 1000*DISCRETE_DTS, condition=condition,
                           color=cpal, marker=marker, markersize=15, lw=3,
                           ci=95, alpha=0.7, ax=ax1)

            ax1.vlines([1.0], np.min(data[0]), 2.0, linestyle='--',
                       color='black', lw=4, alpha=0.7, zorder=0)

            ax1.set_xlabel("Discrete Time-step [ms]", labelpad=20)
            ax1.set_ylabel("Absolute Error", labelpad=20)
            ax1.set_xlim(0, 1000*DISCRETE_DTS[-1] + 0.1)

            ax2.set_title("Delayed Lowpass Improvement").set_y(1.05)

            for i, q in enumerate(orders):
                sys = PadeDelay(theta, order=q)
                mapped = ss2sim(sys, Lowpass(tau), dt=None)
                lambert = lambert_delay(theta, lmbda, tau, q-1, q)

                y_lamb = lambert(Hinvs)
                y_ignore = mapped(Hinvs)

                ax2.semilogy(freqs*theta, abs(y - y_ignore), lw=2, alpha=0.8,
                             zorder=len(orders)-i, c=cmap_ignore[i])
                ax2.semilogy(freqs*theta, abs(y - y_lamb), lw=2, alpha=0.8,
                             zorder=len(orders)-i, c=cmap_lamb[i])

            lc_ignore = LineCollection(
                len(orders) * [[(0, 0)]], lw=10, colors=cmap_ignore)
            lc_lamb = LineCollection(
                len(orders) * [[(0, 0)]], lw=10, colors=cmap_lamb)
            ax2.legend([lc_ignore, lc_lamb], ['Principle 3', 'Extension'],
                       handlelength=2,
                       handler_map={LineCollection: HandlerDashedLines()})

            ax2.set_xlabel(r"Frequency $\times \, \theta$ [Hz $\times$ s]",
                           labelpad=20)

            sns.despine(offset=15)

            savefig(targets[0])


def figure_delay_example(targets):
    npfile = np.load(DELAY_EXAMPLE_SIM)
    theta = npfile['theta']
    dt = npfile['dt']
    t = npfile['t']
    u = npfile['u']
    x = npfile['x']
    a = npfile['a']
    y = npfile['y']
    T = t[-1]

    ideal = ideal_delay(u, theta, dt)
    logging.info("Paper constant: NRMSE: %s", nrmse(y, target=ideal))

    n_encoders = 1000
    rng = np.random.RandomState(seed=0)
    encoders = sphere.sample(n_encoders, x.shape[1], rng=rng)
    sims = np.dot(x, encoders.T)
    order = np.argsort(np.argmax(sims, axis=0))
    intercept = -1
    sims = sims.clip(intercept)
    sims -= np.min(sims, axis=0)
    sims /= np.max(sims, axis=0)
    a = sims[:, order]
    assert np.isfinite(a).all()

    sample_rate = 50  # downsample PDFs to avoid reader lag
    t = t[::sample_rate]
    u = u[::sample_rate]
    x = x[::sample_rate]
    a = a[::sample_rate]
    y = y[::sample_rate]
    ideal = ideal[::sample_rate]

    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    with sns.axes_style('ticks'):
        with sns.plotting_context('paper', font_scale=2.8):
            f, (ax1, ax2, ax3) = pylab.subplots(3, 1, figsize=(18, 16))

            cpal = sns.color_palette(None, 2)
            ax1.plot(t, u, label=r"$u(t)$",
                     c=cpal[0], lw=4, alpha=0.8)
            ax1.plot(t, ideal, label=r"$u(t - %s)$" % theta,
                     c=cpal[0], lw=4, linestyle='--', zorder=2)  # on top
            ax1.plot(t, y,
                     c=cpal[1], lw=4, label=r"$y(t)$", zorder=1)
            ax1.set_xlim(t[0], t[-1])
            ax1.set_ylim(-1, 1)
            ax1.set_xticks([])
            ax1.set_ylabel("Input / Output", labelpad=20)
            ax1.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

            for i in range(x.shape[1]):
                ax2.plot(t, x[:, -i], label=r"$\hat{x}_%d(t)$" % (i + 1),
                         lw=2, alpha=0.8)
            ax2.set_xlim(t[0], t[-1])
            ax2.set_ylim(-1, 1)
            ax2.set_xticks([])
            ax2.set_ylabel("Decoded State", labelpad=20)
            ax2.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

            ax3.imshow(a.T, cmap=cmap, aspect='auto',
                       interpolation='none')
            ax3.set_xlim(0, len(t))
            ax3.set_xticklabels(np.linspace(0, T, 9))
            ax3.set_xlabel("Time [s]", labelpad=20)
            ax3.set_ylabel(r"Cell \#", labelpad=20)

            segs = 100
            lc = LineCollection(segs * [[(0, 0)]], lw=10,
                                colors=cmap(np.linspace(0, 1, segs)))
            ax3.legend([lc], ['Activity'], handlelength=2,
                       handler_map={type(lc): HandlerDashedLines()},
                       bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

            sns.despine(offset=15)

            savefig(targets[0])


def delay_readout(q, thetap, theta):
    # Normalized C matrix
    c = np.zeros(q)
    for i in range(q):
        j = np.arange(i+1, dtype=np.float64)
        c[i] += 1 / binom(q, i) * np.sum(
            binom(q, j) * binom(2*q - 1 - j, i - j) *
            (-thetap/theta)**(i - j))
    return c


def figure_basis_functions(targets):
    q = 12
    theta = 1.0

    thetas = np.linspace(0, theta, 1000)
    data = np.zeros((q, len(thetas)))

    for i, thetap in enumerate(thetas):
        data[:, i] = delay_readout(q, thetap, theta)

    cmap = sns.cubehelix_palette(len(data), light=0.7, reverse=True)

    with sns.axes_style('ticks'):
        with sns.plotting_context('paper', font_scale=2.8):
            pylab.figure(figsize=(18, 5))
            for i in range(len(data)):
                pylab.plot(thetas / theta, data[i], c=cmap[i],
                           lw=3, alpha=1.0, zorder=i)

            pylab.xlabel(r"$\theta' \times \theta^{-1}$ [s / s]", labelpad=20)
            pylab.ylabel(r"$w_{q - 1 - i}$")
            lc = LineCollection(len(cmap) * [[(0, 0)]], lw=10,
                                colors=cmap)
            pylab.legend([lc], [r"$i = 0 \ldots q - 1$"], handlelength=2,
                         handler_map={type(lc): HandlerDashedLines()},
                         bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

            sns.despine(offset=15)

            savefig(targets[0])


def figure_delay_full(targets):
    q = 12
    freq_times_theta = 1.0
    theta = 0.2  # affects resolution of plot alongside dt

    T = 10 * theta
    dt = 0.005
    seed = 5  # chosen to look "interesting"
    u = WhiteSignal(T, high=freq_times_theta/theta, seed=seed,
                    y0=0, rms=0.4).run(T, dt=dt)
    t = np.arange(0, T, dt)

    i = q - 1 - np.arange(q, dtype=np.float64)
    assert np.allclose(delay_readout(q, theta, theta),
                       (-1)**(q - 1 - i) * (i + 1) / q)

    num_thetas = 200
    num_freqs = 200
    props = np.linspace(0, 1.0, num_thetas)
    freqs = np.linspace(0, 10. / theta, num_freqs)
    s = 2.j * np.pi * freqs

    cmap = sns.diverging_palette(h_neg=34, h_pos=215, s=99, l=66, sep=1,
                                 center="dark", n=num_thetas)

    with sns.axes_style('ticks'):
        with sns.plotting_context('paper', font_scale=2.8):
            pylab.figure(figsize=(18, 5))
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.618])
            gs.update(wspace=0.3)
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])

            ax1.set_title(r"Temporal Coding Accuracy").set_y(1.05)
            ax2.set_title(
                r"Decoding at $\theta$ = Frequency$^{-1}$").set_y(1.05)

            for i, thetap in enumerate(props*theta):
                A, B, _, D = PadeDelay(theta, order=q).ss
                C = delay_readout(q, thetap, theta)[::-1]
                tf = LinearSystem((A, B, C, D))

                ax1.plot(freqs*theta, abs(np.exp(-s*thetap) - tf(s)),
                         c=cmap[i], alpha=0.7)
                ax2.plot(t / theta, tf.filt(u, dt=dt), c=cmap[i], alpha=0.5)

            s = 0.4
            pts = np.asarray([[freq_times_theta, 0],
                              [(1-s)*freq_times_theta, -s/2],
                              [(1+s)*freq_times_theta, -s/2]])
            ax1.add_patch(Polygon(pts, closed=True, color='black'))

            ax1.set_xlabel(r"Frequency $\times \, \theta$ [Hz $\times$ s]",
                           labelpad=20)
            ax1.set_ylabel(r"Absolute Error", labelpad=20)

            ax2.set_xlabel(r"Time $\times \, \theta^{-1}$ [s / s]",
                           labelpad=20)
            ax2.set_ylabel(r"Output")

            lc = LineCollection(len(cmap) * [[(0, 0)]], lw=10,
                                colors=cmap)
            ax2.legend([lc], [r"$\theta'$"], handlelength=2,
                       handler_map={type(lc): HandlerDashedLines()},
                       bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

            sns.despine(offset=15)

            savefig(targets[0])


def figure_time_cells(targets, order):
    npfile = np.load(TIME_CELLS_SIM % order)
    t = npfile['t']
    x = npfile['x']
    spikes = npfile['a']

    mus = []
    stds = []
    for i in range(spikes.shape[1]):
        spike_times = t[spikes[:, i] > 0]
        spike_times = spike_times[(spike_times >= 1)] - 1
        total = len(spike_times)
        if total > 0:
            within = len(np.where(spike_times < 4.784)[0])
            if within / float(total) >= 0.9:
                mus.append(np.mean(spike_times))
                stds.append(np.std(spike_times))
    logging.info("Remaining: %d", len(mus))
    logging.info("Pearson's correlation: %s", pearsonr(mus, stds))

    model = sm.OLS(stds, sm.add_constant(mus))
    res = model.fit()
    logging.info("Intercept / Slope: %s", res.params)
    logging.info("Standard errors: %s", res.bse)
    logging.info("p-values: %s", res.pvalues)

    sample_rate = 10
    t = t[::sample_rate]  # downsample PDFs to avoid reader lag
    x = x[::sample_rate]

    n_cells = 73
    intercept = -1

    rng = np.random.RandomState(seed=2)  # chosen to get a good spread
    encoders = []
    while len(encoders) < n_cells:
        e = sphere.sample(1, x.shape[1], rng=rng).squeeze()
        s = np.dot(x, e)
        if np.max(s) > intercept and 1.1 <= t[np.argmax(s)] <= 5.5:
            encoders.append(e)
    encoders = np.asarray(encoders)

    # Compute heat plot (right subfigure)
    a = np.dot(x[t >= 1.0], encoders.T)
    a = a.clip(intercept)
    a -= np.min(a, axis=0)
    assert np.all(np.sum(a, axis=0) > 0)
    normed_a = a / np.max(a, axis=0)
    assert np.allclose(np.min(normed_a, axis=0), 0)
    assert np.allclose(np.max(normed_a, axis=0), 1)

    t_peak = np.argmax(normed_a, axis=0)
    order = np.argsort(t_peak)
    heat = normed_a[:, order]

    # Compute cosine similarities (left subfigure)
    sorted_a = a[:, order]
    norms = np.linalg.norm(sorted_a, axis=1)
    assert np.isfinite(norms).all()
    cos = sorted_a.dot(sorted_a.T) / norms[:, None] / norms[None, :]
    cos -= np.min(cos)  # normalize the color map
    cos /= np.max(cos)

    cmap = 'jet'  # bad, but consistent with Zoran's figure

    with sns.axes_style('white'):
        with sns.plotting_context('paper', font_scale=3.0):
            aspect = 0.790
            t_ticks = ['', 0, 1, 2, 3, 4, '']

            pylab.figure(figsize=(22, 7.4))
            gs = gridspec.GridSpec(1, 2)
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])

            ax1.imshow(cos, cmap=cmap, aspect=aspect,
                       interpolation='none')
            ax1.set_xticklabels(t_ticks)
            ax1.set_yticklabels(t_ticks)
            ax1.set_xlabel("Time [s]", labelpad=20)
            ax1.set_ylabel("Time [s]", labelpad=20)
            ax1.set_anchor('S')

            ax2.imshow(heat.T, cmap=cmap,
                       aspect=aspect * heat.shape[0] / float(heat.shape[1]),
                       interpolation='none')
            ax2.set_xticklabels(t_ticks)
            ax2.set_xlabel("Time [s]", labelpad=20)
            ax2.set_ylabel(r"Cell \#", labelpad=20)
            ax2.set_anchor('S')

            savefig(targets[0])


def figure_pca(targets):
    orders = [3, 6, 9, 12, 15, 27]
    theta = 10.
    dt = 0.01
    T = theta

    length = int(T / dt)
    t = np.linspace(0, T-dt, length)
    t_norm = np.linspace(0, 1, len(t))
    cmap = sns.diverging_palette(h_neg=34, h_pos=215, s=99, l=66, sep=1,
                                 center="dark", as_cmap=True)

    class MidpointNormalize(colors.Normalize):
        """Stolen from http://matplotlib.org/users/colormapnorms.html"""

        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            # I'm ignoring masked values and all kinds of edge cases to make a
            # simple example...
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))

    with sns.axes_style('white'):
        with sns.plotting_context('paper', font_scale=2.8):
            pylab.figure(figsize=(22, 7))
            gs = gridspec.GridSpec(2, len(orders), height_ratios=[1.3, 1])

            for k, order in enumerate(orders):
                F = PadeDelay(theta, order)
                A = F.A
                dA, dB, _, _ = cont2discrete(F, dt=dt).ss

                dx = np.empty((length, len(F)))
                x = np.empty((length, len(F)))
                x[0, :] = dB.squeeze()  # the x0 from delta input

                for j in range(length-1):
                    dx[j, :] = A.dot(x[j, :])
                    x[j + 1, :] = dA.dot(x[j, :])
                dx[-1, :] = A.dot(x[-1, :])

                # Compute PCA of trajectory for top half
                pca = PCA(x, standardize=False)
                p = pca.Y[:, :3]

                logging.info("%d Accounted Variance: %s",
                             order, np.sum(pca.fracs[:3]) / np.sum(pca.fracs))

                # Compute curve for bottom half (and color map center)
                dist = np.cumsum(np.linalg.norm(dx, axis=1))
                dist = dist / np.max(dist)
                infl = np.where((np.diff(np.diff(dist)) >= 0) &
                                (t_norm[:-2] >= 0))[0][-1]
                cnorm = MidpointNormalize(midpoint=t_norm[infl])

                ax = plt.subplot(gs[k], projection='3d')
                ax.set_title(r"$q = %d$" % order).set_y(1.1)

                # Draw in reverse order so the start is on top
                ax.scatter(p[::-1, 0], p[::-1, 1], p[::-1, 2], lw=5,
                           c=t_norm[::-1], cmap=cmap, norm=cnorm, alpha=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                if k == 0:
                    ax.annotate('PCA', xy=(-50, 150), ha='left', va='top',
                                size=22, rotation=90, bbox=None,
                                xycoords='axes points')
                ax.view_init(elev=25, azim=150)

                ax = plt.subplot(gs[len(orders)+k])
                ax.scatter(t, dist, lw=5, c=t_norm, cmap=cmap, norm=cnorm)
                ax.vlines(t[infl], 0, 1, linestyle='--', lw=3, alpha=0.7)
                if k == 0:
                    ax.set_yticks([0, 1])
                    ax.set_ylabel("Length of Curve")
                else:
                    ax.set_yticks([])
                ax.set_xticks([0, theta/2, theta])
                ax.set_xticklabels(
                    [r"$0$", r"$\frac{\theta}{2}$", r"$\theta$"])
                ax.xaxis.set_tick_params(pad=10)
                ax.set_xlabel("Time [s]", labelpad=20)

                sns.despine(offset=10, ax=ax)

            savefig(targets[0])


def task_figure_basis_functions():
    return {
        'actions': [figure_basis_functions],
        'targets': ['figures/basis_functions.pdf'],
        'uptodate': [True],
    }


def task_simulate_lambert():
    return {
        'actions': [simulate_lambert],
        'targets': [LAMBERT_SIM],
        'uptodate': [True]
    }


def task_figure_lambert():
    return {
        'actions': [figure_lambert],
        'file_dep': [LAMBERT_SIM],
        'targets': ['figures/lambert.pdf'],
        'uptodate': [True],
    }


def task_figure_principle3():
    return {
        'actions': [figure_principle3],
        'file_dep': [DISCRETE_SIM % (seed, i)
                     for seed in range(DISCRETE_SEEDS)
                     for i in range(len(DISCRETE_DTS))],
        'targets': ['figures/principle3.pdf'],
        'uptodate': [True],
    }


def task_simulate_delay_example():
    return {
        'actions': [simulate_delay_example],
        'targets': [DELAY_EXAMPLE_SIM],
        'uptodate': [True]
    }


def task_figure_delay_example():
    return {
        'actions': [figure_delay_example],
        'file_dep': [DELAY_EXAMPLE_SIM],
        'targets': ['figures/delay_example.pdf'],
        'uptodate': [True],
    }


def task_simulate_discrete():
    def action(targets, seed, dt):
        delay, dt, t, stim, disc, cont = discrete_example(seed, dt)
        np.savez(targets[0], delay=delay, dt=dt, t=t,
                 stim=stim, disc=disc, cont=cont)
    for seed in range(DISCRETE_SEEDS):
        for i, dt in enumerate(DISCRETE_DTS):
            yield {
                'basename': 'simulate_discrete',
                'name': '%d_%d' % (seed, i),
                'actions': [(action, (), {'seed': seed, 'dt': dt})],
                'targets': [DISCRETE_SIM % (seed, i)],
                'uptodate': [True],
            }


def task_figure_delay_full():
    return {
        'actions': [figure_delay_full],
        'targets': ['figures/delay_full.pdf'],
        'uptodate': [True],
    }


def task_simulate_time_cells():
    for order in TIME_CELLS_ORDERS:
        yield {
            'basename': 'simulate_time_cells_%d' % order,
            'actions': [(simulate_time_cells, [], {'order': order})],
            'targets': [TIME_CELLS_SIM % order],
            'uptodate': [True],
        }


def task_figure_time_cells():
    order = 6
    return {
        'actions': [(figure_time_cells, [], {'order': order})],
        'file_dep': [TIME_CELLS_SIM % order],
        'targets': ['figures/time_cells.pdf'],
        'uptodate': [True],
    }


def task_figure_pca():
    return {
        'actions': [figure_pca],
        'targets': ['figures/pca.pdf'],
        'uptodate': [True],
    }
